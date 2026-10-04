"""Export `TrackMemory` (the Stage 2 GRU that smooths per-frame card identification across a
video stream, see `track_memory.py`'s module docstring) to a standalone ONNX artifact, mirroring
`export_table_detector.py`'s conventions (opset 17, `manifest.json` with per-file sha256, a
`SHA256SUMS` file).

    uv run python -m cardid.export_track_memory --checkpoint data/runs/track-memory-c/best.pt

writes data/track-memory-exports/<version>/ (version defaults to <UTC timestamp>-<run name>):

    manifest.json      version, source checkpoint + its training history, input/output contract
    track_memory.onnx  (hidden, embedding, pose, score) -> (new_hidden, refined_embedding, refined_pose)
    SHA256SUMS

Input/output contract for whoever wires this into an app:

- Inputs, one timestep at a time, per tracked card: `hidden` (HIDDEN_DIM=128 float32, zero vector
  for a brand-new track), `embedding` (128 float32, L2-normalized -- the detector/embedder's own
  per-frame best guess for this track, see `track_memory_dataset.py`'s `_best_frame_embedding` for
  how that's picked out of the embedder's 14 frame-hypothesis outputs; do not feed a raw multi-
  frame-hypothesis tensor here), `pose` (5 float32: cx, cy, short, cos(2*angle), sin(2*angle), in
  the same units `detect_and_embed.DetectAndEmbed.forward(..., return_pose=True)` returns),
  `score` (1 float32, the detector's own confidence for this slot, sigmoid-space).
- Outputs: `new_hidden` (128 float32, feed back in as next frame's `hidden` for this same track --
  carrying it forward IS the smoothing; a lost/new track restarts from a zero vector, not the old
  one), `refined_embedding` (128 float32, L2-normalized -- search the gallery against THIS, not
  the raw per-frame embedding), `refined_pose` (5 float32, same layout as the input pose).
- This graph has no batch dimension and processes one track's one timestep per call; batch
  multiple tracks yourself (stack along a leading dimension before calling, matching how training
  unrolled -- see `train_track_memory.py`'s `step_sequence`) if per-call overhead matters at your
  scale, or just call it once per track per frame if it doesn't (it's a `GRUCell` plus two small
  `Linear` layers -- a few microseconds of real compute either way).
- Track correspondence frame-to-frame (which of this frame's detections is "the same card" as a
  track from last frame) is NOT this graph's job -- that is `track_memory.associate_detections`
  (classical nearest-position matching, not exported since it's not a neural net: port that logic
  directly, it's about a dozen lines).
- User corrections ("this detected card is really Y") are also not part of this graph by design --
  see `track_identity.py` (`TrackIdentity`/`apply_correction`/`step`), a small, dependency-free
  state machine meant to sit between your detector output and this graph's `embedding` input, not
  inside the trained weights.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from . import DATA_DIR
from .export import export_graph, sha256, write_sums
from .track_memory import HIDDEN_DIM, TrackMemory

EXPORT_DIR = DATA_DIR / "track-memory-exports"
OPSET = 17
EMBED_DIM = 128
POSE_DIM = 5


def default_version(checkpoint: Path, now: datetime | None = None) -> str:
    return f"{now or datetime.now(UTC):%Y-%m-%dT%H%M%SZ}-{checkpoint.parent.name}"


def export(checkpoint: Path, out: Path) -> TrackMemory:
    out.mkdir(parents=True, exist_ok=True)
    model = TrackMemory(embed_dim=EMBED_DIM, pose_dim=POSE_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        rng = torch.Generator().manual_seed(0)
        hidden = torch.randn(1, HIDDEN_DIM, generator=rng)
        embedding = F.normalize(torch.randn(1, EMBED_DIM, generator=rng), dim=-1)
        pose = torch.randn(1, POSE_DIM, generator=rng)
        score = torch.rand(1, 1, generator=rng)
        export_graph(
            model,
            (hidden, embedding, pose, score),
            out / "track_memory.onnx",
            ["hidden", "embedding", "pose", "score"],
            ["new_hidden", "refined_embedding", "refined_pose"],
        )

    history_path = checkpoint.parent / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    run_path = checkpoint.parent / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.iterdir()) if p.name not in ("manifest.json", "SHA256SUMS")}
    manifest = {
        "version": out.name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "run_args": run.get("args"),
            "training_history": history[-1] if history else None,
        },
        "contract": {
            "inputs": [
                {"name": "hidden", "shape": [HIDDEN_DIM], "dtype": "float32", "notes": "zero vector for a new track"},
                {"name": "embedding", "shape": [EMBED_DIM], "dtype": "float32", "notes": "L2-normalized, the detector's own per-frame best guess (see module docstring)"},
                {"name": "pose", "shape": [POSE_DIM], "dtype": "float32", "notes": "cx, cy, short, cos(2*angle), sin(2*angle)"},
                {"name": "score", "shape": [1], "dtype": "float32", "notes": "detector confidence, sigmoid-space"},
            ],
            "outputs": [
                {"name": "new_hidden", "shape": [HIDDEN_DIM], "dtype": "float32", "notes": "feed back in as next frame's hidden for this track"},
                {"name": "refined_embedding", "shape": [EMBED_DIM], "dtype": "float32", "notes": "L2-normalized, search the gallery against this"},
                {"name": "refined_pose", "shape": [POSE_DIM], "dtype": "float32", "notes": "same layout as the input pose"},
            ],
            "hidden_dim": HIDDEN_DIM,
            "no_batch_dimension": True,
            "needs_separately": [
                "track_memory.associate_detections (frame-to-frame track correspondence, classical, not exported)",
                "track_identity.TrackIdentity (user corrections, plain Python, not exported)",
            ],
        },
        "opset": OPSET,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_sums(out, manifest)
    total = sum(f["bytes"] for f in files.values())
    print(f"export {out} ({total / 1e6:.2f} MB)")
    return model


def verify(out: Path, model: TrackMemory, steps: int, seed: int) -> bool:
    """Compare the exported graph (onnxruntime) against the torch reference across a short
    random unrolled sequence (feeding each call's `new_hidden` back in as the next call's
    `hidden`, the real usage pattern -- a single-call check wouldn't catch a state-threading
    mistake) rather than one-shot random inputs."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(out / "track_memory.onnx"), providers=["CPUExecutionProvider"])
    rng = torch.Generator().manual_seed(seed)
    torch_hidden = torch.zeros(1, HIDDEN_DIM)
    onnx_hidden = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
    max_embed_diff = max_pose_diff = 0.0
    with torch.no_grad():
        for _ in range(steps):
            embedding = F.normalize(torch.randn(1, EMBED_DIM, generator=rng), dim=-1)
            pose = torch.randn(1, POSE_DIM, generator=rng)
            score = torch.rand(1, 1, generator=rng)

            torch_hidden, torch_embedding, torch_pose = model(torch_hidden, embedding, pose, score)
            onnx_hidden, onnx_embedding, onnx_pose = session.run(
                None, {"hidden": onnx_hidden, "embedding": embedding.numpy(), "pose": pose.numpy(), "score": score.numpy()}
            )
            max_embed_diff = max(max_embed_diff, float(np.abs(torch_embedding.numpy() - onnx_embedding).max()))
            max_pose_diff = max(max_pose_diff, float(np.abs(torch_pose.numpy() - onnx_pose).max()))
            onnx_hidden = onnx_hidden.astype(np.float32)

    print(f"verify over {steps} unrolled steps: max |diff| embedding {max_embed_diff:.2e}, pose {max_pose_diff:.2e}")
    ok = max_embed_diff < 1e-4 and max_pose_diff < 1e-4
    print("verify: OK" if ok else "verify: FAILED (onnx graph diverges from the torch pipeline)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--version", help="export name under data/track-memory-exports (default: <UTC timestamp>-<run name>)")
    parser.add_argument("--out", type=Path, help="export directory (overrides --version)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing export directory")
    parser.add_argument("--verify", type=int, default=50, help="unrolled random steps to compare against the torch pipeline (0 to skip)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    version = args.version or default_version(args.checkpoint)
    out = args.out or EXPORT_DIR / version
    if (out / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --version, --out, or --force")
    model = export(args.checkpoint, out)
    if args.verify and not verify(out, model, args.verify, args.seed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
