"""Export `TableCenterNet` (Strategy A, the dense multi-card detector) to a standalone ONNX
artifact for the rest of the app to consume, mirroring `export.py`'s conventions (opset 17,
`manifest.json` with per-file sha256, a `SHA256SUMS` file) without touching the existing
click-conditioned recogniser bundle that script owns -- Super AI mode's frame transport and
worker plumbing are separate, still-unbuilt work (see `.amp/in/super-ai-mode-plan.md`); this
just hands off a verified, documented model for that work to consume.

    uv run python -m cardid.export_table_detector --checkpoint data/runs/table-a-pretrained/best.pt

writes data/table-detector-exports/<version>/ (version defaults to <UTC timestamp>-<run name>):

    manifest.json        version, source checkpoint + its training history, input/output
                          contract, per-file sha256
    table_detector.onnx  table (TABLE_INPUT, TABLE_INPUT, 4) uint8 RGBA ->
                          quads (max_detections, 4, 2) input px, scores (max_detections,)
    SHA256SUMS

Input/output contract for whoever wires this into the app:

- Input: an RGBA HWC uint8 image, exactly `TABLE_INPUT` x `TABLE_INPUT` (384 by default) --
  resize/letterbox the camera frame to that size first; the model's notion of "how big a card
  looks" was learned at that resolution, so feeding a different size silently hurts accuracy
  even though the graph itself (fully convolutional) would still run.
- Output: exactly `max_detections` (quad, score) pairs, sorted by descending score, in the
  input image's own pixel coordinates, in printed order (corner 0 is the card's top-left).
  Real cards run out before `max_detections` does on any normal table, so the low-score tail is
  padding -- apply your own confidence threshold (0.3 is what training/evaluation here uses).
  This graph's peaks are already CenterNet-style NMS-free, but two adjacent cells can still both
  clear threshold, so a small IoU-based dedup on your side is worth keeping if you see doubles.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from . import DATA_DIR
from .evaluate_tables import load_scenes, read_scene_image
from .export import export_graph, sha256, write_sums
from .graphs import TableDetectorGraph
from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet

EXPORT_DIR = DATA_DIR / "table-detector-exports"
OPSET = 17
MAX_DETECTIONS = 40
SCORE_THRESHOLD = 0.3


def default_version(checkpoint: Path, now: datetime | None = None) -> str:
    return f"{now or datetime.now(UTC):%Y-%m-%dT%H%M%SZ}-{checkpoint.parent.name}"


def export(checkpoint: Path, out: Path, max_detections: int = MAX_DETECTIONS, input_size: int = TABLE_INPUT) -> TableCenterNet:
    out.mkdir(parents=True, exist_ok=True)
    net = TableCenterNet(pretrained=False)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    with torch.no_grad():
        rng = np.random.default_rng(0)
        table = torch.from_numpy(rng.integers(0, 256, (input_size, input_size, 4), dtype=np.uint8))
        export_graph(TableDetectorGraph(net, TABLE_STRIDE, max_detections), (table,), out / "table_detector.onnx", ["table"], ["quads", "scores"])

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
            "input": {"name": "table", "shape": [input_size, input_size, 4], "dtype": "uint8", "layout": "HWC RGBA"},
            "outputs": [
                {"name": "quads", "shape": [max_detections, 4, 2], "dtype": "float32", "units": "input pixels, printed order (corner 0 = top-left)"},
                {"name": "scores", "shape": [max_detections], "dtype": "float32", "units": "sigmoid probability, descending; apply your own threshold"},
            ],
            "table_input": input_size,
            "table_stride": TABLE_STRIDE,
            "max_detections": max_detections,
            "recommended_score_threshold": SCORE_THRESHOLD,
        },
        "opset": OPSET,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_sums(out, manifest)
    total = sum(f["bytes"] for f in files.values())
    print(f"export {out} ({total / 1e6:.1f} MB)")
    return net


def verify(out: Path, net: TableCenterNet, manifest_dir: Path, n: int, score_threshold: float, input_size: int = TABLE_INPUT) -> bool:
    """Compare the exported graph (onnxruntime) against the torch reference
    (`table_detector.decode_detections`) on `n` real `test`-split scenes: same preprocessing
    (resize to `input_size`, the size the graph was actually exported for -- not necessarily
    `TABLE_INPUT`, since `--input-size` can override it), same score threshold, greedy IoU
    matching. The two can differ slightly from onnxruntime's own kernel implementations
    (bilinear resize, argmax tie order), so this checks they agree closely rather than
    bit-for-bit."""
    import cv2
    import onnxruntime as ort

    from .scene_geometry import quad_iou
    from .table_detector import decode_detections

    session = ort.InferenceSession(str(out / "table_detector.onnx"), providers=["CPUExecutionProvider"])
    rows = load_scenes(manifest_dir / "test" / "manifest.jsonl", "test")[:n]
    if not rows:
        raise SystemExit(f"no test scenes in {manifest_dir}")
    matched, torch_total, onnx_total, corner_err = 0, 0, 0, []
    started = time.time()
    for row in rows:
        image = read_scene_image(row)
        scale = input_size / row["width"]
        resized = cv2.resize(image, (input_size, input_size))
        rgba = np.concatenate([resized, np.full((*resized.shape[:2], 1), 255, np.uint8)], axis=-1)
        with torch.no_grad():
            heat, pose, up = net(_to_input(resized))
        torch_dets = [(quad, score) for quad, score in decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold) if score >= score_threshold]
        (onnx_quads, onnx_scores) = session.run(None, {"table": rgba})
        onnx_dets = [(q, float(s)) for q, s in zip(onnx_quads, onnx_scores, strict=True) if s >= score_threshold]
        torch_total += len(torch_dets)
        onnx_total += len(onnx_dets)
        used = set()
        for q, _s in onnx_dets:
            best_iou, best_i = 0.0, -1
            for i, (tq, _ts) in enumerate(torch_dets):
                if i in used:
                    continue
                iou = quad_iou(q, tq)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou > 0.5:
                used.add(best_i)
                matched += 1
                corner_err.append(float(np.abs(q - torch_dets[best_i][0]).max()) / scale)
    elapsed = time.time() - started
    corner_err = np.array(corner_err) if corner_err else np.array([0.0])
    print(
        f"verify on {len(rows)} scenes ({elapsed:.1f}s): torch {torch_total} dets, onnx {onnx_total} dets, {matched} matched "
        f"(corner err max {corner_err.max():.2f}px median {np.median(corner_err):.2f}px, in manifest-pixel units)"
    )
    ok = torch_total > 0 and matched / torch_total > 0.95 and float(np.median(corner_err)) < 2.0
    print("verify: OK" if ok else "verify: FAILED (onnx graph diverges from the torch pipeline)")
    return ok


def _to_input(rgb_uint8: np.ndarray) -> torch.Tensor:
    from .data import to_tensor

    return to_tensor(rgb_uint8)[None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--version", help="export name under data/table-detector-exports (default: <UTC timestamp>-<run name>)")
    parser.add_argument("--out", type=Path, help="export directory (overrides --version)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing export directory")
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    parser.add_argument("--input-size", type=int, default=TABLE_INPUT)
    parser.add_argument("--verify", type=int, default=60, help="test scenes to compare against the torch pipeline (0 to skip)")
    parser.add_argument("--verify-manifest-dir", type=Path, help="table-scenes dataset directory for --verify (needs a test/ split)")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args()

    version = args.version or default_version(args.checkpoint)
    out = args.out or EXPORT_DIR / version
    if (out / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --version, --out, or --force")
    net = export(args.checkpoint, out, args.max_detections, args.input_size)
    if args.verify:
        if not args.verify_manifest_dir:
            raise SystemExit("--verify needs --verify-manifest-dir (a table_scenes dataset with a test/ split)")
        if not verify(out, net, args.verify_manifest_dir, args.verify, args.score_threshold, args.input_size):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
