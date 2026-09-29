"""End-to-end evaluation of `DetectAndEmbed`: detection recall plus identification accuracy
against the real production gallery, on a gallery-verified synthetic dataset (every card's
identity is guaranteed searchable -- see `regen_gallery_verified.py` for why that matters).

Extracts the gallery's raw (embeddings, frames, penalties) directly from a deployed bundle's
`search.onnx` initializers rather than needing a torch `ArtIndex`/ImageBank rebuild, so this
needs no local art-image cache -- only the bundle directory (`arts.json` + `search.onnx`).

Two ways to identify a query embedding against the gallery in this script, both simplifications
of the real `SearchGraph` (`graphs.py`) because `DetectAndEmbed` only produces one (modern-frame)
embedding per card, not the 14 frame hypotheses `search.onnx` expects (see
`detect-and-embed-guide.md`'s "frame-hypothesis gotcha"):

- `--frame-split` (default): report modern-frame and non-modern-frame ground-truth cards
  separately. The modern-frame number is a fair, correct measurement (this module's crop
  assumption matches those cards' actual frame); the non-modern number is expected lower and
  exists to quantify the gap, not to be read as this module's real accuracy on those cards.

    uv run python -m cardid.evaluate_detect_and_embed --detector single-pass --table-checkpoint data/runs/repro-a-hardneg-v4/best.pt --embed-checkpoint data/runs/recogniser-cfbender-oracle/best.pt --native-size 384 --manifest-dir H:\the-gathering-cardid\table-scenes-1920 --gallery-bundle H:\the-gathering-cardid\current
    uv run python -m cardid.evaluate_detect_and_embed --detector tiled-fusion --table-checkpoint data/runs/repro-a-hardneg-v4/best.pt --fusion-checkpoint data/runs/tiled-fusion-1920/best.pt --embed-checkpoint data/runs/recogniser-cfbender-oracle/best.pt --native-size 1920 --score-threshold 0.16 --manifest-dir H:\the-gathering-cardid\table-scenes-1920 --gallery-bundle H:\the-gathering-cardid\current
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnx import numpy_helper

from .data import to_tensor
from .detect_and_embed import DetectAndEmbed
from .evaluate_tables import load_scenes, per_card_hits
from .tiled_fusion import TiledFusionDetector


def load_gallery(bundle_dir: Path) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    """A deployed bundle's gallery, read straight out of `search.onnx`'s constants -- no torch
    checkpoint or local art-image cache needed. Returns (arts, embeddings (N,128), frames (N,),
    penalties (N,)), frames/penalties already aligned 1:1 with gallery rows (not indexed by
    frame-type -- `frame_penalties` in `detect.py` already expands per-art before export)."""
    arts = json.loads((bundle_dir / "arts.json").read_text(encoding="utf-8"))
    m = onnx.load(str(bundle_dir / "search.onnx"))
    tensors = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
    embeddings = torch.from_numpy(tensors["gallery"].astype(np.float32))
    frames = torch.from_numpy(tensors["frames"].flatten().astype(np.int64).copy())
    penalties = torch.from_numpy(tensors["penalties"].copy())
    return arts, embeddings, frames, penalties


def build_model(args: argparse.Namespace) -> DetectAndEmbed:
    if args.detector == "single-pass":
        return DetectAndEmbed(
            table_checkpoint=args.table_checkpoint, embed_checkpoint=args.embed_checkpoint, native_size=args.native_size
        ).eval()
    tiled = TiledFusionDetector(checkpoint=args.table_checkpoint, native_size=args.native_size)
    tiled.fusion.load_state_dict(torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=True))
    return DetectAndEmbed(detector=tiled, embed_checkpoint=args.embed_checkpoint, native_size=args.native_size).eval()


def run(args: argparse.Namespace) -> None:
    arts, gallery, frames, penalties = load_gallery(args.gallery_bundle)
    by_id = {a["id"]: a for a in arts}
    model = build_model(args)

    root = args.manifest_dir / "train"
    rows = load_scenes(root / "manifest.jsonl", "train")[: args.scenes]

    detected, truth_n = 0, 0
    tally = {"modern": [0, 0, 0], "other": [0, 0, 0]}  # top1, top5, n

    for row in rows:
        img = cv2.cvtColor(cv2.imread(str(root / row["image"])), cv2.COLOR_BGR2RGB)
        scale = args.native_size / row["width"]
        small = cv2.resize(img, (args.native_size, args.native_size), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        with torch.no_grad():
            embeddings, scores, quads = model(to_tensor(small).unsqueeze(0))
        embeddings, scores, quads = embeddings[0], scores[0], quads[0]

        truth_quads = [np.float32(c["quad"]) * scale for c in row["cards"]]
        truth_n += len(truth_quads)
        found = [q.numpy() for q, s in zip(quads, scores) if s > args.score_threshold]
        detected += sum(per_card_hits(truth_quads, found, 0.5))

        for quad, score, emb in zip(quads, scores, embeddings):
            if score <= args.score_threshold:
                continue
            match = next((i for i, tq in enumerate(truth_quads) if per_card_hits([tq], [quad.numpy()], 0.5)[0]), None)
            if match is None:
                continue
            true_meta = by_id.get(row["cards"][match]["card_id"])
            if true_meta is None:
                continue  # shouldn't happen on a gallery-verified dataset

            sims = gallery @ emb - penalties
            top5 = [arts[i]["name"] for i in torch.topk(sims, 5).indices.tolist()]
            bucket = tally["modern"] if true_meta.get("frame") == "modern" else tally["other"]
            bucket[0] += top5[0] == true_meta["name"]
            bucket[1] += true_meta["name"] in top5
            bucket[2] += 1

    print(f"detection recall (IoU@0.5): {detected}/{truth_n} = {detected / truth_n:.3f}")
    for name, (top1, top5, n) in tally.items():
        label = "modern-frame" if name == "modern" else "non-modern-frame (expected lower, see module docstring)"
        print(f"{label}: top1={top1}/{n}={top1 / max(n, 1):.3f}  top5={top5}/{n}={top5 / max(n, 1):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["single-pass", "tiled-fusion"], required=True)
    parser.add_argument("--table-checkpoint", type=Path, required=True, help="frozen TableCenterNet weights (both detector modes need this)")
    parser.add_argument("--fusion-checkpoint", type=Path, help="tiled-fusion only: the trained FusionHead")
    parser.add_argument("--embed-checkpoint", type=Path, required=True)
    parser.add_argument("--native-size", type=int, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3, help="0.3 for single-pass; 0.15-0.16 for tiled-fusion-1920 (see tiled_fusion.py)")
    parser.add_argument("--manifest-dir", type=Path, required=True, help="a gallery-verified table-scenes dataset (see regen_gallery_verified.py)")
    parser.add_argument("--gallery-bundle", type=Path, required=True, help="a deployed bundle directory (arts.json + search.onnx)")
    parser.add_argument("--scenes", type=int, default=10)
    args = parser.parse_args()
    if args.detector == "tiled-fusion" and not args.fusion_checkpoint:
        parser.error("--detector tiled-fusion needs --fusion-checkpoint")
    run(args)


if __name__ == "__main__":
    main()
