"""Slice detection recall by card size (short side as a fraction of frame width) and by scene
density, using an existing table-scenes test split -- no new data needed, just a different cut
of what `test`/`challenge` already have.

    uv run python -m cardid.size_density_analysis --checkpoint data/runs/<run>/best.pt --manifest-dir ~/the-gathering-cardid/table-scenes
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import to_tensor
from .evaluate_tables import load_scenes, per_card_hits, read_scene_image
from .table_detector import TABLE_STRIDE, TableCenterNet, decode_detections

SIZE_BUCKETS = (("<1/17", 0, 1 / 17), ("1/17-1/14", 1 / 17, 1 / 14), ("1/14-1/10", 1 / 14, 1 / 10), ("1/10-1/7", 1 / 10, 1 / 7), (">1/7", 1 / 7, 1.0))


def size_bucket(width_frac: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= width_frac < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenes", type=int, default=150)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()

    model = TableCenterNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    rows = load_scenes(args.manifest_dir / args.split / "manifest.jsonl", args.split)[: args.scenes]
    by_size: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_density: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_count: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for row in rows:
            image = read_scene_image(row)
            heat, pose, up = model(to_tensor(image)[None])
            detections = decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, args.score_threshold)
            found = [q for q, _s in detections]
            truth = [np.float32(c["quad"]) for c in row["cards"]]
            hits = per_card_hits(truth, found, 0.5)
            n = len(truth)
            for card, hit in zip(row["cards"], hits, strict=True):
                # card short side / frame width, from the quad directly (not the nominal camera profile)
                quad = np.float32(card["quad"])
                w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
                h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
                width_frac = min(w, h) / row["width"]
                bucket = by_size[size_bucket(width_frac)]
                bucket[0] += hit
                bucket[1] += 1
                dbucket = by_density[row.get("density", "?")]
                dbucket[0] += hit
                dbucket[1] += 1
            cbucket = by_count[n]
            cbucket[0] += sum(hits)
            cbucket[1] += n

    def report(title: str, buckets: dict) -> None:
        print(f"\n{title}")
        for key in sorted(buckets, key=lambda k: (isinstance(k, str), k)):
            hit, total = buckets[key]
            print(f"  {key}: recall {hit / total:.3f} (n={total})" if total else f"  {key}: n=0")

    report("By card width as a fraction of frame width", by_size)
    report("By scene density label", by_density)
    report("By exact card count in the scene", by_count)


if __name__ == "__main__":
    main()
