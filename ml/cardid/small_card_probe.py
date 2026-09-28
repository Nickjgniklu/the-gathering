"""Probe recall at card sizes smaller than any existing camera profile covers (`table_scenes`'s
smallest, `angled_720p`, only reaches 1/14 of frame width; a card held at typical webcam
distance can be 1/17 or smaller). Renders a dedicated eval set at an explicit `short_frac` range
via `render_table_scene` directly, without adding a permanent camera profile until this shows
whether one is actually needed.

    uv run python -m cardid.small_card_probe --checkpoint data/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import to_tensor
from .evaluate_tables import per_card_hits
from .image_bank import ArtBank, CardBank, list_arts
from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet, decode_detections
from .table_scenes import SETUPS, render_table_scene

# 1/20 to 1/14 of frame width, below every existing camera profile's short_frac range.
SMALL_PROFILE = {"short_frac": (0.05, 1 / 14), "severity": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--cards-per-scene", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()

    model = TableCenterNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    cards, arts = CardBank(), ArtBank(list_arts())

    hits_total, truth_total, found_total = 0, 0, 0
    buckets: dict[str, list[int]] = {"1/20-1/17": [0, 0], "1/17-1/14": [0, 0]}
    with torch.no_grad():
        for i in range(args.scenes):
            setup = SETUPS[i % len(SETUPS)]
            count = min(args.cards_per_scene, len(cards))
            image, record = render_table_scene_small(args.seed + i, cards, arts, setup, count)
            heat, pose, up = model(to_tensor(image)[None])
            detections = decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, args.score_threshold)
            found = [q for q, _s in detections]
            truth = [np.float32(c["quad"]) for c in record["cards"]]
            hits = per_card_hits(truth, found, 0.5)
            hits_total += sum(hits)
            truth_total += len(hits)
            found_total += len(found)
            for card, hit in zip(record["cards"], hits, strict=True):
                quad = np.float32(card["quad"])
                w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
                h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
                frac = min(w, h) / TABLE_INPUT
                key = "1/20-1/17" if frac < 1 / 17 else "1/17-1/14"
                buckets[key][0] += hit
                buckets[key][1] += 1

    print(f"overall: recall {hits_total / truth_total:.3f} precision {hits_total / found_total:.3f} (n_truth={truth_total})")
    for key, (hit, total) in buckets.items():
        print(f"  {key}: recall {hit / total:.3f} (n={total})" if total else f"  {key}: n=0")


def render_table_scene_small(seed, cards, arts, setup, count):
    """`render_table_scene` with `SMALL_PROFILE` spliced in as a one-off camera profile."""
    from . import table_scenes

    table_scenes.CAMERA_PROFILES["_small_probe"] = SMALL_PROFILE
    try:
        return render_table_scene(seed, cards, arts, setup, "_small_probe", count=count, size=1280, out=TABLE_INPUT)
    finally:
        del table_scenes.CAMERA_PROFILES["_small_probe"]


if __name__ == "__main__":
    main()
