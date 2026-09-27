"""Score the table detector on each hardware-stress category from `hardware_stress.py generate`,
against the same scenes' clean (un-stressed) baseline. Kept separate so `hardware_stress.py`'s
generation side has no torch dependency.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import torch

from .data import to_tensor
from .evaluate_tables import per_card_hits
from .hardware_stress import STRESS_DIR
from .table_detector import TABLE_STRIDE, TableCenterNet, decode_detections


@torch.no_grad()
def score_rows(model: TableCenterNet, rows: list[dict], score_threshold: float, samples: int, category: str) -> dict:
    hits_total, truth_total, found_total = 0, 0, 0
    for i, row in enumerate(rows):
        image = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        heat, pose, up = model(to_tensor(image)[None])
        detections = decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold)
        found = [q for q, _s in detections]
        truth = [np.float32(c["quad"]) for c in row["cards"]]
        hits = per_card_hits(truth, found, 0.5)
        hits_total += sum(hits)
        truth_total += len(hits)
        found_total += len(found)
        if i < samples:
            canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
            for quad in truth:
                cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
            for quad in found:
                cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 90, 255), 2, cv2.LINE_AA)
            out_dir = STRESS_DIR / "samples"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{category}-{i:02d}.jpg"), canvas)
    return {
        "recall": hits_total / truth_total if truth_total else None,
        "precision": hits_total / found_total if found_total else None,
        "n_truth": truth_total,
        "n_found": found_total,
    }


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = TableCenterNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    manifest = json.loads((STRESS_DIR / "manifest.json").read_text())
    by_category = defaultdict(list)
    for row in manifest:
        by_category[row["category"]].append(row)

    results = {}
    for category in sorted(by_category):
        results[category] = score_rows(model, by_category[category], args.score_threshold, args.samples, category)
        print(f"{category:20s} (n={len(by_category[category])}): {json.dumps(results[category])}")
    (STRESS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"-> {STRESS_DIR / 'results.json'}")
