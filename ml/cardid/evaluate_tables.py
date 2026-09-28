"""Shared manifest/scoring helpers for the table detector's training and export scripts.

`load_scenes` and `read_scene_image` read a `table_scenes` manifest and its JPEGs;
`per_card_hits` is the greedy IoU matching `train_table_detector.evaluate_val` uses for its
per-epoch recall/precision, and `export_table_detector.verify` for comparing the exported ONNX
graph against the torch reference. Kept deliberately small: this project trains one detector
(`table_detector.TableCenterNet`, ImageNet-pretrained backbone) rather than comparing several,
so there is no strategy-comparison harness here to maintain.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .scene_geometry import quad_iou


def load_scenes(manifest: Path, split: str | None = None) -> list[dict]:
    """Manifest rows (optionally filtered to one split), each with an added absolute
    `_image_path`. Pixels are loaded lazily by the caller, not here, so a frozen test
    manifest with thousands of scenes does not have to fit in memory at once."""
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if split is not None and row.get("split") != split:
            continue
        row["_image_path"] = manifest.parent / row["image"]
        rows.append(row)
    return rows


def read_scene_image(row: dict) -> np.ndarray:
    image = cv2.imread(str(row["_image_path"]))
    if image is None:
        raise FileNotFoundError(row["_image_path"])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def per_card_hits(truth: list[np.ndarray], found: list[np.ndarray], threshold: float = 0.5) -> list[bool]:
    """Whether each truth card (by index) was claimed by some found quad, greedy by IoU."""
    pairs = sorted(((quad_iou(t, f), i, j) for i, t in enumerate(truth) for j, f in enumerate(found)), reverse=True)
    used_truth, used_found, hit = set(), set(), [False] * len(truth)
    for score, i, j in pairs:
        if score < threshold or i in used_truth or j in used_found:
            continue
        used_truth.add(i)
        used_found.add(j)
        hit[i] = True
    return hit
