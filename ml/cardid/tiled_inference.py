"""Split a large frame into overlapping tiles, run the existing single-input detector on each at
full resolution, and deduplicate detections in the overlap -- an inference-time-only change that
needs no retraining, to test whether downsampling a whole table into one `TABLE_INPUT`-sized pass
is what actually costs recall on small/distant cards (`size_density_analysis.py` found recall
collapses below 1/14 of frame width), rather than the model itself.

A card at 1/17 of a 4032px-wide real frame is ~237px wide; letterboxed and downsampled to a single
384px pass, that is ~23px, barely 5-6 cells across the stride-4 grid. Split into a 2x2 grid
instead, the same card is ~35px across each 384px tile before its own downsample -- roughly 1.5x
the effective resolution per card, without touching the model.

    uv run python -m cardid.tiled_inference compare --checkpoint data/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from . import DATA_DIR
from .confusion_matrix import letterbox_to_square
from .data import to_tensor
from .scene_geometry import quad_iou
from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet, decode_detections


def tile_boxes(width: float, height: float, grid: tuple[int, int], overlap: float) -> list[tuple[float, float, float, float]]:
    """Axis-aligned (x0, y0, x1, y1) tile boxes covering `width` x `height`, `overlap` as a
    fraction of one tile's own size shared with its neighbour (e.g. 0.2 means each tile repeats
    20% of its width/height with the next one, enough that a card near a seam sits fully inside
    at least one tile as long as it is not wider than that overlap)."""
    rows, cols = grid
    tile_w = width / (cols - (cols - 1) * overlap) if cols > 1 else width
    tile_h = height / (rows - (rows - 1) * overlap) if rows > 1 else height
    step_x = tile_w * (1 - overlap) if cols > 1 else 0.0
    step_y = tile_h * (1 - overlap) if rows > 1 else 0.0
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * step_x, r * step_y
            boxes.append((x0, y0, min(x0 + tile_w, width), min(y0 + tile_h, height)))
    return boxes


def dedupe(detections: list[tuple[np.ndarray, float]], iou_threshold: float = 0.4) -> list[tuple[np.ndarray, float]]:
    """Greedy NMS by score: the same card detected in two overlapping tiles keeps only its
    highest-scoring box."""
    ordered = sorted(detections, key=lambda d: d[1], reverse=True)
    kept: list[tuple[np.ndarray, float]] = []
    for quad, score in ordered:
        if not any(quad_iou(quad, k) > iou_threshold for k, _ in kept):
            kept.append((quad, score))
    return kept


@torch.no_grad()
def detect_tiled(
    model: TableCenterNet,
    image: np.ndarray,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.2,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.4,
    input_size: int = TABLE_INPUT,
) -> list[tuple[np.ndarray, float]]:
    """`image` is a full-resolution (already square, e.g. via `letterbox_to_square`) RGB frame.
    Runs the model once per tile and returns deduplicated (quad, score) pairs in `image`'s own
    pixel coordinates, the same shape `decode_detections` would give for a single pass."""
    h, w = image.shape[:2]
    all_detections: list[tuple[np.ndarray, float]] = []
    for x0, y0, x1, y1 in tile_boxes(w, h, grid, overlap):
        ix0, iy0, ix1, iy1 = round(x0), round(y0), round(x1), round(y1)
        crop = image[iy0:iy1, ix0:ix1]
        ch, cw = crop.shape[:2]
        resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
        heat, pose, up = model(to_tensor(resized)[None])
        for quad, score in decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold):
            full_quad = quad * np.float32([cw / input_size, ch / input_size]) + np.float32([ix0, iy0])
            all_detections.append((full_quad, score))
    return dedupe(all_detections, iou_threshold)


@torch.no_grad()
def detect_multiscale(
    model: TableCenterNet,
    image: np.ndarray,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.2,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.4,
    input_size: int = TABLE_INPUT,
) -> list[tuple[np.ndarray, float]]:
    """A whole-frame single pass (catches cards already at a comfortable trained scale; tiling
    alone over-zooms these past the model's trained size range and can lose them) plus the tiled
    passes (catches cards too small for one whole-frame downsample), deduplicated together."""
    h, w = image.shape[:2]
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    heat, pose, up = model(to_tensor(resized)[None])
    whole = [
        (quad * np.float32([w / input_size, h / input_size]), score)
        for quad, score in decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold)
    ]
    tiled = detect_tiled(model, image, grid, overlap, score_threshold, iou_threshold, input_size)
    return dedupe(whole + tiled, iou_threshold)


def cmd_compare(args: argparse.Namespace) -> None:
    import json

    from .evaluate_tables import per_card_hits

    model = TableCenterNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    real_captures = DATA_DIR / "real-captures"
    tallies = {"single": [0, 0, 0], "tiled": [0, 0, 0], "multiscale": [0, 0, 0]}  # hit, truth, found
    for capture_dir in sorted(real_captures.iterdir()):
        annotation_path = capture_dir / "annotation.json"
        if not annotation_path.exists():
            continue
        annotation = json.loads(annotation_path.read_text())
        image_path = capture_dir / "video_crop.png"
        if not image_path.exists():
            image_path = capture_dir / "frame.jpg"
        raw = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        native_side = max(raw.shape[0], raw.shape[1])
        square, scale, (pad_x, pad_y) = letterbox_to_square(raw, native_side)  # scale == 1.0: full native resolution, just padded square
        pad = np.float32([pad_x, pad_y])
        truth = [(np.float32(c["quad"]) + pad) * scale for c in annotation["cards"]]

        with torch.no_grad():
            small = cv2.resize(square, (TABLE_INPUT, TABLE_INPUT), interpolation=cv2.INTER_AREA)
            heat, pose, up = model(to_tensor(small)[None])
            single = [(q * (square.shape[0] / TABLE_INPUT), s) for q, s in decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, args.score_threshold)]
        grid, overlap = (args.rows, args.cols), args.overlap
        tiled = detect_tiled(model, square, grid, overlap, args.score_threshold, args.iou_threshold)
        multiscale = detect_multiscale(model, square, grid, overlap, args.score_threshold, args.iou_threshold)

        report = []
        for name, dets in (("single", single), ("tiled", tiled), ("multiscale", multiscale)):
            found = [q for q, _s in dets]
            hits = per_card_hits(truth, found, 0.5)
            tally = tallies[name]
            tally[0] += sum(hits)
            tally[1] += len(hits)
            tally[2] += len(found)
            report.append(f"{name}: recall={sum(hits) / max(len(hits), 1):.3f} n_found={len(found)}")
        print(f"{capture_dir.name}: n={len(truth)}  " + "  ".join(report))

    for name, (hit, truth_n, found_n) in tallies.items():
        recall = hit / truth_n if truth_n else 0.0
        precision = hit / found_n if found_n else 0.0
        print(f"OVERALL {name}: recall={recall:.3f} precision={precision:.3f} n_truth={truth_n} n_found={found_n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_compare = sub.add_parser("compare", help="single-pass vs. tiled inference on the golden real-capture dataset")
    p_compare.add_argument("--checkpoint", type=Path, required=True)
    p_compare.add_argument("--rows", type=int, default=2)
    p_compare.add_argument("--cols", type=int, default=2)
    p_compare.add_argument("--overlap", type=float, default=0.2)
    p_compare.add_argument("--score-threshold", type=float, default=0.3)
    p_compare.add_argument("--iou-threshold", type=float, default=0.4)
    args = parser.parse_args()
    cmd_compare(args)


if __name__ == "__main__":
    main()
