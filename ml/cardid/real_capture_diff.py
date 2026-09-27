"""Auto-localize cards in a real-capture photo by diffing it against a clean-desk reference
photo of the same (static-camera) scene, instead of hand-eyeballing every corner.

A plain pixel diff fails here: the camera's auto-exposure/white-balance drifts between shots
(worse the further apart in time they were taken), so a naive threshold picks up the whole
scene as "changed". Dividing each image by a heavily-blurred (sigma ~300px) copy of itself
first (local illumination normalization) cancels that large-scale drift while preserving
card-scale detail, which is what makes the diff usable.

Even normalized, touching/adjacent cards and nearby shiny objects (a deck box's specular
highlight can shift between shots even though the box didn't move) still need a human to
resolve -- this script auto-extracts the clean, well-separated single-card blobs and reports
the rest as a to-do list of numbered residual crops for manual corner-picking, rather than
guessing.

    uv run python -m cardid.real_capture_diff diff --capture 006 --reference 002/desk_clean.jpg
    uv run python -m cardid.real_capture_diff verify --capture 006
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import DATA_DIR

REAL_CAPTURES_DIR = DATA_DIR / "real-captures"

# A single MTG card's on-screen aspect ratio (88:63) varies with rotation/perspective but stays
# in this range across every capture so far; area bounds are loose since card size varies with
# distance from camera.
CARD_ASPECT_RANGE = (1.15, 1.65)
CARD_AREA_RANGE = (35000, 260000)


def normalize(img: np.ndarray, sigma: float = 300.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return img / (blur + 5.0)


def compute_diff(capture_dir: Path, reference: Path, sigma: float = 300.0) -> np.ndarray:
    with_cards = cv2.imread(str(capture_dir / "frame.jpg")).astype(np.float32)
    clean = cv2.imread(str(reference)).astype(np.float32)
    diff = np.abs(normalize(with_cards, sigma) - normalize(clean, sigma)).sum(axis=2)
    return (np.clip(diff / np.percentile(diff, 99.5), 0, 1) * 255).astype(np.uint8)


def card_mask(diff: np.ndarray, threshold: int = 90) -> np.ndarray:
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > 8000:
            cv2.drawContours(filled, [c], -1, 255, -1)
    return filled


def classify_contours(filled_mask: np.ndarray) -> tuple[list[list[list[float]]], list[dict]]:
    """Card-shaped contours -> quads (auto-accepted); everything else -> a residual list with
    each blob's bbox and a saved crop path for a human to resolve (merged cards, specular
    non-card objects, or genuine noise to discard)."""
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads, residuals = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15000:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), _angle = rect
        aspect = max(w, h) / max(min(w, h), 1)
        x, y, bw, bh = cv2.boundingRect(c)
        if CARD_ASPECT_RANGE[0] <= aspect <= CARD_ASPECT_RANGE[1] and CARD_AREA_RANGE[0] <= area <= CARD_AREA_RANGE[1]:
            quads.append(cv2.boxPoints(rect).tolist())
        else:
            residuals.append({"area": area, "aspect": aspect, "bbox": [x, y, x + bw, y + bh]})
    return quads, residuals


def cmd_diff(args: argparse.Namespace) -> None:
    capture_dir = REAL_CAPTURES_DIR / args.capture
    reference = REAL_CAPTURES_DIR / args.reference
    diff = compute_diff(capture_dir, reference, args.sigma)
    (capture_dir / "diff_sigma300.png").write_bytes(cv2.imencode(".png", diff)[1].tobytes())
    filled = card_mask(diff, args.threshold)
    quads, residuals = classify_contours(filled)
    print(f"{args.capture}: {len(quads)} auto card quads, {len(residuals)} residual blob(s) need manual review")
    residual_dir = capture_dir / "residuals"
    residual_dir.mkdir(exist_ok=True)
    for old in residual_dir.glob("*.png"):
        old.unlink()
    for i, r in enumerate(residuals):
        x0, y0, x1, y1 = r["bbox"]
        pad = 100
        crop = diff[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
        path = residual_dir / f"{i:02d}.png"
        cv2.imwrite(str(path), crop)
        print(f"  residual {i}: area={r['area']:.0f} aspect={r['aspect']:.2f} bbox={r['bbox']} -> {path}")
    (capture_dir / "auto_quads.json").write_text(json.dumps(quads))


def cmd_verify(args: argparse.Namespace) -> None:
    capture_dir = REAL_CAPTURES_DIR / args.capture
    quads = json.loads((capture_dir / "auto_quads.json").read_text())
    manual_path = capture_dir / "manual_quads.json"
    if manual_path.exists():
        quads = quads + json.loads(manual_path.read_text())
    img = cv2.imread(str(capture_dir / "frame.jpg"))
    for q in quads:
        cv2.polylines(img, [np.int32(q)], True, (0, 255, 0), 6, cv2.LINE_AA)
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    out = capture_dir / "verify_quads.jpg"
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{args.capture}: {len(quads)} total quads -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_diff = sub.add_parser("diff")
    p_diff.add_argument("--capture", required=True)
    p_diff.add_argument("--reference", default="002/desk_clean.jpg")
    p_diff.add_argument("--sigma", type=float, default=300.0)
    p_diff.add_argument("--threshold", type=int, default=90)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--capture", required=True)

    args = parser.parse_args()
    {"diff": cmd_diff, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
