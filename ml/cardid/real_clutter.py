"""Real desk-clutter crops (dice, a deck box, a mouse, a keyboard corner, headphones, a
ThermoFlask, a phone dock) composited into synthetic table scenes as hard negatives, instead of
`scene_renderer`'s procedural `clutter_object`/`round_object` blobs.

Motivation: a 108-card real-capture golden dataset (see `confusion_matrix.py`'s `real_captures`
category) found the deployed model's real-world precision (75.8%) and false-positive rate on
real clutter (22.2%) were far worse than any synthetic metric predicted, even after training
against procedural clutter shapes (`table-a-hardneg-v4`). The procedural blobs are flat-color
polygons and plain circles -- nothing like the actual glossy dice, a deck box's specular
highlight, or a keyboard's key grid that fool the model in practice. Pasting real crops closes
that visual gap directly instead of hoping procedural shapes generalize.

Crops were extracted once from a real clean-desk photo (`data/real-captures/002/desk_clean.jpg`)
with `extract_real_clutter_crops.py`-equivalent one-off code; see `data/real-clutter-crops/`.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from . import DATA_DIR

REAL_CLUTTER_DIR = DATA_DIR / "real-clutter-crops"


@lru_cache(maxsize=1)
def load_real_clutter_crops() -> tuple[np.ndarray, ...]:
    paths = sorted(REAL_CLUTTER_DIR.glob("*.png"))
    return tuple(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths)


def real_clutter_object(canvas: np.ndarray, rng: np.random.Generator, center: np.ndarray, scale: float) -> None:
    """Paste a random real clutter crop onto `canvas`, long side about `scale` canvas pixels,
    any rotation, with a soft (feathered, rounded-rect) alpha so it blends instead of leaving a
    hard rectangular seam, and a mild colour jitter so the model does not just key on this one
    desk's exact lighting. No ground-truth quad is ever recorded for this (see `table_scenes.py`)."""
    crops = load_real_clutter_crops()
    if not crops:
        return
    crop = crops[int(rng.integers(len(crops)))].astype(np.float32)
    h, w = crop.shape[:2]
    target_long = scale * rng.uniform(0.8, 1.3)
    resize_scale = target_long / max(h, w)
    rw, rh = max(1, round(w * resize_scale)), max(1, round(h * resize_scale))
    crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
    tint = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    crop = np.clip(crop * tint, 0, 255)

    angle = rng.uniform(0, 360)
    diag = int(np.ceil(np.hypot(rw, rh))) + 2
    M = cv2.getRotationMatrix2D((rw / 2, rh / 2), angle, 1.0)
    M[:, 2] += np.float32([diag / 2 - rw / 2, diag / 2 - rh / 2])
    rotated = cv2.warpAffine(crop, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    margin = max(2, round(0.06 * min(rw, rh)))
    mask = np.zeros((rh, rw), np.float32)
    cv2.rectangle(mask, (margin, margin), (rw - 1 - margin, rh - 1 - margin), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), margin * 0.6)
    rotated_mask = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    x0, y0 = int(center[0] - diag / 2), int(center[1] - diag / 2)
    x1, y1 = x0 + diag, y0 + diag
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    src = rotated[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    a = rotated_mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0, None]
    dst = canvas[cy0:cy1, cx0:cx1]
    dst += (src - dst) * a
