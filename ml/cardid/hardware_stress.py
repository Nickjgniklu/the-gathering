"""Post-hoc "bad webcam hardware" degradations layered onto already-rendered table scenes, to
test (and later train against) failure modes `scene_renderer.photometrics` does not cover:
genuinely dark rooms, colored ambient lighting, lens vignetting/chromatic aberration, and desk/
glare reflections that land across several cards at once rather than one card's own gloss.

Post-processing existing `test`-split scenes (not re-rendering) keeps the ground-truth quads
valid for free -- these effects only change pixel values, never geometry.

    uv run python -m cardid.hardware_stress generate --manifest-dir ~/the-gathering-cardid/table-scenes
    uv run python -m cardid.hardware_stress evaluate --checkpoint data/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import DATA_DIR

STRESS_DIR = DATA_DIR / "hardware-stress"


def apply_dark_room(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A single dim lamp, not just underexposure: heavy brightness cut, a warm or cool cast
    (incandescent vs. LED), and the extra sensor noise a camera's auto-gain adds in the dark."""
    gain = rng.uniform(0.15, 0.35)
    warm = rng.uniform(-1, 1) < 0
    tint = np.float32([1.15, 1.0, 0.8]) if warm else np.float32([0.85, 1.0, 1.15])
    x = image.astype(np.float32) * gain * tint
    x += rng.standard_normal(x.shape).astype(np.float32) * rng.uniform(6, 14)
    return np.clip(x, 0, 255).astype(np.uint8)


# RGB-gaming-peripheral hues (purple/magenta, blue, red/pink, cyan, green) as (R, G, B) tint
# targets: a real capture from the deployed feature showed exactly this kind of ambient LED
# lighting (a pink/purple cast across the whole desk), which a uniformly-random tint under-covers
# -- most of the (0.5, 1.6) cube is a color no LED strip or keyboard actually produces.
GAMING_HUES = np.float32(
    [
        [1.5, 0.6, 1.6],  # purple/magenta
        [0.6, 0.7, 1.7],  # blue
        [1.7, 0.5, 0.9],  # red/pink
        [0.5, 1.4, 1.5],  # cyan
        [0.6, 1.7, 0.7],  # green
    ]
)


def apply_color_cast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Strong colored ambient light (RGB room lighting, a monitor's glow, a colored LED strip)
    -- a much larger per-channel tint than photometrics' ordinary white-balance jitter. Mostly
    drawn near a gaming-peripheral hue (see `GAMING_HUES`), since that is what real desks
    actually show, with some fully-random tints kept for generality."""
    if rng.random() < 0.7:
        base = GAMING_HUES[int(rng.integers(len(GAMING_HUES)))]
        tint = base * rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    else:
        tint = rng.uniform(0.5, 1.6, size=3).astype(np.float32)
    tint *= 3.0 / tint.sum()  # keep overall brightness roughly stable; only the balance shifts
    return np.clip(image.astype(np.float32) * tint, 0, 255).astype(np.uint8)


def apply_dark_gaming_cast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dark room *and* a gaming-peripheral color cast together, not just two of five random
    degradations that happen to land on both -- this specific combination (a dim room lit mostly
    by RGB LEDs) is the single most common real-desk condition and was previously only
    ~1/C(5,2)-ish likely to co-occur under `apply_combined`'s uniform pick."""
    return apply_dark_room(apply_color_cast(image, rng), rng)


def apply_vignette(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darkened corners: the commonest cheap-lens artifact, absent from the renderer's uniform
    lighting ramp today."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    strength = rng.uniform(0.35, 0.7)
    mask = 1 - strength * np.clip(r - rng.uniform(0.2, 0.4), 0, None) ** 2
    return np.clip(image.astype(np.float32) * np.clip(mask, 0.15, 1.0)[..., None], 0, 255).astype(np.uint8)


def apply_chromatic_aberration(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Colour fringing at high-contrast edges: a cheap lens focuses each wavelength slightly
    differently, so the red and blue channels are shifted a few pixels outward from centre."""
    h, w = image.shape[:2]
    shift = rng.uniform(1.5, 4.0)
    m_r = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + shift / w)
    m_b = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 - shift / w)
    r = cv2.warpAffine(image[..., 0], m_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(image[..., 2], m_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.stack([r, image[..., 1], b], axis=-1)


def apply_desk_reflection(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A large soft bright patch crossing several cards at once -- an overhead light or window
    reflecting off a glossy desk/glass tabletop -- distinct from `scene_renderer.gloss`'s
    per-card sleeve/foil highlight, which never spans multiple cards."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    sx, sy = rng.uniform(0.25, 0.5) * w, rng.uniform(0.15, 0.35) * h
    angle = rng.uniform(0, np.pi)
    dx, dy = xx - gx, yy - gy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    blob = np.exp(-((rx / sx) ** 2 + (ry / sy) ** 2)) * rng.uniform(80, 160)
    return np.clip(image.astype(np.float32) + blob[..., None], 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "dark_room": apply_dark_room,
    "color_cast": apply_color_cast,
    "dark_gaming_cast": apply_dark_gaming_cast,
    "vignette": apply_vignette,
    "chromatic_aberration": apply_chromatic_aberration,
    "desk_reflection": apply_desk_reflection,
}


def apply_combined(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2-3 degradations stacked, like a genuinely bad cheap webcam in a messy room. Half the
    time starts from the dark+gaming-cast combo (the real-capture-motivated worst case) and
    layers 0-1 more effects on top, instead of picking uniformly from all five/six and under-
    sampling that specific co-occurrence."""
    if rng.random() < 0.5:
        image = apply_dark_gaming_cast(image, rng)
        extra = [n for n in DEGRADATIONS if n not in ("dark_room", "color_cast", "dark_gaming_cast")]
        for name in rng.choice(extra, size=int(rng.integers(0, 2)), replace=False):
            image = DEGRADATIONS[name](image, rng)
        return image
    names = rng.choice(list(DEGRADATIONS), size=int(rng.integers(2, 4)), replace=False)
    for name in names:
        image = DEGRADATIONS[name](image, rng)
    return image


def cmd_generate(args: argparse.Namespace) -> None:
    from .evaluate_tables import load_scenes

    rows = load_scenes(args.manifest_dir / "test" / "manifest.jsonl", "test")[: args.scenes]
    if not rows:
        raise SystemExit(f"no test scenes in {args.manifest_dir}")
    categories = {"clean": lambda image, _rng: image, **DEGRADATIONS, "combined": apply_combined}
    for name in categories:
        (STRESS_DIR / name).mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    rng = np.random.default_rng(args.seed)
    for row in rows:
        image = cv2.cvtColor(cv2.imread(str(row["_image_path"])), cv2.COLOR_BGR2RGB)
        for name, fn in categories.items():
            stressed = fn(image.copy(), np.random.default_rng(rng.integers(0, 2**32)))
            out_path = STRESS_DIR / name / row["image"]
            cv2.imwrite(str(out_path), cv2.cvtColor(stressed, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            manifest_rows.append({"category": name, "image": str(out_path), "cards": row["cards"], "width": row["width"]})
    (STRESS_DIR / "manifest.json").write_text(json.dumps(manifest_rows))
    print(f"{len(rows)} scenes x {len(categories)} categories -> {STRESS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--manifest-dir", type=Path, required=True)
    p_gen.add_argument("--scenes", type=int, default=30)
    p_gen.add_argument("--seed", type=int, default=0)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--score-threshold", type=float, default=0.3)
    p_eval.add_argument("--samples", type=int, default=3, help="save this many overlay images per category")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    else:
        from .evaluate_hardware_stress import cmd_evaluate

        cmd_evaluate(args)


if __name__ == "__main__":
    main()
