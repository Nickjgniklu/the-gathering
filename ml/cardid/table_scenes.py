"""Deterministic full-table synthetic scenes and their JSONL manifests.

Unlike the click-window renderer (`scene_renderer.render_scene`), a table scene contains
0-20 independently labelled cards across sparse, normal, and crowded densities. A real board
is organised, not a pile: every card in a scene shares one physical size (only the
`camera_profile` -- how close/zoomed the rig is -- changes how big cards look, never an
individual card), and `_poses` lays every card out on a non-overlapping grid whose spacing is
sized to that card's on-screen footprint at any of its four rotations. Cards touch only
through the rare aura/equipment/token occluder (`scene_renderer.occluders`), the same rate a
real board actually stacks something on a card. Every scene also carries a `setup` (which
grid shape/region the cards fill), a `camera_profile`, and a background drawn from a
split-specific pool of art crops, so `train`/`val`/`test`/`challenge` never share an
arrangement, a camera profile, or a background image: evaluation holds out whole
spatial/photometric/background combinations instead of merely holding out random pixels from
the same one, which is what would let a model memorise a playmat or a fixed scale instead of
learning to find cards.

Splits:

- ``train`` sees every camera profile except the held-out one, and the "easy" arrangements.
- ``val`` and ``test`` each get one held-out arrangement, plus the held-out camera profile
  and their own background pool. ``val`` is scored every epoch during training; ``test`` is
  frozen and meant for a final, sparingly-run check, not for tuning against.
- ``challenge`` reuses ``test``'s arrangement, camera profile and backgrounds but at higher
  photometric severity and crowded density, for glare/blur/occlusion/tiny-card slices.

Every scene is reproducible from its seed alone (`seed`, `split`, ordinal -> one
`np.random.SeedSequence`), and the dataset header records the renderer version, the catalog
fingerprint (which card images were available), and the split rules, so a manifest is
self-describing even without the code that made it. Generated images and manifests are not
meant for Git (see the `--out` default and `ml/README.md`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .constants import CARD_ASPECT
from .image_bank import ArtBank, CardBank, list_arts
from .scene_geometry import quad_bbox, quad_from_pose, quad_short
from .scene_renderer import background, draw_card, draw_sleeve_ring, draw_toploader, gloss, occluders, photometrics

# Safety margin (over a card's long side, the worst case for any of the 4 rotations) between
# grid cell centres, so neighbouring cards do not touch even with the small per-card angle
# jitter `render_table_scene` adds. A real board is organised, not a pile: cards overlap only
# through the rare aura/equipment/token occluder below, never because the layout ran out of
# room -- a scene that cannot fit its requested count at its card's on-screen size renders
# fewer cards instead (see `_poses`'s `n = min(count, capacity)`).
FOOTPRINT_MARGIN = 1.25
CELL_JITTER = 0.06

RENDERER_VERSION = "table-scenes-v2"

# Spatial arrangements, each a non-overlapping grid over a different region/shape:
# "lanes" packs a tidy contiguous block; "spread" scatters the same grid's cells with gaps;
# "cluster" packs a small central region (a few permanents together); "battlefield" mimics a
# Commander board (a permanents row above a lands row); "duel" mimics two players' zones with
# a gap between them.
SETUPS = ("lanes", "cluster", "spread", "battlefield", "duel")

# Card short side as a fraction of the canvas, and a photometric severity multiplier
# (`scene_renderer.photometrics`'s `severity`) representing a physically different rig, not
# just a different random draw of the same rig.
CAMERA_PROFILES = {
    "overhead_1080p": {"short_frac": (0.09, 0.15), "severity": 1.0},
    "angled_720p": {"short_frac": (0.07, 0.12), "severity": 1.35},
    "closeup_4k": {"short_frac": (0.14, 0.22), "severity": 0.75},
}

# Inclusive card-count ranges for sparse/normal/crowded battlefields.
DENSITIES = {"sparse": (0, 4), "normal": (5, 12), "crowded": (13, 20)}

SPLITS = ("train", "val", "test", "challenge")
SPLIT_INDEX = {name: i for i, name in enumerate(SPLITS)}
SPLIT_SETUPS = {"train": ("lanes", "cluster", "spread"), "val": ("battlefield",), "test": ("duel",), "challenge": ("duel",)}
SPLIT_CAMERA_PROFILES = {
    "train": ("overhead_1080p", "closeup_4k"),
    "val": ("angled_720p",),
    "test": ("angled_720p",),
    "challenge": ("angled_720p",),
}
SPLIT_BACKGROUND_POOL = {"train": "train", "val": "val", "test": "test", "challenge": "test"}
SPLIT_SEVERITY = {"train": 1.0, "val": 1.0, "test": 1.0, "challenge": 1.9}
SPLIT_DENSITY_WEIGHTS = {
    "train": {"sparse": 0.3, "normal": 0.5, "crowded": 0.2},
    "val": {"sparse": 0.3, "normal": 0.5, "crowded": 0.2},
    "test": {"sparse": 0.3, "normal": 0.5, "crowded": 0.2},
    "challenge": {"sparse": 0.0, "normal": 0.25, "crowded": 0.75},
}
DEFAULT_SCENES = {"train": 800, "val": 150, "test": 150, "challenge": 60}

# A card is labelled unidentifiable when later-drawn cards cover most of its face or it is
# too small in the output image to carry recognisable art/text; slicing by this flag is how
# detection metrics separate from identification metrics (see `evaluate_tables.py`).
IDENTIFIABLE_MAX_OCCLUSION = 0.55
IDENTIFIABLE_MIN_SHORT_PX = 24.0

SLEEVE_RATE = 0.35
LOADER_RATE = 0.12
GLOSS_RATE = 0.15
OCCLUDER_RATE = 0.2  # baseline chance an *individual* card gets a nearby die/finger/loose card


@dataclass(frozen=True)
class TableCard:
    card_id: str
    quad: list[list[float]]
    bbox: list[float]
    orientation: int
    occluded_fraction: float
    identifiable: bool


def partition_arts(paths: list[Path], seed: int = 0) -> dict[str, list[Path]]:
    """Split background art paths into disjoint train/val/test pools (~70/15/15) so the same
    background image never appears in more than one split. Falls back to sharing the full
    list when there are too few paths to split meaningfully (small test/dev fixtures)."""
    paths = list(paths)
    if len(paths) < 8:
        return {"train": paths, "val": paths, "test": paths}
    order = np.random.default_rng(seed).permutation(len(paths))
    n_val = max(1, round(len(paths) * 0.15))
    n_test = max(1, round(len(paths) * 0.15))
    val_idx, test_idx, train_idx = order[:n_val], order[n_val : n_val + n_test], order[n_val + n_test :]
    return {"train": [paths[i] for i in train_idx], "val": [paths[i] for i in val_idx], "test": [paths[i] for i in test_idx]}


def _grid_capacity(usable_w: float, usable_h: float, footprint: float) -> tuple[int, int]:
    """Grid dimensions that fit whole `footprint`-sized cells in `usable_w` x `usable_h`.
    Zero in either dimension when the footprint does not fit at all (a duel/battlefield band
    half the canvas tall can be smaller than the footprint at a large camera-profile scale);
    callers place zero cards there rather than forcing one that bleeds into the next band."""
    return int(usable_w // footprint), int(usable_h // footprint)


def _band_cells(
    rng: np.random.Generator, n: int, cols: int, footprint: float, x0: float, y0: float, scattered: bool, capacity: int
) -> list[tuple[float, float]]:
    """`n` non-overlapping cell centres from a `footprint`-spaced grid of `cols` columns
    anchored at `(x0, y0)`, each with a little within-cell jitter. `scattered` draws `n` cells
    at random out of the band's full `capacity` (a loose, gappy look) instead of packing them
    into the first contiguous block (a tidy look) -- the same non-overlapping grid either way."""
    cells = rng.choice(capacity, n, replace=False) if scattered and n < capacity else np.arange(n)
    out = []
    for cell in cells:
        col, row = int(cell) % cols, int(cell) // cols
        jitter = rng.uniform(-CELL_JITTER, CELL_JITTER, 2) * footprint
        out.append((x0 + footprint * (col + 0.5) + jitter[0], y0 + footprint * (row + 0.5) + jitter[1]))
    return out


def _poses(rng: np.random.Generator, setup: str, count: int, size: int, profile: dict) -> list[tuple[float, float, float, int]]:
    """One centre/rotation per card plus a single `short` shared by the whole scene: a real
    card is always the same physical size, so only the camera profile (how close/zoomed the
    rig is) changes how big it looks, never which card or how many share the table."""
    lo, hi = profile["short_frac"]
    short = size * rng.uniform(lo, hi)
    footprint = short * CARD_ASPECT * FOOTPRINT_MARGIN
    margin = 0.08 * size
    usable = size - 2 * margin
    centres: list[tuple[float, float]]
    if setup == "duel":
        band_h = usable / 2 - margin / 2
        cols, rows = _grid_capacity(usable, band_h, footprint)
        capacity = cols * rows
        n_near, n_far = min(count // 2, capacity), min(count - count // 2, capacity)
        centres = _band_cells(rng, n_near, cols, footprint, margin, margin, False, capacity)
        centres += _band_cells(rng, n_far, cols, footprint, margin, size / 2 + margin / 2, False, capacity)
    elif setup == "battlefield":
        band_h = usable / 2 - margin / 2
        cols, rows = _grid_capacity(usable, band_h, footprint)
        capacity = cols * rows
        n_permanents, n_lands = min(count - count // 2, capacity), min(count // 2, capacity)
        centres = _band_cells(rng, n_permanents, cols, footprint, margin, margin, False, capacity)
        centres += _band_cells(rng, n_lands, cols, footprint, margin, size / 2 + margin / 2, False, capacity)
    elif setup == "cluster":
        region = min(usable, usable) * 0.55  # a few permanents close together, not a full board
        cols, rows = _grid_capacity(region, region, footprint)
        capacity = cols * rows
        n = min(count, capacity)
        centres = _band_cells(rng, n, cols, footprint, size / 2 - region / 2, size / 2 - region / 2, False, capacity)
    else:  # lanes (tidy contiguous rows) and spread (same grid, a scattered subset of it)
        cols, rows = _grid_capacity(usable, usable, footprint)
        capacity = cols * rows
        n = min(count, capacity)
        centres = _band_cells(rng, n, cols, footprint, margin, margin, setup == "spread", capacity)
    return [(x, y, short, int(rng.choice((0, 90, 180, 270)))) for x, y in centres]


def render_table_scene(
    seed: int,
    cards: CardBank,
    arts: ArtBank,
    setup: str,
    camera_profile: str = "overhead_1080p",
    count: int = 8,
    size: int = 1280,
    out: int = 640,
    severity: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Render one seeded scene and return RGB pixels plus a portable manifest record."""
    if setup not in SETUPS:
        raise ValueError(f"unknown setup {setup!r}; choose from {SETUPS}")
    if camera_profile not in CAMERA_PROFILES:
        raise ValueError(f"unknown camera profile {camera_profile!r}; choose from {tuple(CAMERA_PROFILES)}")
    if not 0 <= count <= len(cards):
        raise ValueError(f"count must be between 0 and the number of supplied cards ({len(cards)})")
    rng = np.random.default_rng(seed)
    profile = CAMERA_PROFILES[camera_profile]
    canvas = background(rng, arts, size)
    poses = _poses(rng, setup, count, size, profile)
    # A scene cannot fit more cards than its grid has room for at this camera profile's scale
    # (a real close-up camera cannot show 20 cards either); fewer cards render rather than
    # letting them overlap. `placed` is the count actually used everywhere below.
    placed = len(poses)
    indices = rng.choice(len(cards), placed, replace=False) if placed else np.empty(0, dtype=int)
    quads = [quad_from_pose(x, y, short, angle + rng.uniform(-8, 8), rng) for x, y, short, angle in poses]
    masks = []
    for index, quad in zip(indices, quads, strict=True):
        sleeved = rng.random() < SLEEVE_RATE
        loadered = not sleeved and rng.random() < LOADER_RATE
        loader_alpha = draw_toploader(canvas, rng, quad) if loadered else None
        ring_alpha = draw_sleeve_ring(canvas, rng, quad)[1] if sleeved else None
        card_alpha = draw_card(canvas, rng, cards, quad, shadow=not (sleeved or loadered), detail=out / size, index=int(index))
        combined = card_alpha
        if loader_alpha is not None:
            combined = np.maximum(combined, loader_alpha)
            gloss(canvas, rng, combined, quad)
        elif ring_alpha is not None:
            combined = np.maximum(combined, ring_alpha)
            gloss(canvas, rng, combined, quad)
        elif rng.random() < GLOSS_RATE:
            gloss(canvas, rng, card_alpha, quad)
        if rng.random() < min(OCCLUDER_RATE * severity, 0.9):
            occluders(canvas, rng, quad, cards, detail=out / size)
        masks.append(combined > 0.5)
    records = []
    for i, (index, quad, mask) in enumerate(zip(indices, quads, masks, strict=True)):
        later = np.logical_or.reduce(masks[i + 1 :]) if i + 1 < placed else np.zeros_like(mask)
        occluded = float((mask & later).sum() / max(mask.sum(), 1))
        out_quad = quad * (out / size)
        short_px = quad_short(out_quad)
        records.append(
            TableCard(
                card_id=cards.paths[int(index)].stem,
                quad=out_quad.tolist(),
                bbox=list(quad_bbox(out_quad)),
                orientation=int(round(np.degrees(np.arctan2(out_quad[1, 1] - out_quad[0, 1], out_quad[1, 0] - out_quad[0, 0]))) % 360),
                occluded_fraction=occluded,
                identifiable=occluded < IDENTIFIABLE_MAX_OCCLUSION and short_px >= IDENTIFIABLE_MIN_SHORT_PX,
            )
        )
    image = photometrics(
        cv2.resize(np.clip(canvas, 0, 255).astype(np.uint8), (out, out), interpolation=cv2.INTER_AREA),
        rng,
        out / size,
        severity=profile["severity"] * severity,
    )
    return image, {
        "seed": seed,
        "setup": setup,
        "camera_profile": camera_profile,
        "cards": [asdict(record) for record in records],
        "width": out,
        "height": out,
    }


def _scene_seed(seed: int, split: str, ordinal: int) -> int:
    return int(np.random.SeedSequence([seed, SPLIT_INDEX[split], ordinal]).generate_state(1)[0])


def write_split(output: Path, split: str, scenes: int, seed: int, cards: CardBank, arts_by_pool: dict[str, ArtBank], size: int = 1280, out: int = 640) -> dict:
    """Write one split's JPEGs and JSONL manifest under ``output/<split>/``."""
    if split not in SPLIT_SETUPS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    setups, profiles = SPLIT_SETUPS[split], SPLIT_CAMERA_PROFILES[split]
    severity = SPLIT_SEVERITY[split]
    weights = SPLIT_DENSITY_WEIGHTS[split]
    arts = arts_by_pool[SPLIT_BACKGROUND_POOL[split]]
    density_names = list(weights)
    density_probs = np.array([weights[d] for d in density_names], dtype=np.float64)
    density_probs /= density_probs.sum()
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    stats = {"scenes": 0, "cards": 0, "identifiable_cards": 0, "by_density": dict.fromkeys(density_names, 0)}
    with (split_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for ordinal in range(scenes):
            scene_seed = _scene_seed(seed, split, ordinal)
            picker = np.random.default_rng(scene_seed)
            setup = setups[ordinal % len(setups)]
            camera_profile = profiles[ordinal % len(profiles)]
            density = str(picker.choice(density_names, p=density_probs))
            lo, hi = DENSITIES[density]
            count = min(int(picker.integers(lo, hi + 1)), len(cards))
            image, record = render_table_scene(scene_seed, cards, arts, setup, camera_profile, count, size, out, severity)
            name = f"{ordinal:06d}.jpg"
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError(f"failed to encode scene {ordinal} for split {split!r}")
            (split_dir / name).write_bytes(encoded.tobytes())
            row = {**record, "density": density, "split": split, "image": name, "sha256": hashlib.sha256(encoded.tobytes()).hexdigest()}
            file.write(json.dumps(row) + "\n")
            stats["scenes"] += 1
            stats["cards"] += len(record["cards"])
            stats["identifiable_cards"] += sum(c["identifiable"] for c in record["cards"])
            stats["by_density"][density] += 1
    return stats


def catalog_fingerprint(cards: CardBank) -> str:
    return hashlib.sha256("\n".join(p.name for p in cards.paths).encode()).hexdigest()[:16]


def write_dataset(output: Path, seed: int, scenes: dict[str, int], size: int = 1280, out: int = 640) -> dict:
    """Write every requested split plus a dataset-level ``dataset.json`` header recording the
    renderer version, catalog fingerprint, split rules, and per-split counts. Re-running with
    a subset of `scenes` (e.g. just `{"test": 150}`) regenerates only those splits and merges
    the result into the existing header, if any."""
    cards = CardBank()
    pools = partition_arts(list_arts(), seed)
    arts_by_pool = {name: ArtBank(paths=paths) for name, paths in pools.items()}
    output.mkdir(parents=True, exist_ok=True)
    header_path = output / "dataset.json"
    previous = json.loads(header_path.read_text()) if header_path.exists() else {}
    split_rules, split_stats = previous.get("split_rules", {}), previous.get("splits", {})
    for split, count in scenes.items():
        split_stats[split] = write_split(output, split, count, seed, cards, arts_by_pool, size, out)
        split_rules[split] = {
            "setups": SPLIT_SETUPS[split],
            "camera_profiles": SPLIT_CAMERA_PROFILES[split],
            "background_pool": SPLIT_BACKGROUND_POOL[split],
            "severity": SPLIT_SEVERITY[split],
            "density_weights": SPLIT_DENSITY_WEIGHTS[split],
        }
    header = {
        "renderer_version": RENDERER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "catalog_fingerprint": catalog_fingerprint(cards),
        "cards_available": len(cards),
        "background_pool_sizes": {name: len(paths) for name, paths in pools.items()},
        "split_rules": split_rules,
        "splits": split_stats,
    }
    header_path.write_text(json.dumps(header, indent=2))
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = (
        Path(os.environ["CARDID_TABLE_SCENES_DIR"]) if "CARDID_TABLE_SCENES_DIR" in os.environ else Path.home() / "the-gathering-cardid" / "table-scenes"
    )
    parser.add_argument(
        "--out", type=Path, default=default_out, help="output directory (default: ~/the-gathering-cardid/table-scenes, or $CARDID_TABLE_SCENES_DIR)"
    )
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all", help="generate only this split; default generates all four")
    parser.add_argument("--train", type=int, default=DEFAULT_SCENES["train"])
    parser.add_argument("--val", type=int, default=DEFAULT_SCENES["val"])
    parser.add_argument("--test", type=int, default=DEFAULT_SCENES["test"])
    parser.add_argument("--challenge", type=int, default=DEFAULT_SCENES["challenge"])
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--size", type=int, default=1280, help="native render resolution before downscale")
    parser.add_argument("--resolution", type=int, default=640, help="output image resolution")
    args = parser.parse_args()
    scenes = {"train": args.train, "val": args.val, "test": args.test, "challenge": args.challenge}
    if args.split != "all":
        scenes = {args.split: scenes[args.split]}
    header = write_dataset(args.out, args.seed, scenes, args.size, args.resolution)
    print(json.dumps({k: v for k, v in header.items() if k != "split_rules"}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
