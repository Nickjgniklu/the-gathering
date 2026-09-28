"""One report across every failure-mode category this project has ever measured, in place of
re-running `evaluate_nonstandard.py`, `hardware_stress.py`, `size_density_analysis.py`, and
`small_card_probe.py` separately and eyeballing four sets of numbers.

Object detection has one class ("card"), so there is no NxN label-confusion matrix to draw; the
useful analogue is a true/false-positive/false-negative breakdown sliced by *why* a detection
would fail, which is what this prints: one row per category, each scored the same way (greedy
IoU-0.5 matching, `evaluate_tables.per_card_hits`) plus, for categories that have them, a
false-positive-on-hard-negative rate (did the model fire on a round/clutter object that carries
no ground-truth quad at all).

Categories, all motivated by a real deployed-feature capture that showed unmodeled desk clutter,
a false positive on a round object, confused adjacent/stacked cards, and low confidence under
dark pink/purple lighting (see `ml/README.md`):

    baseline              clean test-split scenes, current camera profiles
    size:<bucket>         baseline's cards re-sliced by short-side / frame-width
    density:<label>       baseline's cards re-sliced by scene density
    nonstandard:<cat>     each `nonstandard_cards.py` frame category vs. an ordinary baseline
    hw:<degradation>      each `hardware_stress.py` degradation, applied fresh to test scenes
    stacking:victim/topper/other   recall on the deliberately-stacked pair vs. everything else
    hard_negatives        false-positive rate on scenes forced to include clutter/round objects
    real_captures         the small hand-annotated real-webcam set in data/real-captures/ (n is
                           tiny -- a calibration sanity check, not a reliable estimate)

    uv run python -m cardid.confusion_matrix --checkpoint data/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import DATA_DIR
from .data import to_tensor
from .evaluate_nonstandard import ordinary_card_bank
from .evaluate_tables import per_card_hits
from .hardware_stress import DEGRADATIONS, apply_combined
from .image_bank import ArtBank, CardBank, list_arts
from .nonstandard_cards import CATEGORY_QUERIES, NONSTANDARD_DIR
from .size_density_analysis import size_bucket
from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet, decode_detections
from .table_scenes import DENSITIES, SETUPS, render_table_scene

REAL_CAPTURES_DIR = DATA_DIR / "real-captures"


def detect(model: TableCenterNet, image: np.ndarray, score_threshold: float) -> list[np.ndarray]:
    heat, pose, up = model(to_tensor(image)[None])
    return [q for q, _s in decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold)]


def letterbox_to_square(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Pad `image` to square (matching the deployed frontend's `letterboxToSquare`, not a plain
    resize) and scale to `size`. A non-square real capture squashed directly onto a square input
    would distort its aspect ratio in a way the model never saw in training -- every synthetic
    scene is rendered square from the start (see `table_scenes.render_table_scene`'s `out`).
    Returns (letterboxed image, scale, (pad_x, pad_y)) so ground-truth quads can be mapped the
    same way."""
    h, w = image.shape[:2]
    side = max(h, w)
    pad_x, pad_y = (side - w) / 2, (side - h) / 2
    padded = cv2.copyMakeBorder(image, round(pad_y), round(side - h - pad_y), round(pad_x), round(side - w - pad_x), cv2.BORDER_CONSTANT)
    scale = size / side
    return cv2.resize(padded, (size, size)), scale, (pad_x, pad_y)


def density_label(count: int) -> str:
    for name, (lo, hi) in DENSITIES.items():
        if lo <= count <= hi:
            return name
    return "crowded"


def fp_on_negatives(found: list[np.ndarray], negatives: list[dict]) -> int:
    """How many found quads land centred inside a hard-negative's (slightly expanded) box."""
    count = 0
    for quad in found:
        cx, cy = quad.mean(axis=0)
        for neg in negatives:
            x0, y0, x1, y1 = neg["bbox"]
            pad = 0.15 * max(x1 - x0, y1 - y0)
            if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                count += 1
                break
    return count


class Tally:
    def __init__(self) -> None:
        self.hit = self.truth = self.found = self.neg_fp = self.neg_total = 0

    def add_scene(
        self, model: TableCenterNet, image: np.ndarray, truth: list[np.ndarray], score_threshold: float, negatives: list[dict] | None = None
    ) -> list[bool]:
        found = detect(model, image, score_threshold)
        hits = per_card_hits(truth, found, 0.5)
        self.hit += sum(hits)
        self.truth += len(hits)
        self.found += len(found)
        if negatives:
            self.neg_total += len(negatives)
            self.neg_fp += fp_on_negatives(found, negatives)
        return hits

    def report(self) -> dict:
        out = {
            "recall": self.hit / self.truth if self.truth else None,
            "precision": self.hit / self.found if self.found else None,
            "n_truth": self.truth,
            "n_found": self.found,
        }
        if self.neg_total:
            out["fp_on_negatives_rate"] = self.neg_fp / self.neg_total
            out["n_negatives"] = self.neg_total
        return out


@torch.no_grad()
def run(checkpoint: Path, scenes_per_category: int, cards_per_scene: int, seed: int, score_threshold: float) -> dict:
    model = TableCenterNet(pretrained=False).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    cards, arts = CardBank(), ArtBank(list_arts())

    results: dict[str, dict] = {}

    # --- baseline, plus size/density slices of the same scenes -----------------------------
    baseline, by_size, by_density = Tally(), {}, {}
    for i in range(scenes_per_category):
        setup = SETUPS[i % len(SETUPS)]
        # vary count so density buckets actually get populated, same spread write_split draws
        count = min(int(np.random.default_rng(seed + i).integers(0, min(20, len(cards)) + 1)), len(cards))
        image, record = render_table_scene(seed + i, cards, arts, setup, "overhead_1080p", count=count, size=1280, out=TABLE_INPUT)
        truth = [np.float32(c["quad"]) for c in record["cards"]]
        hits = baseline.add_scene(model, image, truth, score_threshold)
        density = density_label(count)
        by_density.setdefault(density, Tally())
        for card, hit in zip(record["cards"], hits, strict=True):
            quad = np.float32(card["quad"])
            frac = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])) / TABLE_INPUT
            by_size.setdefault(size_bucket(frac), Tally())
            by_size[size_bucket(frac)].hit += hit
            by_size[size_bucket(frac)].truth += 1
            by_density[density].hit += hit
            by_density[density].truth += 1
    results["baseline"] = baseline.report()
    for name, tally in by_size.items():
        results[f"size:{name}"] = tally.report()
    for name, tally in by_density.items():
        results[f"density:{name}"] = tally.report()

    # --- non-standard card frames ------------------------------------------------------------
    ordinary = ordinary_card_bank(max(cards_per_scene * 4, 40), seed)
    ns_baseline = Tally()
    for i in range(scenes_per_category):
        image, record = render_table_scene(
            seed + i, ordinary, arts, SETUPS[i % len(SETUPS)], "overhead_1080p", count=min(cards_per_scene, len(ordinary)), size=1280, out=TABLE_INPUT
        )
        ns_baseline.add_scene(model, image, [np.float32(c["quad"]) for c in record["cards"]], score_threshold)
    results["nonstandard:ordinary"] = ns_baseline.report()
    if NONSTANDARD_DIR.exists():
        for category_dir in sorted(NONSTANDARD_DIR.iterdir()):
            if not category_dir.is_dir() or category_dir.name not in CATEGORY_QUERIES:
                continue
            paths = sorted(category_dir.glob("*.jpg"))
            if len(paths) < 2:
                continue
            ns_cards, tally = CardBank(paths), Tally()
            for i in range(scenes_per_category):
                image, record = render_table_scene(
                    seed + i, ns_cards, arts, SETUPS[i % len(SETUPS)], "overhead_1080p", count=min(cards_per_scene, len(ns_cards)), size=1280, out=TABLE_INPUT
                )
                tally.add_scene(model, image, [np.float32(c["quad"]) for c in record["cards"]], score_threshold)
            results[f"nonstandard:{category_dir.name}"] = tally.report()

    # --- hardware-stress degradations, applied fresh to test-camera-profile scenes -----------
    hw_categories = {**DEGRADATIONS, "combined": apply_combined}
    hw_tallies = {name: Tally() for name in hw_categories}
    for i in range(scenes_per_category):
        image, record = render_table_scene(
            seed + i, cards, arts, SETUPS[i % len(SETUPS)], "angled_720p", count=min(cards_per_scene, len(cards)), size=1280, out=TABLE_INPUT
        )
        truth = [np.float32(c["quad"]) for c in record["cards"]]
        rng = np.random.default_rng(seed + i)
        for name, fn in hw_categories.items():
            stressed = fn(image.copy(), np.random.default_rng(rng.integers(0, 2**32)))
            hw_tallies[name].add_scene(model, stressed, truth, score_threshold)
    for name, tally in hw_tallies.items():
        results[f"hw:{name}"] = tally.report()

    # --- deliberate stacking: victim (occluded, underneath) vs. topper vs. everything else ----
    victim_tally, topper_tally, other_tally = Tally(), Tally(), Tally()
    for i in range(scenes_per_category):
        image, record = render_table_scene(
            seed + i,
            cards,
            arts,
            SETUPS[i % len(SETUPS)],
            "overhead_1080p",
            count=min(max(cards_per_scene, 4), len(cards)),
            size=1280,
            out=TABLE_INPUT,
            stack_rate=1.0,
            clutter_rate=0.0,
            round_negative_rate=0.0,
        )
        truth = [np.float32(c["quad"]) for c in record["cards"]]
        found = detect(model, image, score_threshold)
        hits = per_card_hits(truth, found, 0.5)
        pair = record["stacked_pair"]
        for idx, hit in enumerate(hits):
            tally = other_tally
            if pair and idx == pair[0]:
                tally = victim_tally
            elif pair and idx == pair[1]:
                tally = topper_tally
            tally.hit += hit
            tally.truth += 1
    results["stacking:victim"] = victim_tally.report()
    results["stacking:topper"] = topper_tally.report()
    results["stacking:other"] = other_tally.report()

    # --- hard negatives: clutter blobs and round objects, no card ground truth at all --------
    hard_neg = Tally()
    for i in range(scenes_per_category):
        image, record = render_table_scene(
            seed + i,
            cards,
            arts,
            SETUPS[i % len(SETUPS)],
            "overhead_1080p",
            count=min(cards_per_scene, len(cards)),
            size=1280,
            out=TABLE_INPUT,
            stack_rate=0.0,
            clutter_rate=1.0,
            round_negative_rate=1.0,
        )
        truth = [np.float32(c["quad"]) for c in record["cards"]]
        hard_neg.add_scene(model, image, truth, score_threshold, negatives=record["negatives"])
    results["hard_negatives"] = hard_neg.report()

    # --- real captures: tiny hand-annotated set, if any exist --------------------------------
    if REAL_CAPTURES_DIR.exists():
        real_tally = Tally()
        n_captures = 0
        for capture_dir in sorted(REAL_CAPTURES_DIR.iterdir()):
            annotation_path = capture_dir / "annotation.json"
            if not annotation_path.exists():
                continue
            n_captures += 1
            annotation = json.loads(annotation_path.read_text())
            image_path = capture_dir / "video_crop.png"
            if not image_path.exists():
                image_path = capture_dir / "frame.jpg"
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            resized, scale, (pad_x, pad_y) = letterbox_to_square(image, TABLE_INPUT)
            pad = np.float32([pad_x, pad_y])
            truth = [(np.float32(c["quad"]) + pad) * scale for c in annotation["cards"]]
            negatives = [
                {
                    "kind": n["kind"],
                    "bbox": [(n["box"][0] + pad_x) * scale, (n["box"][1] + pad_y) * scale, (n["box"][2] + pad_x) * scale, (n["box"][3] + pad_y) * scale],
                }
                for n in annotation.get("negatives", [])
            ]
            real_tally.add_scene(model, resized, truth, score_threshold, negatives=negatives)
        if n_captures:
            results["real_captures"] = {**real_tally.report(), "n_captures": n_captures}

    return results


def print_matrix(results: dict) -> None:
    print(f"{'category':24s} {'recall':>8s} {'precision':>10s} {'fp_on_neg':>10s} {'n':>6s}")
    for name, row in results.items():
        recall = f"{row['recall']:.3f}" if row.get("recall") is not None else "-"
        precision = f"{row['precision']:.3f}" if row.get("precision") is not None else "-"
        fp = f"{row['fp_on_negatives_rate']:.3f}" if row.get("fp_on_negatives_rate") is not None else "-"
        n = row.get("n_truth") or row.get("n_negatives") or 0
        print(f"{name:24s} {recall:>8s} {precision:>10s} {fp:>10s} {n:>6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes-per-category", type=int, default=30)
    parser.add_argument("--cards-per-scene", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "confusion-matrix.json")
    args = parser.parse_args()

    results = run(args.checkpoint, args.scenes_per_category, args.cards_per_scene, args.seed, args.score_threshold)
    print_matrix(results)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
