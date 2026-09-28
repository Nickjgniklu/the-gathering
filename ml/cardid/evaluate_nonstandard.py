"""Score the table detector on each non-standard card category downloaded by
`nonstandard_cards.py`, against an ordinary-bordered baseline drawn from the main card pool.
Kept as its own module so `nonstandard_cards.py` (the download side, which has no torch
dependency) can be imported without pulling torch in.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from . import DATA_DIR
from .data import to_tensor
from .evaluate_tables import per_card_hits
from .image_bank import ArtBank, CardBank, list_arts
from .nonstandard_cards import CATEGORY_QUERIES, NONSTANDARD_DIR, categorize_art
from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet, decode_detections
from .table_scenes import SETUPS, render_table_scene


def ordinary_card_bank(n: int, seed: int) -> CardBank:
    """`n` ordinary-bordered cards from the main pool (`data/cards`), as the baseline the
    non-standard categories are compared against -- not the whole pool, which is itself
    already ~88% ordinary and would otherwise just restate that skew."""
    arts = {a["id"]: a for a in json.loads((DATA_DIR / "arts.json").read_text())}
    from .image_bank import list_cards

    ordinary = [p for p in list_cards() if categorize_art(arts.get(p.stem, {})) == "ordinary"]
    if not ordinary:
        raise SystemExit("no ordinary-bordered cards in data/cards; run cardid.scryfall --cards first")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(ordinary), min(n, len(ordinary)), replace=False)
    return CardBank([ordinary[i] for i in picked])


@torch.no_grad()
def score_bank(model: TableCenterNet, cards: CardBank, arts: ArtBank, scenes: int, cards_per_scene: int, seed: int, score_threshold: float, samples_dir=None, label: str = "") -> dict:
    hits_total, truth_total, found_total = 0, 0, 0
    for i in range(scenes):
        setup = SETUPS[i % len(SETUPS)]
        profile = "overhead_1080p"
        count = min(cards_per_scene, len(cards))
        image, record = render_table_scene(seed + i, cards, arts, setup, profile, count=count, size=1280, out=TABLE_INPUT)
        heat, pose, up = model(to_tensor(image)[None])
        detections = decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold)
        found = [q for q, _s in detections]
        truth = [np.float32(c["quad"]) for c in record["cards"]]
        hits = per_card_hits(truth, found, 0.5)
        hits_total += sum(hits)
        truth_total += len(hits)
        found_total += len(found)
        if samples_dir and i < 4:
            samples_dir.mkdir(parents=True, exist_ok=True)
            canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
            for quad in truth:
                cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
            for quad in found:
                cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 90, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(samples_dir / f"{label}-{i:02d}.jpg"), canvas)
    return {
        "recall": hits_total / truth_total if truth_total else None,
        "precision": hits_total / found_total if found_total else None,
        "n_truth": truth_total,
        "n_found": found_total,
    }


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = TableCenterNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    arts = ArtBank(list_arts())

    results = {}
    baseline = ordinary_card_bank(max(args.cards_per_scene * 4, 40), args.seed)
    results["ordinary"] = score_bank(model, baseline, arts, args.scenes_per_category, args.cards_per_scene, args.seed, args.score_threshold, args.samples_dir, "ordinary")
    print(f"ordinary   (n={len(baseline)}): {json.dumps(results['ordinary'])}")

    for category_dir in sorted(NONSTANDARD_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name not in CATEGORY_QUERIES:
            continue
        paths = sorted(category_dir.glob("*.jpg"))
        if len(paths) < 2:
            print(f"{category_dir.name}: skipped, only {len(paths)} images downloaded")
            continue
        cards = CardBank(paths)
        results[category_dir.name] = score_bank(
            model, cards, arts, args.scenes_per_category, min(args.cards_per_scene, len(cards)), args.seed, args.score_threshold, args.samples_dir, category_dir.name
        )
        print(f"{category_dir.name:14s} (n={len(cards)}): {json.dumps(results[category_dir.name])}")

    out = NONSTANDARD_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")
