"""Regenerate a table-scenes dataset restricted to cards that exist in a deployed embedding
gallery's `arts.json`, so every synthetic card can actually be checked against `search.onnx` --
`detect_and_embed.py`'s validation found 196 of the local 767-card pool's cards (25.5%) are not
in the deployed gallery at all, making any card rendered from them permanently "unfindable" and
useless for detect+embed accuracy testing/training, regardless of how good detection or embedding
are. Not a permanent CLI flag: this is a one-off filter for that specific purpose, applied by
monkeypatching `image_bank.list_cards` rather than changing `table_scenes.py`'s own card-loading
(which every other dataset generation should keep using unfiltered).

    uv run python -m cardid.regen_gallery_verified --gallery H:\the-gathering-cardid\current\arts.json --out H:\the-gathering-cardid\table-scenes-1920 --train 4000 --val 300 --size 1920 --resolution 1920
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import image_bank, table_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery", type=Path, required=True, help="a deployed bundle's arts.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=table_scenes.DEFAULT_SCENES["train"])
    parser.add_argument("--val", type=int, default=table_scenes.DEFAULT_SCENES["val"])
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--challenge", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--size", type=int, default=1920, help="native render resolution before downscale")
    parser.add_argument("--resolution", type=int, default=1920, help="output image resolution")
    args = parser.parse_args()

    gallery_ids = {a["id"] for a in json.loads(args.gallery.read_text(encoding="utf-8"))}
    all_cards = image_bank.list_cards()
    filtered = [p for p in all_cards if p.stem.removesuffix("-1") in gallery_ids]
    print(f"cards: {len(all_cards)} local -> {len(filtered)} present in {args.gallery.name} ({len(gallery_ids)} arts)")
    if not filtered:
        raise SystemExit("no local cards overlap with this gallery; check --gallery points at the right bundle")

    image_bank.list_cards = lambda: filtered  # table_scenes.write_dataset's CardBank() reads this

    scenes = {"train": args.train, "val": args.val, "test": args.test, "challenge": args.challenge}
    scenes = {k: v for k, v in scenes.items() if v > 0}
    header = table_scenes.write_dataset(args.out, args.seed, scenes, args.size, args.resolution)
    print(json.dumps({k: v for k, v in header.items() if k != "split_rules"}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
