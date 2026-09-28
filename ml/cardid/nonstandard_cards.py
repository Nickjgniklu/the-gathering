"""Download a dedicated sample of visually non-standard cards (borderless, showcase, extended
art, gold/silver border, full-art, textless, the timeshifted "future" frame) and measure the
table detector's recall on them versus ordinary-bordered cards -- research for whether the
training set (currently ~88% ordinary-bordered, see `ml/README.md`) needs deliberate oversampling
of these categories before the next retrain.

    uv run python -m cardid.nonstandard_cards download --per-category 40
    uv run python -m cardid.nonstandard_cards evaluate --checkpoint data/runs/table-a-pretrained-gpu/best.pt

Downloads go to `data/cards-nonstandard/<category>/<id>.jpg`, kept separate from the main
`data/cards` pool used for training until a decision is made on oversampling them into it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np

from . import DATA_DIR
from .downloads import IMAGE_MAX_BYTES, IMAGE_TYPES, DownloadError, decode_image, fetch_bytes, fetch_json, write_atomic
from .scryfall import HEADERS, REQUEST_GAP_S

NONSTANDARD_DIR = DATA_DIR / "cards-nonstandard"

# Scryfall search syntax per category; `game:paper` excludes digital-only (Arena/MTGO) prints.
CATEGORY_QUERIES = {
    "borderless": "border:borderless game:paper",
    "showcase": "is:showcase game:paper",
    "extended_art": "is:extendedart game:paper",
    "gold_border": "border:gold game:paper",
    "silver_border": "border:silver game:paper",
    "full_art": "is:full game:paper",
    "textless": "is:textless game:paper",
    "future_frame": "frame:future game:paper",
}


def search_cards(client: httpx.Client, query: str, limit: int) -> list[dict]:
    url, params = "https://api.scryfall.com/cards/search", {"q": query, "unique": "prints"}
    cards: list[dict] = []
    while url and len(cards) < limit:
        try:
            page = fetch_json(client, url, params=params)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:  # Scryfall's spelling of "zero matches"
                break
            raise
        cards.extend(page["data"])
        url = page.get("next_page") if page.get("has_more") else None
        params = None
        time.sleep(0.1)
    return cards[:limit]


def download_category(client: httpx.Client, category: str, query: str, limit: int, seed: int) -> int:
    out_dir = NONSTANDARD_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = search_cards(client, query, limit * 3)  # oversample the search, then pick a random subset
    rng = np.random.default_rng(seed)
    if len(cards) > limit:
        cards = [cards[i] for i in rng.choice(len(cards), limit, replace=False)]
    ok = 0
    for card in cards:
        dest = out_dir / f"{card['id']}.jpg"
        if dest.exists():
            ok += 1
            continue
        image_uris = card.get("image_uris") or (card.get("card_faces") or [{}])[0].get("image_uris")
        if not image_uris or "normal" not in image_uris:
            continue
        try:
            data = fetch_bytes(client, image_uris["normal"], max_bytes=IMAGE_MAX_BYTES, content_types=IMAGE_TYPES, timeout=30)
            time.sleep(REQUEST_GAP_S)
            decode_image(data, card=True)
            write_atomic(dest, data)
            ok += 1
        except (httpx.HTTPError, DownloadError):
            continue
    print(f"{category}: {ok}/{len(cards)} downloaded -> {out_dir}")
    return ok


def categorize_art(art: dict) -> str:
    """The same visual-category logic used to characterise the existing training pool
    (see `ml/README.md`'s non-standard-card research note), for filtering an "ordinary"
    baseline out of the main card pool."""
    border = art.get("border_color")
    effects = set(art.get("frame_effects") or [])
    if border == "borderless":
        return "borderless"
    if border in ("gold", "silver"):
        return f"{border}_border"
    if "showcase" in effects:
        return "showcase"
    if "extendedart" in effects:
        return "extended_art"
    if art.get("scryfall_frame") == "future":
        return "future_frame"
    return "ordinary"


def cmd_download(args: argparse.Namespace) -> None:
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        counts = {cat: download_category(client, cat, query, args.per_category, args.seed) for cat, query in CATEGORY_QUERIES.items()}
    (NONSTANDARD_DIR / "counts.json").write_text(json.dumps(counts, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="fetch a sample of each non-standard category")
    p_download.add_argument("--per-category", type=int, default=40)
    p_download.add_argument("--seed", type=int, default=0)

    p_eval = sub.add_parser("evaluate", help="score the table detector on each category vs. ordinary cards")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--scenes-per-category", type=int, default=30)
    p_eval.add_argument("--cards-per-scene", type=int, default=6)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--score-threshold", type=float, default=0.3)
    p_eval.add_argument("--samples-dir", type=Path, help="save a few sample overlay images per category here")

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    else:
        from .evaluate_nonstandard import cmd_evaluate

        cmd_evaluate(args)


if __name__ == "__main__":
    main()
