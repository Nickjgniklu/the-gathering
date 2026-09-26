"""CPU benchmark for table-scene rendering throughput (`table_scenes.render_table_scene`)."""

from __future__ import annotations

import argparse
import time

from .image_bank import ArtBank, CardBank
from .table_scenes import SETUPS, render_table_scene


def bench_render(scenes: int, cards_per_scene: int, seed: int) -> dict:
    cards, arts = CardBank(), ArtBank()
    started = time.perf_counter()
    for i in range(scenes):
        render_table_scene(seed + i, cards, arts, SETUPS[i % len(SETUPS)], count=min(cards_per_scene, len(cards)))
    elapsed = time.perf_counter() - started
    return {"scenes": scenes, "cards_per_scene": cards_per_scene, "ms_per_scene": elapsed * 1000 / scenes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--cards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = bench_render(args.scenes, args.cards, args.seed)
    print(f"table scenes: {result['scenes']} scenes, {result['cards_per_scene']} cards/scene, {result['ms_per_scene']:.1f} ms/scene")


if __name__ == "__main__":
    main()
