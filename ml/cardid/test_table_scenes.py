"""CPU-only contracts for table-scene seeds, split boundaries, and multi-card metrics.

Every test builds its own tiny card bank (a handful of solid-colour JPEGs) and an empty art
bank (`background()` falls back to flat mats when there is no art, see `scene_renderer.py`),
so nothing here touches the Scryfall downloads or a trained checkpoint; the suite stays fast
and runs the same on any machine `mise run ml:test` runs on.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from . import table_scenes
from .evaluate_tables import per_card_hits
from .image_bank import ArtBank, CardBank
from .scene_geometry import quad_from_pose


def make_card_bank(root: Path, n: int = 8) -> CardBank:
    """`n` small, distinctly-coloured portrait JPEGs: enough pixel content for the renderer
    and the classical quad finder to work with, without downloading anything."""
    colors = [(40, 40, 200), (40, 200, 40), (200, 40, 40), (200, 200, 40), (40, 200, 200), (200, 40, 200), (120, 120, 120), (240, 240, 240)]
    paths = []
    for i in range(n):
        img = np.full((140, 100, 3), colors[i % len(colors)], np.uint8)
        cv2.rectangle(img, (4, 4), (95, 135), (0, 0, 0), 3)  # a border, so edge detection has something to find
        path = root / f"card{i}.jpg"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return CardBank(paths)


class TableSceneGeometryTest(unittest.TestCase):
    def test_seeded_geometry_is_repeatable(self):
        first = quad_from_pose(100, 100, 30, 90, np.random.default_rng(9))
        second = quad_from_pose(100, 100, 30, 90, np.random.default_rng(9))
        np.testing.assert_array_equal(first, second)

    def test_unrotated_corners_are_printed_order_top_left_top_right_bottom_right_bottom_left(self):
        tl, tr, br, bl = quad_from_pose(100, 100, 30, 0, None)
        self.assertLess(tl[0], tr[0])
        self.assertAlmostEqual(tl[1], tr[1], places=4)
        self.assertLess(tl[1], bl[1])
        self.assertAlmostEqual(bl[1], br[1], places=4)

    def test_multi_card_matching_is_greedy_one_to_one_not_double_counted(self):
        left = quad_from_pose(40, 40, 20, 0, None)
        right = quad_from_pose(100, 100, 20, 0, None)
        # Two proposals both land on `left`; only one can claim it, and `right` has none.
        hits = per_card_hits([left, right], [left, left.copy()])
        self.assertEqual(hits, [True, False])


class SplitConfigTest(unittest.TestCase):
    def test_every_split_is_configured_the_same_way(self):
        for split in table_scenes.SPLITS:
            self.assertIn(split, table_scenes.SPLIT_SETUPS)
            self.assertIn(split, table_scenes.SPLIT_CAMERA_PROFILES)
            self.assertIn(split, table_scenes.SPLIT_BACKGROUND_POOL)
            self.assertIn(split, table_scenes.SPLIT_SEVERITY)
            self.assertIn(split, table_scenes.SPLIT_DENSITY_WEIGHTS)
            self.assertTrue(set(table_scenes.SPLIT_SETUPS[split]) <= set(table_scenes.SETUPS))
            self.assertTrue(set(table_scenes.SPLIT_CAMERA_PROFILES[split]) <= set(table_scenes.CAMERA_PROFILES))

    def test_train_never_uses_the_setups_or_camera_profiles_held_out_for_val_and_test(self):
        train_setups = set(table_scenes.SPLIT_SETUPS["train"])
        train_profiles = set(table_scenes.SPLIT_CAMERA_PROFILES["train"])
        for split in ("val", "test"):
            self.assertTrue(train_setups.isdisjoint(table_scenes.SPLIT_SETUPS[split]), f"{split} setup leaked into train")
            self.assertTrue(train_profiles.isdisjoint(table_scenes.SPLIT_CAMERA_PROFILES[split]), f"{split} camera profile leaked into train")

    def test_challenge_reuses_test_conditions_at_higher_severity_and_density(self):
        self.assertEqual(table_scenes.SPLIT_SETUPS["challenge"], table_scenes.SPLIT_SETUPS["test"])
        self.assertEqual(table_scenes.SPLIT_BACKGROUND_POOL["challenge"], table_scenes.SPLIT_BACKGROUND_POOL["test"])
        self.assertGreater(table_scenes.SPLIT_SEVERITY["challenge"], table_scenes.SPLIT_SEVERITY["test"])
        self.assertEqual(table_scenes.SPLIT_DENSITY_WEIGHTS["challenge"]["sparse"], 0.0)

    def test_densities_partition_zero_to_twenty_without_gaps_or_overlap(self):
        ranges = sorted(table_scenes.DENSITIES.values())
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 20)
        for (_, hi), (lo2, _) in pairwise(ranges):
            self.assertEqual(hi + 1, lo2)


class GridCapacityTest(unittest.TestCase):
    def test_zero_rows_when_the_footprint_does_not_fit_the_band_height(self):
        # Regression: duel/battlefield split the canvas into two half-height bands: forcing at
        # least one row here (the old behaviour) let a card bleed past its band into the other.
        cols, rows = table_scenes._grid_capacity(1000.0, 100.0, 150.0)
        self.assertEqual(rows, 0)
        self.assertGreater(cols, 0)  # the other dimension is unaffected

    def test_nonzero_when_the_footprint_fits(self):
        cols, rows = table_scenes._grid_capacity(1000.0, 300.0, 150.0)
        self.assertGreaterEqual(cols, 6)
        self.assertGreaterEqual(rows, 2)


class PartitionArtsTest(unittest.TestCase):
    def test_pools_are_disjoint_and_cover_every_path(self):
        paths = [Path(f"art{i}.jpg") for i in range(40)]
        pools = table_scenes.partition_arts(paths, seed=3)
        self.assertEqual(set(pools), {"train", "val", "test"})
        seen = [p for pool in pools.values() for p in pool]
        self.assertEqual(sorted(seen), sorted(paths))
        self.assertEqual(len(set(seen)), len(paths))  # no path in more than one pool

    def test_deterministic_for_the_same_seed(self):
        paths = [Path(f"art{i}.jpg") for i in range(40)]
        self.assertEqual(table_scenes.partition_arts(paths, seed=5), table_scenes.partition_arts(paths, seed=5))

    def test_falls_back_to_sharing_when_too_few_paths_to_split(self):
        paths = [Path("a.jpg"), Path("b.jpg")]
        pools = table_scenes.partition_arts(paths, seed=0)
        self.assertEqual(pools["train"], pools["val"])
        self.assertEqual(pools["train"], pools["test"])


class RenderTableSceneTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cards = make_card_bank(self.root)
        self.arts = ArtBank([])

    def test_rejects_unknown_setup_camera_profile_or_out_of_range_count(self):
        with self.assertRaises(ValueError):
            table_scenes.render_table_scene(0, self.cards, self.arts, "not-a-setup")
        with self.assertRaises(ValueError):
            table_scenes.render_table_scene(0, self.cards, self.arts, "lanes", camera_profile="not-a-profile")
        with self.assertRaises(ValueError):
            table_scenes.render_table_scene(0, self.cards, self.arts, "lanes", count=len(self.cards) + 1)
        with self.assertRaises(ValueError):
            table_scenes.render_table_scene(0, self.cards, self.arts, "lanes", count=-1)

    def test_zero_cards_renders_an_empty_but_valid_scene(self):
        image, record = table_scenes.render_table_scene(0, self.cards, self.arts, "spread", count=0, size=256, out=128)
        self.assertEqual(image.shape, (128, 128, 3))
        self.assertEqual(record["cards"], [])

    def test_every_setup_and_camera_profile_produces_valid_non_overlapping_fields(self):
        # Count is a request, not a promise: a scene never overlaps cards to hit it (see
        # `_poses`), so a small canvas at a large camera-profile scale may place fewer.
        for setup in table_scenes.SETUPS:
            for profile in table_scenes.CAMERA_PROFILES:
                with self.subTest(setup=setup, profile=profile):
                    image, record = table_scenes.render_table_scene(1, self.cards, self.arts, setup, profile, count=5, size=1024, out=256)
                    self.assertEqual(image.dtype, np.uint8)
                    self.assertEqual(image.shape, (256, 256, 3))
                    self.assertTrue(1 <= len(record["cards"]) <= 5)
                    for card in record["cards"]:
                        self.assertEqual(len(card["quad"]), 4)
                        self.assertEqual(len(card["bbox"]), 4)
                        self.assertTrue(0 <= card["orientation"] < 360)
                        self.assertTrue(0.0 <= card["occluded_fraction"] <= 1.0)
                        self.assertIsInstance(card["identifiable"], bool)
                        x0, _y0, _x1, y1 = card["bbox"]
                        quad = np.float32(card["quad"])
                        self.assertAlmostEqual(x0, float(quad[:, 0].min()), places=3)
                        self.assertAlmostEqual(y1, float(quad[:, 1].max()), places=3)

    def test_generous_canvas_reaches_the_requested_count(self):
        _, record = table_scenes.render_table_scene(1, self.cards, self.arts, "lanes", "overhead_1080p", count=5, size=1280, out=256)
        self.assertEqual(len(record["cards"]), 5)

    def test_cards_never_overlap_regardless_of_setup_or_density(self):
        # stack_rate=0: this is the grid layout's own non-overlap invariant, not the separate
        # deliberate-stacking feature (see StackingTest below), which intentionally overlaps.
        from .scene_geometry import quad_iou

        for setup in table_scenes.SETUPS:
            with self.subTest(setup=setup):
                _, record = table_scenes.render_table_scene(1, self.cards, self.arts, setup, "overhead_1080p", count=8, size=1280, out=256, stack_rate=0.0)
                quads = [np.float32(c["quad"]) for c in record["cards"]]
                for i, a in enumerate(quads):
                    for b in quads[i + 1 :]:
                        self.assertEqual(quad_iou(a, b), 0.0)

    def test_duel_bands_never_overlap_even_at_closeup_scale(self):
        # Regression: closeup_4k's largest short_frac makes the footprint bigger than half the
        # canvas height, the exact case that used to let a duel band's row bleed into the other.
        from .scene_geometry import quad_iou

        for seed in range(20):
            _, record = table_scenes.render_table_scene(
                seed, self.cards, self.arts, "duel", "closeup_4k", count=len(self.cards), size=1280, out=256, stack_rate=0.0
            )
            quads = [np.float32(c["quad"]) for c in record["cards"]]
            for i, a in enumerate(quads):
                for b in quads[i + 1 :]:
                    self.assertEqual(quad_iou(a, b), 0.0, f"seed {seed}: overlapping duel cards")

    def test_a_tight_canvas_clips_the_count_instead_of_overlapping_cards(self):
        _, record = table_scenes.render_table_scene(1, self.cards, self.arts, "cluster", "closeup_4k", count=len(self.cards), size=512, out=256)
        self.assertLess(len(record["cards"]), len(self.cards))

    def test_same_seed_is_byte_identical(self):
        image_a, record_a = table_scenes.render_table_scene(42, self.cards, self.arts, "cluster", count=4, size=256, out=128)
        image_b, record_b = table_scenes.render_table_scene(42, self.cards, self.arts, "cluster", count=4, size=256, out=128)
        np.testing.assert_array_equal(image_a, image_b)
        self.assertEqual(json.dumps(record_a, sort_keys=True), json.dumps(record_b, sort_keys=True))

    def test_full_occlusion_is_flagged_not_identifiable(self):
        # Two cards at the same pose (short=60 native px -> 30 output px, above the
        # identifiable-size floor): the earlier one is entirely covered by the later one.
        with patch.object(table_scenes, "_poses", return_value=[(100, 100, 60, 0), (100, 100, 60, 0)]):
            _, record = table_scenes.render_table_scene(
                7,
                self.cards,
                self.arts,
                "spread",
                count=2,
                size=256,
                out=128,
                stack_rate=0.0,
                clutter_rate=0.0,
                round_negative_rate=0.0,
                real_clutter_rate=0.0,
            )
        first, second = record["cards"]
        self.assertGreater(first["occluded_fraction"], 0.9)
        self.assertFalse(first["identifiable"])
        self.assertTrue(second["identifiable"])

    def test_tiny_cards_are_flagged_not_identifiable_even_when_unoccluded(self):
        _, record = table_scenes.render_table_scene(3, self.cards, self.arts, "spread", "angled_720p", count=1, size=4000, out=64)
        card = record["cards"][0]
        self.assertEqual(card["occluded_fraction"], 0.0)
        from .scene_geometry import quad_short

        self.assertLess(quad_short(np.float32(card["quad"])), table_scenes.IDENTIFIABLE_MIN_SHORT_PX)
        self.assertFalse(card["identifiable"])

    def test_stack_rate_zero_never_reports_a_stacked_pair(self):
        for seed in range(10):
            _, record = table_scenes.render_table_scene(seed, self.cards, self.arts, "lanes", count=6, size=1280, out=256, stack_rate=0.0)
            self.assertIsNone(record["stacked_pair"])

    def test_stack_rate_one_forces_a_heavily_occluded_pair(self):
        _, record = table_scenes.render_table_scene(1, self.cards, self.arts, "lanes", count=6, size=1280, out=256, stack_rate=1.0)
        victim, topper = record["stacked_pair"]
        self.assertGreater(record["cards"][victim]["occluded_fraction"], 0.5)
        self.assertEqual(record["cards"][topper]["occluded_fraction"], 0.0)

    def test_clutter_and_round_negatives_carry_no_card_ground_truth(self):
        _, record = table_scenes.render_table_scene(2, self.cards, self.arts, "lanes", count=4, size=1280, out=256, clutter_rate=1.0, round_negative_rate=1.0)
        kinds = {n["kind"] for n in record["negatives"]}
        self.assertEqual(kinds, {"clutter", "round_object"})
        self.assertEqual(len(record["cards"]), 4)  # negatives never become (or displace) a card


class WriteDatasetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cards = make_card_bank(self.root / "bank")
        self.out = self.root / "out"

    def test_writes_every_split_with_a_matching_dataset_header(self):
        with patch.object(table_scenes, "CardBank", return_value=self.cards), patch.object(table_scenes, "list_arts", return_value=[]):
            header = table_scenes.write_dataset(self.out, seed=11, scenes={"train": 3, "val": 2, "test": 2, "challenge": 1}, size=256, out=128)
        self.assertEqual(header["renderer_version"], table_scenes.RENDERER_VERSION)
        self.assertEqual(header["cards_available"], len(self.cards))
        for split, count in (("train", 3), ("val", 2), ("test", 2), ("challenge", 1)):
            self.assertEqual(header["splits"][split]["scenes"], count)
            rows = (self.out / split / "manifest.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), count)
            for line in rows:
                row = json.loads(line)
                self.assertEqual(row["split"], split)
                self.assertIn(row["setup"], table_scenes.SPLIT_SETUPS[split])
                self.assertIn(row["camera_profile"], table_scenes.SPLIT_CAMERA_PROFILES[split])
                self.assertEqual(len(row["sha256"]), 64)
                self.assertTrue((self.out / split / row["image"]).exists())

    def test_regenerating_one_split_preserves_the_others_in_the_header(self):
        with patch.object(table_scenes, "CardBank", return_value=self.cards), patch.object(table_scenes, "list_arts", return_value=[]):
            table_scenes.write_dataset(self.out, seed=1, scenes={"train": 2, "val": 1, "test": 1, "challenge": 1}, size=256, out=128)
            header = table_scenes.write_dataset(self.out, seed=1, scenes={"test": 3}, size=256, out=128)
        self.assertEqual(header["splits"]["train"]["scenes"], 2)  # untouched
        self.assertEqual(header["splits"]["test"]["scenes"], 3)  # regenerated


if __name__ == "__main__":
    unittest.main()
