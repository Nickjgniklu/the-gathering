"""Geometry-only tests for `tiled_inference.py` (no torch/model needed)."""

from __future__ import annotations

import unittest

import numpy as np

from .tiled_inference import dedupe, tile_boxes


class TileBoxesTest(unittest.TestCase):
    def test_single_tile_covers_the_whole_image(self):
        boxes = tile_boxes(100, 100, (1, 1), overlap=0.2)
        self.assertEqual(boxes, [(0.0, 0.0, 100.0, 100.0)])

    def test_two_by_two_grid_has_four_tiles_all_within_bounds(self):
        boxes = tile_boxes(100, 100, (2, 2), overlap=0.2)
        self.assertEqual(len(boxes), 4)
        for x0, y0, x1, y1 in boxes:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, 100)
            self.assertLessEqual(y1, 100)

    def test_neighbouring_tiles_actually_overlap(self):
        boxes = tile_boxes(100, 100, (1, 2), overlap=0.2)
        left, right = boxes
        self.assertGreater(left[2], right[0])  # left tile's right edge passes right tile's left edge

    def test_zero_overlap_tiles_touch_without_gaps_or_double_coverage(self):
        boxes = tile_boxes(100, 100, (1, 2), overlap=0.0)
        left, right = boxes
        self.assertAlmostEqual(left[2], right[0], places=4)


class DedupeTest(unittest.TestCase):
    def test_two_overlapping_detections_of_the_same_card_merge_into_one_reporting_the_top_score(self):
        quad_a = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
        quad_b = np.float32([[1, 1], [11, 1], [11, 11], [1, 11]])  # near-identical box, high IoU
        kept = dedupe([(quad_a, 0.6), (quad_b, 0.9)], iou_threshold=0.4)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0][1], 0.9, places=5)
        # the merged quad is a score-weighted average, not either raw input, so it should land
        # strictly between the two (closer to b, the higher-weighted one) on every corner
        merged = kept[0][0]
        self.assertTrue(np.all(merged > quad_a))
        self.assertTrue(np.all(merged < quad_b))

    def test_two_far_apart_detections_are_both_kept(self):
        quad_a = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
        quad_b = np.float32([[50, 50], [60, 50], [60, 60], [50, 60]])
        kept = dedupe([(quad_a, 0.6), (quad_b, 0.9)], iou_threshold=0.4)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
