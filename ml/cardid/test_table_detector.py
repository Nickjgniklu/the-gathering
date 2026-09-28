"""CPU-only contracts for the dense multi-card detector (`table_detector.py`).

`TableCenterNet(pretrained=False)` never touches the network, so this suite is fast and
offline like the rest of `mise run ml:test`; the architecture, target construction, loss, and
decode are exercised independently of any real training run or checkpoint.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .table_detector import TableCenterNet, build_targets, decode_detections, table_detector_loss


class TableCenterNetTest(unittest.TestCase):
    def test_forward_shapes_match_the_stride(self):
        model = TableCenterNet(pretrained=False).eval()
        x = torch.zeros(2, 3, 64, 64)
        heat, pose, up = model(x)
        self.assertEqual(heat.shape, (2, 1, 16, 16))  # stride 4
        self.assertEqual(pose.shape, (2, 3, 16, 16))
        self.assertEqual(up.shape, (2, 2, 16, 16))

    def test_backbone_and_head_parameters_partition_all_parameters(self):
        model = TableCenterNet(pretrained=False)
        backbone_ids = {id(p) for p in model.backbone_parameters()}
        head_ids = {id(p) for p in model.head_parameters()}
        all_ids = {id(p) for p in model.parameters()}
        self.assertEqual(backbone_ids | head_ids, all_ids)
        self.assertEqual(backbone_ids & head_ids, set())

    def test_heat_bias_starts_at_the_focal_loss_prior(self):
        model = TableCenterNet(pretrained=False)
        self.assertAlmostEqual(float(model.head.bias[0].detach()), -2.19, places=2)


class BuildTargetsTest(unittest.TestCase):
    def test_single_card_marks_its_nearest_cell_and_no_other(self):
        heat, pose, up, mask, _hard_neg = build_targets([(100.0, 100.0, 60.0, 0.0)], [np.float32([0, -1])], image_size=256, stride=4)
        self.assertEqual(mask.sum(), 1.0)
        cy, cx = 24, 24  # cell (i+0.5)*4 = 100 -> i = round(25.0-0.5) = round(24.5) = 24
        self.assertEqual(mask[0, cy, cx], 1.0)
        np.testing.assert_allclose(pose[:, cy, cx], [np.log(60.0), 1.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(up[:, cy, cx], [0, -1], atol=1e-6)
        self.assertGreater(heat[0, cy, cx], 0.99)

    def test_heatmap_falls_off_away_from_the_centre_but_stays_within_bounds(self):
        heat, _pose, _up, _mask, _hard_neg = build_targets([(32.0, 32.0, 20.0, 0.0)], [np.float32([0, -1])], image_size=64, stride=4)
        self.assertTrue((heat >= 0).all() and (heat <= 1).all())
        self.assertLess(heat[0, 0, 0], heat[0, 8, 8])  # far corner is cooler than the centre cell

    def test_no_cards_gives_all_zero_targets(self):
        heat, pose, up, mask, hard_neg = build_targets([], [], image_size=64, stride=4)
        self.assertEqual(heat.sum(), 0.0)
        self.assertEqual(pose.sum(), 0.0)
        self.assertEqual(up.sum(), 0.0)
        self.assertEqual(mask.sum(), 0.0)
        self.assertEqual(hard_neg.sum(), 0.0)

    def test_negative_box_marks_covered_cells_as_hard_negative(self):
        _heat, _pose, _up, _mask, hard_neg = build_targets([], [], image_size=64, stride=4, negatives=[(20.0, 20.0, 28.0, 28.0)])
        self.assertGreater(hard_neg.sum(), 0.0)
        self.assertEqual(hard_neg[0, 5, 5], 1.0)  # cell (5+0.5)*4 = 22, inside [20, 28]
        self.assertEqual(hard_neg[0, 0, 0], 0.0)


class TableDetectorLossTest(unittest.TestCase):
    def test_perfect_prediction_gives_near_zero_pose_and_up_loss(self):
        # Off-grid coordinates (not exactly halfway between cells) so the normalised Gaussian
        # has one unambiguous peak cell instead of a tie between several equidistant ones.
        heat, pose, up, mask, _hard_neg = build_targets([(33.4, 29.7, 20.0, 15.0)], [np.float32([0.3, -0.9])], image_size=64, stride=4)
        t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
        heat_logits = torch.full_like(t(heat), -10.0)
        cy, cx = np.argwhere(mask[0] == 1.0)[0]
        heat_logits[0, 0, cy, cx] = 10.0
        loss, parts = table_detector_loss(heat_logits, t(pose), t(up), t(heat), t(pose), t(up), t(mask))
        self.assertLess(parts["pose"], 1e-4)
        self.assertLess(parts["up"], 1e-4)
        self.assertLess(float(loss), 0.1)

    def test_wrong_prediction_costs_more_than_correct(self):
        heat, pose, up, mask, _hard_neg = build_targets([(32.0, 32.0, 20.0, 0.0)], [np.float32([0, -1])], image_size=64, stride=4)
        t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
        heat_logits = torch.zeros_like(t(heat))
        good, _ = table_detector_loss(heat_logits, t(pose), t(up), t(heat), t(pose), t(up), t(mask))
        bad, _ = table_detector_loss(heat_logits, t(pose) + 1.0, t(up) * -1, t(heat), t(pose), t(up), t(mask))
        self.assertGreater(float(bad), float(good))


class DecodeDetectionsTest(unittest.TestCase):
    def test_recovers_a_single_known_peak(self):
        size = 16
        heat = torch.full((1, size, size), -10.0)
        heat[0, 8, 8] = 10.0
        pose = torch.zeros(3, size, size)
        pose[:, 8, 8] = torch.tensor([np.log(40.0), 1.0, 0.0])
        up = torch.zeros(2, size, size)
        up[:, 8, 8] = torch.tensor([0.0, -1.0])
        detections = decode_detections(heat, pose, up, stride=4, score_threshold=0.5)
        self.assertEqual(len(detections), 1)
        quad, score = detections[0]
        self.assertEqual(quad.shape, (4, 2))
        self.assertGreater(score, 0.99)
        cx, cy = quad.mean(axis=0)
        self.assertAlmostEqual(float(cx), 8.5 * 4, delta=1e-3)
        self.assertAlmostEqual(float(cy), 8.5 * 4, delta=1e-3)

    def test_below_threshold_finds_nothing(self):
        size = 8
        heat = torch.full((1, size, size), -10.0)
        pose = torch.zeros(3, size, size)
        up = torch.zeros(2, size, size)
        self.assertEqual(decode_detections(heat, pose, up, stride=4, score_threshold=0.5), [])

    def test_two_far_apart_peaks_are_both_recovered(self):
        size = 24
        heat = torch.full((1, size, size), -10.0)
        heat[0, 2, 2] = 10.0
        heat[0, 20, 20] = 10.0
        pose = torch.zeros(3, size, size)
        pose[:, 2, 2] = pose[:, 20, 20] = torch.tensor([np.log(30.0), 1.0, 0.0])
        up = torch.zeros(2, size, size)
        up[:, 2, 2] = up[:, 20, 20] = torch.tensor([0.0, -1.0])
        detections = decode_detections(heat, pose, up, stride=4, score_threshold=0.5)
        self.assertEqual(len(detections), 2)


if __name__ == "__main__":
    unittest.main()
