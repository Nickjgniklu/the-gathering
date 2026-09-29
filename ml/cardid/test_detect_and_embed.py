"""Geometry tests for `detect_and_embed.py`. The crop math was validated once by comparing
against `detect.warp_card` + `detect.art_crop` on a real rendered scene (mean abs pixel diff 2.7
of 255, entirely explained by interpolation method differences -- cv2's two-step warp+resize vs.
one `grid_sample` call), and again for all 14 frame hypotheses against `detect.art_crops`
(cosine similarity 0.985+ on every frame, including the six with a 90/180/270-degree rotation);
these tests lock in the underlying orientation-resolution and per-frame-box logic that made both
comparisons pass, without needing a real checkpoint or a full scene render."""

from __future__ import annotations

import math
import unittest

import torch

from .detect import FRAME_NAMES
from .detect_and_embed import N_FRAMES, _FRAME_BOXES, _FRAME_RAW_SIZE, _FRAME_ROTATION_K, _resolve_orientation, _topk_peaks


class ResolveOrientationTest(unittest.TestCase):
    def test_angle_kept_when_naive_direction_already_matches_up(self):
        # angle=0 -> naive direction (sin 0, -cos 0) = (0, -1): matches an up vector of (0, -1).
        angle = torch.tensor([0.0])
        up_x, up_y = torch.tensor([0.0]), torch.tensor([-1.0])
        result = _resolve_orientation(angle, up_x, up_y)
        self.assertAlmostEqual(float(result[0]), 0.0, places=5)

    def test_angle_flipped_by_pi_when_naive_direction_opposes_up(self):
        # Same angle=0 (naive direction (0,-1)) but up now points the opposite way (0, 1).
        angle = torch.tensor([0.0])
        up_x, up_y = torch.tensor([0.0]), torch.tensor([1.0])
        result = _resolve_orientation(angle, up_x, up_y)
        self.assertAlmostEqual(float(result[0]), math.pi, places=5)

    def test_flipped_angle_direction_actually_matches_up(self):
        # For arbitrary angles, whichever of (angle, angle+pi) resolve_orientation picks should
        # itself produce a naive direction with a *non-negative* dot product against up -- i.e.
        # the fixed-up orientation is self-consistent, not just "different from before".
        torch.manual_seed(0)
        angle = torch.rand(20) * 2 * math.pi - math.pi
        up_angle = torch.rand(20) * 2 * math.pi
        up_x, up_y = torch.cos(up_angle), torch.sin(up_angle)
        resolved = _resolve_orientation(angle, up_x, up_y)
        naive_x, naive_y = torch.sin(resolved), -torch.cos(resolved)
        dot = naive_x * up_x + naive_y * up_y
        self.assertTrue(torch.all(dot >= -1e-4))


class TopkPeaksTest(unittest.TestCase):
    def test_returns_exactly_k_slots_padding_with_zero_score_when_fewer_peaks_exist(self):
        size, stride, k = 12, 4, 20
        heat = torch.full((1, 1, size, size), -10.0)  # one strong peak, everything else quiet
        heat[0, 0, 5, 5] = 10.0
        pose = torch.zeros(1, 3, size, size)
        up = torch.zeros(1, 2, size, size)
        up[0, 1, 5, 5] = -1.0  # a well-formed unit up vector at the peak
        cx, cy, short, angle, up_x, up_y, scores = _topk_peaks(heat, pose, up, stride, k)
        self.assertEqual(scores.shape, (1, k))
        self.assertGreater(float(scores[0, 0]), 0.9)  # the real peak, sigmoid(10) ~= 1
        self.assertTrue(torch.all(scores[0, 1:] < 1e-3))  # every other slot is padding

    def test_top_slot_locates_the_correct_grid_cell(self):
        size, stride, k = 12, 4, 5
        heat = torch.full((1, 1, size, size), -10.0)
        heat[0, 0, 3, 7] = 10.0  # (y=3, x=7)
        pose = torch.zeros(1, 3, size, size)
        up = torch.zeros(1, 2, size, size)
        cx, cy, short, angle, up_x, up_y, scores = _topk_peaks(heat, pose, up, stride, k)
        best = int(torch.argmax(scores[0]))
        self.assertAlmostEqual(float(cx[0, best]), (7 + 0.5) * stride, places=4)
        self.assertAlmostEqual(float(cy[0, best]), (3 + 0.5) * stride, places=4)


class FrameGeometryTest(unittest.TestCase):
    def test_one_box_size_and_rotation_per_frame_name(self):
        self.assertEqual(N_FRAMES, 14)
        self.assertEqual(len(FRAME_NAMES), N_FRAMES)
        self.assertEqual(len(_FRAME_BOXES), N_FRAMES)
        self.assertEqual(len(_FRAME_RAW_SIZE), N_FRAMES)
        self.assertEqual(len(_FRAME_ROTATION_K), N_FRAMES)

    def test_every_raw_size_is_a_positive_pixel_count(self):
        for w, h in _FRAME_RAW_SIZE:
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    def test_rotation_counts_are_quarter_turns(self):
        for k in _FRAME_ROTATION_K:
            self.assertIn(k, (0, 1, 2, 3))

    def test_two_part_frames_are_the_ones_with_a_rotation(self):
        # Matches detect.FRAME_ROTATIONS exactly: every two-part frame except aftermath_0 and
        # flip_0 rotates; no single-box frame (modern/old/extended/tall/right/left) does.
        rotated = {name for name, k in zip(FRAME_NAMES, _FRAME_ROTATION_K) if k}
        self.assertEqual(rotated, {"room_0", "room_1", "split_0", "split_1", "aftermath_1", "flip_1"})


if __name__ == "__main__":
    unittest.main()
