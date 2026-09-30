"""Contracts for `track_memory.py`'s two stages: `associate_detections` (classical, not learned)
and `TrackMemory` (the GRU). Neither needs a real checkpoint or rendered scene -- both operate on
hand-built tensors/arrays, same convention as `test_detect_and_embed.py`."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .track_memory import HIDDEN_DIM, Track, TrackMemory, associate_detections


def _track(track_id: int, center: tuple[float, float]) -> Track:
    return Track(track_id=track_id, hidden=torch.zeros(HIDDEN_DIM), last_center=np.array(center, dtype=np.float64))


class AssociateDetectionsTest(unittest.TestCase):
    def test_empty_tracks_makes_every_detection_unmatched(self):
        matched, unmatched = associate_detections([], np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(matched, {})
        self.assertEqual(unmatched, [0, 1])

    def test_empty_detections_matches_nothing(self):
        matched, unmatched = associate_detections([_track(0, (0, 0))], np.empty((0, 2)))
        self.assertEqual(matched, {})
        self.assertEqual(unmatched, [])

    def test_resolves_the_module_docstrings_rank_swap_example(self):
        # frame 1: slot 0 -> track 0 at (100,200), slot 1 -> track 1 at (400,150), slot 2 -> track
        # 2 at (250,500). frame 2, cards untouched: detections arrive in a different rank order.
        tracks = [_track(0, (100, 200)), _track(1, (400, 150)), _track(2, (250, 500))]
        detections = np.array([[400.0, 148.0], [101.0, 199.0], [251.0, 501.0]])
        matched, unmatched = associate_detections(tracks, detections)
        self.assertEqual(unmatched, [])
        self.assertEqual(matched, {0: 1, 1: 0, 2: 2})  # by nearest position, not by rank/order

    def test_a_detection_far_from_every_track_is_unmatched_not_forced(self):
        tracks = [_track(0, (0, 0))]
        detections = np.array([[1000.0, 1000.0]])
        matched, unmatched = associate_detections(tracks, detections, gate=40.0)
        self.assertEqual(matched, {})
        self.assertEqual(unmatched, [0])

    def test_closer_detection_wins_a_contested_track_the_other_stays_unmatched(self):
        tracks = [_track(0, (0, 0))]
        detections = np.array([[5.0, 0.0], [1.0, 0.0]])  # both near the one track; only one can claim it
        matched, unmatched = associate_detections(tracks, detections)
        self.assertEqual(matched, {1: 0})
        self.assertEqual(unmatched, [0])


class TrackMemoryTest(unittest.TestCase):
    def test_output_shapes(self):
        model = TrackMemory()
        b = 3
        hidden = torch.zeros(b, HIDDEN_DIM)
        embedding_t = torch.nn.functional.normalize(torch.randn(b, 128), dim=-1)
        pose_t = torch.randn(b, 5)
        score_t = torch.rand(b, 1)
        new_hidden, refined_embedding, refined_pose = model(hidden, embedding_t, pose_t, score_t)
        self.assertEqual(new_hidden.shape, (b, HIDDEN_DIM))
        self.assertEqual(refined_embedding.shape, (b, 128))
        self.assertEqual(refined_pose.shape, (b, 5))

    def test_refined_embedding_is_l2_normalized(self):
        model = TrackMemory()
        hidden = torch.zeros(2, HIDDEN_DIM)
        embedding_t = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        pose_t, score_t = torch.randn(2, 5), torch.rand(2, 1)
        _, refined_embedding, _ = model(hidden, embedding_t, pose_t, score_t)
        norms = refined_embedding.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_gradients_flow_to_every_parameter(self):
        # A from-scratch GRUCell with an unlucky init can silently zero a gate; this is the cheap
        # regression check for that, not a claim about what the model has learned. A genuinely
        # zero starting `hidden` would make weight_hh's gradient trivially (and correctly) zero
        # for this one step, so use a nonzero one like a real mid-sequence step would have.
        model = TrackMemory()
        hidden = torch.randn(2, HIDDEN_DIM)
        embedding_t = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        pose_t, score_t = torch.randn(2, 5, requires_grad=True), torch.rand(2, 1)
        new_hidden, refined_embedding, refined_pose = model(hidden, embedding_t, pose_t, score_t)
        (new_hidden.sum() + refined_embedding.sum() + refined_pose.sum()).backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")
            self.assertGreater(float(param.grad.abs().sum()), 0.0, f"{name} got an all-zero gradient")


if __name__ == "__main__":
    unittest.main()
