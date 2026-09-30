"""Contracts for `train_track_memory.py`'s two pure(ish) building blocks: `step_sequence`
(unrolling `TrackMemory` across a padded batch, with padding frozen out of the hidden-state
trajectory) and `track_memory_loss` (masked cross-entropy against the fixed gallery plus a pose
regularizer). `main()`'s CLI/IO orchestration isn't unit-tested here, matching this project's
convention for its other trainers (`train_tiled_fusion.py` has no test file of its own either)."""

from __future__ import annotations

import unittest

import torch

from .track_memory import HIDDEN_DIM, TrackMemory
from .train_track_memory import step_sequence, track_memory_loss


class StepSequenceTest(unittest.TestCase):
    def test_output_shapes(self):
        model = TrackMemory()
        b, t = 3, 4
        embeddings = torch.nn.functional.normalize(torch.randn(b, t, 128), dim=-1)
        poses, scores = torch.randn(b, t, 5), torch.rand(b, t, 1)
        valid = torch.ones(b, t, dtype=torch.bool)
        refined_embeddings, refined_poses = step_sequence(model, embeddings, poses, scores, valid)
        self.assertEqual(refined_embeddings.shape, (b, t, 128))
        self.assertEqual(refined_poses.shape, (b, t, 5))

    def test_hidden_state_is_frozen_through_a_padding_gap(self):
        # Two runs, identical except for what garbage sits at an invalid mid-sequence timestep:
        # since that timestep is masked invalid, it must not influence anything computed *after*
        # it -- the whole point of freezing `hidden` back to its pre-step value on padding.
        model = TrackMemory()
        torch.manual_seed(0)
        b, t = 1, 3
        embeddings = torch.nn.functional.normalize(torch.randn(b, t, 128), dim=-1)
        poses, scores = torch.randn(b, t, 5), torch.rand(b, t, 1)
        valid = torch.tensor([[True, False, True]])

        embeddings_b = embeddings.clone()
        embeddings_b[0, 1] = torch.nn.functional.normalize(torch.randn(128), dim=-1)  # different garbage at the gap
        poses_b, scores_b = poses.clone(), scores.clone()
        poses_b[0, 1] = torch.randn(5)
        scores_b[0, 1] = torch.rand(1)

        with torch.no_grad():
            refined_a, _ = step_sequence(model, embeddings, poses, scores, valid)
            refined_b, _ = step_sequence(model, embeddings_b, poses_b, scores_b, valid)

        torch.testing.assert_close(refined_a[0, 0], refined_b[0, 0])  # before the gap: identical inputs -> identical
        torch.testing.assert_close(refined_a[0, 2], refined_b[0, 2])  # after the gap: gap content shouldn't matter
        # the gap timestep itself is free to differ (its output is discarded by the loss's mask)

    def test_gradients_flow_through_a_multi_step_unroll(self):
        model = TrackMemory()
        embeddings = torch.nn.functional.normalize(torch.randn(2, 3, 128), dim=-1)
        poses, scores = torch.randn(2, 3, 5), torch.rand(2, 3, 1)
        valid = torch.ones(2, 3, dtype=torch.bool)
        refined_embeddings, refined_poses = step_sequence(model, embeddings, poses, scores, valid)
        (refined_embeddings.sum() + refined_poses.sum()).backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient across the unroll")


class TrackMemoryLossTest(unittest.TestCase):
    def setUp(self):
        self.gallery = torch.nn.functional.normalize(torch.randn(5, 128), dim=-1)

    def test_all_invalid_gives_exactly_zero_loss(self):
        b, t = 2, 3
        refined_embeddings = torch.nn.functional.normalize(torch.randn(b, t, 128), dim=-1)
        refined_poses, true_poses = torch.randn(b, t, 5), torch.randn(b, t, 5)
        valid = torch.zeros(b, t, dtype=torch.bool)
        gallery_index = torch.tensor([0, 1])
        loss, parts = track_memory_loss(refined_embeddings, refined_poses, true_poses, valid, gallery_index, self.gallery, temperature=0.1, pose_weight=0.1)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(parts, {"embed": 0.0, "pose": 0.0})

    def test_embedding_loss_is_lower_when_closer_to_the_true_gallery_row(self):
        b, t = 1, 1
        valid = torch.ones(b, t, dtype=torch.bool)
        gallery_index = torch.tensor([2])
        true_poses = torch.zeros(b, t, 5)

        matching = self.gallery[2].reshape(1, 1, 128).clone()
        mismatched = self.gallery[0].reshape(1, 1, 128).clone()
        loss_matching, _ = track_memory_loss(matching, torch.zeros(b, t, 5), true_poses, valid, gallery_index, self.gallery, temperature=0.1, pose_weight=0.0)
        loss_mismatched, _ = track_memory_loss(mismatched, torch.zeros(b, t, 5), true_poses, valid, gallery_index, self.gallery, temperature=0.1, pose_weight=0.0)
        self.assertLess(float(loss_matching), float(loss_mismatched))

    def test_pose_loss_grows_with_distance_from_the_true_pose(self):
        b, t = 1, 1
        valid = torch.ones(b, t, dtype=torch.bool)
        gallery_index = torch.tensor([0])
        refined_embeddings = self.gallery[0].reshape(1, 1, 128).clone()  # perfect embedding: isolates the pose term
        true_poses = torch.zeros(b, t, 5)

        _, parts_close = track_memory_loss(refined_embeddings, torch.zeros(b, t, 5) + 0.01, true_poses, valid, gallery_index, self.gallery, 0.1, 1.0)
        _, parts_far = track_memory_loss(refined_embeddings, torch.ones(b, t, 5), true_poses, valid, gallery_index, self.gallery, 0.1, 1.0)
        self.assertLess(parts_close["pose"], parts_far["pose"])

    def test_padding_rows_do_not_affect_the_loss(self):
        gallery_index = torch.tensor([0, 1])
        true_poses = torch.zeros(2, 2, 5)
        refined_poses = torch.zeros(2, 2, 5)
        refined_embeddings = torch.nn.functional.normalize(torch.randn(2, 2, 128), dim=-1)
        refined_embeddings[0, 0] = self.gallery[0]

        valid_a = torch.tensor([[True, False], [False, False]])
        loss_a, _ = track_memory_loss(refined_embeddings, refined_poses, true_poses, valid_a, gallery_index, self.gallery, 0.1, 0.1)

        # Change the padding-only positions' content; the loss must be unaffected since none of
        # them are marked valid.
        refined_embeddings_b = refined_embeddings.clone()
        refined_embeddings_b[0, 1] = torch.nn.functional.normalize(torch.randn(128), dim=-1)
        refined_embeddings_b[1] = torch.nn.functional.normalize(torch.randn(2, 128), dim=-1)
        loss_b, _ = track_memory_loss(refined_embeddings_b, refined_poses, true_poses, valid_a, gallery_index, self.gallery, 0.1, 0.1)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)


if __name__ == "__main__":
    unittest.main()
