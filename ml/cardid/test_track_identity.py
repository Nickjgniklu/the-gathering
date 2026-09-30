"""Pin/release contracts for `track_identity.py`, against synthetic embeddings -- no checkpoint
or gallery needed, same convention as this project's other hand-built-tensor tests."""

from __future__ import annotations

import unittest

import numpy as np

from .track_identity import TrackIdentity, apply_correction, clear_correction, step

A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])


class TrackIdentityTest(unittest.TestCase):
    def test_unpinned_track_passes_the_raw_embedding_through_unchanged(self):
        identity = TrackIdentity()
        fed, pinned = step(identity, raw_embedding=A, gallery_embedding_for_pin=None)
        np.testing.assert_array_equal(fed, A)
        self.assertFalse(pinned)

    def test_pinned_track_feeds_the_pinned_embedding_not_the_raw_one(self):
        identity = TrackIdentity()
        apply_correction(identity, gallery_index=7)
        # raw detector output still looks like A, but the user says this track is really B's card
        fed, pinned = step(identity, raw_embedding=A, gallery_embedding_for_pin=B)
        np.testing.assert_array_equal(fed, B)
        self.assertTrue(pinned)
        self.assertEqual(identity.pinned_gallery_index, 7)

    def test_manual_clear_returns_control_to_the_model(self):
        identity = TrackIdentity()
        apply_correction(identity, gallery_index=1)
        clear_correction(identity)
        fed, pinned = step(identity, raw_embedding=A, gallery_embedding_for_pin=None)
        self.assertIsNone(identity.pinned_gallery_index)
        np.testing.assert_array_equal(fed, A)
        self.assertFalse(pinned)

    def test_agreeing_frames_reset_the_mismatch_streak(self):
        identity = TrackIdentity()
        apply_correction(identity, gallery_index=1)
        for _ in range(3):
            step(identity, raw_embedding=A, gallery_embedding_for_pin=A)  # keeps agreeing
        self.assertEqual(identity.mismatch_streak, 0)
        self.assertEqual(identity.pinned_gallery_index, 1)  # still pinned: never fell below the floor

    def test_auto_releases_after_enough_consecutive_mismatched_frames(self):
        identity = TrackIdentity()
        apply_correction(identity, gallery_index=1)
        # raw embedding (A) disagrees with the pinned card's embedding (B) every frame: the
        # physical card in this slot was probably swapped for a different one.
        for _ in range(4):
            _, pinned = step(identity, raw_embedding=A, gallery_embedding_for_pin=B, auto_release_frames=5)
            self.assertTrue(pinned)  # not released yet
        fed, pinned = step(identity, raw_embedding=A, gallery_embedding_for_pin=B, auto_release_frames=5)
        self.assertFalse(pinned)
        self.assertIsNone(identity.pinned_gallery_index)
        np.testing.assert_array_equal(fed, A)  # falls back to the raw embedding the same frame it releases

    def test_an_isolated_mismatch_does_not_release_the_pin(self):
        identity = TrackIdentity()
        apply_correction(identity, gallery_index=1)
        step(identity, raw_embedding=A, gallery_embedding_for_pin=B, auto_release_frames=5)  # one bad frame
        _, pinned = step(identity, raw_embedding=B, gallery_embedding_for_pin=B, auto_release_frames=5)  # then agrees again
        self.assertTrue(pinned)
        self.assertEqual(identity.mismatch_streak, 0)


if __name__ == "__main__":
    unittest.main()
