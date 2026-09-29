"""Tests `DetectEmbedAndSearch`'s batched search math against a synthetic gallery, without a real
checkpoint or a `DetectAndEmbed` -- a tiny stand-in module supplies fixed (embeddings, det_scores,
quads) so only the search step itself (the new code in this module) is under test. Correctness
against the *real* gallery and a real `DetectAndEmbed` was validated separately (see
`detect_embed_search.py`'s module docstring) by comparing every detected slot's result against
`evaluate_detect_and_embed.search`'s independently-written per-slot loop: 0 mismatches."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from .detect_and_embed import EMBED_DIM, MAX_CARDS, N_FRAMES
from .detect_embed_search import DetectEmbedAndSearch


class _FixedDetectAndEmbed(nn.Module):
    """Returns whatever (embeddings, det_scores, quads) it was constructed with, ignoring its
    input -- a stand-in so `DetectEmbedAndSearchTest` exercises only the search math."""

    def __init__(self, embeddings: torch.Tensor, det_scores: torch.Tensor, quads: torch.Tensor):
        super().__init__()
        self.embeddings, self.det_scores, self.quads = embeddings, det_scores, quads

    def forward(self, images: torch.Tensor):
        return self.embeddings, self.det_scores, self.quads


class DetectEmbedAndSearchTest(unittest.TestCase):
    def _build(self, gallery: torch.Tensor, frames: torch.Tensor, penalties: torch.Tensor, embeddings: torch.Tensor, topk: int = 3) -> DetectEmbedAndSearch:
        n = 1
        det_scores = torch.ones(n, MAX_CARDS)
        quads = torch.zeros(n, MAX_CARDS, 4, 2)
        dae = _FixedDetectAndEmbed(embeddings, det_scores, quads)
        return DetectEmbedAndSearch(dae, gallery, frames, penalties, topk=min(topk, gallery.shape[0]))

    def test_finds_the_exact_gallery_match_with_similarity_one(self):
        torch.manual_seed(0)
        n_gallery = 50
        gallery = torch.nn.functional.normalize(torch.randn(n_gallery, EMBED_DIM), dim=-1)
        frames = torch.zeros(n_gallery, dtype=torch.int64)  # every gallery art is frame 0
        penalties = torch.zeros(n_gallery)

        embeddings = torch.zeros(1, MAX_CARDS, N_FRAMES, EMBED_DIM)
        target_row = 7
        embeddings[0, 0, 0] = gallery[target_row]  # slot 0's frame-0 embedding == gallery row 7 exactly

        model = self._build(gallery, frames, penalties, embeddings)
        with torch.no_grad():
            indices, scores, det_scores, quads = model(torch.zeros(1, 3, 4, 4))
        self.assertEqual(int(indices[0, 0, 0]), target_row)
        self.assertAlmostEqual(float(scores[0, 0, 0]), 1.0, places=5)

    def test_gathers_each_gallery_row_s_own_frame_not_a_fixed_one(self):
        # Two gallery rows, one native to frame 0 and one to frame 1. A query whose frame-0 and
        # frame-1 embeddings point at *different* directions must be compared against each
        # gallery row using that row's own frame -- not e.g. always frame 0.
        gallery = torch.eye(EMBED_DIM)[:2]  # row 0 = unit vector e0, row 1 = unit vector e1
        frames = torch.tensor([0, 1], dtype=torch.int64)
        penalties = torch.zeros(2)

        embeddings = torch.zeros(1, MAX_CARDS, N_FRAMES, EMBED_DIM)
        embeddings[0, 0, 0] = gallery[0]  # this slot's frame-0 hypothesis matches gallery row 0 exactly
        embeddings[0, 0, 1] = gallery[1]  # and its frame-1 hypothesis matches gallery row 1 exactly

        model = self._build(gallery, frames, penalties, embeddings)
        with torch.no_grad():
            indices, scores, _, _ = model(torch.zeros(1, 3, 4, 4))
        # Both gallery rows should score a perfect match (each against its own frame hypothesis).
        self.assertAlmostEqual(float(scores[0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(scores[0, 0, 1]), 1.0, places=5)
        self.assertEqual(set(indices[0, 0, :2].tolist()), {0, 1})

    def test_penalty_can_demote_an_otherwise_better_match(self):
        gallery = torch.eye(EMBED_DIM)[:2]
        frames = torch.zeros(2, dtype=torch.int64)
        penalties = torch.tensor([0.0, 0.5])  # row 1 is a "rare frame", penalised

        embeddings = torch.zeros(1, MAX_CARDS, N_FRAMES, EMBED_DIM)
        # A query slightly closer to row 1 than row 0, but not by more than the penalty.
        query = torch.nn.functional.normalize(gallery[0] * 0.9 + gallery[1] * 0.95, dim=-1)
        embeddings[0, 0, 0] = query

        model = self._build(gallery, frames, penalties, embeddings)
        with torch.no_grad():
            indices, _, _, _ = model(torch.zeros(1, 3, 4, 4))
        self.assertEqual(int(indices[0, 0, 0]), 0)  # row 0 wins once row 1's penalty is applied


if __name__ == "__main__":
    unittest.main()
