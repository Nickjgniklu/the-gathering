"""Contracts for `track_memory_dataset.py`: `_match_frame`'s IoU correspondence directly (pure
function, hand-built quads), and `TrackMemorySequenceDataset`'s shape/padding/gallery-filtering
contract against a tiny rendered sequence and random-weight checkpoints. The detector's own
accuracy is meaningless with random weights, so the "a real detection gets matched and its fields
land in the right slot" path is tested with a fixed stand-in model instead (same technique
`test_detect_embed_search.py` uses), not by hoping a random detector happens to find anything."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import track_memory_dataset
from .image_bank import ArtBank
from .model import Embedder
from .table_detector import TableCenterNet
from .table_scenes import write_sequence_split
from .test_table_scenes import make_card_bank
from .track_memory_dataset import TrackMemorySequenceDataset, _match_frame


class MatchFrameTest(unittest.TestCase):
    def test_matches_by_nearest_iou_not_by_slot_order(self):
        truth = [
            {"quad": [[0, 0], [10, 0], [10, 10], [0, 10]], "track_id": 5},
            {"quad": [[100, 100], [110, 100], [110, 110], [100, 110]], "track_id": 9},
        ]
        # detected slot 0 actually overlaps track 9's quad, slot 1 overlaps track 5's -- reversed order
        detected = [
            np.float32([[100, 100], [110, 100], [110, 110], [100, 110]]),
            np.float32([[0, 0], [10, 0], [10, 10], [0, 10]]),
        ]
        matched = _match_frame(detected, [0.9, 0.9], truth, score_threshold=0.3)
        self.assertEqual(matched, {9: 0, 5: 1})

    def test_low_score_slots_are_never_matched(self):
        truth = [{"quad": [[0, 0], [10, 0], [10, 10], [0, 10]], "track_id": 1}]
        detected = [np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])]
        matched = _match_frame(detected, [0.1], truth, score_threshold=0.3)
        self.assertEqual(matched, {})

    def test_a_truth_card_with_no_overlapping_detection_is_a_genuine_miss(self):
        truth = [{"quad": [[0, 0], [10, 0], [10, 10], [0, 10]], "track_id": 1}]
        detected = [np.float32([[1000, 1000], [1010, 1000], [1010, 1010], [1000, 1010]])]
        matched = _match_frame(detected, [0.9], truth, score_threshold=0.3)
        self.assertEqual(matched, {})


class TrackMemorySequenceDatasetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "bank").mkdir()
        self.cards = make_card_bank(self.root / "bank", n=6)
        self.arts = ArtBank([])

        self.out = self.root / "sequences"
        with patch("cardid.table_scenes.list_arts", return_value=[]):
            write_sequence_split(self.out, "train", sequences=2, n_frames=4, seed=1, cards=self.cards, arts_by_pool={"train": self.arts}, size=128, out=128)
        self.manifest_path = self.out / "train" / "manifest.jsonl"

        self.table_checkpoint = self.root / "table.pt"
        self.embed_checkpoint = self.root / "embed.pt"
        torch.save(TableCenterNet(pretrained=False).state_dict(), self.table_checkpoint)
        torch.save(Embedder(pretrained=False).state_dict(), self.embed_checkpoint)

        # A synthetic gallery matching the tiny card bank's own ids, standing in for load_gallery
        # (a real search.onnx isn't needed to test this dataset's contract).
        card_ids = [p.stem for p in self.cards.paths]
        self.arts_json = [{"id": cid, "name": cid} for cid in card_ids]
        self.gallery = torch.nn.functional.normalize(torch.randn(len(card_ids), 128), dim=-1)
        self.frames = torch.zeros(len(card_ids), dtype=torch.int64)
        self.penalties = torch.zeros(len(card_ids))

    def _build_dataset(self, arts_json=None, gallery=None, frames=None, penalties=None) -> TrackMemorySequenceDataset:
        gallery_tuple = (
            arts_json if arts_json is not None else self.arts_json,
            gallery if gallery is not None else self.gallery,
            frames if frames is not None else self.frames,
            penalties if penalties is not None else self.penalties,
        )
        with patch.object(track_memory_dataset, "load_gallery", return_value=gallery_tuple):
            return TrackMemorySequenceDataset(
                self.manifest_path,
                table_checkpoint=self.table_checkpoint,
                embed_checkpoint=self.embed_checkpoint,
                gallery_bundle=Path("unused"),
                native_size=128,
            )

    def test_one_example_per_track_across_every_sequence(self):
        dataset = self._build_dataset()
        rows = [json.loads(line) for line in self.manifest_path.read_text().splitlines()]
        expected = sum(len({c["track_id"] for f in row["frames"] for c in f["cards"]}) for row in rows)
        self.assertEqual(len(dataset), expected)
        self.assertGreater(len(dataset), 0)

    def test_getitem_shapes(self):
        dataset = self._build_dataset()
        embeddings, poses, scores, true_poses, valid, gallery_index = dataset[0]
        self.assertEqual(embeddings.shape, (dataset.max_frames, 128))
        self.assertEqual(poses.shape, (dataset.max_frames, 5))
        self.assertEqual(scores.shape, (dataset.max_frames, 1))
        self.assertEqual(true_poses.shape, (dataset.max_frames, 5))
        self.assertEqual(valid.shape, (dataset.max_frames,))
        self.assertIsInstance(gallery_index, int)

    def test_skips_every_track_when_the_gallery_has_none_of_the_cards(self):
        empty = ([], torch.zeros(0, 128), torch.zeros(0, dtype=torch.int64), torch.zeros(0))
        dataset = self._build_dataset(*empty)
        self.assertEqual(len(dataset), 0)

    def test_a_matching_detection_lands_in_the_right_slot(self):
        # Fixed stand-in model, not the random-weight one: deterministically makes slot 0 the
        # frame-0 match for whichever track __getitem__(0) resolves to, so the match + pose/score/
        # embedding plumbing is under test without depending on an untrained detector's output.
        dataset = self._build_dataset()
        row, track_id, _gallery_index = dataset._examples[0]
        truth_quad = next(c["quad"] for c in row["frames"][0]["cards"] if c["track_id"] == track_id)

        max_cards = 3
        fixed_embeddings = torch.nn.functional.normalize(torch.randn(1, max_cards, 14, 128), dim=-1)
        fixed_scores = torch.zeros(1, max_cards)
        fixed_scores[0, 0] = 0.99
        fixed_quads = torch.zeros(1, max_cards, 4, 2)
        fixed_quads[0, 0] = torch.tensor(truth_quad, dtype=torch.float32)  # native_size == out here, no rescale needed
        fixed_pose = torch.zeros(1, max_cards, 5)
        fixed_pose[0, 0, 2] = 42.0  # a recognizable "short" value to confirm it lands in slot 0

        class _Fixed(nn.Module):
            def forward(self, images, return_pose=False):
                return fixed_embeddings, fixed_scores, fixed_quads, fixed_pose

        dataset.model = _Fixed()
        embeddings, poses, scores, true_poses, valid, _ = dataset[0]
        # The stand-in returns the same fixed quad every frame, and per-frame jitter is small
        # enough that it may well also overlap later frames' (slightly moved) ground truth --
        # only frame 0's exact match is asserted here, not that later frames must miss.
        self.assertTrue(bool(valid[0]))
        self.assertEqual(float(poses[0, 2]), 42.0)
        self.assertAlmostEqual(float(scores[0, 0]), 0.99, places=5)
        # true_poses comes from the ground-truth quad, not the fixed stand-in's pose output --
        # its "short" (index 2) should be a real positive card size, not the stand-in's 42.0.
        self.assertGreater(float(true_poses[0, 2]), 0.0)
        self.assertNotEqual(float(true_poses[0, 2]), 42.0)


if __name__ == "__main__":
    unittest.main()
