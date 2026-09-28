"""CPU-only contracts for the batched graph helpers behind `TableDetectorGraph`
(`graphs.card_rect_graph_k`/`cyclic_order_graph_k`/`orient_quad_graph_k`): each must agree with
the existing scalar reference (`detector.card_rect`/`cyclic_order`/`orient_quad`) that
`table_detector.decode_detections` already uses, just batched over `K` poses at once. No
network weights or ONNX runtime involved, so this stays fast; the actual export is verified
against a real checkpoint by `export_table_detector.py --verify` instead.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .detector import card_rect, cyclic_order, orient_quad
from .graphs import card_rect_graph_k, cyclic_order_graph_k, orient_quad_graph_k


class CardRectGraphKTest(unittest.TestCase):
    def test_matches_the_scalar_reference_for_several_poses(self):
        poses = [(100.0, 100.0, 60.0, 0.0), (50.0, 200.0, 30.0, 37.0), (300.0, 150.0, 90.0, -15.5)]
        cx, cy, short, angle = (torch.tensor([p[i] for p in poses]) for i in range(4))
        batched = card_rect_graph_k(cx, cy, short, torch.deg2rad(angle))
        for i, (x, y, s, a) in enumerate(poses):
            expected = card_rect(x, y, s, a)
            np.testing.assert_allclose(batched[i].numpy(), expected, atol=1e-3)


class CyclicOrderGraphKTest(unittest.TestCase):
    def test_matches_the_scalar_reference(self):
        quads = np.stack(
            [
                card_rect(100.0, 100.0, 60.0, 20.0),
                np.roll(card_rect(50.0, 80.0, 40.0, 0.0), 2, axis=0),  # already rotated input order
            ]
        )
        batched = cyclic_order_graph_k(torch.from_numpy(quads))
        for i, quad in enumerate(quads):
            np.testing.assert_allclose(batched[i].numpy(), cyclic_order(quad), atol=1e-4)


class OrientQuadGraphKTest(unittest.TestCase):
    def test_matches_the_scalar_reference(self):
        quads = np.stack([cyclic_order(card_rect(100.0, 100.0, 60.0, 20.0)), cyclic_order(card_rect(50.0, 80.0, 40.0, 70.0))])
        ups = np.float32([[0.3, -0.95], [-0.8, 0.6]])
        batched = orient_quad_graph_k(torch.from_numpy(quads), torch.from_numpy(ups))
        for i, (quad, up) in enumerate(zip(quads, ups, strict=True)):
            np.testing.assert_allclose(batched[i].numpy(), orient_quad(quad, up), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
