"""A dual-input variant of `TableCenterNet`: takes a *background* frame (captured once, e.g. when
the user starts a table session, before any cards are on it) alongside the *current* frame, so the
network can directly compare against a known-clean reference instead of having to recognise every
possible real desk object as "not a card" from single-frame appearance alone.

Motivation: a 108-card real-capture golden dataset (see `confusion_matrix.py`'s `real_captures`
category) found the single-frame model's real-world precision (75.8%) and false-positive rate on
static desk clutter (22.2%) were far worse than synthetic metrics predicted, even after training
against both procedural and real-photographed hard-negative objects (see `real_clutter.py` in the
sibling `feat/table-detector-nonstandard-cards` branch). Those approaches teach the network what
clutter *looks like* in general; a background reference instead tells it exactly what is already
sitting on *this* desk, which should suppress any static object regardless of what it is.

Architecture: `TableCenterNet`'s backbone unmodified except the first conv, which is "inflated"
from 3 to 6 input channels (background RGB concatenated with current RGB) by duplicating the
ImageNet-pretrained weights and halving them, so the layer starts from the same response
statistics as the pretrained 3-channel filter applied to either image alone, rather than random
extra-channel weights (the standard trick for adapting a pretrained conv to more input channels,
e.g. Xie & Girshick's inflated-3D-conv-from-2D-conv initialisation for video models). Everything
downstream of the stem -- decoder, head, targets, loss, decode -- is identical to `TableCenterNet`.

The background/current pair is never pixel-registered in deployment (auto-exposure and white
balance drift between a startup capture and a later gameplay frame -- see `table_scenes.py`'s
`hardware_stress` reuse in `render_table_scene_pair`), so training must never hand the network a
clean, perfectly-matched pair either; each half of a training pair gets independent photometric
augmentation.
"""

from __future__ import annotations

import torch
from torch import nn

from .table_detector import TableCenterNet


class TableCenterNetDual(TableCenterNet):
    """`TableCenterNet` with a 6-channel (background RGB + current RGB) input."""

    def __init__(self, pretrained: bool = True):
        super().__init__(pretrained=pretrained)
        old_conv: nn.Conv2d = self.stem[0][0]
        new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(torch.cat([old_conv.weight, old_conv.weight], dim=1) * 0.5)
        self.stem[0][0] = new_conv

    def forward(self, background: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`background`/`current` are each (N,3,H,W); returns the same (heat, pose, up) as
        `TableCenterNet.forward`."""
        return super().forward(torch.cat([background, current], dim=1))
