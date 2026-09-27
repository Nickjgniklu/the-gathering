"""The homemade dense multi-card detector: Strategy A's real, trainable model.

`TableCenterNet` reuses `detector.CornerNet`'s exact backbone/decoder shape (MobileNetV3-Small
stem+top, the lateral/reduce/squeeze fuse at stride 16, then the lat8/lat4/up16/dec8/dec4
decoder down to stride 4) so an ImageNet-pretrained backbone transfers the same way it does for
the click-conditioned detector. The difference is the head: instead of flattening to one global
pose (one card per window), every stride-4 cell predicts its own (card-present, pose, up)
independently -- a CenterNet-style dense head, which is what lets one forward pass find every
card on the table instead of one per click.

Per cell the head predicts:
- a card-presence logit (`heat`), trained with the same penalty-reduced focal loss as the
  corner heatmap (`detector.heat_loss`);
- a pose (`log short side`, `cos 2theta`, `sin 2theta`) -- the card's centre is implicit in the
  cell location, so only size and (180-degree-symmetric) rotation need regressing;
- an up vector, exactly like `CornerNet`'s, for printed-orientation disambiguation.

Only cells at an instance's centre are supervised for pose/up (`build_targets`'s `mask`); the
heatmap loss covers every cell, positive and negative, the way CenterNet trains detection
without anchors or NMS-during-training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .detector import card_rect, cyclic_order, heat_loss, orient_quad

TABLE_INPUT = 384  # model input side; table_scenes renders at 640 and this downscales at load time
TABLE_STRIDE = 4  # output grid stride (TABLE_INPUT/TABLE_STRIDE = 96x96 cells)


class TableCenterNet(nn.Module):
    """Dense multi-instance detector. See module docstring for the architecture rationale."""

    FUSE = 64
    SQUEEZE = 32
    DECODE = 32
    OUT_CHANNELS = 6  # 1 heat + 3 pose (log_short, cos2t, sin2t) + 2 up

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.stem = backbone.features[:9]  # stride 16, 48ch at the end
        self.top = backbone.features[9:]  # stride 32, 576ch
        self.lateral = nn.Conv2d(48, self.FUSE, 1)
        self.reduce = nn.Conv2d(576, self.FUSE, 1)
        self.up16 = nn.Conv2d(self.FUSE, self.DECODE, 1)
        self.lat8 = nn.Conv2d(24, self.DECODE, 1)
        self.lat4 = nn.Conv2d(16, self.DECODE, 1)
        self.dec8 = nn.Sequential(nn.Conv2d(self.DECODE, self.DECODE, 3, padding=1), nn.Hardswish())
        self.dec4 = nn.Sequential(nn.Conv2d(self.DECODE, self.DECODE, 3, padding=1), nn.Hardswish())
        self.head = nn.Conv2d(self.DECODE, self.OUT_CHANNELS, 1)
        with torch.no_grad():
            self.head.bias[0] = -2.19  # focal-loss prior: card centres are rare (see CornerNet)

    def backbone_parameters(self):
        yield from self.stem.parameters()
        yield from self.top.parameters()

    def head_parameters(self):
        for m in (self.lateral, self.reduce, self.up16, self.lat8, self.lat4, self.dec8, self.dec4, self.head):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (heat logits (N,1,S,S), pose (N,3,S,S), up (N,2,S,S)) at stride `TABLE_STRIDE`."""
        s4 = self.stem[:2](x)
        s8 = self.stem[2:4](s4)
        mid = self.stem[4:](s8)
        top = self.reduce(self.top(mid))
        fused = self.lateral(mid) + nn.functional.interpolate(top, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        d8 = self.dec8(self.lat8(s8) + nn.functional.interpolate(self.up16(fused), scale_factor=2, mode="bilinear", align_corners=False))
        d4 = self.dec4(self.lat4(s4) + nn.functional.interpolate(d8, scale_factor=2, mode="bilinear", align_corners=False))
        out = self.head(d4)
        return out[:, :1], out[:, 1:4], out[:, 4:6]


def build_targets(
    poses: list[tuple[float, float, float, float]],
    ups: list[np.ndarray],
    image_size: int,
    stride: int,
    negatives: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`poses` are (cx, cy, short, angle_deg) in input-image pixels, `ups` the matching unit
    vectors. Returns (heat, pose, up, mask, hard_neg), each float32 and channel-first, at
    `image_size / stride` resolution. Cell `i`'s centre sits at pixel `(i + 0.5) * stride`
    (matches `decode_detections`'s inverse mapping). `negatives`, when given, are (x0, y0, x1,
    y1) hard-negative object boxes (clutter/round objects with no card there at all, see
    `table_scenes.py`); `hard_neg` marks the cells whose centre falls inside one, for
    `table_detector_loss` to penalise a false activation there more than over plain background."""
    size = image_size // stride
    heat = np.zeros((1, size, size), np.float32)
    pose = np.zeros((3, size, size), np.float32)
    up = np.zeros((2, size, size), np.float32)
    mask = np.zeros((1, size, size), np.float32)
    hard_neg = np.zeros((1, size, size), np.float32)
    grid = np.arange(size, dtype=np.float32) + 0.5
    for (cx, cy, short, angle), up_vec in zip(poses, ups, strict=True):
        gx, gy = cx / stride, cy / stride
        sigma = max(0.05 * short / stride, 0.8)
        g = np.exp(-(((grid[None, :] - gx) ** 2) + ((grid[:, None] - gy) ** 2)) / (2 * sigma**2)).astype(np.float32)
        g /= max(g.max(), 1e-6)  # the nearest cell always reaches exactly 1.0 (the focal loss's
        # positive), whatever the sub-cell offset between the continuous centre and the grid
        np.maximum(heat[0], g, out=heat[0])
        cxi, cyi = int(np.clip(round(gx - 0.5), 0, size - 1)), int(np.clip(round(gy - 0.5), 0, size - 1))
        t = np.radians(angle)
        pose[:, cyi, cxi] = (np.log(max(short, 1e-3)), np.cos(2 * t), np.sin(2 * t))
        up[:, cyi, cxi] = up_vec
        mask[0, cyi, cxi] = 1.0
    for x0, y0, x1, y1 in negatives or []:
        gx0, gx1 = int(np.clip(np.floor(x0 / stride), 0, size - 1)), int(np.clip(np.ceil(x1 / stride), 0, size - 1))
        gy0, gy1 = int(np.clip(np.floor(y0 / stride), 0, size - 1)), int(np.clip(np.ceil(y1 / stride), 0, size - 1))
        hard_neg[0, gy0 : gy1 + 1, gx0 : gx1 + 1] = 1.0
    return heat, pose, up, mask, hard_neg


HARD_NEG_WEIGHT = 3.0  # how much harder a false activation over a clutter/round object is
# penalised than over plain background (see `detector.heat_loss`'s `neg_weight`)


def table_detector_loss(
    heat_logits: torch.Tensor,
    pose: torch.Tensor,
    up: torch.Tensor,
    heat_target: torch.Tensor,
    pose_target: torch.Tensor,
    up_target: torch.Tensor,
    mask: torch.Tensor,
    pose_weight: float = 1.0,
    up_weight: float = 1.0,
    hard_neg_mask: torch.Tensor | None = None,
    hard_neg_weight: float = HARD_NEG_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Focal loss on every cell's heatmap plus pose/up regression masked to instance centres.
    `hard_neg_mask`, when given, upweights the heatmap's background penalty by `hard_neg_weight`
    (default `HARD_NEG_WEIGHT`) over clutter/round-object cells (see `table_scenes.py`'s hard
    negatives and `build_targets`)."""
    n_pos = mask.sum().clamp(min=1)
    neg_weight = 1 + (hard_neg_weight - 1) * hard_neg_mask if hard_neg_mask is not None else None
    hl = heat_loss(heat_logits, heat_target, neg_weight=neg_weight)
    size_l = ((pose[:, :1] - pose_target[:, :1]).abs() * mask).sum() / n_pos
    angle_l = (((pose[:, 1:] - pose_target[:, 1:]) ** 2).sum(dim=1, keepdim=True) * mask).sum() / n_pos
    pose_l = size_l + 0.5 * angle_l
    up_l = (((up - up_target) ** 2).sum(dim=1, keepdim=True) * mask).sum() / n_pos
    total = hl + pose_weight * pose_l + up_weight * up_l
    return total, {"heat": hl.item(), "pose": pose_l.item(), "up": up_l.item()}


@torch.no_grad()
def decode_detections(
    heat_logits: torch.Tensor, pose: torch.Tensor, up: torch.Tensor, stride: int = TABLE_STRIDE, score_threshold: float = 0.3, max_detections: int = 60
) -> list[tuple[np.ndarray, float]]:
    """One sample's (heat (1,S,S), pose (3,S,S), up (2,S,S)) -> [(quad, score), ...] in input
    pixels, highest score first. Peaks are local maxima of the sigmoid heatmap (a 3x3 max-pool
    equality test, CenterNet's NMS-free peak picking), so no separate box-NMS is needed."""
    prob = torch.sigmoid(heat_logits)[0]  # (S, S)
    pooled = nn.functional.max_pool2d(prob[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = (prob == pooled) & (prob > score_threshold)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    if len(ys) == 0:
        return []
    scores = prob[ys, xs]
    order = torch.argsort(scores, descending=True)[:max_detections]
    results = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        score = float(scores[idx])
        log_short, c2, s2 = (float(v) for v in pose[:, y, x])
        cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
        short = float(np.exp(log_short))
        angle = float(np.degrees(0.5 * np.arctan2(s2, c2)))
        quad = card_rect(cx, cy, short, angle)
        up_vec = up[:, y, x].cpu().numpy()
        results.append((orient_quad(cyclic_order(quad), up_vec), score))
    return results


__all__ = ["TABLE_INPUT", "TABLE_STRIDE", "TableCenterNet", "build_targets", "decode_detections", "table_detector_loss"]
