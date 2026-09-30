"""One combined model: `TableCenterNet` (frozen) finds every card on the table, a new batched
crop layer warps each one straight to the recogniser's input for *every* frame hypothesis, and
`Embedder` (frozen) embeds all of them in one parallel pass -- the "multi-model, one accelerated
forward pass" pipeline, wiring together two already-separately-trained models rather than training
anything new (see `tiled_fusion.py` for the separate, also-already-shipped "frozen backbone + new
trainable layers" idea; this module trains nothing at all itself).

One remaining deliberate simplification: **fixed top-`MAX_CARDS` detections, not a score
threshold.** `decode_detections` returns a variable-length list (however many peaks clear
`score_threshold`), which is awkward for a single static-shaped exported graph. This takes the
top `MAX_CARDS` local-maxima by score unconditionally (`_topk_peaks`) and returns their scores
alongside; a low-confidence slot's score is still there for the caller to threshold at display
time, same as today, just not baked into the graph's control flow.

**All 14 frame hypotheses (`detect.FRAME_NAMES`), not just "modern".** An earlier version of this
module cropped only the "modern" window per card, which made its embedding output incompatible
with the real `search.onnx` (it expects 14 embeddings per query -- one per `FRAME_NAMES` entry --
and looks up each gallery art's own frame index into that batch). This version crops and embeds
all 14 the same way `detect.art_crops` does, so `embeddings` is now `(N, MAX_CARDS, 14, 128)` and
`embeddings[n, k]` can be fed to `search.onnx` directly, exactly like `embed.onnx`'s own
`(scene, quad) -> (14, 128)` output -- no separate `embed.onnx` call needed:

    DetectAndEmbed -> search.onnx   # instead of: DetectAndEmbed -> embed.onnx -> search.onnx

The crop itself is *exactly* representable as an affine warp for the card-to-canonical step, not
an approximation of one: every quad this pipeline ever produces comes from `TableCenterNet`'s own
(center, short-side, angle) pose, which has no skew/perspective term at all (unlike
`detect.warp_card`, which perspective-warps an arbitrary externally-supplied quad because
contour-detected real geometry can be slightly keystoned). A rotation+uniform-scale+translation is
the exact inverse of how the quad was built. Each frame's own window is then cut from that
canonical card at its own native (possibly non-square) aspect ratio, rotated by
`detect.FRAME_ROTATIONS` where the reference pipeline does (`torch.rot90`, matching `numpy.rot90`'s
convention exactly), and *then* squashed to `INPUT_SIZE`-square -- replicating
`detect.frame_crop`'s crop-then-rotate-then-resize order exactly, not resize-then-rotate, which
would squash a non-square box along the wrong axis for the 6 frames with a 90/270-degree rotation.

See `ml/detect-and-embed-guide.md` for the full API contract, construction recipes for both
detector variants, and the ONNX export recipe.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .detect import CARD_ASPECT, CARD_H, CARD_W, FRAME_NAMES, FRAME_ROTATIONS, INPUT_SIZE, frame_box
from .model import EMBED_DIM, Embedder
from .table_detector import TABLE_STRIDE, TableCenterNet

MAX_CARDS = 20  # matches table_scenes.py's own per-scene card-count ceiling
N_FRAMES = len(FRAME_NAMES)  # 14

# Per frame, fixed at import time (frame boxes are constants, not data-dependent): its own
# (x0, y0, x1, y1) card-fraction box, its own *native* (unrotated, non-square) pixel footprint
# (a fixed rendering resolution for the pre-rotation crop -- reusing CARD_W/CARD_H as the
# reference scale, same as `detect.py`'s own 250x350 canonical card, not tied to any card's
# actual runtime `short` value), and its CCW quarter-turn count.
_FRAME_BOXES = [frame_box(name) for name in FRAME_NAMES]
_FRAME_RAW_SIZE = [(max(round((x1 - x0) * CARD_W), 1), max(round((y1 - y0) * CARD_H), 1)) for x0, y0, x1, y1 in _FRAME_BOXES]
_FRAME_ROTATION_K = [FRAME_ROTATIONS.get(name, 0) for name in FRAME_NAMES]


def _topk_peaks(heat_logits: torch.Tensor, pose: torch.Tensor, up: torch.Tensor, stride: int, k: int):
    """Batched, fixed-`k` stand-in for `table_detector.decode_detections`'s variable-length,
    thresholded peak list -- same NMS-free local-max peak picking (a 3x3 max-pool equality
    test), just always returning exactly `k` slots (score 0 where fewer than `k` real peaks
    exist) instead of however many clear a threshold. Returns (cx, cy, short, angle_rad, up_x,
    up_y, score), each (B, k)."""
    b, _, s, _ = heat_logits.shape
    prob = torch.sigmoid(heat_logits)
    pooled = F.max_pool2d(prob, 3, stride=1, padding=1)
    peak_scores = torch.where(prob == pooled, prob, torch.zeros_like(prob)).view(b, -1)
    scores, idx = torch.topk(peak_scores, k, dim=1)
    y, x = torch.div(idx, s, rounding_mode="floor").float(), (idx % s).float()
    cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
    idx_pose = idx.unsqueeze(1).expand(-1, 3, -1)
    idx_up = idx.unsqueeze(1).expand(-1, 2, -1)
    log_short, c2, s2 = pose.view(b, 3, -1).gather(2, idx_pose).unbind(1)
    up_x, up_y = up.view(b, 2, -1).gather(2, idx_up).unbind(1)
    short = torch.exp(log_short)
    angle = 0.5 * torch.atan2(s2, c2)
    return cx, cy, short, angle, up_x, up_y, scores


def _resolve_orientation(angle: torch.Tensor, up_x: torch.Tensor, up_y: torch.Tensor) -> torch.Tensor:
    """`angle` (from `0.5*atan2`) is only defined up to a 180-degree rotation -- the same
    ambiguity `table_detector.decode_detections` resolves with `orient_quad(cyclic_order(...))`
    on the *constructed* quad's corners. Since this pipeline builds the quad from (center,
    short, angle) rather than receiving one from elsewhere, the equivalent check can run
    directly on `angle` and `up` instead: a rectangle built at `angle` has its own "top edge
    points this way" direction at `(sin(angle), -cos(angle))` (see `table_detector.card_rect`'s
    rotation convention); when that disagrees with the predicted up vector, the true orientation
    is `angle + pi`, not `angle`. Returns the disambiguated angle."""
    naive_up_x, naive_up_y = torch.sin(angle), -torch.cos(angle)
    flip = (naive_up_x * up_x + naive_up_y * up_y) < 0
    return angle + flip * torch.pi


def _rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    """`torch.rot90(x, k, dims=(-2,-1))`, reimplemented with `transpose`/`flip` because
    `aten::rot90` itself has no ONNX opset-18 exporter (confirmed by trying: `UnsupportedOperatorError`).
    Verified equal to `torch.rot90` for every `k` in 0..3 before this replaced it."""
    if k == 0:
        return x
    if k == 1:
        return x.transpose(-2, -1).flip(-2)
    if k == 2:
        return x.flip(-2).flip(-1)
    return x.transpose(-2, -1).flip(-1)  # k == 3


def _sampling_grid(
    cx: torch.Tensor, cy: torch.Tensor, short: torch.Tensor, angle: torch.Tensor, native_size: int, box: tuple[float, float, float, float], out_w: int, out_h: int
) -> torch.Tensor:
    """(B, K) pose parameters -> (B*K, out_h, out_w, 2) `grid_sample` grid, normalised to
    `native_size`, that cuts `box`'s (x0, y0, x1, y1) card-fraction window directly out of the
    full frame in one step, at `box`'s own native (out_w, out_h) resolution -- composing the
    card-rect warp and the frame crop, before any rotation/resize (see module docstring)."""
    x0, y0, x1, y1 = box
    device, dtype = cx.device, cx.dtype
    n = cx.numel()
    u = torch.linspace(x0, x1, out_w, device=device, dtype=dtype)
    v = torch.linspace(y0, y1, out_h, device=device, dtype=dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")  # (out_h, out_w), card-fraction coords
    local_x = (grid_u - 0.5) * short.view(n, 1, 1)  # local box coords, PRINTED-template convention
    local_y = (grid_v - 0.5) * (short.view(n, 1, 1) * CARD_ASPECT)
    cos_a, sin_a = torch.cos(angle).view(n, 1, 1), torch.sin(angle).view(n, 1, 1)
    world_x = local_x * cos_a - local_y * sin_a + cx.view(n, 1, 1)  # card_rect's rot.T, applied forward
    world_y = local_x * sin_a + local_y * cos_a + cy.view(n, 1, 1)
    norm_x = world_x / (native_size - 1) * 2 - 1
    norm_y = world_y / (native_size - 1) * 2 - 1
    return torch.stack([norm_x, norm_y], dim=-1)


class DetectAndEmbed(nn.Module):
    """Wraps a frozen `TableCenterNet` and a frozen `Embedder`. `forward` takes native
    `native_size`-square RGB images (N,3,H,W) and returns (embeddings
    (N,MAX_CARDS,N_FRAMES,EMBED_DIM), scores (N,MAX_CARDS), quads (N,MAX_CARDS,4,2)) --
    `MAX_CARDS` fixed slots per image, sorted by score, in the *same* full-image pixel
    coordinates `decode_detections` would use. `embeddings[n, k]` is ready to pass to
    `search.onnx` as-is, in `detect.FRAME_NAMES` order, for slot `k`'s detection."""

    def __init__(
        self,
        embed_checkpoint: str,
        native_size: int,
        max_cards: int = MAX_CARDS,
        table_checkpoint: str | None = None,
        detector: nn.Module | None = None,
    ):
        """Either pass `table_checkpoint` (builds a plain, single-pass `TableCenterNet`, the
        original behaviour) or pass a pre-built `detector` module directly -- e.g. a
        `tiled_fusion.TiledFusionDetector` -- to swap in the tiled/fused detection path instead.
        Any `detector` must return (heat logits, pose, up) at stride `TABLE_STRIDE` (true for
        both `TableCenterNet` and `TiledFusionDetector`, since the latter keeps the same physical
        stride, just over a larger canonical grid) given `native_size`-square images."""
        super().__init__()
        self.native_size = native_size
        self.max_cards = max_cards
        if detector is not None:
            self.table = detector
        else:
            if table_checkpoint is None:
                raise ValueError("pass either table_checkpoint or detector")
            self.table = TableCenterNet(pretrained=False)
            self.table.load_state_dict(torch.load(table_checkpoint, map_location="cpu", weights_only=True))
        self.embed = Embedder(pretrained=False)
        self.embed.load_state_dict(torch.load(embed_checkpoint, map_location="cpu", weights_only=True))
        for m in (self.table, self.embed):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.table.eval()
        self.embed.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = images.shape[0]
        heat, pose, up = self.table(images)
        cx, cy, short, angle, up_x, up_y, scores = _topk_peaks(heat, pose, up, TABLE_STRIDE, self.max_cards)
        angle = _resolve_orientation(angle, up_x, up_y)
        cx_flat, cy_flat, short_flat, angle_flat = (t.reshape(-1) for t in (cx, cy, short, angle))
        images_expanded = images.unsqueeze(1).expand(-1, self.max_cards, -1, -1, -1).reshape(n * self.max_cards, 3, self.native_size, self.native_size)

        # One crop per frame hypothesis, each at that frame's own native (possibly non-square)
        # aspect before rotation/resize -- see module docstring for why the order matters. All 14
        # frames' crops are concatenated into one batch for a single `self.embed` call (one kernel
        # launch instead of 14) rather than embedding each frame separately.
        per_frame_crops = []
        for box, (raw_w, raw_h), k in zip(_FRAME_BOXES, _FRAME_RAW_SIZE, _FRAME_ROTATION_K):
            grid = _sampling_grid(cx_flat, cy_flat, short_flat, angle_flat, self.native_size, box, raw_w, raw_h)
            crop = F.grid_sample(images_expanded, grid, mode="bilinear", align_corners=True)
            crop = _rot90(crop, k)
            per_frame_crops.append(F.interpolate(crop, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False))
        all_crops = torch.cat(per_frame_crops, dim=0)  # (N_FRAMES * n * max_cards, 3, INPUT_SIZE, INPUT_SIZE)
        all_embeddings = self.embed(all_crops)  # one batched call instead of N_FRAMES separate ones
        embeddings = all_embeddings.view(N_FRAMES, n, self.max_cards, EMBED_DIM).permute(1, 2, 0, 3)

        half_short, half_long = short / 2, short * CARD_ASPECT / 2
        local = torch.stack([torch.stack([-half_short, -half_long], -1), torch.stack([half_short, -half_long], -1),
                              torch.stack([half_short, half_long], -1), torch.stack([-half_short, half_long], -1)], dim=-2)  # (B,K,4,2)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot = torch.stack([torch.stack([cos_a, -sin_a], -1), torch.stack([sin_a, cos_a], -1)], dim=-2)  # (B,K,2,2)
        quads = torch.einsum("bkij,bkcj->bkci", rot, local) + torch.stack([cx, cy], -1).unsqueeze(-2)
        return embeddings, scores, quads
