"""One combined model: `TableCenterNet` (frozen) finds every card on the table, a new batched
crop layer warps each one straight to the recogniser's input, and `Embedder` (frozen) embeds all
of them in one parallel pass -- the "multi-model, one accelerated forward pass" pipeline, wiring
together two already-separately-trained models rather than training anything new for this first
version (see `tiled_fusion.py` for the separate, also-already-shipped "frozen backbone + new
trainable layers" idea; this module trains nothing at all yet).

Two deliberate simplifications for this first version, both because the ask was to keep it
simple before combining further, not because they are the only way to do this:

  * **Fixed top-`MAX_CARDS` detections, not a score threshold.** `decode_detections` returns a
    variable-length list (however many peaks clear `score_threshold`), which is awkward for a
    single static-shaped exported graph. This takes the top `MAX_CARDS` local-maxima by score
    unconditionally (`_topk_peaks`) and returns their scores alongside; a low-confidence slot's
    score is still there for the caller to threshold at display time, same as today, just not
    baked into the graph's control flow.

  * **One frame hypothesis (`DEFAULT_FRAME = "modern"`), not all six.** The real pipeline tries
    every entry in `detect.FRAME_NAMES` per card because a card's frame style (which fraction of
    it is art vs. border) isn't known from its quad alone, and picks whichever scores best
    against the gallery (`index.frame_similarities`). Modern is the most common frame by far;
    trying the rest is a straightforward extension (stack more crops per card the same way this
    stacks tiles) once this simpler version is validated, not a redesign.

The crop itself is *exactly* representable as an affine warp, not an approximation of one: every
quad this pipeline ever produces comes from `TableCenterNet`'s own (center, short-side, angle)
pose, which has no skew/perspective term at all (unlike `detect.warp_card`, which perspective-warps
an arbitrary externally-supplied quad because contour-detected real geometry can be slightly
keystoned). A rotation+uniform-scale+translation is the exact inverse of how the quad was built,
so this composes the canonical-card warp and the frame-window crop into one pixel-space sampling
grid and does the whole thing in a single `grid_sample`, skipping the intermediate 250x350
canonical card image `warp_card` materialises.

    uv run python -m cardid.detect_and_embed compare --table-checkpoint data/runs/<run>/best.pt --embed-checkpoint data/runs/<run>/best.pt
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .detect import CARD_ASPECT, CARD_H, CARD_W, FRAME_ASPECT, FRAMES, INPUT_SIZE
from .model import EMBED_DIM, Embedder
from .table_detector import TABLE_STRIDE, TableCenterNet

MAX_CARDS = 20  # matches table_scenes.py's own per-scene card-count ceiling
DEFAULT_FRAME = "modern"
FRAME_X0, FRAME_Y0, FRAME_X1 = FRAMES[DEFAULT_FRAME]
FRAME_Y1 = FRAME_Y0 + (FRAME_X1 - FRAME_X0) * CARD_W / FRAME_ASPECT[DEFAULT_FRAME] / CARD_H


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


def _sampling_grid(cx: torch.Tensor, cy: torch.Tensor, short: torch.Tensor, angle: torch.Tensor, native_size: int) -> torch.Tensor:
    """(B, K) pose parameters -> (B*K, INPUT_SIZE, INPUT_SIZE, 2) `grid_sample` grid, normalised
    to `native_size`, that cuts `DEFAULT_FRAME`'s art window directly out of the full frame in
    one step (composing the card-rect warp and the frame crop; see module docstring)."""
    device, dtype = cx.device, cx.dtype
    n = cx.numel()
    u = torch.linspace(FRAME_X0, FRAME_X1, INPUT_SIZE, device=device, dtype=dtype)
    v = torch.linspace(FRAME_Y0, FRAME_Y1, INPUT_SIZE, device=device, dtype=dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")  # (INPUT_SIZE, INPUT_SIZE), card-fraction coords
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
    `native_size`-square RGB images (N,3,H,W) and returns (embeddings (N,MAX_CARDS,EMBED_DIM),
    scores (N,MAX_CARDS), quads (N,MAX_CARDS,4,2)) -- `MAX_CARDS` fixed slots per image, sorted
    by score, in the *same* full-image pixel coordinates `decode_detections` would use."""

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
        grid = _sampling_grid(cx.reshape(-1), cy.reshape(-1), short.reshape(-1), angle.reshape(-1), self.native_size)
        images_expanded = images.unsqueeze(1).expand(-1, self.max_cards, -1, -1, -1).reshape(n * self.max_cards, 3, self.native_size, self.native_size)
        crops = F.grid_sample(images_expanded, grid, mode="bilinear", align_corners=True)
        embeddings = self.embed(crops).view(n, self.max_cards, EMBED_DIM)

        half_short, half_long = short / 2, short * CARD_ASPECT / 2
        local = torch.stack([torch.stack([-half_short, -half_long], -1), torch.stack([half_short, -half_long], -1),
                              torch.stack([half_short, half_long], -1), torch.stack([-half_short, half_long], -1)], dim=-2)  # (B,K,4,2)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot = torch.stack([torch.stack([cos_a, -sin_a], -1), torch.stack([sin_a, cos_a], -1)], dim=-2)  # (B,K,2,2)
        quads = torch.einsum("bkij,bkcj->bkci", rot, local) + torch.stack([cx, cy], -1).unsqueeze(-2)
        return embeddings, scores, quads
