"""A tiled detector where the tiling and overlap-deduping are part of the model, not a post-hoc
inference-time script. `tiled_inference.py` proved the idea works (split the frame into
overlapping tiles so no card gets downsampled as hard as a single whole-frame pass, greedily
merge duplicate detections in the overlap) entirely without retraining; this module replaces that
hand-coded merge step with a small trained one, while keeping the actual card-recognising work in
the existing, already-trained `TableCenterNet` unchanged and frozen.

Architecture: `TABLE_GRID` (2x2, 20% overlap, same geometry `tiled_inference.py` validated) tiles
are each resized to `TABLE_INPUT` and run through one frozen `TableCenterNet` (a single shared
instance, not four separate networks -- "4 frozen copies" means 4 forward passes of the same
weights, not 4x the parameters). Each tile's raw (pre-sigmoid heat, pose, up) output is projected
back onto a shared canonical grid at the *native* image's own stride-4 resolution (`NATIVE_SIZE`,
`CANONICAL_GRID`) -- not resized down to `TABLE_INPUT` first, so the fusion step still sees
whatever extra resolution tiling bought. Every canonical cell gets one (heat, pose, up, valid)
slot per tile that covers it (a corner cell can have up to 4, most cells 1-2), zero-filled where a
tile doesn't reach; `FusionHead` is a small trainable conv stack over the stacked 4-tile slots
that outputs one final (heat, pose, up) at canonical resolution -- a learned replacement for
`tiled_inference.dedupe`'s greedy IoU-clustering, operating on dense features instead of a sparse
decoded detection list.

Why native-resolution stitching, not a resize back to `TABLE_INPUT`: the whole point of tiling is
to stop losing resolution to the frozen detector's `TABLE_INPUT`-sized input; collapsing back to
that same size for fusion would throw away exactly what tiling bought. The stored table-scenes
images are 640x640 (`table_scenes.py`'s `--resolution` default) and the frozen detector's input is
384, so a straight whole-frame pass already downsamples 640->384 (1.67x); each 2x2/20%-overlap
tile instead crops roughly 356px of that same 640px image and *upscales* it to 384 (~1.08x), a
real resolution gain per card, and the training data already has enough headroom for this without
regenerating anything.

    uv run python -m cardid.train_tiled_fusion --checkpoint data/runs/<run>/best.pt --manifest-dir ~/the-gathering-cardid/table-scenes
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet
from .tiled_inference import tile_boxes

NATIVE_SIZE = 640  # table_scenes.py's stored image resolution (manifest width/height)
TILE_GRID = (2, 2)
TILE_OVERLAP = 0.2
CANONICAL_STRIDE = TABLE_STRIDE  # keep the same physical stride (pixels/cell) as the frozen model
CANONICAL_GRID = NATIVE_SIZE // CANONICAL_STRIDE  # 160
N_TILES = TILE_GRID[0] * TILE_GRID[1]
SLOT_CHANNELS = 1 + 3 + 2 + 1  # heat + pose(3) + up(2) + valid mask


def _tile_geometry() -> list[tuple[int, int, int, int, int, int, float]]:
    """Precomputed, fixed for every image (same `NATIVE_SIZE`/`TILE_GRID`/`TILE_OVERLAP`
    everywhere): per tile, (ix0, iy0, ix1, iy1) in native pixels, (gx0, gy0) canonical-grid
    placement origin, and `log_scale` = log(native tile width / `TABLE_INPUT`) -- the additive
    correction `build_targets`' log-short pose channel needs when a tile's own `TABLE_INPUT`-space
    "short side" length is expressed in native-image pixels instead (see module docstring)."""
    geometry = []
    for x0, y0, x1, y1 in tile_boxes(NATIVE_SIZE, NATIVE_SIZE, TILE_GRID, TILE_OVERLAP):
        ix0, iy0, ix1, iy1 = round(x0), round(y0), round(x1), round(y1)
        gx0, gy0 = round(ix0 / CANONICAL_STRIDE), round(iy0 / CANONICAL_STRIDE)
        log_scale = float(np.log((ix1 - ix0) / TABLE_INPUT))
        geometry.append((ix0, iy0, ix1, iy1, gx0, gy0, log_scale))
    return geometry


TILE_GEOMETRY = _tile_geometry()


class FusionHead(nn.Module):
    """Trainable-only part of `TiledFusionDetector`. Input: the 4 tiles' projected (heat, pose,
    up, valid) stacked in the channel dim at `CANONICAL_GRID` resolution (`N_TILES *
    SLOT_CHANNELS` channels). Output: one fused (heat logit, pose, up) at the same resolution."""

    HIDDEN = 64

    def __init__(self):
        super().__init__()
        in_ch = N_TILES * SLOT_CHANNELS
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, self.HIDDEN, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.HIDDEN, self.HIDDEN, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(self.HIDDEN, 6, 1)
        with torch.no_grad():
            self.out.bias[0] = -2.19  # same focal-loss prior as TableCenterNet.head

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.out(self.net(slots))
        return out[:, :1], out[:, 1:4], out[:, 4:6]


class TiledFusionDetector(nn.Module):
    """Wraps a frozen `TableCenterNet` (loaded from `checkpoint`, never updated by training) with
    a trainable `FusionHead`. `forward` takes native `NATIVE_SIZE`x`NATIVE_SIZE` RGB images
    (N,3,H,W) already normalised the way `data.to_tensor` produces, and returns fused (heat
    logits, pose, up) at `CANONICAL_GRID` resolution -- a drop-in replacement for
    `TableCenterNet.forward`'s return shape, just at a different, larger grid size."""

    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        self.frozen = TableCenterNet(pretrained=False)
        if checkpoint is not None:
            self.frozen.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.fusion = FusionHead()

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()  # always frozen, even under model.train()
        return self

    @torch.no_grad()
    def _tile_crops(self, images: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) native images -> (N*N_TILES,3,TABLE_INPUT,TABLE_INPUT), tile-major within
        each image (tile 0 of every image, then tile 1 of every image, ...) so a single
        `TableCenterNet` forward pass covers everything and the output can be split back by tile
        with a plain `.view`."""
        n = images.shape[0]
        crops = images.new_empty((N_TILES, n, 3, TABLE_INPUT, TABLE_INPUT))
        for t, (ix0, iy0, ix1, iy1, *_rest) in enumerate(TILE_GEOMETRY):
            crop = images[:, :, iy0:iy1, ix0:ix1]
            crops[t] = F.interpolate(crop, size=(TABLE_INPUT, TABLE_INPUT), mode="bilinear", align_corners=False)
        return crops.reshape(N_TILES * n, 3, TABLE_INPUT, TABLE_INPUT)

    def _project_to_canonical(self, heat: torch.Tensor, pose: torch.Tensor, up: torch.Tensor, n: int) -> torch.Tensor:
        """(N_TILES*n, ...) raw per-tile outputs at `TABLE_INPUT`/`TABLE_STRIDE` resolution ->
        (n, N_TILES*SLOT_CHANNELS, CANONICAL_GRID, CANONICAL_GRID), each tile's block resized to
        its own canonical footprint and placed at its own offset, zero elsewhere."""
        heat = heat.view(N_TILES, n, *heat.shape[1:])
        pose = pose.view(N_TILES, n, *pose.shape[1:])
        up = up.view(N_TILES, n, *up.shape[1:])
        slots = heat.new_zeros((n, N_TILES * SLOT_CHANNELS, CANONICAL_GRID, CANONICAL_GRID))
        for t, (ix0, iy0, ix1, iy1, gx0, gy0, log_scale) in enumerate(TILE_GEOMETRY):
            fw, fh = round(ix1 / CANONICAL_STRIDE) - gx0, round(iy1 / CANONICAL_STRIDE) - gy0
            tile_pose = pose[t].clone()
            tile_pose[:, :1] = tile_pose[:, :1] + log_scale  # log-short: tile-space -> native-space
            block = torch.cat([heat[t], tile_pose, up[t], heat.new_ones((n, 1, *heat.shape[3:]))], dim=1)
            resized = F.interpolate(block, size=(fh, fw), mode="bilinear", align_corners=False)
            base = t * SLOT_CHANNELS
            slots[:, base : base + SLOT_CHANNELS, gy0 : gy0 + fh, gx0 : gx0 + fw] = resized
        return slots

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = images.shape[0]
        crops = self._tile_crops(images)
        with torch.no_grad():
            heat, pose, up = self.frozen(crops)
        slots = self._project_to_canonical(heat, pose, up, n)
        return self.fusion(slots)
