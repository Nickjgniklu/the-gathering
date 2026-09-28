"""Torch `Dataset` for `train_tiled_fusion.py`: reuses the same already-rendered table-scenes
manifest `table_scene_dataset.py` reads, but returns the image at a chosen native resolution
(unresized when the manifest already matches it) plus dense targets built at that same native
resolution's own stride-4 grid, instead of downsampling to `TABLE_INPUT` first. `TiledFusionDetector`
does its own resizing per tile; collapsing to `TABLE_INPUT` here would throw away the resolution
tiling exists to keep (see `tiled_fusion.py`'s module docstring)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import to_tensor
from .table_detector import build_targets
from .table_scene_dataset import card_pose_and_up
from .tiled_fusion import CANONICAL_STRIDE, NATIVE_SIZE


class TiledFusionDataset(Dataset):
    """One item per manifest row: the native-resolution scene image and its dense detection
    targets at `native_size // CANONICAL_STRIDE` resolution. Rows whose stored resolution does
    not match `native_size` are resized to it (with their quads scaled to match) so this
    tolerates a manifest generated at a different `--resolution` without silently mis-scaling
    targets."""

    def __init__(self, rows: list[dict], root: Path, native_size: int = NATIVE_SIZE):
        self.rows = rows
        self.root = Path(root)
        self.native_size = native_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[i]
        path = self.root / row["image"]
        raw = cv2.imread(str(path))
        if raw is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        scale = self.native_size / row["width"]
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (self.native_size, self.native_size), interpolation=interp)
        poses, ups = [], []
        for card in row["cards"]:
            pose, up = card_pose_and_up(np.float32(card["quad"]) * scale)
            poses.append(pose)
            ups.append(up)
        negatives = [tuple(v * scale for v in n["bbox"]) for n in row.get("negatives", [])]
        heat, pose_t, up_t, mask, hard_neg = build_targets(poses, ups, self.native_size, CANONICAL_STRIDE, negatives)
        return (
            to_tensor(image),
            torch.from_numpy(heat),
            torch.from_numpy(pose_t),
            torch.from_numpy(up_t),
            torch.from_numpy(mask),
            torch.from_numpy(hard_neg),
        )
