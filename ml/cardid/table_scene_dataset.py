"""Torch `Dataset` over an already-rendered table-scene manifest (see `table_scenes.py`).

Unlike `scene_datasets.SceneDataset`, scenes are not rendered on the fly here: `table_scenes`
already wrote JPEGs and a JSONL manifest to disk (typically on a separate data drive, see
`ml/README.md`), so this dataset only reads and resizes them and builds `table_detector`'s
dense targets.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import to_tensor
from .detector import fit_card_pose
from .hardware_stress import apply_combined, apply_dark_room
from .table_detector import TABLE_INPUT, TABLE_STRIDE, build_targets

# Found by `hardware_stress.py`: dark rooms and stacked degradations cost 25-30 points of
# recall while individual lens/lighting artifacts (vignette, chromatic aberration, colour
# cast) cost nothing, so only those two are worth the training-time cost of augmenting for.
HARDWARE_AUGMENTATIONS = (apply_dark_room, apply_combined)


def card_pose_and_up(quad: np.ndarray) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """A manifest card's quad (already in printed order) -> (pose, up unit vector)."""
    cx, cy, short, angle = fit_card_pose(quad)
    top = (quad[0] + quad[1]) / 2 - quad.mean(axis=0)
    return (cx, cy, short, angle), top / (np.linalg.norm(top) + 1e-6)


class TableSceneDetectionDataset(Dataset):
    """One item per manifest row: the resized scene image and its dense detection targets."""

    def __init__(self, rows: list[dict], root: Path, input_size: int = TABLE_INPUT, stride: int = TABLE_STRIDE, hardware_stress_rate: float = 0.0):
        self.rows = rows
        self.root = Path(root)
        self.input_size = input_size
        self.stride = stride
        self.hardware_stress_rate = hardware_stress_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[i]
        path = self.root / row["image"]
        raw = cv2.imread(str(path))
        if raw is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        scale = self.input_size / row["width"]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=interp)
        if self.hardware_stress_rate and np.random.random() < self.hardware_stress_rate:
            augment = HARDWARE_AUGMENTATIONS[np.random.randint(len(HARDWARE_AUGMENTATIONS))]
            image = augment(image, np.random.default_rng())
        poses, ups = [], []
        for card in row["cards"]:
            pose, up = card_pose_and_up(np.float32(card["quad"]) * scale)
            poses.append(pose)
            ups.append(up)
        heat, pose_t, up_t, mask = build_targets(poses, ups, self.input_size, self.stride)
        return to_tensor(image), torch.from_numpy(heat), torch.from_numpy(pose_t), torch.from_numpy(up_t), torch.from_numpy(mask)
