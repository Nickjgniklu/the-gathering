"""On-the-fly `Dataset` of (background, current) scene pairs for `TableCenterNetDual`.

Unlike `table_scene_dataset.TableSceneDetectionDataset` (pre-rendered scenes read from disk),
pairs are rendered fresh per sample -- `render_table_scene_pair` is cheap enough (no sleeves,
loaders, gloss, or stacking, since this model's job is background suppression, not those other
detection cases) and a pre-rendered pair set would double the disk/generation cost for a still-
unproven architecture. `[seed, epoch, i]` makes every sample reproducible without needing a
frozen manifest, the same convention `scene_datasets.SceneDataset` uses.

Both halves get independent `hardware_stress` degradation draws (never the same one, and at
different rates -- see `BACKGROUND_STRESS_RATE`/`CURRENT_STRESS_RATE`): a background captured
once at session start and a current frame minutes or hours later are never pixel-registered in
deployment, so training must never hand the network a clean, matched pair. Background is stressed
*more* often than current, since a quick startup capture (the user glancing at a "capture
background" prompt) is more likely to be under bad lighting than a frame during active play.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import to_tensor
from .hardware_stress import apply_combined, apply_dark_room
from .image_bank import ArtBank, CardBank
from .table_detector import TABLE_INPUT, TABLE_STRIDE, build_targets
from .table_scene_dataset import card_pose_and_up
from .table_scenes import SETUPS, render_table_scene_pair

HARDWARE_AUGMENTATIONS = (apply_dark_room, apply_combined)
BACKGROUND_STRESS_RATE = 0.3
CURRENT_STRESS_RATE = 0.15


def _maybe_stress(image: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    if rate and rng.random() < rate:
        augment = HARDWARE_AUGMENTATIONS[rng.integers(len(HARDWARE_AUGMENTATIONS))]
        return augment(image, rng)
    return image


class TableScenePairDataset(Dataset):
    """One item per rendered pair: (background tensor, current tensor, dense detection targets
    for the cards on `current`)."""

    def __init__(
        self,
        length: int,
        cards: CardBank | None = None,
        arts: ArtBank | None = None,
        seed: int = 0,
        input_size: int = TABLE_INPUT,
        stride: int = TABLE_STRIDE,
        cards_per_scene: tuple[int, int] = (0, 14),
    ):
        self.length = length
        self.cards = cards or CardBank()
        self.arts = arts if arts is not None else ArtBank()
        self.cards.build()
        self.arts.build()
        self.seed = seed
        self.input_size = input_size
        self.stride = stride
        self.cards_per_scene = cards_per_scene
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.epoch, i])
        setup = SETUPS[int(rng.integers(len(SETUPS)))]
        count = min(int(rng.integers(*self.cards_per_scene)), len(self.cards))
        bg, cur, record = render_table_scene_pair(int(rng.integers(2**31)), self.cards, self.arts, setup, count=count, size=1280, out=self.input_size)
        bg = _maybe_stress(bg, rng, BACKGROUND_STRESS_RATE)
        cur = _maybe_stress(cur, rng, CURRENT_STRESS_RATE)

        poses, ups = [], []
        for card in record["cards"]:
            pose, up = card_pose_and_up(np.float32(card["quad"]))
            poses.append(pose)
            ups.append(up)
        negatives = [tuple(n["bbox"]) for n in record.get("negatives", [])]
        heat, pose_t, up_t, mask, hard_neg = build_targets(poses, ups, self.input_size, self.stride, negatives)
        return (
            to_tensor(bg),
            to_tensor(cur),
            torch.from_numpy(heat),
            torch.from_numpy(pose_t),
            torch.from_numpy(up_t),
            torch.from_numpy(mask),
            torch.from_numpy(hard_neg),
        )
