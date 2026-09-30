"""Turns `table_scenes.render_table_scene_sequence` output into per-track training examples for
`TrackMemory`: for every physical card that appears in a sequence, one padded (embedding_t,
pose_t, score_t, true_pose_t, valid) time series plus its true gallery row index. `true_pose_t` is
read off the ground-truth quad (`_quad_pose`), independent of the detector's own noisy `pose_t` --
the training target for `refined_pose`'s regularizer, not something to feed into `TrackMemory`.

Ground-truth correspondence between the detector's per-frame ranked slots and the sequence's
persistent `track_id`s is IoU-matching against the manifest's own quads (`_match_frame`) -- more
reliable than `track_memory.associate_detections`'s heuristic, and keeps Stage 1 association bugs
from contaminating what Stage 2 learns from. `associate_detections` is inference-time only (see
`track_memory.py`'s module docstring); this dataset never calls it.

`DetectAndEmbed` produces 14 frame-hypothesis embeddings per detected slot (one per
`detect.FRAME_NAMES` entry), but `TrackMemory.forward`'s design takes a single 128-d `embedding_t`.
Resolved the same way the raw (unsmoothed) pipeline already would: for each matched slot, find
which of its 14 hypotheses the real gallery search would actually have scored best (same math as
`evaluate_detect_and_embed.search`, argmax'd), and use that one -- `embedding_t` is "the raw
pipeline's own best guess this frame," and `TrackMemory`'s job is to smooth *that* time series, not
to additionally arbitrate between frame hypotheses itself.

A sequence's card pool must already be gallery-verified (every `card_id` present in the gallery
`gallery_bundle` points at) -- the same requirement `regen_gallery_verified.py` exists to satisfy
for the non-sequential dataset, and for the same reason (see that module and the
`gallery_printing_gap` note in `ml/README.md`'s history): a card absent from the gallery has no
true row index to train against. A track whose `card_id` isn't found is skipped with a count
printed at load time, but the intended workflow renders sequences from a gallery-verified pool in
the first place so this count is normally zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import to_tensor
from .detect_and_embed import DetectAndEmbed
from .evaluate_detect_and_embed import load_gallery
from .scene_geometry import quad_iou, quad_short
from .tiled_fusion import TiledFusionDetector

SCORE_THRESHOLD = 0.3  # a detected slot below this is treated as noise, not a real card, when matching
IOU_THRESHOLD = 0.5  # same convention as evaluate_tables.per_card_hits


def _match_frame(detected_quads: list[np.ndarray], detected_scores: list[float], truth_cards: list[dict], score_threshold: float, iou_threshold: float = IOU_THRESHOLD) -> dict[int, int]:
    """Greedy IoU matching, same style as `evaluate_tables.per_card_hits`, generalized to return
    which detected slot matched which ground-truth `track_id` rather than just a hit/miss bool.
    Returns {track_id: detected_slot_index}; a ground-truth card with no matching detection this
    frame (a genuine miss -- e.g. total occlusion, or the detector simply missed it) is absent
    from the result, not forced."""
    candidates = [(i, q) for i, (q, s) in enumerate(zip(detected_quads, detected_scores)) if s > score_threshold]
    pairs = sorted(
        ((quad_iou(np.float32(truth["quad"]), q), truth["track_id"], slot) for truth in truth_cards for slot, q in candidates),
        reverse=True,
    )
    used_tracks: set[int] = set()
    used_slots: set[int] = set()
    matched: dict[int, int] = {}
    for iou, track_id, slot in pairs:
        if iou < iou_threshold or track_id in used_tracks or slot in used_slots:
            continue
        matched[track_id] = slot
        used_tracks.add(track_id)
        used_slots.add(slot)
    return matched


def _quad_pose(quad: np.ndarray) -> torch.Tensor:
    """A quad's true `(cx, cy, short, cos(2*angle), sin(2*angle))` -- the training target for
    `refined_pose`'s regularizer, independent of the detector's own noisy `pose_t`. Corner-1-minus-
    corner-0's direction equals `DetectAndEmbed`'s own `angle` convention exactly (both are built
    the same way from a card-rect rotation -- see `detect_and_embed.py`'s quad construction and
    `table_scenes.TableCard.orientation`, which uses this identical `atan2`), so this is the same
    quantity the network predicts, just read off the ground-truth quad instead of estimated."""
    cx, cy = quad.mean(axis=0)
    angle = float(np.arctan2(quad[1, 1] - quad[0, 1], quad[1, 0] - quad[0, 0]))
    return torch.tensor([cx, cy, quad_short(quad), np.cos(2 * angle), np.sin(2 * angle)], dtype=torch.float32)


def _best_frame_embedding(slot_embeddings: torch.Tensor, gallery: torch.Tensor, frames: torch.Tensor, penalties: torch.Tensor) -> torch.Tensor:
    """`slot_embeddings` (14, 128), one detected slot's frame hypotheses -> the single (128,)
    embedding whichever gallery row the real search would rank first actually used. Same math as
    `evaluate_detect_and_embed.search` (cosine to each gallery row's own frame, minus its
    penalty), stopping at the argmax instead of a top-k list since only the winning hypothesis
    is needed here."""
    sims = gallery @ slot_embeddings.T  # (N_gallery, 14)
    own_frame_sims = sims.gather(1, frames[:, None])[:, 0] - penalties
    best_gallery_row = int(torch.argmax(own_frame_sims))
    best_frame_idx = int(frames[best_gallery_row])
    return slot_embeddings[best_frame_idx]


class TrackMemorySequenceDataset(Dataset):
    """One example per physical card (track) across every sequence under `manifest_path`
    (`table_scenes.write_sequence_split`'s layout: `<dir>/<sequence>/frame_NNN.jpg` plus a shared
    `manifest.jsonl`). `native_size` must match the frozen detector these sequences are scored
    with. Every example is padded/masked to `max_frames` (default: the longest sequence present),
    since a track's own true length can be anywhere from 1 frame to the sequence's `n_frames`."""

    def __init__(
        self,
        manifest_path: Path,
        table_checkpoint: Path,
        embed_checkpoint: Path,
        gallery_bundle: Path,
        native_size: int,
        detector: str = "single-pass",
        fusion_checkpoint: Path | None = None,
        score_threshold: float = SCORE_THRESHOLD,
        max_frames: int | None = None,
        limit: int | None = None,
    ):
        self.root = manifest_path.parent
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        self.rows = rows[:limit] if limit else rows
        self.native_size = native_size
        self.score_threshold = score_threshold
        self.max_frames = max_frames or max((row["n_frames"] for row in self.rows), default=1)

        if detector == "single-pass":
            self.model = DetectAndEmbed(str(embed_checkpoint), native_size, table_checkpoint=str(table_checkpoint)).eval()
        else:
            if fusion_checkpoint is None:
                raise ValueError("detector='tiled-fusion' needs fusion_checkpoint")
            tiled = TiledFusionDetector(checkpoint=str(table_checkpoint), native_size=native_size)
            tiled.fusion.load_state_dict(torch.load(fusion_checkpoint, map_location="cpu", weights_only=True))
            self.model = DetectAndEmbed(str(embed_checkpoint), native_size, detector=tiled).eval()

        arts, self.gallery, self.frames, self.penalties = load_gallery(gallery_bundle)
        id_to_index = {a["id"]: i for i, a in enumerate(arts)}

        # Flatten (sequence, track_id) into one example list up front, dropping any track whose
        # card isn't in the gallery (see module docstring) -- so __len__/__getitem__ are trivial
        # and every worker sees the same fixed index space.
        self._examples: list[tuple[dict, int, int]] = []
        skipped = 0
        for row in self.rows:
            card_id_by_track: dict[int, str] = {}
            for frame in row["frames"]:
                for c in frame["cards"]:
                    card_id_by_track.setdefault(c["track_id"], c["card_id"])
            for track_id, card_id in card_id_by_track.items():
                gallery_index = id_to_index.get(card_id)
                if gallery_index is None:
                    skipped += 1
                    continue
                self._examples.append((row, track_id, gallery_index))
        if skipped:
            print(f"TrackMemorySequenceDataset: skipped {skipped} tracks whose card isn't in the gallery (render from a gallery-verified pool to avoid this)")

    def __len__(self) -> int:
        return len(self._examples)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row, track_id, gallery_index = self._examples[idx]
        embeddings = torch.zeros(self.max_frames, 128)
        poses = torch.zeros(self.max_frames, 5)
        scores = torch.zeros(self.max_frames, 1)
        true_poses = torch.zeros(self.max_frames, 5)  # from the ground-truth quad, not the noisy detector
        valid = torch.zeros(self.max_frames, dtype=torch.bool)

        seq_dir = self.root / row["sequence"]
        scale = self.native_size / row["width"]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        for t, (frame_name, frame_manifest) in enumerate(zip(row["frame_files"], row["frames"])):
            if t >= self.max_frames:
                break
            image = cv2.cvtColor(cv2.imread(str(seq_dir / frame_name)), cv2.COLOR_BGR2RGB)
            small = cv2.resize(image, (self.native_size, self.native_size), interpolation=interp)
            emb, sc, quads, pose = self.model(to_tensor(small).unsqueeze(0), return_pose=True)
            emb, sc, quads, pose = emb[0], sc[0], quads[0], pose[0]  # (MAX_CARDS,14,128) (MAX_CARDS,) (MAX_CARDS,4,2) (MAX_CARDS,5)

            # The model's own quads are already in native_size pixel space; scale the manifest's
            # ground-truth quads *down* to match (evaluate_detect_and_embed.py's convention),
            # rather than scaling the model's output up.
            scaled_truth = [{"quad": (np.float32(c["quad"]) * scale).tolist(), "track_id": c["track_id"]} for c in frame_manifest["cards"]]
            matched = _match_frame([q.numpy() for q in quads], sc.tolist(), scaled_truth, self.score_threshold)
            slot = matched.get(track_id)
            if slot is None:
                continue  # a genuine miss this frame: stays invalid/padding

            embeddings[t] = _best_frame_embedding(emb[slot], self.gallery, self.frames, self.penalties)
            poses[t] = pose[slot]
            scores[t] = sc[slot]
            true_poses[t] = _quad_pose(np.float32(next(c for c in scaled_truth if c["track_id"] == track_id)["quad"]))
            valid[t] = True

        return embeddings, poses, scores, true_poses, valid, gallery_index
