"""DESIGN DOCUMENT, NOT YET IMPLEMENTED. Stubs below sketch the intended shape so a future
session can pick this up without re-deriving the design; nothing here runs yet.

Goal: stabilize `detect_and_embed.py`'s per-frame output across a live video stream. Today every
frame is scored independently, so a card's embedding is a noisy one-shot sample (motion blur,
exposure flicker, a slightly different angle) that occasionally crosses a decision boundary in
the gallery search and briefly resolves to the *wrong* card. The fix is to stop trusting any
single frame in isolation -- carry a running belief about each physical card forward across
frames, for both its embedding (identity) and its pose (location), and only let each new frame
nudge that belief rather than replace it outright.

This combines two independently-existing pieces:

  * `tiled_fusion.TiledFusionDetector` -- 4 frozen `TableCenterNet` tile passes + a trained
    `FusionHead`, for the higher-effective-resolution dense (heat, pose, up) this design's input
    should come from instead of a plain single-pass `TableCenterNet`.
  * `detect_and_embed.py`'s top-`MAX_CARDS` peak decode + crop + `Embedder` -- unchanged, still
    the per-frame (quad, score, embedding) producer this module consumes one frame at a time.

Everything below is new.

## Stage 1: track association (classical bookkeeping, not learned, not in this file's nn.Module)

The detector doesn't output "card A, card B, card C" -- it outputs up to `MAX_CARDS` detections
ranked by confidence, and which physical card lands in which rank can shuffle frame to frame from
score noise alone even when nothing moved. Concrete example: frame 1 gives slot 0 = quad at
(100, 200), slot 1 = quad at (400, 150), slot 2 = quad at (250, 500). Frame 2, cards untouched,
gives slot 0 = quad at (400, 148), slot 1 = quad at (101, 199), slot 2 = quad at (251, 501) --
slots 0 and 1 swapped identity purely because their confidence scores traded places. Feeding
"slot 0's new embedding" into "slot 0's carried-forward state" would blend two different cards'
history together.

The fix is ordinary nearest-position matching, done *before* anything reaches a learned
component: for each of this frame's detections, find whichever of last frame's tracked positions
is closest; within a gate distance (roughly half a card-width), that's the same physical card,
just a different rank. A detection with no close match is a new card (start fresh state). A
track with no match this frame is possibly a momentary miss or occlusion, not necessarily gone --
keep its state alive for a couple of frames before dropping it, rather than deleting on the first
miss. This is deliberately *not* a neural network: it's a discrete correspondence problem
(SORT/DeepSORT/ByteTrack-style tracking-by-detection), cheap and easy to debug, and there's no
accuracy benefit to making it learned.

## Stage 2: per-track recurrent state (the actual new model, GRU-based)

One shared-weight `GRUCell` (not one GRU per card -- the same cell applied to every track, every
frame, learning a general "how much should this frame move my belief" policy, not memorizing
specific cards):

    input  = concat(embedding_t (128), pose_t (cx, cy, short, cos2t, sin2t), score_t)  # ~135-d
    hidden_t = GRUCell(input, hidden_{t-1})                                            # e.g. 128-d
    refined_embedding = L2Normalize(Linear(hidden_t))          # search the gallery on THIS
    refined_pose      = pose_t + Linear(hidden_t)              # small residual correction, not
                                                                # a from-scratch re-prediction

A new track's `hidden_0` starts at zero (or a learned initial vector). Search the embedding
gallery against `refined_embedding`, not the raw per-frame embedding -- a single noisy frame can
nudge it but can't flip an identification on its own the way it can today, which is the specific
failure mode this whole design exists to fix.

Considered and deliberately not chosen for v1: an explicit fixed 3-frame window (a causal
temporal conv or small attention over the last 3 raw embeddings per track) instead of a GRU's
implicit, smoothly-decaying memory. The GRU is the more standard choice for this kind of problem
and deploys the same way either way (the frontend already holds detections across frames for
overlay stability; one 128-d hidden vector per track alongside that is a small addition, not a
new category of complexity) -- but if training data or stability problems make the GRU hard to
get working, the fixed-window variant is the direct fallback and needs no architectural rethink,
just a different Stage 2 module with the same inputs/outputs.

## What this needs that doesn't exist yet

Training needs *sequences*, not independent scenes -- `table_scenes.py` only renders one-off
frames today. The nearest existing building block is `render_table_scene_pair` (parked on
`feat/background-subtraction-detector`), which renders two frames sharing a scene with
independent photometric drift. The natural extension is a `render_table_scene_sequence(n_frames)`
that keeps card identity and layout fixed across N frames while independently perturbing each
frame's camera jitter, lighting, and card angle by a small amount -- enough to reproduce the
same-card-different-embedding-per-frame noise this module exists to smooth over, without
simulating real physics. This is the one genuinely missing prerequisite; everything else
(the frozen tile detector, the embedder, the crop layer) already exists and works.

Loss sketch: across a rendered sequence, `refined_embedding` should match the true card's
canonical gallery embedding more often / more confidently than the raw per-frame embedding does
(e.g. a contrastive or triplet loss against the gallery, evaluated at every timestep, compared as
a same-seed before/after against the un-smoothed baseline the way every other change in this
project has been measured); `refined_pose` should reduce frame-to-frame jitter against
ground-truth quads without lagging behind genuine motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

MAX_CARDS = 20  # matches detect_and_embed.MAX_CARDS
GATE_DISTANCE = 40.0  # placeholder: roughly half a card-width at the resolution tracks operate in
MAX_MISSES_BEFORE_DROP = 2
HIDDEN_DIM = 128


@dataclass
class Track:
    """One physical card's carried-forward state. `hidden` is the GRU's running belief;
    `last_quad`/`last_score` are only used for Stage 1's nearest-position matching, not fed to
    the GRU directly (the GRU sees `pose_t`, not corner coordinates)."""

    track_id: int
    hidden: torch.Tensor  # (HIDDEN_DIM,)
    last_center: np.ndarray  # (2,), for next frame's nearest-position match
    misses: int = 0


def associate_detections(tracks: list[Track], detection_centers: np.ndarray, gate: float = GATE_DISTANCE) -> tuple[dict[int, int], list[int]]:
    """Match this frame's `detection_centers` (K, 2) against `tracks` by nearest position.
    Returns (matched: {detection_index: track_index}, unmatched_detection_indices). Greedy
    nearest-neighbour is enough here (typical frame-to-frame card displacement is small relative
    to card spacing); switch to a proper assignment solver only if greedy proves to misassign in
    practice. NOT YET IMPLEMENTED -- see module docstring's Stage 1 example."""
    raise NotImplementedError("design stub -- see module docstring")


class TrackMemory(nn.Module):
    """The Stage 2 GRU. Operates on ONE track's history at a time (call once per matched track,
    weights shared); the surrounding loop that maintains `Track` objects across frames, calls
    `associate_detections`, and ages out missed tracks lives outside this class, in the eventual
    per-frame driver (not written yet)."""

    def __init__(self, embed_dim: int = 128, pose_dim: int = 5, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.cell = nn.GRUCell(embed_dim + pose_dim + 1, hidden_dim)
        self.to_embedding = nn.Linear(hidden_dim, embed_dim)
        self.to_pose_residual = nn.Linear(hidden_dim, pose_dim)

    def forward(self, hidden: torch.Tensor, embedding_t: torch.Tensor, pose_t: torch.Tensor, score_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(hidden, embedding_t (B,128), pose_t (B,5), score_t (B,1)) -> (new_hidden,
        refined_embedding (B,128) L2-normalised, refined_pose (B,5)). NOT YET IMPLEMENTED."""
        raise NotImplementedError("design stub -- see module docstring")
