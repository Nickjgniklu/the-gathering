"""User corrections for a tracked card ("this detected card is really Y"), deliberately kept
outside `TrackMemory`'s trained weights.

A correction must be authoritative and inspectable, not something a GRU has learned discretion
over -- so `TrackIdentity` is plain Python, not an `nn.Module`, and `step()` below is the whole
mechanism: it is not a step towards a correction, it *is* the correction applied, every frame,
until it releases. Two things happen at once when a track is pinned:

  1. The identity actually shown to the user is forced to `pinned_gallery_index`, full stop --
     `TrackMemory`'s own `refined_embedding`/gallery search is not consulted while a pin is active.
  2. `TrackMemory`'s hidden state keeps being fed the *pinned card's own canonical gallery
     embedding* as this frame's "observation" (`step()`'s return value), not the raw per-frame
     embedding the detector actually produced. Otherwise the GRU's belief would keep drifting from
     whatever the detector is actually seeing, and releasing the pin later (manually, or via
     auto-release below) would cause a jarring snap back to a belief that never agreed with the
     correction in the first place.

A pin auto-releases if the raw per-frame embedding stops looking anything like the pinned card for
`auto_release_frames` consecutive frames: the most likely explanation at that point is that the
*physical* card in this track's spot was swapped for a different one (picked up, something else
played in the same spot), not that the correction was wrong -- so releasing hands the slot back to
the model rather than keeping a now-stale pin forever. A manual `clear_correction` is always
available too, for a user who corrects, then reconsiders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackIdentity:
    """One track's correction state. `pinned_gallery_index` is `None` when nothing has been
    corrected -- the normal, model-driven path. `mismatch_streak` only moves while pinned; it
    resets to 0 the moment a pin starts or a frame agrees with it again."""

    pinned_gallery_index: int | None = None
    mismatch_streak: int = 0


def apply_correction(identity: TrackIdentity, gallery_index: int) -> None:
    """The user said this track is really `gallery_index`. Takes effect immediately and resets
    any in-progress auto-release countdown from a previous pin."""
    identity.pinned_gallery_index = gallery_index
    identity.mismatch_streak = 0


def clear_correction(identity: TrackIdentity) -> None:
    """Hand the track back to the model, whether because the user asked to undo a correction or
    because `step()` auto-released it."""
    identity.pinned_gallery_index = None
    identity.mismatch_streak = 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def step(
    identity: TrackIdentity,
    raw_embedding: np.ndarray,
    gallery_embedding_for_pin: np.ndarray | None,
    auto_release_floor: float = 0.5,
    auto_release_frames: int = 5,
) -> tuple[np.ndarray, bool]:
    """Called once per frame, per track, before `TrackMemory` sees this frame's embedding.

    When not pinned, returns `(raw_embedding, False)` unchanged -- the model path, untouched.

    When pinned, returns `(gallery_embedding_for_pin, True)` -- feed *that*, not `raw_embedding`,
    into `TrackMemory` this frame (see module docstring) -- unless the pin has just auto-released
    this frame, in which case it falls back to returning `(raw_embedding, False)` immediately so
    the caller doesn't need a separate check.

    `gallery_embedding_for_pin` must be supplied (non-`None`) whenever `identity` is pinned; it is
    the pinned card's own canonical embedding, looked up by the caller from `ArtIndex`/the
    gallery, not computed here.
    """
    if identity.pinned_gallery_index is None:
        return raw_embedding, False
    assert gallery_embedding_for_pin is not None, "a pinned track needs its pinned card's gallery embedding"
    if _cosine(raw_embedding, gallery_embedding_for_pin) < auto_release_floor:
        identity.mismatch_streak += 1
        if identity.mismatch_streak >= auto_release_frames:
            clear_correction(identity)
            return raw_embedding, False
    else:
        identity.mismatch_streak = 0
    return gallery_embedding_for_pin, True
