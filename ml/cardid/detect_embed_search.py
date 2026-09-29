"""The final stage of "one accelerated forward pass": wraps `DetectAndEmbed` with the gallery
search itself, so a single exported graph goes from a raw frame straight to "here are the top-k
gallery matches for every detected card" -- `image -> indices, scores, det_scores, quads` -- with
no separate `search.onnx` call needed, the same way `DetectAndEmbed` already removed the need for
a separate `embed.onnx` call.

Reimplements `graphs.SearchGraph.forward`'s own math (gather each gallery row's similarity to its
own frame hypothesis among a query's 14, then subtract that row's penalty) rather than importing
`SearchGraph` itself, because that class expects one query's `(14, 128)` at a time; here every one
of `MAX_CARDS` detected slots needs the same search run against the same fixed gallery in one
pass, batched via `torch.matmul`'s broadcasting instead of a `MAX_CARDS`-times Python loop (both
are equally valid ways to "run search.onnx as many times as needed in one graph" -- broadcasting
was chosen for a smaller exported graph, not because a loop would be wrong).

The gallery itself is fixed at construction time, exactly like `SearchGraph` -- extract one from a
deployed bundle's `search.onnx` via `evaluate_detect_and_embed.load_gallery` (no torch checkpoint
or local art-image cache needed) and pass its three tensors straight in.

    uv run python -m cardid.evaluate_detect_and_embed --detector single-pass ...   # first validate DetectAndEmbed alone
"""

from __future__ import annotations

import torch
from torch import nn

from .detect_and_embed import MAX_CARDS, N_FRAMES, DetectAndEmbed
from .model import EMBED_DIM


class DetectEmbedAndSearch(nn.Module):
    """Wraps a `DetectAndEmbed` (already itself wrapping a frozen detector + frozen `Embedder`)
    with a fixed gallery. `forward` takes native `native_size`-square images and returns
    (indices (N,MAX_CARDS,topk), scores (N,MAX_CARDS,topk), det_scores (N,MAX_CARDS),
    quads (N,MAX_CARDS,4,2)) -- `indices[n,k]` are gallery row indices for detected slot `k`,
    already sorted best-first, exactly what `SearchGraph.forward` would return for that slot's
    own 14 embeddings called separately."""

    def __init__(self, detect_and_embed: DetectAndEmbed, gallery: torch.Tensor, frames: torch.Tensor, penalties: torch.Tensor, topk: int = 5):
        super().__init__()
        self.dae = detect_and_embed
        self.topk = topk
        self.register_buffer("gallery", gallery.to(torch.float32))  # (N_gallery, EMBED_DIM)
        self.register_buffer("frames", frames.to(torch.int64).view(1, -1, 1))  # (1, N_gallery, 1), for gather + broadcast
        self.register_buffer("penalties", penalties.to(torch.float32).view(1, -1))  # (1, N_gallery), for broadcast

    def train(self, mode: bool = True):
        super().train(mode)
        self.dae.eval()  # DetectAndEmbed's own train() already pins its detector/embedder eval; keep it that way here too
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings, det_scores, quads = self.dae(images)  # (n, MAX_CARDS, N_FRAMES, EMBED_DIM), (n, MAX_CARDS), (n, MAX_CARDS, 4, 2)
        n = embeddings.shape[0]
        flat = embeddings.reshape(n * MAX_CARDS, N_FRAMES, EMBED_DIM)
        # (1, N_gallery, EMBED_DIM) @ (B, EMBED_DIM, N_FRAMES) broadcasts to (B, N_gallery, N_FRAMES),
        # same matmul SearchGraph does per-query, batched here over every detected slot at once.
        sims = torch.matmul(self.gallery.unsqueeze(0), flat.transpose(-1, -2))
        own_frame_sims = sims.gather(-1, self.frames.expand(sims.shape[0], -1, -1))[..., 0]  # (B, N_gallery)
        adjusted = own_frame_sims - self.penalties
        top = torch.topk(adjusted, self.topk, dim=-1)
        indices = top.indices.view(n, MAX_CARDS, self.topk)
        scores = top.values.view(n, MAX_CARDS, self.topk)
        return indices, scores, det_scores, quads
