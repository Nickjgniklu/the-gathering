"""Train `TrackMemory`'s GRU -- the only trainable part; `DetectAndEmbed` (whichever detector
variant) and the embedder stay frozen, exactly like `train_tiled_fusion.py` trains only its
`FusionHead` on top of a frozen `TableCenterNet`. Reuses `track_memory_dataset.
TrackMemorySequenceDataset` for ground-truth-associated per-track sequences.

    uv run python -m cardid.train_track_memory --manifest-dir ~/the-gathering-cardid/table-sequences --table-checkpoint data/runs/repro-a-hardneg-v4/best.pt --embed-checkpoint data/runs/recogniser-cfbender-oracle/best.pt --gallery-bundle H:\the-gathering-cardid\current --native-size 384 --run track-memory-a --epochs 15

At every real (non-padding) timestep: cross-entropy of `refined_embedding @ gallery.T /
temperature` against the track's true gallery row -- the fixed real gallery is the classification
target matrix directly (mirrors `model.ArcFaceHead`'s "compare against every class" precedent,
without a separate learned weight matrix, since the gallery here is already real and fixed) --
plus a small-weight L1 term keeping `refined_pose` close to the ground-truth pose (`true_pose_t`,
read off the manifest's quad, not the detector's own noisy `pose_t`) so the residual doesn't drift
from correct geometry while chasing the embedding objective. The GRU is unrolled across
`max_frames` per batch, carrying `hidden` forward only across real timesteps (padding neither
advances nor is scored).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import RUNS_DIR
from .track_memory import HIDDEN_DIM, TrackMemory
from .track_memory_dataset import TrackMemorySequenceDataset
from .training_runtime import add_runtime_args, make_loader, setup, write_run_metadata


def step_sequence(model: TrackMemory, embeddings: torch.Tensor, poses: torch.Tensor, scores: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unroll `model` across one batch's `max_frames` timesteps. Returns (refined_embeddings,
    refined_poses), each `(B, max_frames, D)` -- masking/loss is the caller's job, since what
    counts as "real" differs between the embedding loss (every valid timestep) and metrics
    (only the last valid one matters for "final belief" style numbers)."""
    b, t, _ = embeddings.shape
    hidden = torch.zeros(b, HIDDEN_DIM, device=embeddings.device)
    refined_embeddings, refined_poses = [], []
    for i in range(t):
        old_hidden = hidden
        new_hidden, refined_embedding, refined_pose = model(old_hidden, embeddings[:, i], poses[:, i], scores[:, i])
        # Padding timesteps must not advance a track's belief: freeze `hidden` back to what it
        # was for any batch row where this timestep isn't real (a row already past its own
        # sequence length, e.g. one track is 3 frames long, another in the same batch is 8).
        hidden = torch.where(valid[:, i : i + 1], new_hidden, old_hidden)
        refined_embeddings.append(refined_embedding)
        refined_poses.append(refined_pose)
    return torch.stack(refined_embeddings, dim=1), torch.stack(refined_poses, dim=1)


def track_memory_loss(
    refined_embeddings: torch.Tensor,
    refined_poses: torch.Tensor,
    true_poses: torch.Tensor,
    valid: torch.Tensor,
    gallery_index: torch.Tensor,
    gallery: torch.Tensor,
    temperature: float,
    pose_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    b, t, _ = refined_embeddings.shape
    flat_valid = valid.reshape(-1)
    if not bool(flat_valid.any()):
        zero = refined_embeddings.sum() * 0.0
        return zero, {"embed": 0.0, "pose": 0.0}
    flat_embeddings = refined_embeddings.reshape(b * t, -1)[flat_valid]
    flat_targets = gallery_index.unsqueeze(1).expand(-1, t).reshape(-1)[flat_valid]
    logits = flat_embeddings @ gallery.T / temperature
    embed_loss = F.cross_entropy(logits, flat_targets)

    flat_poses = refined_poses.reshape(b * t, -1)[flat_valid]
    flat_true_poses = true_poses.reshape(b * t, -1)[flat_valid]
    pose_loss = F.l1_loss(flat_poses, flat_true_poses)

    return embed_loss + pose_weight * pose_loss, {"embed": float(embed_loss), "pose": float(pose_loss)}


@torch.no_grad()
def evaluate_val(model: TrackMemory, rows, device: torch.device, gallery: torch.Tensor, limit: int) -> dict:
    """Raw (unsmoothed) vs. refined top-1 accuracy, and a flicker-rate metric (fraction of
    within-track frame-to-frame identity changes) for both, at every real timestep -- the same
    same-seed before/after comparison this project reports for every other change."""
    model.eval()
    raw_top1, refined_top1, n = 0, 0, 0
    raw_flickers, refined_flickers, flicker_opportunities = 0, 0, 0
    for i in range(min(limit, len(rows))):
        embeddings, poses, scores, _true_poses, valid, gallery_index = rows[i]
        embeddings, poses, scores, valid = (t.unsqueeze(0).to(device) for t in (embeddings, poses, scores, valid))
        gallery_index = int(gallery_index)
        refined_embeddings, _ = step_sequence(model, embeddings, poses, scores, valid)

        raw_choices, refined_choices = [], []
        for t in range(embeddings.shape[1]):
            if not bool(valid[0, t]):
                continue
            raw_choice = int(torch.argmax(embeddings[0, t] @ gallery.T))
            refined_choice = int(torch.argmax(refined_embeddings[0, t] @ gallery.T))
            raw_top1 += raw_choice == gallery_index
            refined_top1 += refined_choice == gallery_index
            n += 1
            raw_choices.append(raw_choice)
            refined_choices.append(refined_choice)
        for a, b in zip(raw_choices, raw_choices[1:]):
            flicker_opportunities += 1
            raw_flickers += a != b
        for a, b in zip(refined_choices, refined_choices[1:]):
            refined_flickers += a != b
    model.train()
    return {
        "raw_top1": raw_top1 / max(n, 1),
        "refined_top1": refined_top1 / max(n, 1),
        "raw_flicker_rate": raw_flickers / max(flicker_opportunities, 1),
        "refined_flicker_rate": refined_flickers / max(flicker_opportunities, 1),
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True, help="a table_scenes.write_sequence_dataset output directory")
    parser.add_argument("--detector", choices=["single-pass", "tiled-fusion"], default="single-pass")
    parser.add_argument("--table-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, help="tiled-fusion only: the trained FusionHead")
    parser.add_argument("--embed-checkpoint", type=Path, required=True)
    parser.add_argument("--gallery-bundle", type=Path, required=True, help="a deployed bundle directory (arts.json + search.onnx)")
    parser.add_argument("--native-size", type=int, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--run", default="track-memory-a")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1, help="softmax temperature for the gallery cross-entropy (see model.info_nce)")
    parser.add_argument("--pose-weight", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, help="cap training sequences (smoke tests)")
    parser.add_argument("--val-limit", type=int, default=60, help="tracks scored per epoch, not sequences")
    parser.add_argument("--resume", help="warm-start TrackMemory only (weights, not optimizer/schedule/history)")
    add_runtime_args(parser, "dataset-loading worker processes")
    args = parser.parse_args()
    if args.detector == "tiled-fusion" and not args.fusion_checkpoint:
        parser.error("--detector tiled-fusion needs --fusion-checkpoint")
    runtime = setup(args, "loading")
    device = runtime.device

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, args, runtime)

    def build_dataset(split: str, limit: int | None) -> TrackMemorySequenceDataset:
        return TrackMemorySequenceDataset(
            args.manifest_dir / split / "manifest.jsonl",
            table_checkpoint=args.table_checkpoint,
            embed_checkpoint=args.embed_checkpoint,
            gallery_bundle=args.gallery_bundle,
            native_size=args.native_size,
            detector=args.detector,
            fusion_checkpoint=args.fusion_checkpoint,
            score_threshold=args.score_threshold,
            limit=limit,
        )

    train_set = build_dataset("train", args.train_limit)
    val_set = build_dataset("val", None)
    if len(train_set) == 0:
        raise SystemExit(f"no training tracks in {args.manifest_dir} (check it is gallery-verified -- see regen_gallery_verified.py)")
    loader = make_loader(train_set, args.batch, runtime)
    gallery = train_set.gallery.to(device)
    print(f"train: {len(train_set)} tracks, {len(loader)} batches/epoch; val: {len(val_set)} tracks ({min(args.val_limit, len(val_set))} scored/epoch)")

    model = TrackMemory().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"warm-started TrackMemory from {args.resume}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(args.epochs * len(loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        started, losses, parts_sum = time.time(), [], {"embed": 0.0, "pose": 0.0}
        for embeddings, poses, scores, true_poses, valid, gallery_index in loader:
            embeddings, poses, scores, true_poses, valid, gallery_index = (
                t.to(device, non_blocking=True) for t in (embeddings, poses, scores, true_poses, valid, gallery_index)
            )
            refined_embeddings, refined_poses = step_sequence(model, embeddings, poses, scores, valid)
            loss, parts = track_memory_loss(refined_embeddings, refined_poses, true_poses, valid, gallery_index, gallery, args.temperature, args.pose_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
            for k, v in parts.items():
                parts_sum[k] += v
        metrics = evaluate_val(model, val_set, device, gallery, args.val_limit)
        entry = {
            "epoch": epoch,
            "loss": sum(losses) / max(len(losses), 1),
            **{f"loss_{k}": v / max(len(loader), 1) for k, v in parts_sum.items()},
            "seconds": round(time.time() - started, 1),
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, run_dir / "last.pt")
        score = metrics["refined_top1"] - metrics["refined_flicker_rate"]
        if score >= best:
            best = score
            torch.save(state, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best -> {run_dir / 'best.pt'}" if (run_dir / "best.pt").exists() else f"no epoch beat init; last -> {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
