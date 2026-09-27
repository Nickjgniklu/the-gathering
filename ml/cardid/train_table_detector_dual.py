"""Train `TableCenterNetDual`, the background+current dual-input table detector, on freshly
rendered (background, current) pairs (see `table_scene_pair_dataset.py` -- there is no pre-
rendered manifest for this data yet, unlike `train_table_detector.py`).

    uv run python -m cardid.train_table_detector_dual --run bgsub-a --epochs 40 --device cuda

Warm-starts the backbone from a single-input `TableCenterNet` checkpoint with `--resume-single`:
the first conv is inflated (see `TableCenterNetDual`) and every other layer loads unchanged, so
training does not have to relearn card detection from scratch, only background suppression.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import RUNS_DIR
from .data import to_tensor
from .image_bank import ArtBank, CardBank
from .table_detector import TABLE_INPUT, TABLE_STRIDE, decode_detections, table_detector_loss
from .table_detector_dual import TableCenterNetDual
from .table_scene_pair_dataset import TableScenePairDataset
from .table_scenes import SETUPS, render_table_scene_pair
from .training_runtime import add_runtime_args, make_loader, setup, write_run_metadata


def load_single_input_checkpoint(model: TableCenterNetDual, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    old_weight = state.pop("stem.0.0.weight")
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        model.stem[0][0].weight.copy_(torch.cat([old_weight, old_weight], dim=1) * 0.5)


@torch.no_grad()
def evaluate_val(model: TableCenterNetDual, cards: CardBank, arts: ArtBank, device: torch.device, seed: int, scenes: int, score_threshold: float = 0.3) -> dict:
    from .evaluate_tables import per_card_hits

    model.eval()
    hits_total, truth_total, found_total = 0, 0, 0
    for i in range(scenes):
        setup_name = SETUPS[i % len(SETUPS)]
        count = min(10, len(cards))
        bg, cur, record = render_table_scene_pair(seed + i, cards, arts, setup_name, count=count, size=1280, out=TABLE_INPUT)
        bg_t = to_tensor(bg).unsqueeze(0).to(device)
        cur_t = to_tensor(cur).unsqueeze(0).to(device)
        heat, pose, up = model(bg_t, cur_t)
        detections = decode_detections(heat[0], pose[0], up[0], TABLE_STRIDE, score_threshold)
        found = [q for q, _s in detections]
        truth = [np.float32(c["quad"]) for c in record["cards"]]
        hits = per_card_hits(truth, found, 0.5)
        hits_total += sum(hits)
        truth_total += len(hits)
        found_total += len(found)
    model.train()
    return {
        "recall": hits_total / truth_total if truth_total else 0.0,
        "precision": hits_total / found_total if found_total else 0.0,
        "n_truth": truth_total,
        "n_found": found_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="bgsub-a")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=750, help="rendered pairs per epoch (no fixed dataset size; on-the-fly)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--up-weight", type=float, default=1.0)
    parser.add_argument(
        "--hard-neg-weight", type=float, default=1.5, help="see table_detector.HARD_NEG_WEIGHT; lower than the 3.0 that cost broad single-input recall"
    )
    parser.add_argument("--val-scenes", type=int, default=60)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--resume-single", help="warm-start from a single-input TableCenterNet checkpoint (see load_single_input_checkpoint)")
    add_runtime_args(parser, "rendering")
    args = parser.parse_args()
    runtime = setup(args, "rendering")
    device = runtime.device

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, args, runtime)

    cards, arts = CardBank(), ArtBank()
    train_set = TableScenePairDataset(args.steps_per_epoch * args.batch, cards, arts, seed=runtime.seed)
    loader = make_loader(train_set, args.batch, runtime)
    print(f"train: {args.steps_per_epoch} batches/epoch (on-the-fly); val: {args.val_scenes} fresh scenes/epoch")

    model = TableCenterNetDual(pretrained=True).to(device)
    if args.resume_single:
        load_single_input_checkpoint(model, args.resume_single)
        print(f"warm-started from single-input checkpoint {args.resume_single} (first conv inflated 3->6 channels)")
    opt = torch.optim.AdamW(
        [{"params": model.backbone_parameters(), "lr": args.backbone_lr}, {"params": model.head_parameters(), "lr": args.lr}],
        weight_decay=1e-4,
    )
    steps = max(args.epochs * args.steps_per_epoch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.lr], total_steps=steps, pct_start=0.1)

    history, best = [], -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        started, losses, parts_sum = time.time(), [], {"heat": 0.0, "pose": 0.0, "up": 0.0}
        for bg, cur, heat_t, pose_t, up_t, mask, hard_neg in loader:
            bg, cur, heat_t, pose_t, up_t, mask, hard_neg = (t.to(device, non_blocking=True) for t in (bg, cur, heat_t, pose_t, up_t, mask, hard_neg))
            heat, pose, up = model(bg, cur)
            loss, parts = table_detector_loss(heat, pose, up, heat_t, pose_t, up_t, mask, args.pose_weight, args.up_weight, hard_neg, args.hard_neg_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            for k, v in parts.items():
                parts_sum[k] += v
        metrics = evaluate_val(model, cards, arts, device, seed=10_000_000, scenes=args.val_scenes, score_threshold=args.score_threshold)
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{f"loss_{k}": v / len(loader) for k, v in parts_sum.items()},
            "seconds": round(time.time() - started, 1),
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry))
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, run_dir / "last.pt")
        score = metrics["recall"] + metrics["precision"]
        if score >= best:
            best = score
            torch.save(state, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best -> {run_dir / 'best.pt'}" if (run_dir / "best.pt").exists() else f"no epoch beat init; last -> {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
