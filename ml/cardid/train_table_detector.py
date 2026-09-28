"""Train `TableCenterNet`, the dense multi-card detector, on a pre-rendered table-scene dataset.

Unlike `train_detector.py`, scenes are not rendered on the fly: `table_scenes.write_dataset`
already wrote JPEGs and manifests to `--manifest-dir` (typically on a separate data drive; see
`ml/README.md`). Each epoch reports the training loss and, on a subset of `val`, recall/precision
at IoU 0.5 (`evaluate_tables.per_card_hits`):

    uv run python -m cardid.train_table_detector --manifest-dir ~/the-gathering-cardid/table-scenes --run table-a --epochs 80

The backbone always starts from ImageNet-pretrained MobileNetV3 weights: an early from-scratch
(random-init backbone) comparison run reached only 78% recall / 99.5% precision after 63 epochs,
versus the pretrained backbone's 95.8% recall / 99.9% precision converged by epoch 120 -- clearly
and consistently the better starting point on this dataset size, so training a from-scratch
variant is no longer supported here.

`--resume` is a warm start, not a full resume: it loads the checkpoint's model weights only.
The optimizer, the `OneCycleLR` schedule, and the epoch count all start over from `--epochs`
worth of fresh warmup, and `history.json`/`best.pt` in `--run`'s directory start over too
(`best` resets to unset, so the first post-resume epoch is always written as the new best).
This is deliberate for quick continuations (e.g. moving a checkpoint from a slower device to a
faster one, as this project's CPU-to-GPU runs did) but means a resumed run's loss/LR curve is
not a continuation of the original run's, and its history.json should not be concatenated with
the original's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import RUNS_DIR
from .data import to_tensor
from .evaluate_tables import load_scenes, per_card_hits
from .table_detector import TABLE_INPUT, TABLE_STRIDE, TableCenterNet, decode_detections, table_detector_loss
from .table_scene_dataset import TableSceneDetectionDataset
from .training_runtime import add_runtime_args, make_loader, setup, write_run_metadata


@torch.no_grad()
def evaluate_val(
    model: TableCenterNet, rows: list[dict], root: Path, device: torch.device, input_size: int, stride: int, score_threshold: float = 0.3, limit: int = 60
) -> dict:
    model.eval()
    hits_total, truth_total, found_total = 0, 0, 0
    for row in rows[:limit]:
        image = cv2.cvtColor(cv2.imread(str(root / row["image"])), cv2.COLOR_BGR2RGB)
        scale = input_size / row["width"]
        resized = cv2.resize(image, (input_size, input_size))
        x = to_tensor(resized).unsqueeze(0).to(device)
        heat, pose, up = model(x)
        detections = decode_detections(heat[0], pose[0], up[0], stride, score_threshold)
        found = [quad / scale for quad, _score in detections]  # back to manifest pixel space
        truth = [np.float32(c["quad"]) for c in row["cards"]]
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
    parser.add_argument("--manifest-dir", type=Path, required=True, help="table-scenes output directory (contains train/, val/)")
    parser.add_argument("--run", default="table-a")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--up-weight", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=TABLE_INPUT)
    parser.add_argument("--train-limit", type=int, help="cap training scenes per epoch (smoke tests)")
    parser.add_argument("--val-limit", type=int, default=60, help="val scenes scored per epoch")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--hardware-stress-rate",
        type=float,
        default=0.0,
        help="fraction of training scenes augmented with a bad-webcam degradation (dark room or a stacked combo); see hardware_stress.py",
    )
    parser.add_argument(
        "--resume", help="warm-start from a checkpoint's model weights only; optimizer/schedule/epoch/history all restart (see module docstring)"
    )
    add_runtime_args(parser, "dataset-loading worker processes")
    args = parser.parse_args()
    runtime = setup(args, "loading")
    device = runtime.device

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, args, runtime)

    train_rows = load_scenes(args.manifest_dir / "train" / "manifest.jsonl", "train")
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    val_rows = load_scenes(args.manifest_dir / "val" / "manifest.jsonl", "val")
    if not train_rows:
        raise SystemExit(f"no train scenes in {args.manifest_dir}")
    train_set = TableSceneDetectionDataset(train_rows, args.manifest_dir / "train", args.input_size, TABLE_STRIDE, args.hardware_stress_rate)
    loader = make_loader(train_set, args.batch, runtime)
    print(f"train: {len(train_rows)} scenes, {len(loader)} batches/epoch; val: {len(val_rows)} scenes ({args.val_limit} scored/epoch)")

    model = TableCenterNet(pretrained=True).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"warm-started from {args.resume} (weights only; optimizer/schedule/epoch/history all restart, see --help)")
    opt = torch.optim.AdamW(
        [{"params": model.backbone_parameters(), "lr": args.backbone_lr}, {"params": model.head_parameters(), "lr": args.lr}],
        weight_decay=1e-4,
    )
    steps = max(args.epochs * len(loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.lr], total_steps=steps, pct_start=0.1)

    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        started, losses, parts_sum = time.time(), [], {"heat": 0.0, "pose": 0.0, "up": 0.0}
        for x, heat_t, pose_t, up_t, mask, hard_neg in loader:
            x, heat_t, pose_t, up_t, mask, hard_neg = (t.to(device, non_blocking=True) for t in (x, heat_t, pose_t, up_t, mask, hard_neg))
            heat, pose, up = model(x)
            loss, parts = table_detector_loss(heat, pose, up, heat_t, pose_t, up_t, mask, args.pose_weight, args.up_weight, hard_neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            for k, v in parts.items():
                parts_sum[k] += v
        metrics = evaluate_val(model, val_rows, args.manifest_dir / "val", device, args.input_size, TABLE_STRIDE, args.score_threshold, args.val_limit)
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
