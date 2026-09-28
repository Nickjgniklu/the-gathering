"""Train `TiledFusionDetector`'s `FusionHead` -- the only trainable part; the wrapped
`TableCenterNet` is loaded from `--checkpoint` and frozen (see `tiled_fusion.py`'s module
docstring). Reuses the same pre-rendered `table_scenes` manifest `train_table_detector.py` does,
just without downsampling to `TABLE_INPUT` first (`tiled_fusion_dataset.py`).

    uv run python -m cardid.train_tiled_fusion --manifest-dir ~/the-gathering-cardid/table-scenes --checkpoint data/runs/<run>/best.pt --run tiled-fusion-a --epochs 15
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
from .table_detector import decode_detections, table_detector_loss
from .tiled_fusion import CANONICAL_STRIDE, NATIVE_SIZE, TiledFusionDetector
from .tiled_fusion_dataset import TiledFusionDataset
from .training_runtime import add_runtime_args, make_loader, setup, write_run_metadata


@torch.no_grad()
def evaluate_val(model: TiledFusionDetector, rows: list[dict], root: Path, device: torch.device, score_threshold: float, limit: int) -> dict:
    model.eval()
    hits_total, truth_total, found_total = 0, 0, 0
    for row in rows[:limit]:
        image = cv2.cvtColor(cv2.imread(str(root / row["image"])), cv2.COLOR_BGR2RGB)
        scale = NATIVE_SIZE / row["width"]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (NATIVE_SIZE, NATIVE_SIZE), interpolation=interp)
        x = to_tensor(resized).unsqueeze(0).to(device)
        heat, pose, up = model(x)
        detections = decode_detections(heat[0], pose[0], up[0], CANONICAL_STRIDE, score_threshold)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen TableCenterNet weights each tile pass uses")
    parser.add_argument("--run", default="tiled-fusion-a")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--up-weight", type=float, default=1.0)
    parser.add_argument("--train-limit", type=int, help="cap training scenes per epoch (smoke tests)")
    parser.add_argument("--val-limit", type=int, default=60)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--resume", help="warm-start the fusion head only (weights, not optimizer/schedule/history)")
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
    train_set = TiledFusionDataset(train_rows, args.manifest_dir / "train")
    loader = make_loader(train_set, args.batch, runtime)
    print(f"train: {len(train_rows)} scenes, {len(loader)} batches/epoch; val: {len(val_rows)} scenes ({args.val_limit} scored/epoch)")

    model = TiledFusionDetector(checkpoint=args.checkpoint).to(device)
    if args.resume:
        model.fusion.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"warm-started fusion head from {args.resume}")
    opt = torch.optim.AdamW(model.fusion.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(args.epochs * len(loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

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
        metrics = evaluate_val(model, val_rows, args.manifest_dir / "val", device, args.score_threshold, args.val_limit)
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{f"loss_{k}": v / len(loader) for k, v in parts_sum.items()},
            "seconds": round(time.time() - started, 1),
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        fusion_state = {k: v.detach().cpu() for k, v in model.fusion.state_dict().items()}
        torch.save(fusion_state, run_dir / "last.pt")
        score = metrics["recall"] + metrics["precision"]
        if score >= best:
            best = score
            torch.save(fusion_state, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best -> {run_dir / 'best.pt'}" if (run_dir / "best.pt").exists() else f"no epoch beat init; last -> {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
