"""Reproduce the full `table-a-*` training lineage from scratch, one phase per warm-started
`cardid.train_table_detector` run, ending in a checkpoint intended to match
`table-a-realclutter-v5`'s real-world performance (recall ~89.8%, precision ~87.4%,
fp_on_negatives_rate ~18.5% on the `real_captures` category of `confusion_matrix.py`).

This intentionally does NOT replay history move-for-move. Two simplifications, both because the
things those historical steps were *for* are now permanent parts of the current codebase rather
than one-off fixes bolted on mid-lineage:

  * No separate data regeneration per phase. Historically the synthetic scene renderer gained
    features between phases (nonstandard frames, hardware-stress augmentation, stacking, clutter/
    round hard negatives, real desk-clutter compositing) and `RENDERER_VERSION` moved v3->v4->v5
    accordingly. All of that is now baked into the *current* `table_scenes.py` and the dataset at
    `manifest_dir` was already generated with it (`renderer_version: table-scenes-v5`), so every
    phase below trains on the same, fully-featured dataset instead of a growing subset of it.

  * No `table-a-combined` phase. That historical run's only purpose was retrofitting the
    `distant_wide` camera profile into an already-trained model after the fix landed mid-session;
    `distant_wide` has been a permanent member of `SPLIT_CAMERA_PROFILES["train"]` ever since, so
    a from-scratch run already sees it from epoch 1 of the very first phase. Skipped here.

One consequence: `HARD_NEG_WEIGHT` (the loss's clutter/round-object upweighting) is now a
permanent module constant rather than something introduced only starting at `table-a-hardneg-v4`,
so phases 4-6 below train on literally identical data and loss code and mostly just spend the
historical epoch budget re-confirming convergence. Kept anyway because the ask was to replicate
the *steps*, and it costs nothing but GPU time to stay faithful to them.

Each phase warm-starts from the previous phase's `best.pt` (score-gated: `recall + precision`),
matching how every historical `table-a-*` run resumed. Idempotent: re-running skips any phase
whose run directory already has a `best.pt` and a `history.json` with at least as many completed
epochs as configured, so a killed/interrupted invocation can just be re-run.

    uv run python -m cardid.reproduce_table_detector run
    uv run python -m cardid.reproduce_table_detector run --from-phase repro-a-hardware-stress
    uv run python -m cardid.reproduce_table_detector evaluate --run repro-a-realclutter-v5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import DATA_DIR

MANIFEST_DIR = Path(r"H:\the-gathering-cardid\table-scenes")


@dataclass
class Phase:
    name: str
    epochs: int
    batch: int
    val_limit: int
    workers: int
    hardware_stress_rate: float | None = None
    extra: list[str] = field(default_factory=list)


# Mirrors each historical run's own `run.json` CLI args (manifest_dir/lr/backbone_lr/pose_weight/
# up_weight/input_size/score_threshold were identical across every phase and are hardcoded into
# `build_command` below). `table-a-combined` is skipped -- see module docstring. No `--pretrained`
# flag: `train_table_detector.py` hardcodes an ImageNet-pretrained backbone unconditionally now
# (an early from-scratch comparison lost decisively; see that module's own docstring), so the
# first phase's `pretrained: True` in its historical run.json reflects that default, not a flag.
PHASES: list[Phase] = [
    Phase("repro-a-pretrained", epochs=120, batch=32, val_limit=150, workers=2),
    Phase("repro-a-nonstandard-finetune", epochs=50, batch=32, val_limit=150, workers=3),
    Phase("repro-a-hardware-stress", epochs=25, batch=8, val_limit=100, workers=1, hardware_stress_rate=0.25),
    Phase("repro-a-realcapture-hardening", epochs=20, batch=16, val_limit=150, workers=1, hardware_stress_rate=0.15),
    Phase("repro-a-hardneg-v4", epochs=20, batch=16, val_limit=150, workers=2, hardware_stress_rate=0.15),
    Phase("repro-a-realclutter-v5", epochs=20, batch=16, val_limit=150, workers=2, hardware_stress_rate=0.15),
]

RUNS_DIR = DATA_DIR / "runs"


def history_epochs(run_dir: Path) -> int:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return 0
    try:
        return len(json.loads(history_path.read_text()))
    except (json.JSONDecodeError, OSError):
        return 0


def phase_done(phase: Phase) -> bool:
    run_dir = RUNS_DIR / phase.name
    return (run_dir / "best.pt").exists() and history_epochs(run_dir) >= phase.epochs


def build_command(phase: Phase, resume: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cardid.train_table_detector",
        "--manifest-dir",
        str(MANIFEST_DIR),
        "--run",
        phase.name,
        "--epochs",
        str(phase.epochs),
        "--batch",
        str(phase.batch),
        "--lr",
        "0.001",
        "--backbone-lr",
        "0.0003",
        "--pose-weight",
        "1.0",
        "--up-weight",
        "1.0",
        "--input-size",
        "384",
        "--val-limit",
        str(phase.val_limit),
        "--score-threshold",
        "0.3",
        "--device",
        "cuda",
        "--workers",
        str(phase.workers),
        "--seed",
        "0",
    ]
    if phase.hardware_stress_rate is not None:
        cmd += ["--hardware-stress-rate", str(phase.hardware_stress_rate)]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    cmd += phase.extra
    return cmd


def run_phase(phase: Phase, resume: Path | None) -> None:
    if phase_done(phase):
        print(f"[skip] {phase.name}: already has best.pt and {history_epochs(RUNS_DIR / phase.name)}/{phase.epochs} epochs")
        return
    cmd = build_command(phase, resume)
    print(f"[run] {phase.name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent)
    if not (RUNS_DIR / phase.name / "best.pt").exists():
        raise SystemExit(f"{phase.name} finished but produced no best.pt")


def cmd_run(args: argparse.Namespace) -> None:
    if not MANIFEST_DIR.exists():
        raise SystemExit(f"expected an already-generated table-scenes dataset at {MANIFEST_DIR}")
    manifest = json.loads((MANIFEST_DIR / "dataset.json").read_text())
    print(f"reusing existing dataset: renderer_version={manifest.get('renderer_version')} generated_at={manifest.get('generated_at')}")

    start = 0
    if args.from_phase:
        names = [p.name for p in PHASES]
        if args.from_phase not in names:
            raise SystemExit(f"unknown phase {args.from_phase!r}; choices: {names}")
        start = names.index(args.from_phase)

    resume: Path | None = None
    if start > 0:
        resume = RUNS_DIR / PHASES[start - 1].name / "best.pt"
        if not resume.exists():
            raise SystemExit(f"--from-phase {args.from_phase} needs {resume} to already exist")

    for phase in PHASES[start:]:
        run_phase(phase, resume)
        resume = RUNS_DIR / phase.name / "best.pt"

    print(f"done. final checkpoint: {resume}")
    if not args.skip_evaluate:
        evaluate(resume, PHASES[-1].name)


def evaluate(checkpoint: Path, run_name: str) -> dict:
    out_path = DATA_DIR / f"confusion-matrix-{run_name}.json"
    cmd = [sys.executable, "-m", "cardid.confusion_matrix", "--checkpoint", str(checkpoint), "--out", str(out_path)]
    print(f"[run] evaluate: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent)
    report = json.loads(out_path.read_text())
    real = report.get("real_captures")
    print(f"real_captures: {json.dumps(real)}")
    print("compare against table-a-realclutter-v5: recall~=0.898 precision~=0.874 fp_on_negatives_rate~=0.185")
    return report


def cmd_evaluate(args: argparse.Namespace) -> None:
    checkpoint = RUNS_DIR / args.run / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    evaluate(checkpoint, args.run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run every phase in order, warm-starting each from the previous, then evaluate")
    p_run.add_argument("--from-phase", help="resume starting at this phase name, warm-starting from the prior phase's best.pt")
    p_run.add_argument("--skip-evaluate", action="store_true")

    p_eval = sub.add_parser("evaluate", help="run confusion_matrix.py against one phase's best.pt and compare to the historical baseline")
    p_eval.add_argument("--run", default=PHASES[-1].name)

    args = parser.parse_args()
    {"run": cmd_run, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
