"""Standard launcher for inherited DeepLabV3+ training code."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from atldsd_seg.configs import EXPERIMENTS, get_experiment
from atldsd_seg.paths import DEEPLABV3PLUS_ROOT


def str_bool(value: bool) -> str:
    return "true" if value else "false"


def build_command(experiment_name: str, python_executable: str = sys.executable) -> list[str]:
    cfg = get_experiment(experiment_name)
    if cfg.model != "deeplabv3plus":
        raise ValueError(f"Experiment {experiment_name} is not a DeepLabV3+ experiment")

    cfg.weights_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        str(DEEPLABV3PLUS_ROOT / "train.py"),
        "--cuda",
        "true",
        "--seed",
        str(cfg.seed),
        "--num-classes",
        str(cfg.num_classes),
        "--backbone",
        cfg.backbone,
        "--pretrained",
        str_bool(cfg.pretrained),
        "--downsample-factor",
        "16",
        "--attention-type",
        cfg.attention_type,
        "--use-ppm",
        str_bool(cfg.use_ppm),
        "--input-shape",
        str(cfg.input_size[0]),
        str(cfg.input_size[1]),
        "--init-epoch",
        "0",
        "--freeze-epoch",
        "50",
        "--freeze-batch-size",
        str(cfg.freeze_batch_size),
        "--unfreeze-epoch",
        str(cfg.epochs),
        "--unfreeze-batch-size",
        str(cfg.batch_size),
        "--freeze-train",
        "true",
        "--init-lr",
        str(cfg.init_lr),
        "--optimizer-type",
        "sgd",
        "--lr-decay-type",
        "cos",
        "--save-period",
        "10",
        "--eval-period",
        "10",
        "--dataset-name",
        "ATLDSD",
        "--vocdevkit-path",
        str(cfg.vocdevkit_path),
        "--dice-loss",
        str_bool(cfg.dice_loss),
        "--focal-loss",
        str_bool(cfg.focal_loss),
        "--num-workers",
        "0",
        "--auto-export-report",
        "true",
        "--report-dir",
        str(cfg.report_dir),
        "--report-split",
        "val",
        "--report-fps-interval",
        "100",
        "--save-dir",
        str(cfg.weights_dir),
        "--log-dir",
        str(cfg.logs_dir),
        "--class-names",
        *cfg.class_names,
    ]
    if cfg.model_path is not None:
        command.extend(["--model-path", str(cfg.model_path)])
    command.extend(cfg.extra_args)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a named ATLDSD DeepLabV3+ experiment.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the legacy trainer.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_command(args.experiment, args.python)
    print(" ".join(f'"{item}"' if " " in item else item for item in command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(DEEPLABV3PLUS_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
