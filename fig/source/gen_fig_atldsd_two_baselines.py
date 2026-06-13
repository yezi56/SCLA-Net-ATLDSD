from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "outputs" / "atldsd"
FIG_DIR = PROJECT_ROOT / "fig" / "source"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "MobileNetV2": {
        "dir": OUT_ROOT / "deeplabv3plus_mobilenet_150",
        "report": "ep150_val",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "EfficientNet-B4": {
        "dir": OUT_ROOT / "deeplabv3plus_efficientnet_b4_150",
        "report": "ep150_val",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}

CLASS_LABELS = {
    "background": "Background",
    "leaf": "Leaf",
    "rust": "Rust",
    "alternaria_leaf_spot": "Alternaria",
    "gray_spot": "Gray spot",
    "brown_spot": "Brown spot",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
        }
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_per_class(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["class_name"]] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"class_id", "class_name"}
            }
    return rows


def parse_training_log(path: Path) -> dict[str, list[float]]:
    epoch_re = re.compile(r"Epoch:(\d+)/(\d+)")
    loss_re = re.compile(r"Total Loss:\s*([0-9.]+)\s*\|\|\s*Val Loss:\s*([0-9.]+)")
    miou_re = re.compile(r"mIoU:\s*([0-9.]+);\s*mPA:\s*([0-9.]+);\s*Accuracy:\s*([0-9.]+)")

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    eval_epoch: list[int] = []
    miou: list[float] = []
    mpa: list[float] = []
    acc: list[float] = []

    current_epoch: int | None = None
    eval_count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = epoch_re.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        loss_match = loss_re.search(line)
        if loss_match and current_epoch is not None:
            epochs.append(current_epoch)
            train_loss.append(float(loss_match.group(1)))
            val_loss.append(float(loss_match.group(2)))
            continue

        miou_match = miou_re.search(line)
        if miou_match:
            eval_count += 1
            eval_epoch.append(eval_count * 10)
            miou.append(float(miou_match.group(1)))
            mpa.append(float(miou_match.group(2)))
            acc.append(float(miou_match.group(3)))

    return {
        "epoch": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "eval_epoch": eval_epoch,
        "miou": miou,
        "mpa": mpa,
        "acc": acc,
    }


def load_runs() -> dict[str, dict]:
    data: dict[str, dict] = {}
    for name, cfg in RUNS.items():
        run_dir = cfg["dir"]
        report_dir = run_dir / "reports" / cfg["report"]
        data[name] = {
            "cfg": cfg,
            "log": parse_training_log(run_dir / "train_stdout.log"),
            "metrics": read_json(report_dir / "metrics_summary.json"),
            "complexity": read_json(report_dir / "complexity.json"),
            "per_class": read_per_class(report_dir / "per_class_metrics.csv"),
        }
    return data


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7))
    ax_loss, ax_val, ax_miou, ax_acc = axes.ravel()

    for name, item in data.items():
        cfg = item["cfg"]
        log = item["log"]
        color = cfg["color"]
        ax_loss.plot(
            log["epoch"],
            log["train_loss"],
            color=color,
            linewidth=1.7,
            label=name,
            linestyle=cfg["linestyle"],
        )
        ax_val.plot(
            log["epoch"],
            log["val_loss"],
            color=color,
            linewidth=1.7,
            label=name,
            linestyle=cfg["linestyle"],
        )
        ax_miou.plot(
            log["eval_epoch"],
            log["miou"],
            color=color,
            marker=cfg["marker"],
            markersize=4.0,
            linewidth=1.7,
            label=name,
            linestyle=cfg["linestyle"],
        )
        ax_acc.plot(
            log["eval_epoch"],
            log["acc"],
            color=color,
            marker=cfg["marker"],
            markersize=4.0,
            linewidth=1.7,
            label=name,
            linestyle=cfg["linestyle"],
        )

    panels = [
        (ax_loss, "Training loss", "Loss"),
        (ax_val, "Validation loss", "Loss"),
        (ax_miou, "Validation mIoU", "mIoU (%)"),
        (ax_acc, "Pixel accuracy", "Accuracy (%)"),
    ]
    for ax, title, ylabel in panels:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.set_xlim(0, 150)

    ax_loss.legend(frameon=False, loc="upper right")
    save_figure(fig, "fig_atldsd_training_curves")


def plot_final_metric_summary(data: dict[str, dict]) -> None:
    metrics = [
        ("miou_all", "mIoU"),
        ("mdice_all", "mDice"),
        ("pixel_accuracy", "Pixel Acc."),
        ("miou_foreground", "FG mIoU"),
    ]
    names = list(data.keys())
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for i, name in enumerate(names):
        cfg = data[name]["cfg"]
        values = [data[name]["metrics"][key] * 100.0 for key, _ in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.6,
            label=name,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )

    ax.set_title("Final validation performance at epoch 150")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, "fig_atldsd_final_metrics")


def plot_per_class_iou(data: dict[str, dict]) -> None:
    classes = ["leaf", "rust", "alternaria_leaf_spot", "gray_spot", "brown_spot"]
    names = list(data.keys())
    x = np.arange(len(classes))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, name in enumerate(names):
        cfg = data[name]["cfg"]
        values = [data[name]["per_class"][klass]["iou"] * 100.0 for klass in classes]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.6,
            label=name,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )

    ax.set_title("Per-class IoU on ATLDSD validation set")
    ax.set_ylabel("IoU (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, "fig_atldsd_per_class_iou")


def plot_efficiency_tradeoff(data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax_scatter, ax_bar = axes

    for name, item in data.items():
        cfg = item["cfg"]
        params_m = item["complexity"]["params"] / 1e6
        fps = item["complexity"]["fps"]
        miou = item["metrics"]["miou_all"] * 100.0
        ax_scatter.scatter(
            params_m,
            miou,
            s=max(90, min(260, fps * 1.8)),
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            label=f"{name} ({fps:.1f} FPS)",
        )
        ax_scatter.text(
            params_m + 0.45,
            miou,
            name,
            va="center",
            ha="left",
            fontsize=8,
            color="#222222",
        )

    ax_scatter.set_title("Accuracy-efficiency trade-off")
    ax_scatter.set_xlabel("Parameters (M)")
    ax_scatter.set_ylabel("mIoU (%)")
    ax_scatter.grid(True)

    names = list(data.keys())
    fps_values = [data[name]["complexity"]["fps"] for name in names]
    flops_values = [data[name]["complexity"]["flops_estimate_2x_macs"] / 1e9 for name in names]
    x = np.arange(len(names))
    width = 0.36
    fps_bars = ax_bar.bar(
        x - width / 2,
        fps_values,
        width=width,
        color="#009E73",
        edgecolor="white",
        linewidth=0.6,
        label="FPS",
    )
    ax_flops = ax_bar.twinx()
    flops_bars = ax_flops.bar(
        x + width / 2,
        flops_values,
        width=width,
        color="#E69F00",
        edgecolor="white",
        linewidth=0.6,
        label="FLOPs",
    )
    ax_bar.set_title("Inference cost")
    ax_bar.set_ylabel("FPS")
    ax_flops.set_ylabel("FLOPs (G)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=15, ha="right")
    ax_bar.grid(True, axis="y")
    for bars, axis, fmt in [(fps_bars, ax_bar, "{:.1f}"), (flops_bars, ax_flops, "{:.1f}")]:
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(1.2, value * 0.025),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
    handles = [fps_bars, flops_bars]
    labels = ["FPS", "FLOPs"]
    ax_bar.legend(handles, labels, frameon=False, loc="upper right")

    save_figure(fig, "fig_atldsd_efficiency_tradeoff")


def write_summary_table(data: dict[str, dict]) -> None:
    rows = []
    for name, item in data.items():
        metrics = item["metrics"]
        complexity = item["complexity"]
        rows.append(
            {
                "model": name,
                "miou_all": f"{metrics['miou_all'] * 100:.2f}",
                "mdice_all": f"{metrics['mdice_all'] * 100:.2f}",
                "pixel_accuracy": f"{metrics['pixel_accuracy'] * 100:.2f}",
                "miou_foreground": f"{metrics['miou_foreground'] * 100:.2f}",
                "params_m": f"{complexity['params'] / 1e6:.2f}",
                "flops_g": f"{complexity['flops_estimate_2x_macs'] / 1e9:.2f}",
                "fps": f"{complexity['fps']:.2f}",
            }
        )

    out = FIG_DIR / "atldsd_two_baselines_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_paper_style()
    data = load_runs()
    plot_training_curves(data)
    plot_final_metric_summary(data)
    plot_per_class_iou(data)
    plot_efficiency_tradeoff(data)
    write_summary_table(data)


if __name__ == "__main__":
    main()
