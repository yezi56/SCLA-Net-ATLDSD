from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_REPORT_DIR = Path(
    r"D:\Code\all\outputs\riceseg\deeplabv3plus_baseline\report_source"
)

CLASS_LABELS = {
    "background": "Background",
    "leaf": "Leaf",
    "rust": "Rust",
    "alternaria_leaf_spot": "Alternaria",
    "gray_spot": "Gray spot",
    "brown_spot": "Brown spot",
    "Bacterialblight": "Bact.\nblight",
    "Blast": "Blast",
    "Brownspot": "Brown\nspot",
    "Tungro": "Tungro",
}

METRIC_LABELS = {
    "iou": "IoU",
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Recall",
}

PALETTE = {
    "IoU": "#2C6B55",
    "Dice": "#C97B33",
    "Precision": "#2F6F8F",
    "Recall": "#D8A23A",
    "FG mIoU": "#2C6B55",
    "FG mDice": "#6FA58A",
    "FG Precision": "#2F6F8F",
    "FG Recall": "#D8A23A",
    "Background": "#B8B3A7",
    "Leaf": "#6FA58A",
    "Rust": "#C97B33",
    "Alternaria": "#2F6F8F",
    "Gray spot": "#8E7CC3",
    "Brown spot": "#D8A23A",
    "Params": "#2C6B55",
    "MACs": "#C97B33",
    "FLOPs": "#2F6F8F",
    "FPS": "#D8A23A",
    "Bact.\nblight": "#2C6B55",
    "Blast": "#6FA58A",
    "Brown\nspot": "#C97B33",
    "Tungro": "#2F6F8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-ready charts from a segmentation report directory.")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory containing metrics_summary.json, per_class_metrics.csv and complexity.json.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Defaults to a sibling figures directory for report_source, "
            "otherwise <report-dir>/figures."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg", "tiff"],
        help="Export formats for the paper composite figure.",
    )
    parser.add_argument(
        "--paper-prefix",
        default="paper_model_summary",
        help="Filename stem for the paper composite figure.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title. Pass an empty string to omit it.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also export the old separate bar charts. Disabled by default to avoid PNG clutter.",
    )
    parser.add_argument(
        "--include-efficiency",
        action="store_true",
        help="Also export a compact efficiency summary from complexity.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Arial",
        rc={
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.45,
        },
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei", "SimSun"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, out: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_multi(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    out_paths = []
    fig.tight_layout()
    for fmt in formats:
        out = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        out_paths.append(out)
    plt.close(fig)
    return out_paths


def default_out_dir(report_dir: Path) -> Path:
    if report_dir.name.lower() in {"report_source", "best_val"}:
        return report_dir.parent / "figures"
    return report_dir / "figures"


def display_class_names(values: pd.Series) -> pd.Series:
    return values.map(lambda value: CLASS_LABELS.get(value, value))


def annotate_bars(ax, bars, fmt="{:.2f}", offset=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def annotate_axis_bars(ax, fmt="{:.3f}", dy=0.018, fontsize=6.3) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height) or height <= 1e-12:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def annotate_horizontal_bars(ax, fmt="{:.1f}%", dx=1.0, fontsize=6.3) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if np.isnan(width):
            continue
        ax.text(
            width + dx,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(width),
            ha="left",
            va="center",
            fontsize=fontsize,
            color="#222222",
        )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#111111",
    )


def clean_axis(ax) -> None:
    ax.grid(axis="y", alpha=0.75)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)


def plot_fg_summary(ax, metrics: dict) -> None:
    data = pd.DataFrame(
        {
            "Metric": ["FG mIoU", "FG mDice", "FG Precision", "FG Recall"],
            "Score": [
                metrics["miou_foreground"],
                metrics["mdice_foreground"],
                metrics["mprecision_foreground"],
                metrics["mrecall_foreground"],
            ],
        }
    )
    sns.barplot(data=data, x="Metric", y="Score", hue="Metric", palette=PALETTE, legend=False, ax=ax)
    annotate_axis_bars(ax, dy=0.015)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Foreground mean")
    ax.tick_params(axis="x", rotation=18)
    clean_axis(ax)


def plot_per_class_metric_pair(ax, per_class: pd.DataFrame, metrics: list[str], title: str) -> None:
    plot_df = per_class[per_class["class_name"] != "background"].copy()
    plot_df["Class"] = display_class_names(plot_df["class_name"])
    long_df = plot_df.melt(
        id_vars=["Class"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    long_df["Metric"] = long_df["Metric"].map(METRIC_LABELS)
    sns.barplot(
        data=long_df,
        x="Class",
        y="Score",
        hue="Metric",
        palette=PALETTE,
        ax=ax,
        width=0.72,
    )
    annotate_axis_bars(ax, dy=0.012, fontsize=5.8)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=len(metrics), handlelength=1.2, columnspacing=0.9)
    clean_axis(ax)


def plot_class_distribution(ax, per_class: pd.DataFrame) -> None:
    plot_df = per_class.copy()
    total = plot_df["gt_pixels"].sum()
    plot_df["Pixel ratio (%)"] = plot_df["gt_pixels"] / total * 100
    background_ratio = plot_df.loc[plot_df["class_name"] == "background", "Pixel ratio (%)"]
    background_text = (
        f"Background: {background_ratio.iloc[0]:.1f}%"
        if not background_ratio.empty
        else "Background not found"
    )
    plot_df = plot_df[plot_df["class_name"] != "background"].copy()
    plot_df["Class"] = display_class_names(plot_df["class_name"])
    plot_df = plot_df.iloc[::-1]
    sns.barplot(
        data=plot_df,
        y="Class",
        x="Pixel ratio (%)",
        hue="Class",
        palette=PALETTE,
        legend=False,
        ax=ax,
    )
    max_ratio = float(plot_df["Pixel ratio (%)"].max()) if not plot_df.empty else 1.0
    annotate_horizontal_bars(ax, fmt="{:.2f}%", dx=max_ratio * 0.04)
    ax.set_xlim(0, max_ratio * 1.28)
    ax.set_xlabel("Ground-truth pixel ratio (%)")
    ax.set_ylabel("")
    ax.set_title("Foreground pixel distribution")
    ax.text(
        0.98,
        0.92,
        background_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color="#6E6A61",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    ax.grid(axis="x", alpha=0.75)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)


def plot_paper_summary(metrics: dict, per_class: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=False)
    if args.title:
        fig.suptitle(args.title, fontsize=9.5, fontweight="semibold", y=1.015)

    plot_fg_summary(axes[0, 0], metrics)
    panel_label(axes[0, 0], "A")

    plot_per_class_metric_pair(axes[0, 1], per_class, ["iou", "dice"], "Per-class overlap")
    panel_label(axes[0, 1], "B")

    plot_per_class_metric_pair(axes[1, 0], per_class, ["precision", "recall"], "Error balance")
    panel_label(axes[1, 0], "C")

    plot_class_distribution(axes[1, 1], per_class)
    panel_label(axes[1, 1], "D")

    return save_multi(fig, out_dir, args.paper_prefix, args.formats, args.dpi)


def plot_efficiency_summary(complexity: dict, out_dir: Path, dpi: int, formats: list[str]) -> list[Path]:
    data = pd.DataFrame(
        {
            "Metric": ["Params", "MACs", "FLOPs", "FPS"],
            "Value": [
                complexity["params"] / 1e6,
                complexity["macs"] / 1e9 if complexity.get("macs") is not None else 0,
                complexity["flops_estimate_2x_macs"] / 1e9
                if complexity.get("flops_estimate_2x_macs") is not None
                else 0,
                complexity["fps"],
            ],
            "Unit": ["M", "G", "G", "img/s"],
        }
    )

    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    sns.barplot(data=data, x="Metric", y="Value", hue="Metric", palette=PALETTE, legend=False, ax=ax)
    for patch, (_, row) in zip(ax.patches, data.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + data["Value"].max() * 0.025,
            f"{row['Value']:.2f} {row['Unit']}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    ax.set_title("Model efficiency")
    clean_axis(ax)
    return save_multi(fig, out_dir, "paper_efficiency_summary", formats, dpi)


def plot_overall_metrics(metrics: dict, out_dir: Path, dpi: int) -> None:
    names = ["mIoU", "mDice", "mPrecision", "mRecall", "Pixel Acc."]
    values = [
        metrics["miou_all"],
        metrics["mdice_all"],
        metrics["mprecision_all"],
        metrics["mrecall_all"],
        metrics["pixel_accuracy"],
    ]
    colors = ["#2f7d57", "#5a9c76", "#c9792b", "#d9a441", "#286b8f"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, values, color=colors, width=0.64)
    annotate_bars(ax, bars, fmt="{:.3f}", offset=0.008)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Overall Validation Metrics")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    save(fig, out_dir / "overall_metrics_bar.png", dpi)


def plot_foreground_metrics(metrics: dict, out_dir: Path, dpi: int) -> None:
    names = ["FG mIoU", "FG mDice", "FG Precision", "FG Recall"]
    values = [
        metrics["miou_foreground"],
        metrics["mdice_foreground"],
        metrics["mprecision_foreground"],
        metrics["mrecall_foreground"],
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(names, values, color=["#2f7d57", "#5a9c76", "#c9792b", "#d9a441"], width=0.58)
    annotate_bars(ax, bars, fmt="{:.3f}", offset=0.008)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Foreground-Class Average Metrics")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    save(fig, out_dir / "foreground_metrics_bar.png", dpi)


def plot_per_class_iou_dice(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    plot_df = df[df["class_name"] != "background"].copy()
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars1 = ax.bar(x - width / 2, plot_df["iou"], width, label="IoU", color="#2f7d57")
    bars2 = ax.bar(x + width / 2, plot_df["dice"], width, label="Dice", color="#c9792b")
    annotate_bars(ax, bars1, fmt="{:.3f}", offset=0.006)
    annotate_bars(ax, bars2, fmt="{:.3f}", offset=0.006)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["class_name"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class IoU and Dice")
    ax.legend()
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    save(fig, out_dir / "per_class_iou_dice_bar.png", dpi)


def plot_precision_recall(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    plot_df = df[df["class_name"] != "background"].copy()
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars1 = ax.bar(x - width / 2, plot_df["precision"], width, label="Precision", color="#286b8f")
    bars2 = ax.bar(x + width / 2, plot_df["recall"], width, label="Recall", color="#d9a441")
    annotate_bars(ax, bars1, fmt="{:.3f}", offset=0.006)
    annotate_bars(ax, bars2, fmt="{:.3f}", offset=0.006)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["class_name"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision and Recall")
    ax.legend()
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    save(fig, out_dir / "per_class_precision_recall_bar.png", dpi)


def plot_class_pixel_ratio(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    plot_df = df.copy()
    total = plot_df["gt_pixels"].sum()
    plot_df["gt_ratio"] = plot_df["gt_pixels"] / total
    colors = ["#b6b0a3", "#2f7d57", "#5a9c76", "#c9792b", "#286b8f"]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(plot_df["class_name"], plot_df["gt_ratio"], color=colors, width=0.62)
    annotate_bars(ax, bars, fmt="{:.3f}", offset=0.004)
    ax.set_ylabel("Ground-truth pixel ratio")
    ax.set_title("Class Pixel Distribution on Validation Set")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["class_name"], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    save(fig, out_dir / "class_pixel_distribution_bar.png", dpi)


def plot_complexity(complexity: dict, out_dir: Path, dpi: int) -> None:
    names = ["Params (M)", "MACs (G)", "FLOPs (G)", "FPS"]
    values = [
        complexity["params"] / 1e6,
        complexity["macs"] / 1e9 if complexity.get("macs") is not None else 0,
        complexity["flops_estimate_2x_macs"] / 1e9 if complexity.get("flops_estimate_2x_macs") is not None else 0,
        complexity["fps"],
    ]
    colors = ["#2f7d57", "#c9792b", "#286b8f", "#d9a441"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, values, color=colors, width=0.58)
    annotate_bars(ax, bars, fmt="{:.2f}", offset=max(values) * 0.012)
    ax.set_ylabel("Value")
    ax.set_title("Model Complexity and Inference Speed")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    save(fig, out_dir / "complexity_bar.png", dpi)


def main() -> None:
    args = parse_args()
    setup_style()

    report_dir = Path(args.report_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(report_dir / "metrics_summary.json")
    per_class = pd.read_csv(report_dir / "per_class_metrics.csv")

    saved_paths = plot_paper_summary(metrics, per_class, out_dir, args)

    if args.include_efficiency:
        complexity = load_json(report_dir / "complexity.json")
        saved_paths.extend(plot_efficiency_summary(complexity, out_dir, args.dpi, args.formats))

    if args.legacy:
        complexity = load_json(report_dir / "complexity.json")
        plot_overall_metrics(metrics, out_dir, args.dpi)
        plot_foreground_metrics(metrics, out_dir, args.dpi)
        plot_per_class_iou_dice(per_class, out_dir, args.dpi)
        plot_precision_recall(per_class, out_dir, args.dpi)
        plot_class_pixel_ratio(per_class, out_dir, args.dpi)
        plot_complexity(complexity, out_dir, args.dpi)
        saved_paths.extend(sorted(out_dir.glob("*_bar.png")))

    print(f"Saved figures to: {out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
