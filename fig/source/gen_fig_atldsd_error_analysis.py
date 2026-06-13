"""Generate paper-ready error analysis for the ATLDSD final mainline.

Inputs are the existing LesionDice2 seed23 report files:
metrics_summary.json, per_class_metrics.csv, and confusion_matrix.csv.
The script does not train, evaluate, or run inference.

Usage:
python fig/source/gen_fig_atldsd_error_analysis.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "atldsd_fast"
    / "long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23"
    / "reports"
    / "best_miou"
)
OUT_DIR = PROJECT_ROOT / "fig" / "results" / "paper_error_analysis"

CLASS_LABELS = {
    "background": "Background",
    "leaf": "Leaf",
    "rust": "Rust",
    "alternaria_leaf_spot": "Alternaria",
    "gray_spot": "Gray spot",
    "brown_spot": "Brown spot",
}
LESION_CLASSES = ["rust", "alternaria_leaf_spot", "gray_spot", "brown_spot"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    lines = [
        "| " + " | ".join(field.replace("_", " ") for field in fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def per_class_by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out = {}
    for row in rows:
        name = row["class_name"]
        out[name] = {
            "iou": float(row["iou"]),
            "dice": float(row["dice"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "gt_pixels": float(row["gt_pixels"]),
            "pred_pixels": float(row["pred_pixels"]),
            "true_positive_pixels": float(row["true_positive_pixels"]),
        }
    return out


def confusion_to_matrix(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray]:
    class_names = [row["class_name"] for row in rows]
    matrix = []
    for row in rows:
        matrix.append([int(row[f"pred_{name}"]) for name in class_names])
    return class_names, np.asarray(matrix, dtype=np.float64)


def dominant_offdiag(class_names: list[str], matrix: np.ndarray, class_name: str) -> tuple[str, float]:
    row_idx = class_names.index(class_name)
    row = matrix[row_idx]
    total = row.sum()
    if total <= 0:
        return "none", 0.0
    offdiag = row.copy()
    offdiag[row_idx] = 0
    pred_idx = int(np.argmax(offdiag))
    return class_names[pred_idx], float(offdiag[pred_idx] / total)


def build_error_rows(per_class: dict[str, dict[str, float]], class_names: list[str], matrix: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for class_name in LESION_CLASSES:
        metrics = per_class[class_name]
        pred_gt_ratio = metrics["pred_pixels"] / metrics["gt_pixels"] if metrics["gt_pixels"] else 0.0
        dominant_name, dominant_rate = dominant_offdiag(class_names, matrix, class_name)
        rows.append(
            {
                "class_name": class_name,
                "label": CLASS_LABELS[class_name],
                "iou": f"{metrics['iou'] * 100:.2f}",
                "dice": f"{metrics['dice'] * 100:.2f}",
                "precision": f"{metrics['precision'] * 100:.2f}",
                "recall": f"{metrics['recall'] * 100:.2f}",
                "pred_gt_ratio": f"{pred_gt_ratio:.2f}",
                "dominant_error_target": dominant_name,
                "dominant_error_rate": f"{dominant_rate * 100:.2f}",
                "gt_pixels": int(metrics["gt_pixels"]),
                "pred_pixels": int(metrics["pred_pixels"]),
            }
        )
    return rows


def plot_precision_recall_bias(error_rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in error_rows]
    precision = [float(str(row["precision"])) for row in error_rows]
    recall = [float(str(row["recall"])) for row in error_rows]
    pred_gt = [float(str(row["pred_gt_ratio"])) for row in error_rows]
    x = np.arange(len(labels))
    width = 0.26

    fig, ax1 = plt.subplots(figsize=(7.0, 3.8))
    ax1.bar(x - width / 2, precision, width, color="#2F6F8F", label="Precision")
    ax1.bar(x + width / 2, recall, width, color="#D8A23A", label="Recall")
    ax1.set_ylabel("Precision / Recall (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="x", visible=False)

    ax2 = ax1.twinx()
    ax2.plot(x, pred_gt, color="#B24C63", marker="o", linewidth=2.0, label="Pred/GT pixels")
    ax2.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Predicted / GT pixels")
    ax2.set_ylim(0, max(1.45, max(pred_gt) + 0.15))
    ax2.spines["top"].set_visible(False)

    for idx, ratio in enumerate(pred_gt):
        ax2.text(idx, ratio + 0.03, f"{ratio:.2f}x", ha="center", fontsize=7.5, color="#6D2236")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax1.set_title("Lesion-class error balance in final LesionDice2 seed23")
    save_figure(fig, "fig_paper_error_balance")


def plot_confusion_heatmap(class_names: list[str], matrix: np.ndarray) -> None:
    selected = ["leaf", *LESION_CLASSES]
    idxs = [class_names.index(name) for name in selected]
    sub = matrix[np.ix_(idxs, idxs)]
    row_sums = sub.sum(axis=1, keepdims=True)
    norm = np.divide(sub, row_sums, out=np.zeros_like(sub), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=100)
    labels = [CLASS_LABELS[name] for name in selected]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Row-normalized foreground confusion (%)")
    ax.grid(False)

    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            value = norm[i, j]
            color = "white" if value > 55 else "#1A1A1A"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("% of GT pixels", rotation=270, labelpad=12)
    save_figure(fig, "fig_paper_foreground_confusion_heatmap")


def write_discussion(metrics: dict, error_rows: list[dict[str, object]], class_names: list[str], matrix: np.ndarray) -> None:
    rows_by_class = {str(row["class_name"]): row for row in error_rows}
    alt_to_leaf = dominant_offdiag(class_names, matrix, "alternaria_leaf_spot")
    brown_to_leaf = dominant_offdiag(class_names, matrix, "brown_spot")
    gray_error = dominant_offdiag(class_names, matrix, "gray_spot")
    lines = [
        "# ATLDSD final error analysis notes",
        "",
        "Source: LesionDice2 full/e80 seed23 final report. This is analysis only; no training or inference was run.",
        "",
        "## Summary",
        f"- Overall mIoU: {metrics['miou_all'] * 100:.2f}; FG mIoU: {metrics['miou_foreground'] * 100:.2f}.",
        f"- Rust is the most stable lesion class: IoU {rows_by_class['rust']['iou']}, recall {rows_by_class['rust']['recall']}.",
        f"- Alternaria remains recall-limited: IoU {rows_by_class['alternaria_leaf_spot']['iou']}, recall {rows_by_class['alternaria_leaf_spot']['recall']}; dominant confusion is `{alt_to_leaf[0]}` at {alt_to_leaf[1] * 100:.2f}% of GT pixels.",
        f"- Brown spot has the largest prediction-area bias: pred/GT {rows_by_class['brown_spot']['pred_gt_ratio']}x, precision {rows_by_class['brown_spot']['precision']}, recall {rows_by_class['brown_spot']['recall']}; dominant confusion is `{brown_to_leaf[0]}` at {brown_to_leaf[1] * 100:.2f}%.",
        f"- Gray spot is recall-friendly but precision-limited: precision {rows_by_class['gray_spot']['precision']}, recall {rows_by_class['gray_spot']['recall']}; dominant off-diagonal target is `{gray_error[0]}` at {gray_error[1] * 100:.2f}%.",
        "",
        "## Paper wording",
        "The final model improves lesion segmentation overall, but the residual errors are class-specific. Alternaria and brown spot still suffer from small-lesion missed detections, while gray spot tends to over-expand into visually similar local texture. This supports future lightweight refinements that target lesion recall/precision balance instead of adding generic attention stacks.",
        "",
        "## Next experiment hint",
        "- Do not start from a detection-style branch.",
        "- If opening a new run, begin with 128/64 seed11 and use LesionDice2 as the reference.",
        "- Candidate directions should target alternaria/brown missed small lesions and gray over-prediction with a light boundary/context gate or conservative class-weight/Dice tuning.",
    ]
    (OUT_DIR / "paper_error_analysis_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    metrics = read_json(REPORT_DIR / "metrics_summary.json")
    per_class = per_class_by_name(read_csv(REPORT_DIR / "per_class_metrics.csv"))
    class_names, matrix = confusion_to_matrix(read_csv(REPORT_DIR / "confusion_matrix.csv"))
    error_rows = build_error_rows(per_class, class_names, matrix)

    fields = [
        "class_name",
        "label",
        "iou",
        "dice",
        "precision",
        "recall",
        "pred_gt_ratio",
        "dominant_error_target",
        "dominant_error_rate",
        "gt_pixels",
        "pred_pixels",
    ]
    write_csv(OUT_DIR / "paper_error_analysis_metrics.csv", error_rows, fields)
    write_markdown(OUT_DIR / "paper_error_analysis_metrics.md", error_rows, fields)
    plot_precision_recall_bias(error_rows)
    plot_confusion_heatmap(class_names, matrix)
    write_discussion(metrics, error_rows, class_names, matrix)

    print("Generated ATLDSD error analysis pack:")
    for name in [
        "paper_error_analysis_metrics.csv",
        "paper_error_analysis_metrics.md",
        "paper_error_analysis_notes.md",
        "fig_paper_error_balance.png",
        "fig_paper_error_balance.pdf",
        "fig_paper_foreground_confusion_heatmap.png",
        "fig_paper_foreground_confusion_heatmap.pdf",
    ]:
        print(f"- {OUT_DIR / name}")


if __name__ == "__main__":
    main()
