"""Generate qualitative ATLDSD segmentation success/failure panels.

The script reads the final LesionDice2 seed23 report and VOC-style images.
It does not run inference. It uses existing pred_masks from the report.

Usage:
python fig/source/gen_fig_atldsd_qualitative_cases.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "atldsd_fast"
    / "long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23"
    / "reports"
    / "best_miou"
)
SUMMARY_DIR = PROJECT_ROOT / "fig" / "results"
OUT_DIR = SUMMARY_DIR / "paper_qualitative_cases"

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (82, 160, 92),
    2: (211, 109, 42),
    3: (125, 90, 180),
    4: (88, 133, 166),
    5: (128, 83, 51),
}
ERROR_COLORS = {
    "tp": (87, 171, 90),
    "fn": (215, 69, 69),
    "fp": (33, 151, 188),
}


@dataclass(frozen=True)
class CaseMetric:
    class_id: int
    class_name: str
    image_id: str
    iou: float
    dice: float
    precision: float
    recall: float
    gt_pixels: int
    pred_pixels: int
    tp_pixels: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_image_ids(vocdevkit_path: Path, split: str) -> list[str]:
    path = vocdevkit_path / "VOC2007" / "ImageSets" / "Segmentation" / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def class_metrics(gt: np.ndarray, pred: np.ndarray, class_id: int, class_name: str, image_id: str) -> CaseMetric:
    gt_mask = gt == class_id
    pred_mask = pred == class_id
    tp = int(np.logical_and(gt_mask, pred_mask).sum())
    gt_pixels = int(gt_mask.sum())
    pred_pixels = int(pred_mask.sum())
    fp = pred_pixels - tp
    fn = gt_pixels - tp
    iou = safe_div(tp, gt_pixels + pred_pixels - tp)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, pred_pixels)
    recall = safe_div(tp, gt_pixels)
    return CaseMetric(
        class_id=class_id,
        class_name=class_name,
        image_id=image_id,
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        gt_pixels=gt_pixels,
        pred_pixels=pred_pixels,
        tp_pixels=tp,
    )


def colorize_mask(mask: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    if alpha is None:
        return rgb
    rgb[~alpha] = 245
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = colorize_mask(mask)
    active = mask > 0
    out = image.astype(np.float32).copy()
    out[active] = (1.0 - alpha) * out[active] + alpha * colored[active].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def error_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, class_id: int) -> np.ndarray:
    out = (image.astype(np.float32) * 0.58 + 255 * 0.42).astype(np.uint8)
    gt_mask = gt == class_id
    pred_mask = pred == class_id
    tp = np.logical_and(gt_mask, pred_mask)
    fn = np.logical_and(gt_mask, ~pred_mask)
    fp = np.logical_and(~gt_mask, pred_mask)
    for key, mask in [("tp", tp), ("fn", fn), ("fp", fp)]:
        color = np.array(ERROR_COLORS[key], dtype=np.float32)
        out[mask] = (0.25 * out[mask].astype(np.float32) + 0.75 * color).astype(np.uint8)
    return out


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


def collect_metrics(config: dict) -> dict[int, list[CaseMetric]]:
    vocdevkit_path = Path(config["vocdevkit_path"])
    split = config["split"]
    class_names = config["class_names"]
    image_ids = read_image_ids(vocdevkit_path, split)
    gt_dir = vocdevkit_path / "VOC2007" / "SegmentationClass"
    pred_dir = REPORT_DIR / "pred_masks"
    lesion_class_ids = list(config.get("lesion_class_ids") or range(2, len(class_names)))

    metrics: dict[int, list[CaseMetric]] = {class_id: [] for class_id in lesion_class_ids}
    for image_id in image_ids:
        gt_path = gt_dir / f"{image_id}.png"
        pred_path = pred_dir / f"{image_id}.png"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        for class_id in lesion_class_ids:
            item = class_metrics(gt, pred, int(class_id), class_names[int(class_id)], image_id)
            if item.gt_pixels > 0:
                metrics[int(class_id)].append(item)
    return metrics


def select_cases(metrics: dict[int, list[CaseMetric]]) -> tuple[list[CaseMetric], list[CaseMetric]]:
    success_cases: list[CaseMetric] = []
    failure_cases: list[CaseMetric] = []
    used_success: set[str] = set()
    used_failure: set[str] = set()
    for class_id, items in metrics.items():
        eligible = [m for m in items if m.gt_pixels >= 50]
        if not eligible:
            eligible = items
        success_sorted = sorted(eligible, key=lambda m: (m.iou, m.gt_pixels), reverse=True)
        failure_sorted = sorted(eligible, key=lambda m: (m.iou, -m.gt_pixels))

        success = next((m for m in success_sorted if m.image_id not in used_success), success_sorted[0])
        failure = next((m for m in failure_sorted if m.image_id not in used_failure), failure_sorted[0])
        success_cases.append(success)
        failure_cases.append(failure)
        used_success.add(success.image_id)
        used_failure.add(failure.image_id)
    return success_cases, failure_cases


def case_to_row(case_type: str, metric: CaseMetric) -> dict[str, object]:
    return {
        "case_type": case_type,
        "class_id": metric.class_id,
        "class_name": metric.class_name,
        "image_id": metric.image_id,
        "iou": f"{metric.iou:.4f}",
        "dice": f"{metric.dice:.4f}",
        "precision": f"{metric.precision:.4f}",
        "recall": f"{metric.recall:.4f}",
        "gt_pixels": metric.gt_pixels,
        "pred_pixels": metric.pred_pixels,
        "tp_pixels": metric.tp_pixels,
    }


def add_case_row(fig: plt.Figure, axes: np.ndarray, row: int, metric: CaseMetric, config: dict) -> None:
    vocdevkit_path = Path(config["vocdevkit_path"])
    image_path = vocdevkit_path / "VOC2007" / "JPEGImages" / f"{metric.image_id}.jpg"
    gt_path = vocdevkit_path / "VOC2007" / "SegmentationClass" / f"{metric.image_id}.png"
    pred_path = REPORT_DIR / "pred_masks" / f"{metric.image_id}.png"

    image = load_rgb(image_path)
    gt = load_mask(gt_path)
    pred = load_mask(pred_path)

    panels = [
        ("Image", image),
        ("Ground truth", overlay_mask(image, gt)),
        ("Prediction", overlay_mask(image, pred)),
        ("TP/FN/FP", error_overlay(image, gt, pred, metric.class_id)),
    ]
    for col, (title, panel) in enumerate(panels):
        ax = axes[row, col]
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=10, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    row_label = (
        f"{metric.class_name}\n"
        f"{metric.image_id}\n"
        f"IoU {metric.iou:.2f}  P {metric.precision:.2f}  R {metric.recall:.2f}"
    )
    axes[row, 0].set_ylabel(row_label, fontsize=8.5, rotation=0, labelpad=52, va="center", ha="right")


def save_case_figure(cases: list[CaseMetric], config: dict, stem: str, title: str) -> None:
    fig, axes = plt.subplots(len(cases), 4, figsize=(8.2, 2.15 * len(cases)), squeeze=False)
    for row, metric in enumerate(cases):
        add_case_row(fig, axes, row, metric, config)
    fig.suptitle(title, fontsize=12, y=0.995)
    legend_text = "Error panel: green=TP, red=FN, blue=FP for the target lesion class"
    fig.text(0.5, 0.012, legend_text, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=[0.08, 0.035, 1.0, 0.975])
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_json(REPORT_DIR / "run_config.json")
    metrics = collect_metrics(config)
    success_cases, failure_cases = select_cases(metrics)

    rows = [case_to_row("success", m) for m in success_cases]
    rows.extend(case_to_row("failure", m) for m in failure_cases)
    fields = [
        "case_type",
        "class_id",
        "class_name",
        "image_id",
        "iou",
        "dice",
        "precision",
        "recall",
        "gt_pixels",
        "pred_pixels",
        "tp_pixels",
    ]
    write_csv(OUT_DIR / "paper_qualitative_case_selection.csv", rows, fields)
    write_markdown(OUT_DIR / "paper_qualitative_case_selection.md", rows, fields)
    save_case_figure(
        success_cases,
        config,
        "fig_paper_qualitative_success_cases",
        "Representative successful lesion segmentation cases",
    )
    save_case_figure(
        failure_cases,
        config,
        "fig_paper_qualitative_failure_cases",
        "Representative failure cases by lesion class",
    )

    print("Generated qualitative case evidence:")
    for name in [
        "paper_qualitative_case_selection.csv",
        "paper_qualitative_case_selection.md",
        "fig_paper_qualitative_success_cases.png",
        "fig_paper_qualitative_success_cases.pdf",
        "fig_paper_qualitative_failure_cases.png",
        "fig_paper_qualitative_failure_cases.pdf",
    ]:
        print(f"- {OUT_DIR / name}")


if __name__ == "__main__":
    main()
