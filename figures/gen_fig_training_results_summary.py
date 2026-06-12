"""Generate ATLDSD training-result summary table and figures.

Update ROWS whenever a training run finishes, then rerun:
python figures/gen_fig_training_results_summary.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "atldsd" / "summary"


ROWS = [
    {
        "id": "Old-MNet",
        "method": "DeepLabV3+ MobileNet",
        "change": "old backbone",
        "status": "done",
        "miou": 67.58,
        "fg_miou": 61.75,
        "acc": 97.22,
        "severity_mae": None,
        "grade_acc": None,
        "params_m": None,
        "flops_g": None,
        "fps": None,
        "decision": "old baseline",
    },
    {
        "id": "B4",
        "method": "DeepLabV3+ EfficientNet-B4",
        "change": "heavier backbone",
        "status": "done",
        "miou": 65.59,
        "fg_miou": 59.59,
        "acc": 96.44,
        "severity_mae": None,
        "grade_acc": None,
        "params_m": 32.48,
        "flops_g": 51.30,
        "fps": 52.77,
        "decision": "discard as main backbone",
    },
    {
        "id": "CLCS-V3",
        "method": "CLCS + MobileNetV3-Large",
        "change": "early compositional trial",
        "status": "done",
        "miou": 70.50,
        "fg_miou": 65.05,
        "acc": 97.97,
        "severity_mae": None,
        "grade_acc": None,
        "params_m": None,
        "flops_g": None,
        "fps": None,
        "decision": "archival trial",
    },
    {
        "id": "SCLP-0.7",
        "method": "Mainline0 + SCLP",
        "change": "strong copy-paste",
        "status": "done",
        "miou": 68.97,
        "fg_miou": 63.28,
        "acc": 97.71,
        "severity_mae": 0.01542,
        "grade_acc": 90.24,
        "params_m": None,
        "flops_g": None,
        "fps": None,
        "decision": "failed augmentation",
    },
    {
        "id": "SCLP-0.3",
        "method": "Mainline0 + weak SCLP",
        "change": "weak copy-paste",
        "status": "done",
        "miou": 69.90,
        "fg_miou": 64.67,
        "acc": 96.76,
        "severity_mae": 0.01592,
        "grade_acc": 90.65,
        "params_m": None,
        "flops_g": None,
        "fps": None,
        "decision": "failed augmentation",
    },
    {
        "id": "Mainline0",
        "method": "DeepLabV3+ MobileNetV3-Large",
        "change": "strong baseline",
        "status": "done",
        "miou": 71.72,
        "fg_miou": 66.58,
        "acc": 97.76,
        "severity_mae": 0.01241,
        "grade_acc": 95.12,
        "params_m": 11.73,
        "flops_g": 15.28,
        "fps": 98.80,
        "decision": "baseline to beat",
    },
    {
        "id": "Mainline1",
        "method": "Mainline0 + component heads",
        "change": "lesion/boundary/center heads",
        "status": "done",
        "miou": 72.11,
        "fg_miou": 67.03,
        "acc": 97.82,
        "severity_mae": 0.01212,
        "grade_acc": 94.31,
        "params_m": 11.73,
        "flops_g": 15.29,
        "fps": 101.10,
        "decision": "current structural anchor",
    },
    {
        "id": "Aux-A",
        "method": "Mainline1 + severity loss",
        "change": "loss ablation",
        "status": "done",
        "miou": 72.12,
        "fg_miou": 67.06,
        "acc": 97.78,
        "severity_mae": 0.01147,
        "grade_acc": 93.90,
        "params_m": 11.73,
        "flops_g": 15.29,
        "fps": 98.68,
        "decision": "best severity MAE; not main structure",
    },
    {
        "id": "Mainline2",
        "method": "Mainline1 + PConv",
        "change": "decoder locality",
        "status": "done",
        "miou": 71.76,
        "fg_miou": 66.62,
        "acc": 97.80,
        "severity_mae": 0.01373,
        "grade_acc": 93.09,
        "params_m": 10.65,
        "flops_g": 6.51,
        "fps": 92.14,
        "decision": "lighter; test PConv+LBSB synergy",
    },
    {
        "id": "Boundary1",
        "method": "Mainline1 + LBSB",
        "change": "boundary sharpening",
        "status": "done",
        "miou": 72.86,
        "fg_miou": 67.89,
        "acc": 97.97,
        "severity_mae": 0.01177,
        "grade_acc": 93.90,
        "params_m": 11.73,
        "flops_g": 15.29,
        "fps": 106.89,
        "decision": "best mIoU; promote to current best",
    },
    {
        "id": "Boundary2",
        "method": "Mainline1 + PConv + LBSB",
        "change": "decoder locality + boundary sharpening",
        "status": "done",
        "miou": 71.68,
        "fg_miou": 66.54,
        "acc": 97.72,
        "severity_mae": 0.01281,
        "grade_acc": 93.50,
        "params_m": 10.65,
        "flops_g": 6.52,
        "fps": 39.81,
        "decision": "no PConv-LBSB synergy; do not keep PConv",
    },
    {
        "id": "Fusion1",
        "method": "Mainline1 + LBSB + LCAF",
        "change": "lesion-aware cross-scale fusion",
        "status": "done",
        "miou": 72.68,
        "fg_miou": 67.70,
        "acc": 97.86,
        "severity_mae": 0.01169,
        "grade_acc": 93.90,
        "params_m": 11.76,
        "flops_g": 15.53,
        "fps": 88.16,
        "decision": "close, but below Boundary1; do not replace LBSB-only",
    },
    {
        "id": "Context1",
        "method": "Mainline1 + LBSB + LGLC",
        "change": "local-global lesion context",
        "status": "done",
        "miou": 72.31,
        "fg_miou": 67.26,
        "acc": 97.87,
        "severity_mae": 0.01170,
        "grade_acc": 93.50,
        "params_m": 11.84,
        "flops_g": 15.33,
        "fps": 42.85,
        "decision": "below Boundary1; keep as negative context ablation",
    },
    {
        "id": "Final-LGC-LCSF",
        "method": "Boundary1 + SP decoder + LGC + LCSF",
        "change": "balanced-prefix full e80, dual-seed avg",
        "status": "done",
        "miou": 76.60,
        "fg_miou": 72.22,
        "acc": None,
        "severity_mae": 0.00965,
        "grade_acc": 94.92,
        "params_m": 12.14,
        "flops_g": 38.37,
        "fps": 54.99,
        "decision": "previous pre-RepConv mainline",
    },
    {
        "id": "Final-RepConv",
        "method": "Final-LGC-LCSF + RepConv decoder",
        "change": "full e80, dual-seed avg",
        "status": "done",
        "miou": 76.94,
        "fg_miou": 72.63,
        "acc": 98.56,
        "severity_mae": 0.01030,
        "grade_acc": 93.29,
        "params_m": 12.28,
        "flops_g": 41.07,
        "fps": 63.61,
        "decision": "previous official mainline",
    },
    {
        "id": "Final-LesionDice2",
        "method": "Final-RepConv + lesion-only Dice",
        "change": "full e80, dual-seed avg",
        "status": "done",
        "miou": 77.10,
        "fg_miou": 72.83,
        "acc": 98.53,
        "severity_mae": 0.01012,
        "grade_acc": 94.51,
        "params_m": 12.28,
        "flops_g": 41.07,
        "fps": 52.36,
        "decision": "current official mainline",
    },
]


PAPER_ABLATION_ROWS = [
    {
        "id": "Baseline",
        "method": "DeepLabV3+ MobileNetV3-Large",
        "scale": "full/e150",
        "seeds": "best_mIoU",
        "miou": 71.72,
        "fg_miou": 66.58,
        "rust_iou": 81.23,
        "alternaria_iou": 52.88,
        "gray_iou": 56.88,
        "brown_iou": 48.42,
        "params_m": 11.73,
        "flops_g": 15.28,
        "fps": 86.32,
        "note": "baseline to beat",
    },
    {
        "id": "+LGC",
        "method": "Baseline + LGC",
        "scale": "full/e80",
        "seeds": "11,23 avg",
        "miou": 76.17,
        "fg_miou": 71.68,
        "rust_iou": 84.74,
        "alternaria_iou": 55.22,
        "gray_iou": 67.73,
        "brown_iou": 54.79,
        "params_m": 12.11,
        "flops_g": 37.83,
        "fps": 76.92,
        "note": "formal full/e80 dual-seed result",
    },
    {
        "id": "+LGC+LCSF",
        "method": "Baseline + LGC + LCSF",
        "scale": "128/64 e24",
        "seeds": "11,23 avg",
        "miou": 63.43,
        "fg_miou": 56.89,
        "rust_iou": 72.25,
        "alternaria_iou": 43.56,
        "gray_iou": 34.98,
        "brown_iou": 43.32,
        "params_m": 12.14,
        "flops_g": 38.37,
        "fps": 63.40,
        "note": "fast-screen evidence only",
    },
    {
        "id": "+BalancedPrefix",
        "method": "+LGC+LCSF + BalancedPrefix",
        "scale": "full/e80",
        "seeds": "11,23 avg",
        "miou": 76.60,
        "fg_miou": 72.22,
        "rust_iou": 84.90,
        "alternaria_iou": 56.10,
        "gray_iou": 68.16,
        "brown_iou": 56.31,
        "params_m": 12.14,
        "flops_g": 38.37,
        "fps": 55.00,
        "note": "formal dual-seed result",
    },
    {
        "id": "+RepConv",
        "method": "+BalancedPrefix + RepConv decoder",
        "scale": "full/e80",
        "seeds": "11,23 avg",
        "miou": 76.94,
        "fg_miou": 72.63,
        "rust_iou": 84.60,
        "alternaria_iou": 58.57,
        "gray_iou": 68.35,
        "brown_iou": 55.93,
        "params_m": 12.28,
        "flops_g": 41.07,
        "fps": 63.61,
        "note": "previous official mainline",
    },
    {
        "id": "+LesionDice2",
        "method": "+RepConv + lesion-only Dice",
        "scale": "full/e80",
        "seeds": "11,23 avg",
        "miou": 77.10,
        "fg_miou": 72.83,
        "rust_iou": 84.44,
        "alternaria_iou": 58.92,
        "gray_iou": 68.27,
        "brown_iou": 56.95,
        "params_m": 12.28,
        "flops_g": 41.07,
        "fps": 52.36,
        "note": "current official mainline",
    },
]


DEPLOY_FUSED_ROWS = [
    {
        "id": "LesionDice2-RepConv",
        "checkpoint": "best_miou_deploy_fused_weights.pth",
        "source": "full/e80 seed23 best_mIoU checkpoint",
        "blocks_fused": 2,
        "params_before": 12284955,
        "params_after": 12139547,
        "macs_after_g": 19.1675,
        "flops_after_g": 38.3351,
        "cpu_fps_smoke": 2.84,
        "cpu_time_smoke": 0.351812,
        "cuda_train_fps_300": 88.08,
        "cuda_fused_fps_300": 46.96,
        "cuda_train_time_300": 0.011473,
        "cuda_fused_time_300": 0.021293,
        "cuda_train_fps_1000_mean": 113.22,
        "cuda_train_fps_1000_std": 1.82,
        "cuda_fused_fps_1000_mean": 118.82,
        "cuda_fused_fps_1000_std": 1.16,
        "cuda_train_time_1000_mean": 0.008834,
        "cuda_fused_time_1000_mean": 0.008417,
        "cuda_fused_speedup_1000": 1.050,
        "equivalence": "CPU max diff 3.05e-05; CPU argmax mismatch 0; CUDA1000 avg argmax mismatch 2.94e-05",
        "note": "CUDA1000 stable repeats show fused +5.0% FPS in this environment; older CUDA300 current-load repeats were slower",
    },
]


def fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_csv_and_markdown(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "training_results_summary.csv"
    md_path = OUT_DIR / "training_results_summary.md"
    fields = [
        "id",
        "method",
        "change",
        "status",
        "miou",
        "fg_miou",
        "acc",
        "severity_mae",
        "grade_acc",
        "params_m",
        "flops_g",
        "fps",
        "decision",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    headers = ["ID", "Method", "Change", "Status", "mIoU", "FG mIoU", "Sev. MAE", "Grade Acc", "Params", "FLOPs", "Decision"]
    lines = [
        "# ATLDSD Training Results Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["id"],
                    r["method"],
                    r["change"],
                    r["status"],
                    fmt(r["miou"]),
                    fmt(r["fg_miou"]),
                    fmt(r["severity_mae"], 5),
                    fmt(r["grade_acc"]),
                    fmt(r["params_m"]),
                    fmt(r["flops_g"]),
                    r["decision"],
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_paper_ablation(rows):
    csv_path = OUT_DIR / "paper_ablation_chain.csv"
    md_path = OUT_DIR / "paper_ablation_chain.md"
    fields = [
        "id",
        "method",
        "scale",
        "seeds",
        "miou",
        "fg_miou",
        "rust_iou",
        "alternaria_iou",
        "gray_iou",
        "brown_iou",
        "params_m",
        "flops_g",
        "fps",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "ID",
        "Method",
        "Scale",
        "Seeds",
        "mIoU",
        "FG",
        "Rust",
        "Alternaria",
        "Gray",
        "Brown",
        "Params",
        "FLOPs",
        "FPS",
        "Note",
    ]
    lines = [
        "# ATLDSD Paper Ablation Chain",
        "",
        "Formal claims should use full/e80 dual-seed rows. The 128/64 rows are retained as fast-screen evidence only. FPS in this table is the report benchmark average; deploy-fused CUDA1000 stability evidence is recorded separately in deploy_fused_summary.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["id"],
                    r["method"],
                    r["scale"],
                    r["seeds"],
                    fmt(r["miou"]),
                    fmt(r["fg_miou"]),
                    fmt(r["rust_iou"]),
                    fmt(r["alternaria_iou"]),
                    fmt(r["gray_iou"]),
                    fmt(r["brown_iou"]),
                    fmt(r["params_m"]),
                    fmt(r["flops_g"]),
                    fmt(r["fps"]),
                    r["note"],
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_deploy_fused_summary(rows):
    csv_path = OUT_DIR / "deploy_fused_summary.csv"
    md_path = OUT_DIR / "deploy_fused_summary.md"
    fields = [
        "id",
        "checkpoint",
        "source",
        "blocks_fused",
        "params_before",
        "params_after",
        "macs_after_g",
        "flops_after_g",
        "cpu_fps_smoke",
        "cpu_time_smoke",
        "cuda_train_fps_300",
        "cuda_fused_fps_300",
        "cuda_train_time_300",
        "cuda_fused_time_300",
        "cuda_train_fps_1000_mean",
        "cuda_train_fps_1000_std",
        "cuda_fused_fps_1000_mean",
        "cuda_fused_fps_1000_std",
        "cuda_train_time_1000_mean",
        "cuda_fused_time_1000_mean",
        "cuda_fused_speedup_1000",
        "equivalence",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "ID",
        "Checkpoint",
        "Source",
        "Fused Blocks",
        "Params Before",
        "Params After",
        "MACs After",
        "FLOPs After",
        "CPU FPS",
        "CPU Time",
        "CUDA Train FPS",
        "CUDA Fused FPS",
        "CUDA Train Time",
        "CUDA Fused Time",
        "CUDA1000 Train FPS",
        "CUDA1000 Train FPS Std",
        "CUDA1000 Fused FPS",
        "CUDA1000 Fused FPS Std",
        "CUDA1000 Train Time",
        "CUDA1000 Fused Time",
        "CUDA1000 Speedup",
        "Equivalence",
        "Note",
    ]
    lines = [
        "# ATLDSD Deploy-Fused Summary",
        "",
        "This table records the actual deploy-fused RepConv structure. CPU FPS is a smoke benchmark only. CUDA300 records an older current-load repeat; CUDA1000 records the stable final three-repeat benchmark on the current GPU environment.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["id"],
                    r["checkpoint"],
                    r["source"],
                    str(r["blocks_fused"]),
                    str(r["params_before"]),
                    str(r["params_after"]),
                    fmt(r["macs_after_g"], 4),
                    fmt(r["flops_after_g"], 4),
                    fmt(r["cpu_fps_smoke"]),
                    fmt(r["cpu_time_smoke"], 6),
                    fmt(r["cuda_train_fps_300"]),
                    fmt(r["cuda_fused_fps_300"]),
                    fmt(r["cuda_train_time_300"], 6),
                    fmt(r["cuda_fused_time_300"], 6),
                    fmt(r["cuda_train_fps_1000_mean"]),
                    fmt(r["cuda_train_fps_1000_std"]),
                    fmt(r["cuda_fused_fps_1000_mean"]),
                    fmt(r["cuda_fused_fps_1000_std"]),
                    fmt(r["cuda_train_time_1000_mean"], 6),
                    fmt(r["cuda_fused_time_1000_mean"], 6),
                    fmt(r["cuda_fused_speedup_1000"], 3),
                    r["equivalence"],
                    r["note"],
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18)


def plot_miou(rows):
    done = [r for r in rows if r["status"] == "done" and r["miou"] is not None]
    labels = [r["id"] for r in done]
    values = [r["miou"] for r in done]
    colors = ["#B0BEC5"] * len(done)
    for i, r in enumerate(done):
        if r["id"] == "Mainline1":
            colors[i] = "#2A9D8F"
        if r["id"] == "Aux-A":
            colors[i] = "#E76F51"
        if r["id"] == "Mainline2":
            colors[i] = "#E9C46A"
        if r["id"] == "Boundary1":
            colors[i] = "#D62828"
        if r["id"] == "Final-LGC-LCSF":
            colors[i] = "#7C3AED"
        if r["id"] == "Final-RepConv":
            colors[i] = "#2563EB"
        if r["id"] == "Final-LesionDice2":
            colors[i] = "#059669"

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU (%)")
    ax.set_title("ATLDSD Training Results: mIoU Comparison", weight="bold")
    ax.axvline(72.86, color="#D62828", linestyle="--", linewidth=1.2, alpha=0.8, label="Boundary1")
    ax.axvline(76.60, color="#7C3AED", linestyle="-.", linewidth=1.2, alpha=0.8, label="BalancedPrefix")
    ax.axvline(77.10, color="#059669", linestyle="-.", linewidth=1.2, alpha=0.8, label="LesionDice2")
    for idx, value in enumerate(values):
        ax.text(value + 0.15, idx, f"{value:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(max(min(values) - 3.0, 55), max(values) + 2.8)
    style_axes(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_training_miou_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_training_miou_comparison.pdf")
    plt.close(fig)


def plot_table(rows):
    display_rows = []
    for r in rows:
        display_rows.append(
            [
                r["id"],
                r["change"],
                r["status"],
                fmt(r["miou"]),
                fmt(r["fg_miou"]),
                fmt(r["severity_mae"], 5),
                fmt(r["grade_acc"]),
                fmt(r["params_m"]),
                fmt(r["flops_g"]),
                r["decision"],
            ]
        )
    headers = ["ID", "Change", "Status", "mIoU", "FG", "Sev.MAE", "Grade", "Params", "FLOPs", "Decision"]
    fig, ax = plt.subplots(figsize=(13.4, 6.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.08, 0.18, 0.08, 0.07, 0.07, 0.08, 0.08, 0.07, 0.07, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    best_miou = max(r["miou"] for r in rows if r["miou"] is not None)
    best_sev = min(r["severity_mae"] for r in rows if r["severity_mae"] is not None)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        if row_idx == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
            continue
        r = rows[row_idx - 1]
        cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "#FFFFFF")
        if r["status"] == "running":
            cell.set_facecolor("#FFF3CD")
        if col_idx == 3 and r["miou"] == best_miou:
            cell.set_facecolor("#F4A261")
            cell.set_text_props(weight="bold")
        if col_idx == 5 and r["severity_mae"] == best_sev:
            cell.set_facecolor("#E9C46A")
            cell.set_text_props(weight="bold")
        if r["id"] in {"Mainline1", "Final-LGC-LCSF", "Final-RepConv", "Final-LesionDice2"} and col_idx == 0:
            final_colors = {
                "Mainline1": "#2A9D8F",
                "Final-LGC-LCSF": "#7C3AED",
                "Final-RepConv": "#2563EB",
                "Final-LesionDice2": "#059669",
            }
            cell.set_facecolor(final_colors[r["id"]])
            cell.set_text_props(color="white", weight="bold")

    ax.set_title("ATLDSD Training Summary (update after every completed run)", fontsize=13, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_training_results_table.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_training_results_table.pdf")
    plt.close(fig)


def plot_model_tradeoff(rows):
    rows_with_cost = [
        r
        for r in rows
        if r["status"] == "done"
        and r["miou"] is not None
        and r["params_m"] is not None
        and r["flops_g"] is not None
        and r["fps"] is not None
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    groups = [
        {
            "ids": ["B4"],
            "color": "#8D99AE",
            "label": "Backbone trial",
            "marker": "s",
            "line": "--",
        },
        {
            "ids": ["Mainline0", "Mainline1", "Aux-A"],
            "color": "#D62828",
            "label": "Component mainline",
            "marker": "D",
            "line": "-",
        },
        {
            "ids": ["Mainline1", "Mainline2", "Boundary1", "Boundary2", "Fusion1", "Context1"],
            "color": "#1D4E89",
            "label": "Decoder/boundary path",
            "marker": "o",
            "line": "-",
        },
        {
            "ids": ["Boundary1", "Final-LGC-LCSF", "Final-RepConv", "Final-LesionDice2"],
            "color": "#7C3AED",
            "label": "Final path",
            "marker": "*",
            "line": "-",
        },
    ]
    by_id = {r["id"]: r for r in rows_with_cost}
    for group in groups:
        group_rows = [by_id[i] for i in group["ids"] if i in by_id]
        if not group_rows:
            continue
        xs = [r["params_m"] for r in group_rows]
        ys = [r["miou"] for r in group_rows]
        ax.plot(
            xs,
            ys,
            group["line"],
            color=group["color"],
            linewidth=1.8,
            marker=group["marker"],
            markersize=5.0,
            label=group["label"],
        )

    offsets = {
        "B4": (0.35, -0.24),
        "Mainline0": (0.20, -0.30),
        "Mainline1": (0.26, 0.16),
        "Aux-A": (0.24, 0.34),
        "Mainline2": (-1.65, -0.28),
        "Boundary1": (0.28, 0.16),
        "Fusion1": (0.28, -0.28),
        "Context1": (0.30, -0.20),
        "Final-LGC-LCSF": (0.35, 0.18),
        "Final-RepConv": (0.35, 0.05),
        "Final-LesionDice2": (0.35, 0.18),
    }
    for r in rows_with_cost:
        dx, dy = offsets.get(r["id"], (0.2, 0.2))
        weight = "bold" if r["id"] in {"Boundary1", "Aux-A", "Final-LGC-LCSF", "Final-RepConv", "Final-LesionDice2"} else "normal"
        ax.scatter(r["params_m"], r["miou"], s=32, color="#111827", zorder=4)
        ax.text(r["params_m"] + dx, r["miou"] + dy, r["id"], fontsize=8.5, weight=weight)

    ax.set_xlabel("Params (Millions)")
    ax.set_ylabel("ATLDSD mIoU (%)")
    ax.set_title("ATLDSD Model Accuracy-Efficiency Comparison", weight="bold")
    ax.set_xlim(8.2, 34.5)
    ax.set_ylim(65.0, 77.8)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    table_rows = [
        [
            r["id"],
            fmt(r["miou"]),
            f"{r['params_m']:.2f}M",
            f"{r['flops_g']:.2f}G",
            f"{r['fps']:.1f}",
        ]
        for r in rows_with_cost
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Method", "mIoU", "Params", "FLOPs", "FPS"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.45, 0.07, 0.52, 0.34],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#6B7280")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F3F4F6")
        elif table_rows[row_idx - 1][0] in {"Mainline1", "Aux-A", "Final-LGC-LCSF", "Final-RepConv", "Final-LesionDice2"}:
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_training_model_tradeoff.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_training_model_tradeoff.pdf")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    csv_path, md_path = write_csv_and_markdown(ROWS)
    ablation_csv, ablation_md = write_paper_ablation(PAPER_ABLATION_ROWS)
    deploy_csv, deploy_md = write_deploy_fused_summary(DEPLOY_FUSED_ROWS)
    plot_miou(ROWS)
    plot_table(ROWS)
    plot_model_tradeoff(ROWS)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {ablation_csv}")
    print(f"Wrote {ablation_md}")
    print(f"Wrote {deploy_csv}")
    print(f"Wrote {deploy_md}")
    print(f"Wrote {OUT_DIR / 'fig_training_results_table.png'}")
    print(f"Wrote {OUT_DIR / 'fig_training_miou_comparison.png'}")
    print(f"Wrote {OUT_DIR / 'fig_training_model_tradeoff.png'}")


if __name__ == "__main__":
    main()
