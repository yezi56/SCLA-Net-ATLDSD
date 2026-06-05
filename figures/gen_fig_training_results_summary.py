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
        "status": "running",
        "miou": None,
        "fg_miou": None,
        "acc": None,
        "severity_mae": None,
        "grade_acc": None,
        "params_m": None,
        "flops_g": None,
        "fps": None,
        "decision": "running; update after completion",
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

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU (%)")
    ax.set_title("ATLDSD Training Results: mIoU Comparison", weight="bold")
    ax.axvline(72.11, color="#2A9D8F", linestyle="--", linewidth=1.2, alpha=0.8, label="Mainline1")
    for idx, value in enumerate(values):
        ax.text(value + 0.15, idx, f"{value:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(max(min(values) - 3.0, 55), max(values) + 2.2)
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
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
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
        if r["id"] == "Mainline1" and col_idx == 0:
            cell.set_facecolor("#2A9D8F")
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
            "ids": ["Mainline1", "Mainline2"],
            "color": "#1D4E89",
            "label": "PConv decoder",
            "marker": "o",
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
    }
    for r in rows_with_cost:
        dx, dy = offsets.get(r["id"], (0.2, 0.2))
        weight = "bold" if r["id"] in {"Mainline1", "Aux-A"} else "normal"
        ax.scatter(r["params_m"], r["miou"], s=32, color="#111827", zorder=4)
        ax.text(r["params_m"] + dx, r["miou"] + dy, r["id"], fontsize=8.5, weight=weight)

    ax.set_xlabel("Params (Millions)")
    ax.set_ylabel("ATLDSD mIoU (%)")
    ax.set_title("ATLDSD Model Accuracy-Efficiency Comparison", weight="bold")
    ax.set_xlim(8.2, 34.5)
    ax.set_ylim(65.0, 72.8)
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
        elif table_rows[row_idx - 1][0] in {"Mainline1", "Aux-A"}:
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
    plot_miou(ROWS)
    plot_table(ROWS)
    plot_model_tradeoff(ROWS)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {OUT_DIR / 'fig_training_results_table.png'}")
    print(f"Wrote {OUT_DIR / 'fig_training_miou_comparison.png'}")
    print(f"Wrote {OUT_DIR / 'fig_training_model_tradeoff.png'}")


if __name__ == "__main__":
    main()
