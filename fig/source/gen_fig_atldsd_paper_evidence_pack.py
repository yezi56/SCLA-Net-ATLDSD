"""Generate lesion-focused evidence tables and figures for the ATLDSD paper.

The script reads existing experiment summaries only. It does not train,
evaluate, or touch model code.

Usage:
python fig/source/gen_fig_atldsd_paper_evidence_pack.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "fig" / "results"
ABLATION_CSV = SUMMARY_DIR / "paper_ablation_chain.csv"

LESION_COLUMNS = [
    ("rust_iou", "Rust"),
    ("alternaria_iou", "Alternaria"),
    ("gray_iou", "Gray spot"),
    ("brown_iou", "Brown spot"),
]

FORMAL_IDS = ["Baseline", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
LESION_FOCUS_IDS = ["Baseline", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
FULL_CHAIN_IDS = [
    "Baseline",
    "+LGC",
    "+LGC+LCSF",
    "+BalancedPrefix",
    "+RepConv",
    "+LesionDice2",
]

METHOD_LABELS = {
    "Baseline": "Baseline",
    "+LGC": "+LGC",
    "+LGC+LCSF": "+LGC+LCSF",
    "+BalancedPrefix": "+Balanced",
    "+RepConv": "+RepConv",
    "+LesionDice2": "+LesionDice2",
}


def read_rows() -> list[dict[str, str]]:
    with ABLATION_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["id"]: row for row in rows}


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    headers = [field.replace("_", " ") for field in fields]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(SUMMARY_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(SUMMARY_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_lesion_iou_table(rows_by_id: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for method_id in LESION_FOCUS_IDS:
        row = rows_by_id[method_id]
        lesion_values = [fnum(row, key) for key, _ in LESION_COLUMNS]
        out_rows.append(
            {
                "id": method_id,
                "method": row["method"],
                "scale": row["scale"],
                "seeds": row["seeds"],
                "miou": f"{fnum(row, 'miou'):.2f}",
                "fg_miou": f"{fnum(row, 'fg_miou'):.2f}",
                "rust_iou": f"{fnum(row, 'rust_iou'):.2f}",
                "alternaria_iou": f"{fnum(row, 'alternaria_iou'):.2f}",
                "gray_iou": f"{fnum(row, 'gray_iou'):.2f}",
                "brown_iou": f"{fnum(row, 'brown_iou'):.2f}",
                "lesion_avg_iou": f"{np.mean(lesion_values):.2f}",
                "note": row["note"],
            }
        )
    return out_rows


def make_full_chain_table(rows_by_id: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    baseline = rows_by_id["Baseline"]
    base_miou = fnum(baseline, "miou")
    base_fg = fnum(baseline, "fg_miou")
    for method_id in FULL_CHAIN_IDS:
        row = rows_by_id[method_id]
        lesion_values = [fnum(row, key) for key, _ in LESION_COLUMNS]
        out_rows.append(
            {
                "id": method_id,
                "method": row["method"],
                "scale": row["scale"],
                "seeds": row["seeds"],
                "miou": f"{fnum(row, 'miou'):.2f}",
                "fg_miou": f"{fnum(row, 'fg_miou'):.2f}",
                "delta_miou_vs_baseline": f"{fnum(row, 'miou') - base_miou:+.2f}",
                "delta_fg_vs_baseline": f"{fnum(row, 'fg_miou') - base_fg:+.2f}",
                "lesion_avg_iou": f"{np.mean(lesion_values):.2f}",
                "params_m": row["params_m"],
                "flops_g": row["flops_g"],
                "fps": row["fps"],
                "note": row["note"],
            }
        )
    return out_rows


def make_lesiondice_delta_table(rows_by_id: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    repconv = rows_by_id["+RepConv"]
    lesiondice = rows_by_id["+LesionDice2"]
    metric_pairs = [
        ("mIoU", "miou"),
        ("FG mIoU", "fg_miou"),
        ("Rust IoU", "rust_iou"),
        ("Alternaria IoU", "alternaria_iou"),
        ("Gray IoU", "gray_iou"),
        ("Brown IoU", "brown_iou"),
    ]
    out_rows: list[dict[str, object]] = []
    for label, key in metric_pairs:
        before = fnum(repconv, key)
        after = fnum(lesiondice, key)
        out_rows.append(
            {
                "metric": label,
                "repconv": f"{before:.2f}",
                "lesiondice2": f"{after:.2f}",
                "delta": f"{after - before:+.2f}",
            }
        )
    return out_rows


def plot_lesion_iou(rows_by_id: dict[str, dict[str, str]]) -> None:
    methods = LESION_FOCUS_IDS
    labels = [METHOD_LABELS[mid] for mid in methods]
    x = np.arange(len(labels))
    width = 0.18
    colors = ["#557A95", "#D9822B", "#4C956C", "#B24C63"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, (key, lesion_label) in enumerate(LESION_COLUMNS):
        vals = [fnum(rows_by_id[mid], key) for mid in methods]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=lesion_label, color=colors[idx])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.6,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
            )

    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 94)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Lesion-class IoU across formal ATLDSD stages")
    ax.legend(ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", visible=False)
    save_figure(fig, "fig_paper_lesion_iou_comparison")


def plot_formal_progression(rows_by_id: dict[str, dict[str, str]]) -> None:
    methods = FORMAL_IDS
    labels = [METHOD_LABELS[mid] for mid in methods]
    x = np.arange(len(labels))
    miou = [fnum(rows_by_id[mid], "miou") for mid in methods]
    fg = [fnum(rows_by_id[mid], "fg_miou") for mid in methods]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, miou, marker="o", linewidth=2.2, color="#2F5D8C", label="mIoU")
    ax.plot(x, fg, marker="s", linewidth=2.2, color="#C45A3C", label="FG mIoU")
    for idx, val in enumerate(miou):
        ax.text(idx, val + 0.28, f"{val:.2f}", ha="center", fontsize=8)
    for idx, val in enumerate(fg):
        ax.text(idx, val - 0.58, f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(64, 79)
    ax.set_title("Formal full-resolution performance progression")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "fig_paper_formal_progression")


def plot_lesiondice_delta(delta_rows: list[dict[str, object]]) -> None:
    labels = [str(row["metric"]) for row in delta_rows]
    deltas = [float(str(row["delta"])) for row in delta_rows]
    colors = ["#4C956C" if val >= 0 else "#B24C63" for val in deltas]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.barh(y, deltas, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.9)
    for bar, val in zip(bars, deltas):
        x_pos = val + (0.04 if val >= 0 else -0.04)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha=ha, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta vs +RepConv (percentage points)")
    ax.set_title("Effect of lesion-only Dice on final RepConv mainline")
    max_abs = max(abs(v) for v in deltas) if deltas else 1.0
    ax.set_xlim(-max_abs - 0.35, max_abs + 0.35)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig_paper_lesiondice2_delta")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    rows = read_rows()
    rows_by_id = by_id(rows)

    missing = [method_id for method_id in FULL_CHAIN_IDS if method_id not in rows_by_id]
    if missing:
        raise KeyError(f"Missing methods in {ABLATION_CSV}: {missing}")

    lesion_rows = make_lesion_iou_table(rows_by_id)
    chain_rows = make_full_chain_table(rows_by_id)
    delta_rows = make_lesiondice_delta_table(rows_by_id)

    lesion_fields = [
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
        "lesion_avg_iou",
        "note",
    ]
    chain_fields = [
        "id",
        "method",
        "scale",
        "seeds",
        "miou",
        "fg_miou",
        "delta_miou_vs_baseline",
        "delta_fg_vs_baseline",
        "lesion_avg_iou",
        "params_m",
        "flops_g",
        "fps",
        "note",
    ]
    delta_fields = ["metric", "repconv", "lesiondice2", "delta"]

    write_csv(SUMMARY_DIR / "paper_lesion_iou_comparison.csv", lesion_rows, lesion_fields)
    write_markdown_table(SUMMARY_DIR / "paper_lesion_iou_comparison.md", lesion_rows, lesion_fields)
    write_csv(SUMMARY_DIR / "paper_full_chain_progression.csv", chain_rows, chain_fields)
    write_markdown_table(SUMMARY_DIR / "paper_full_chain_progression.md", chain_rows, chain_fields)
    write_csv(SUMMARY_DIR / "paper_lesiondice2_delta.csv", delta_rows, delta_fields)
    write_markdown_table(SUMMARY_DIR / "paper_lesiondice2_delta.md", delta_rows, delta_fields)

    plot_lesion_iou(rows_by_id)
    plot_formal_progression(rows_by_id)
    plot_lesiondice_delta(delta_rows)

    print("Generated paper evidence pack:")
    for name in [
        "paper_lesion_iou_comparison.csv",
        "paper_lesion_iou_comparison.md",
        "paper_full_chain_progression.csv",
        "paper_full_chain_progression.md",
        "paper_lesiondice2_delta.csv",
        "paper_lesiondice2_delta.md",
        "fig_paper_lesion_iou_comparison.png",
        "fig_paper_lesion_iou_comparison.pdf",
        "fig_paper_formal_progression.png",
        "fig_paper_formal_progression.pdf",
        "fig_paper_lesiondice2_delta.png",
        "fig_paper_lesiondice2_delta.pdf",
    ]:
        print(f"- {SUMMARY_DIR / name}")


if __name__ == "__main__":
    main()
