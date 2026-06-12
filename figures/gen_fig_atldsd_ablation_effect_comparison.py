from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "atldsd" / "summary"
PICTURES_DIR = Path("D:/Pictures")
OUT_STEM = "fig_paper_ablation_effect_comparison"


COLORS = {
    "Baseline": "#94A3B8",
    "+LGC": "#22B8A9",
    "+LGC+LCSF": "#A78BFA",
    "+BalancedPrefix": "#FDBA74",
    "+RepConv": "#FBBF24",
    "+LesionDice2": "#4ADE80",
}

FORMAL_ORDER = ["Baseline", "+LGC", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
FULL_ORDER_WITH_PARTIAL = ["Baseline", "+LGC", "+LGC+LCSF", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
LESION_COLS = ["rust_iou", "alternaria_iou", "gray_iou", "brown_iou"]
LESION_LABELS = ["Rust", "Alternaria", "Gray", "Brown"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "axes.labelcolor": "#0F172A",
            "text.color": "#0F172A",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def annotate_bar(ax: plt.Axes, bars, fmt="{:.2f}", dy=0.25, size=6.5) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=size,
            color="#334155",
        )


def load_data() -> dict[str, dict[str, float | str]]:
    csv_path = SUMMARY_DIR / "paper_ablation_chain.csv"
    rows: dict[str, dict[str, float | str]] = {}
    numeric = {
        "miou",
        "fg_miou",
        "rust_iou",
        "alternaria_iou",
        "gray_iou",
        "brown_iou",
        "params_m",
        "flops_g",
        "fps",
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item: dict[str, float | str] = {}
            for key, value in row.items():
                item[key] = float(value) if key in numeric else value
            item["lesion_avg_iou"] = float(np.mean([item[c] for c in LESION_COLS]))
            rows[str(row["id"])] = item
    return rows


def plot_metric_progression(ax: plt.Axes, df: dict[str, dict[str, float | str]]) -> None:
    x = np.arange(len(FULL_ORDER_WITH_PARTIAL))
    width = 0.36
    miou = np.array([float(df[m]["miou"]) for m in FULL_ORDER_WITH_PARTIAL])
    fg = np.array([float(df[m]["fg_miou"]) for m in FULL_ORDER_WITH_PARTIAL])
    colors = [COLORS[m] for m in FULL_ORDER_WITH_PARTIAL]

    bars1 = ax.bar(x - width / 2, miou, width, label="mIoU", color=colors, edgecolor="#0F172A", linewidth=0.45)
    bars2 = ax.bar(x + width / 2, fg, width, label="FG mIoU", color=colors, edgecolor="#0F172A", linewidth=0.45, alpha=0.56)

    partial_idx = FULL_ORDER_WITH_PARTIAL.index("+LGC+LCSF")
    for bar in (bars1[partial_idx], bars2[partial_idx]):
        bar.set_hatch("//")
        bar.set_alpha(0.34)

    annotate_bar(ax, bars1, dy=0.18, size=5.8)
    annotate_bar(ax, bars2, dy=0.18, size=5.8)

    ax.axhline(float(df["Baseline"]["miou"]), color="#94A3B8", linestyle="--", linewidth=0.9, zorder=0)
    ax.text(
        0.02,
        0.92,
        "Final: +5.38 mIoU / +6.25 FG over baseline",
        transform=ax.transAxes,
        fontsize=6.8,
        fontweight="bold",
        color="#15803D",
    )
    ax.text(
        partial_idx,
        60.2,
        "partial\nfast-screen",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#7C3AED",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FULL_ORDER_WITH_PARTIAL, rotation=22, ha="right", fontsize=6.8)
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(54, 80.5)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.legend(loc="upper right", ncols=2, fontsize=6.6)
    ax.set_title("A. Overall and foreground mIoU progression", loc="left", fontweight="bold")


def plot_lesion_iou(ax: plt.Axes, df: dict[str, dict[str, float | str]]) -> None:
    methods = ["Baseline", "+LGC", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
    x = np.arange(len(LESION_LABELS))
    width = 0.15
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

    for method, off in zip(methods, offsets):
        vals = np.array([float(df[method][c]) for c in LESION_COLS])
        ax.bar(
            x + off,
            vals,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="#0F172A",
            linewidth=0.35,
        )

    final = np.array([float(df["+LesionDice2"][c]) for c in LESION_COLS])
    for i, v in enumerate(final):
        ax.text(i, v + 1.0, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2, color="#15803D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LESION_LABELS)
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(40, 90)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3, fontsize=6.5)
    ax.set_title("B. Per-lesion-class IoU", loc="left", fontweight="bold")


def plot_delta(ax: plt.Axes, df: dict[str, dict[str, float | str]]) -> None:
    baseline = df["Baseline"]
    final = df["+LesionDice2"]
    metrics = [
        ("mIoU", float(final["miou"]) - float(baseline["miou"])),
        ("FG mIoU", float(final["fg_miou"]) - float(baseline["fg_miou"])),
        ("Rust", float(final["rust_iou"]) - float(baseline["rust_iou"])),
        ("Alternaria", float(final["alternaria_iou"]) - float(baseline["alternaria_iou"])),
        ("Gray", float(final["gray_iou"]) - float(baseline["gray_iou"])),
        ("Brown", float(final["brown_iou"]) - float(baseline["brown_iou"])),
    ]
    labels, vals = zip(*metrics)
    colors = ["#4ADE80" if v >= 0 else "#FB7185" for v in vals]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, edgecolor="#0F172A", linewidth=0.35)
    ax.axvline(0, color="#334155", linewidth=0.8)
    for yi, v in enumerate(vals):
        ax.text(v + (0.12 if v >= 0 else -0.12), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta IoU vs baseline (points)")
    ax.set_xlim(-1, 13)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_title("C. Final model gains over baseline", loc="left", fontweight="bold")


def plot_tradeoff(ax: plt.Axes, df: dict[str, dict[str, float | str]]) -> None:
    methods = ["Baseline", "+LGC", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
    label_offsets = {
        "Baseline": (0.55, -0.10, "left"),
        "+LGC": (-3.6, -0.18, "right"),
        "+BalancedPrefix": (-1.4, -0.45, "right"),
        "+RepConv": (0.55, 0.18, "left"),
        "+LesionDice2": (0.55, 0.48, "left"),
    }
    for method in methods:
        row = df[method]
        ax.scatter(
            float(row["flops_g"]),
            float(row["miou"]),
            s=120 + (float(row["params_m"]) - 11.5) * 160,
            color=COLORS[method],
            edgecolor="#0F172A",
            linewidth=0.55,
            zorder=3,
        )
        dx, dy, ha = label_offsets[method]
        ax.annotate(
            f"{method}\n{float(row['fps']):.1f} FPS",
            xy=(float(row["flops_g"]), float(row["miou"])),
            xytext=(float(row["flops_g"]) + dx, float(row["miou"]) + dy),
            textcoords="data",
            fontsize=6.2,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.55, shrinkA=3, shrinkB=3),
        )

    ax.plot(
        [float(df[m]["flops_g"]) for m in methods],
        [float(df[m]["miou"]) for m in methods],
        color="#CBD5E1",
        linewidth=1.2,
        zorder=1,
    )
    ax.set_xlabel("FLOPs (G)")
    ax.set_ylabel("mIoU (%)")
    ax.set_xlim(13, 46)
    ax.set_ylim(70.6, 78.4)
    ax.grid(color="#E2E8F0", linewidth=0.7)
    ax.text(0.02, 0.05, "Bubble size: Params; label: reported FPS", transform=ax.transAxes, fontsize=6.4, color="#64748B")
    ax.set_title("D. Accuracy-complexity tradeoff", loc="left", fontweight="bold")


def main() -> None:
    setup_style()
    df = load_data()
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)
    plot_metric_progression(axs[0, 0], df)
    plot_lesion_iou(axs[0, 1], df)
    plot_delta(axs[1, 0], df)
    plot_tradeoff(axs[1, 1], df)

    fig.suptitle(
        "ATLDSD ablation effect comparison",
        fontsize=10,
        fontweight="bold",
        x=0.02,
        y=1.01,
        ha="left",
    )
    fig.text(
        0.02,
        -0.01,
        "Note: +LGC+LCSF is shown as partial fast-screen evidence only; formal claims use full/e80 dual-seed rows.",
        fontsize=6.8,
        color="#64748B",
    )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in [SUMMARY_DIR, PICTURES_DIR]:
        fig.savefig(out_dir / f"{OUT_STEM}.png", dpi=600, bbox_inches="tight")
        fig.savefig(out_dir / f"{OUT_STEM}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{OUT_STEM}.svg", bbox_inches="tight")
    print(f"saved {SUMMARY_DIR / (OUT_STEM + '.png')}")
    print(f"saved {PICTURES_DIR / (OUT_STEM + '.png')}")


if __name__ == "__main__":
    main()
