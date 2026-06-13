import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_values(path: Path) -> list[float]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(float(line))
    return values


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready training metric figures.")
    parser.add_argument(
        "--log-dir",
        required=True,
        action="append",
        help="Training log directory. Pass multiple times for resumed runs; earlier dirs drop their final overlapping epoch.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="Training Metrics")
    parser.add_argument("--miou-period", type=int, default=10)
    parser.add_argument("--prefix", default="e00")
    args = parser.parse_args()

    log_dirs = [Path(path) for path in args.log_dir]
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_loss = []
    val_loss = []
    for idx, log_dir in enumerate(log_dirs):
        loss_part = read_values(log_dir / "epoch_loss.txt")
        val_loss_part = read_values(log_dir / "epoch_val_loss.txt")
        if idx < len(log_dirs) - 1:
            loss_part = loss_part[:-1]
            val_loss_part = val_loss_part[:-1]
        train_loss.extend(loss_part)
        val_loss.extend(val_loss_part)

    miou = read_values(log_dirs[-1] / "epoch_miou.txt")

    epochs = list(range(1, len(train_loss) + 1))
    miou_epochs = [i * args.miou_period for i in range(len(miou))]

    best_miou = max(miou)
    best_miou_epoch = miou_epochs[miou.index(best_miou)]
    min_val_loss = min(val_loss)
    min_val_loss_epoch = val_loss.index(min_val_loss) + 1

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.9,
            "lines.markersize": 4.5,
        }
    )

    colors = {
        "train": "#264653",
        "val": "#2A9D8F",
        "miou": "#E76F51",
        "marker": "#D97706",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(epochs, train_loss, color=colors["train"], label="Train loss")
    axes[0].plot(epochs, val_loss, color=colors["val"], label="Val loss")
    axes[0].scatter([min_val_loss_epoch], [min_val_loss], color=colors["marker"], zorder=4)
    axes[0].annotate(
        f"min {min_val_loss:.3f}@{min_val_loss_epoch}",
        xy=(min_val_loss_epoch, min_val_loss),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        color="#444444",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(miou_epochs, miou, color=colors["miou"], marker="o", label="mIoU")
    axes[1].scatter([best_miou_epoch], [best_miou], color=colors["marker"], zorder=4)
    axes[1].annotate(
        f"best {best_miou:.2f}@{best_miou_epoch}",
        xy=(best_miou_epoch, best_miou),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=8,
        color="#444444",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU (%)")
    axes[1].set_title("Validation mIoU")
    axes[1].legend()
    fig.suptitle(args.title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, f"fig_{args.prefix}_metrics_overview")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(epochs, train_loss, color=colors["train"], label="Train loss")
    ax.plot(epochs, val_loss, color=colors["val"], label="Val loss")
    ax.scatter([min_val_loss_epoch], [min_val_loss], color=colors["marker"], zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("E00 Loss")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, f"fig_{args.prefix}_loss")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(miou_epochs, miou, color=colors["miou"], marker="o", label="mIoU")
    ax.scatter([best_miou_epoch], [best_miou], color=colors["marker"], zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("E00 mIoU")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, f"fig_{args.prefix}_miou")
    plt.close(fig)

    summary = {
        "epochs_recorded": len(train_loss),
        "last_epoch": len(train_loss),
        "last_train_loss": train_loss[-1],
        "last_val_loss": val_loss[-1],
        "last_miou_epoch": miou_epochs[-1],
        "last_miou": miou[-1],
        "best_miou": best_miou,
        "best_miou_epoch": best_miou_epoch,
        "min_val_loss": min_val_loss,
        "min_val_loss_epoch": min_val_loss_epoch,
    }
    (out_dir / f"{args.prefix}_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
