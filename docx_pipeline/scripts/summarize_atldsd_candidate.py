"""Summarize ATLDSD candidate runs against a reference.

The script reads report folders produced by export_segmentation_report.py.  If a
candidate report is not ready yet, it falls back to weights/best_miou.txt and
marks that seed as running instead of making an upgrade decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


CLASS_ORDER = [
    "leaf",
    "rust",
    "alternaria_leaf_spot",
    "gray_spot",
    "brown_spot",
]


@dataclass
class RunMetrics:
    name: str
    seed: str
    path: Path
    status: str
    miou: float | None
    fg_miou: float | None
    per_class_iou: dict[str, float]


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def signed_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}"


def infer_seed(path: Path) -> str:
    match = re.search(r"(?:^|_)s(?:eed)?(\d+)(?:_|$)", path.name)
    if match:
        return match.group(1)
    match = re.search(r"seed(\d+)", path.name)
    if match:
        return match.group(1)
    return "?"


def read_per_class(report_dir: Path) -> dict[str, float]:
    csv_path = report_dir / "per_class_metrics.csv"
    if not csv_path.exists():
        return {}
    values: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["class_name"]] = float(row["iou"])
    return values


def read_run(path: Path, name: str) -> RunMetrics:
    report_dir = path / "reports" / "best_miou"
    summary_path = report_dir / "metrics_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        return RunMetrics(
            name=name,
            seed=infer_seed(path),
            path=path,
            status="done",
            miou=float(summary["miou_all"]),
            fg_miou=float(summary["miou_foreground"]),
            per_class_iou=read_per_class(report_dir),
        )

    best_path = path / "weights" / "best_miou.txt"
    best = None
    if best_path.exists():
        text = best_path.read_text(encoding="utf-8").strip()
        if text:
            best = float(text) / 100.0
    return RunMetrics(
        name=name,
        seed=infer_seed(path),
        path=path,
        status="running" if best is not None else "missing",
        miou=best,
        fg_miou=None,
        per_class_iou={},
    )


def average(values: list[float | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    if len(concrete) != len(values) or not concrete:
        return None
    return sum(concrete) / len(concrete)


def group_by_seed(runs: list[RunMetrics]) -> dict[str, RunMetrics]:
    return {run.seed: run for run in runs}


def build_markdown(
    reference: list[RunMetrics],
    candidate: list[RunMetrics],
    reference_name: str,
    candidate_name: str,
    min_miou_delta: float,
    min_fg_delta: float,
) -> str:
    reference_by_seed = group_by_seed(reference)
    candidate_by_seed = group_by_seed(candidate)
    seeds = sorted(set(reference_by_seed) | set(candidate_by_seed), key=lambda item: (item == "?", item))

    lines = [
        f"# {candidate_name} vs {reference_name}",
        "",
        "| seed | ref mIoU | cand mIoU | delta | ref FG | cand FG | delta FG | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for seed in seeds:
        ref = reference_by_seed.get(seed)
        cand = candidate_by_seed.get(seed)
        ref_miou = ref.miou if ref else None
        ref_fg = ref.fg_miou if ref else None
        cand_miou = cand.miou if cand else None
        cand_fg = cand.fg_miou if cand else None
        lines.append(
            "| "
            + " | ".join(
                [
                    seed,
                    percent(ref_miou),
                    percent(cand_miou),
                    signed_percent(None if ref_miou is None or cand_miou is None else cand_miou - ref_miou),
                    percent(ref_fg),
                    percent(cand_fg),
                    signed_percent(None if ref_fg is None or cand_fg is None else cand_fg - ref_fg),
                    cand.status if cand else "missing",
                ]
            )
            + " |"
        )

    ref_avg_miou = average([run.miou for run in reference])
    ref_avg_fg = average([run.fg_miou for run in reference])
    cand_avg_miou = average([run.miou for run in candidate])
    cand_avg_fg = average([run.fg_miou for run in candidate])
    all_done = all(run.status == "done" for run in candidate)

    lines.extend(
        [
            "",
            "| group | avg mIoU | avg FG | delta mIoU | delta FG |",
            "|---|---:|---:|---:|---:|",
            f"| {reference_name} | {percent(ref_avg_miou)} | {percent(ref_avg_fg)} | - | - |",
            (
                f"| {candidate_name} | {percent(cand_avg_miou)} | {percent(cand_avg_fg)} | "
                f"{signed_percent(None if ref_avg_miou is None or cand_avg_miou is None else cand_avg_miou - ref_avg_miou)} | "
                f"{signed_percent(None if ref_avg_fg is None or cand_avg_fg is None else cand_avg_fg - ref_avg_fg)} |"
            ),
            "",
        ]
    )

    if not all_done:
        decision = "WAIT: at least one candidate report is not complete."
    elif cand_avg_miou is None or cand_avg_fg is None or ref_avg_miou is None or ref_avg_fg is None:
        decision = "WAIT: missing metrics prevent a valid decision."
    elif cand_avg_miou - ref_avg_miou >= min_miou_delta and cand_avg_fg - ref_avg_fg >= min_fg_delta:
        decision = "UPGRADE: candidate beats the reference gates."
    else:
        decision = "REJECT: candidate does not beat the reference gates."
    lines.extend(["Decision: " + decision, ""])

    class_rows: list[str] = []
    for class_name in CLASS_ORDER:
        ref_class_avg = average([run.per_class_iou.get(class_name) for run in reference])
        cand_class_avg = average([run.per_class_iou.get(class_name) for run in candidate])
        if ref_class_avg is None and cand_class_avg is None:
            continue
        class_rows.append(
            f"| {class_name} | {percent(ref_class_avg)} | {percent(cand_class_avg)} | "
            f"{signed_percent(None if ref_class_avg is None or cand_class_avg is None else cand_class_avg - ref_class_avg)} |"
        )

    if class_rows:
        lines.extend(
            [
                "| class | ref IoU | cand IoU | delta |",
                "|---|---:|---:|---:|",
                *class_rows,
                "",
            ]
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--min-miou-delta", type=float, default=0.0)
    parser.add_argument("--min-fg-delta", type=float, default=0.0)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = [read_run(path, args.reference_name) for path in args.reference]
    candidate = [read_run(path, args.candidate_name) for path in args.candidate]
    markdown = build_markdown(
        reference=reference,
        candidate=candidate,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        min_miou_delta=args.min_miou_delta,
        min_fg_delta=args.min_fg_delta,
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
