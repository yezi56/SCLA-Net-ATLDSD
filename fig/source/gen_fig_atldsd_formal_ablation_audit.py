"""Audit formal full/e80 evidence for ATLDSD paper ablations.

This script reads existing summary/report artifacts only. It does not train,
evaluate, or modify model code.

Usage:
python fig/source/gen_fig_atldsd_formal_ablation_audit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "fig" / "results"
ABLATION_CSV = SUMMARY_DIR / "paper_ablation_chain.csv"
AUDIT_CSV = SUMMARY_DIR / "paper_formal_ablation_audit.csv"
AUDIT_MD = SUMMARY_DIR / "paper_formal_ablation_audit.md"


EXPECTED = [
    {
        "id": "Baseline",
        "formal_required": "baseline reference",
        "seed11_report": "",
        "seed23_report": "",
        "note": "Historical full/e150 baseline best_mIoU row; dual-seed baseline is not available in current artifacts.",
    },
    {
        "id": "+LGC",
        "formal_required": "component ablation",
        "seed11_report": "outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s11/reports/best_miou/metrics_summary.json",
        "seed23_report": "outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s23/reports/best_miou/metrics_summary.json",
        "note": "Planned formal full/e80 paths exist in the launcher; missing reports remain fast-screen-only evidence.",
    },
    {
        "id": "+LGC+LCSF",
        "formal_required": "component ablation",
        "seed11_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s11/reports/best_miou/metrics_summary.json",
        "seed23_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s23/reports/best_miou/metrics_summary.json",
        "note": "Planned formal full/e80 paths exist in the launcher; missing reports remain fast-screen-only evidence.",
    },
    {
        "id": "+BalancedPrefix",
        "formal_required": "formal dual-seed",
        "seed11_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s11/reports/best_miou/metrics_summary.json",
        "seed23_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s23/reports/best_miou/metrics_summary.json",
        "note": "Formal full/e80 dual-seed evidence exists.",
    },
    {
        "id": "+RepConv",
        "formal_required": "formal dual-seed",
        "seed11_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s11/reports/best_miou/metrics_summary.json",
        "seed23_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s23/reports/best_miou/metrics_summary.json",
        "note": "Formal full/e80 dual-seed evidence exists.",
    },
    {
        "id": "+LesionDice2",
        "formal_required": "formal dual-seed",
        "seed11_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s11/reports/best_miou/metrics_summary.json",
        "seed23_report": "outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23/reports/best_miou/metrics_summary.json",
        "note": "Current official mainline full/e80 dual-seed evidence exists.",
    },
]


def read_ablation_rows() -> dict[str, dict[str, str]]:
    with ABLATION_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def read_metric(path_text: str) -> dict[str, float] | None:
    if not path_text:
        return None
    path = PROJECT_ROOT / path_text
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "miou": float(data["miou_all"]) * 100.0,
        "fg_miou": float(data["miou_foreground"]) * 100.0,
    }


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def build_rows() -> list[dict[str, str]]:
    paper_rows = read_ablation_rows()
    out: list[dict[str, str]] = []
    for item in EXPECTED:
        paper = paper_rows.get(item["id"], {})
        seed11 = read_metric(item["seed11_report"])
        seed23 = read_metric(item["seed23_report"])
        if seed11 and seed23:
            avg_miou = (seed11["miou"] + seed23["miou"]) / 2.0
            avg_fg = (seed11["fg_miou"] + seed23["fg_miou"]) / 2.0
            evidence_level = "formal full/e80 dual-seed"
            status = "PASS"
            action = "Use as formal row."
        elif seed11 or seed23:
            avg_miou = None
            avg_fg = None
            evidence_level = "partial full/e80"
            status = "PARTIAL"
            action = "Wait for both seed11 and seed23 before using as a formal dual-seed row."
        elif "fast-screen" in paper.get("note", ""):
            avg_miou = None
            avg_fg = None
            evidence_level = "fast-screen only"
            status = "GAP"
            action = "Keep label as fast-screen evidence; run full/e80 seed11/23 only if strict component-by-component formal ablation is required."
        else:
            avg_miou = None
            avg_fg = None
            evidence_level = "reference / non-dual"
            status = "CHECK"
            action = "Use as historical baseline reference unless the paper requires a new dual-seed baseline rerun."

        if status == "PASS":
            note = "Formal full/e80 dual-seed evidence exists."
        elif status == "PARTIAL":
            note = "One full/e80 seed exists; wait for the second seed before using as a formal dual-seed row."
        elif status == "GAP":
            note = "Missing full/e80 seed reports; keep as fast-screen-only evidence until both seeds are added."
        else:
            note = item["note"]

        out.append(
            {
                "id": item["id"],
                "paper_scale": paper.get("scale", ""),
                "paper_seeds": paper.get("seeds", ""),
                "paper_miou": paper.get("miou", ""),
                "paper_fg_miou": paper.get("fg_miou", ""),
                "formal_required": item["formal_required"],
                "evidence_level": evidence_level,
                "status": status,
                "seed11_miou": fmt(seed11["miou"] if seed11 else None),
                "seed23_miou": fmt(seed23["miou"] if seed23 else None),
                "dual_avg_miou": fmt(avg_miou),
                "seed11_fg": fmt(seed11["fg_miou"] if seed11 else None),
                "seed23_fg": fmt(seed23["fg_miou"] if seed23 else None),
                "dual_avg_fg": fmt(avg_fg),
                "action": action,
                "note": note,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "id",
        "paper_scale",
        "paper_seeds",
        "evidence_level",
        "status",
        "dual_avg_miou",
        "dual_avg_fg",
        "action",
    ]
    headers = [
        "ID",
        "Paper Scale",
        "Paper Seeds",
        "Evidence Level",
        "Status",
        "Dual Avg mIoU",
        "Dual Avg FG",
        "Action",
    ]
    lines = [
        "# ATLDSD Formal Ablation Audit",
        "",
        "This audit distinguishes formal full/e80 dual-seed evidence from fast-screen-only ablation rows. It reads existing reports only and does not train or evaluate models.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(AUDIT_CSV, rows)
    write_markdown(AUDIT_MD, rows)
    print(f"Wrote {AUDIT_CSV}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
