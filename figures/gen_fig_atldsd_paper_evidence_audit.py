"""Audit ATLDSD paper evidence artifacts.

This script reads existing summary artifacts only. It does not train,
evaluate, or modify model code.

Usage:
python figures/gen_fig_atldsd_paper_evidence_audit.py
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "atldsd" / "summary"
ABLATION_CSV = SUMMARY_DIR / "paper_ablation_chain.csv"
FORMAL_AUDIT_CSV = SUMMARY_DIR / "paper_formal_ablation_audit.csv"
DEPLOY_MD = SUMMARY_DIR / "deploy_fused_summary.md"
AUDIT_CSV = SUMMARY_DIR / "paper_evidence_audit.csv"
AUDIT_MD = SUMMARY_DIR / "paper_evidence_audit.md"

REQUIRED_ROWS = [
    "Baseline",
    "+LGC",
    "+LGC+LCSF",
    "+RepConv",
    "+BalancedPrefix",
    "+LesionDice2",
]

REQUIRED_COLUMNS = [
    "miou",
    "fg_miou",
    "rust_iou",
    "alternaria_iou",
    "gray_iou",
    "brown_iou",
    "params_m",
    "flops_g",
    "fps",
]


def read_ablation_rows() -> list[dict[str, str]]:
    with ABLATION_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_formal_audit_rows() -> dict[str, dict[str, str]]:
    if not FORMAL_AUDIT_CSV.exists():
        return {}
    with FORMAL_AUDIT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def add_row(rows: list[dict[str, str]], item: str, status: str, evidence: str, caveat: str) -> None:
    rows.append(
        {
            "item": item,
            "status": status,
            "evidence": evidence,
            "caveat": caveat,
        }
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["item", "status", "evidence", "caveat"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# ATLDSD Paper Evidence Audit",
        "",
        "This audit is generated from existing summary artifacts only. It is a paper-writing guardrail: it separates formal claims from fast-screen evidence and records RepConv deploy-fusion caveats.",
        "",
        "| Item | Status | Evidence | Caveat |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['item']} | {row['status']} | {row['evidence']} | {row['caveat']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evidence_rows: list[dict[str, str]] = []
    ablation_rows = read_ablation_rows()
    rows_by_id = {row["id"]: row for row in ablation_rows}
    formal_rows = read_formal_audit_rows()

    missing_rows = [row_id for row_id in REQUIRED_ROWS if row_id not in rows_by_id]
    add_row(
        evidence_rows,
        "Required ablation rows",
        "PASS" if not missing_rows else "FAIL",
        "present: " + ", ".join(row_id for row_id in REQUIRED_ROWS if row_id in rows_by_id),
        "missing: " + ", ".join(missing_rows) if missing_rows else "none",
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if any(rows_by_id[row_id].get(column, "") == "" for row_id in REQUIRED_ROWS if row_id in rows_by_id)
    ]
    add_row(
        evidence_rows,
        "Required metrics",
        "PASS" if not missing_columns else "FAIL",
        "mIoU, FG mIoU, four lesion IoUs, Params, FLOPs, FPS",
        "missing columns/values: " + ", ".join(missing_columns) if missing_columns else "none",
    )

    formal = rows_by_id.get("+LesionDice2", {})
    add_row(
        evidence_rows,
        "Current formal mainline",
        "PASS" if formal.get("scale") == "full/e80" and "avg" in formal.get("seeds", "") else "CHECK",
        f"+LesionDice2 {formal.get('scale', '')} {formal.get('seeds', '')}: mIoU {formal.get('miou', '')}, FG {formal.get('fg_miou', '')}",
        "Use only dual-seed average as formal result; RepConv seed23 remains best single-seed only.",
    )

    lgc_formal = formal_rows.get("+LGC", {}).get("status") == "PASS"
    lgc_lcsf_status = formal_rows.get("+LGC+LCSF", {}).get("status", "")
    lgc_lcsf_formal = lgc_lcsf_status == "PASS"
    lgc_lcsf_partial = lgc_lcsf_status == "PARTIAL"
    lgc_lcsf_fast_screen = "fast-screen" in rows_by_id.get("+LGC+LCSF", {}).get("note", "")
    lgc_lcsf_caveat = (
        "+LGC+LCSF is partial full/e80 evidence; seed23 is still required for a formal dual-seed row."
        if lgc_lcsf_partial
        else "+LGC+LCSF remains fast-screen-only until full/e80 seed11/23 reports are added."
    )
    add_row(
        evidence_rows,
        "Early module ablation boundary",
        "PASS" if lgc_formal and (lgc_lcsf_formal or lgc_lcsf_partial or lgc_lcsf_fast_screen) else "CHECK",
        "+LGC is now formal full/e80 dual-seed; +LGC+LCSF row is present",
        lgc_lcsf_caveat,
    )

    formal_ids = ["+LGC", "+BalancedPrefix", "+RepConv", "+LesionDice2"]
    formal_ok = all(
        formal_rows.get(mid, {}).get("status") == "PASS"
        or (rows_by_id[mid].get("scale") == "full/e80" and "avg" in rows_by_id[mid].get("seeds", ""))
        for mid in formal_ids
    )
    add_row(
        evidence_rows,
        "Formal ablation coverage",
        "PASS" if formal_ok and lgc_lcsf_formal else ("CHECK" if formal_ok else "FAIL"),
        "+LGC, +BalancedPrefix, +RepConv, and +LesionDice2 have full/e80 dual-seed rows",
        "Do not present +LGC+LCSF as a formal full/e80 component ablation until seed23 is added.",
    )

    deploy_text = DEPLOY_MD.read_text(encoding="utf-8") if DEPLOY_MD.exists() else ""
    deploy_ok = all(
        token in deploy_text
        for token in ["Fused Blocks", "CPU max diff", "argmax mismatch 0", "CUDA1000", "Speedup"]
    )
    add_row(
        evidence_rows,
        "RepConv deploy-fusion evidence",
        "PASS" if deploy_ok else "CHECK",
        "deploy_fused_summary.md records fused blocks, params/FLOPs after fusion, equivalence, FPS/time, and CUDA1000 stable repeats",
        "CUDA1000 supports a small speedup in the current environment; cite hardware/load and avoid generalizing beyond the final benchmark.",
    )

    add_row(
        evidence_rows,
        "Publication complexity caveat",
        "PASS" if "CUDA1000 stable" in deploy_text else "CHECK",
        "Params/FLOPs/FPS are present in paper_ablation_chain.csv; deploy_fused_summary.md contains stable CUDA1000 repeats",
        "Rerun the benchmark if hardware, CUDA/PyTorch, or GPU load changes before camera-ready.",
    )

    write_csv(AUDIT_CSV, evidence_rows)
    write_markdown(AUDIT_MD, evidence_rows)
    print(f"Wrote {AUDIT_CSV}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
