# Figures

This folder stores paper-ready figures and scripts used to regenerate result summaries.

```text
fig/results/  Tracked result figures and summary tables for the paper
fig/source/   Figure-generation scripts, Draw.io files, and source plots
```

Common commands:

```bash
python fig/source/gen_fig_training_results_summary.py
python fig/source/gen_fig_atldsd_formal_ablation_audit.py
python fig/source/gen_fig_atldsd_paper_evidence_audit.py
python fig/source/gen_fig_atldsd_qualitative_cases.py
```

Local untracked training outputs remain under `outputs/`.
