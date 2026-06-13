# ATLDSD Lesion-Focused Segmentation

Official research code and paper materials for lightweight single-stage semantic segmentation on the Apple Tree Leaf Disease Segmentation Dataset (ATLDSD).

## Repository Layout

```text
data/                   Dataset preparation utilities and data-layout notes
docx_pipeline/scripts/  Training, evaluation, reporting, and paper utility scripts
fig/                    Paper figures, Draw.io sources, and result summaries
model/                  Place trained checkpoints or exported deployment weights here
paper/                  Paper notes, experiment logs, plans, and writing materials
src/                    Model source code and reusable modules
README.md               Project overview and reproduction guide
requirements.txt        Python dependencies used by this project
```

Large local artifacts such as training runs, checkpoints, and generated reports are intentionally kept out of the release tree by `.gitignore`. Local runs still write to `outputs/` by default.

## Task

ATLDSD is treated as a six-class semantic segmentation task:

```text
0 background
1 leaf
2 rust
3 alternaria_leaf_spot
4 gray_spot
5 brown_spot
```

The paper story is not "adding modules to DeepLabV3+". The project frames apple leaf disease analysis as lesion-focused, lightweight, single-stage multi-class segmentation, where the model must preserve small lesions, fuzzy boundaries, class-imbalanced disease spots, and lesion-area evidence for severity assessment.

## Main Result

Historical baseline:

```text
DeepLabV3+ + MobileNetV3-Large
mIoU     71.72
FG mIoU  66.58
```

Current official mainline:

```text
LGC-LCSF-SP384-RepConv-BalancedPrefix-LesionDice2
dual-seed full/e80 average
mIoU     77.10
FG mIoU  72.83
```

Lesion-class IoU:

```text
rust                  84.44
alternaria_leaf_spot  58.92
gray_spot             68.27
brown_spot            56.95
```

## Method Components

```text
LGC:          local-global context modeling after ASPP
LCSF:         cross-scale lesion semantic-boundary fusion before decoder concat
SP384:        384 input resolution setting for small-lesion preservation
RepConv:      deployment-friendly decoder refinement
BalancedPrefix: mild lesion-oriented class weighting
LesionDice2:  CE supervises all classes; Dice focuses on lesion classes 2-5
```

## Dataset

The dataset is not included in this repository.

Recommended local layout:

```text
D:\dataset\ATLDSD\VOCdevkit\VOC2012
D:\dataset\ATLDSD\VOCdevkit\VOC2007
```

On Linux:

```text
/path/to/ATLDSD/VOCdevkit/VOC2012
/path/to/ATLDSD/VOCdevkit/VOC2007
```

You may set:

```bash
export ATLDSD_VOCDEVKIT_PATH=/absolute/path/to/VOCdevkit
```

## Environment

```bash
pip install -r requirements.txt
```

For server setup helpers:

```bash
chmod +x docx_pipeline/scripts/*.sh
./docx_pipeline/scripts/setup_ubuntu_env.sh cu121
```

Use `cpu`, `cu118`, `cu121`, `cu124`, or `skip` according to your environment.

## Training

Run the final LesionDice2 mainline on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File docx_pipeline\scripts\run_long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80.ps1 `
  -Seeds 11,23
```

Run the formal `+LGC+LCSF` ablation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File docx_pipeline\scripts\run_long_formal_ablation_lgc_lcsf_full_e80.ps1 `
  -Variant LGC_LCSF `
  -Seeds 11,23
```

Run the quick screening harness:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File docx_pipeline\scripts\run_fast_deeplabv3plus_screen.ps1 `
  -ExperimentName smoke_lgc_lcsf_sp384 `
  -TrainCount 128 `
  -ValCount 64 `
  -Seed 11 `
  -Epochs 24 `
  -InputSize 384 `
  -DecoderConvType repconv `
  -AttentionDecoderType sp `
  -LesionCrossScaleFusion true `
  -LesionLocalGlobalContext true `
  -ExtraArgs @("--dice-class-start","2")
```

## Evaluation and Reports

Export a segmentation report:

```powershell
python docx_pipeline/scripts/export_segmentation_report.py --help
```

Check/fuse RepConv deployment weights:

```powershell
python docx_pipeline/scripts/check_deeplab_repconv_deploy.py --help
python docx_pipeline/scripts/export_deeplab_repconv_deploy.py --help
```

Generate paper summaries and figures:

```powershell
python fig/source/gen_fig_training_results_summary.py
python fig/source/gen_fig_atldsd_formal_ablation_audit.py
python fig/source/gen_fig_atldsd_paper_evidence_audit.py
python fig/source/gen_fig_atldsd_qualitative_cases.py
```

Tracked summary figures are stored under:

```text
fig/results/
```

Source scripts and Draw.io files are stored under:

```text
fig/source/
```

## Paper Notes

The most relevant writing and experiment notes are:

```text
paper/notes/ATLDSD_context_state.md
paper/notes/ATLDSD论文问题定义与故事线_2026-06-11.md
paper/notes/ATLDSD论文图件清单_2026-06-12.md
paper/notes/ATLDSD训练接力日志_2026-06-10.md
paper/notes/ATLDSD快速模块筛选记录_2026-06-09.md
```

The current Obsidian vault also contains the synchronized writing note:

```text
D:\soft\obsidian_notion\seg\ATLDSD涨点可能性与论文故事线_2026-06-12.md
```

## Important Caveats

- Do not report single-seed best values as the official result; use dual-seed averages.
- Do not describe this repository as a detection method. It is semantic segmentation only.
- Do not claim SP384 as a module; it is a training resolution setting.
- Do not present partial ablations as formal full/e80 dual-seed evidence.
- Datasets, checkpoints, and local `outputs/` are not part of the tracked release.
