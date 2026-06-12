# Beyond Boundary1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replan ATLDSD experiments after recent failed attempts so the next runs have the highest chance of exceeding `Boundary1 = Mainline1 + LBSB`.

**Architecture:** Stop adding generic modules. Keep `DeepLabV3+ MobileNetV3-Large + component auxiliary heads + LBSB` as the anchor, first test whether 256x256 input is suppressing small lesions, then test one task-specific structural idea: component feedback refinement.

**Tech Stack:** PyTorch, DeepLabV3+, ATLDSD VOC-format data, PowerShell launchers, Ubuntu shell launchers, repository notes under `seg`.

---

## Current Evidence

The current best model is:

```text
Boundary1 = Mainline1 + LBSB
mIoU = 72.86
FG mIoU = 67.89
Accuracy = 97.97
Severity MAE = 0.01177
Grade Acc = 93.90
Params = 11.73M
FLOPs = 15.29G
FPS = 106.89
```

---

## 2026-06-09 Fast Screening Update: gray=3,rust=1.5 Promoted to Retention Gate

Discovery-tier two-seed average:

| Candidate | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg gray IoU | avg brown IoU | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Normal sampling | 36.50 | 25.51 | 0.04133 | 73.44 | 44.83 | 0.00 | 1.63 | 96.27 | baseline |
| gray=3 | 37.08 | 26.59 | 0.02553 | 79.69 | 38.89 | 8.14 | 6.65 | 92.10 | gray recovery but rust drop |
| gray=2 | 36.62 | 25.73 | 0.03197 | 78.12 | 44.37 | 0.00 | 2.66 | 90.80 | too weak |
| gray=3,rust=1.5 | 37.78 | 27.05 | 0.03078 | 81.25 | 44.76 | 2.44 | 6.22 | 95.95 | promote |

Decision:

```text
Promote gray=3,rust=1.5 to the 128/64 retention gate.

Rationale:
1. It gives the best discovery-tier mIoU and foreground mIoU.
2. It keeps rust almost unchanged versus normal sampling.
3. It keeps severity/grade gains and recovers some gray/brown signal.

If 128/64 passes:
Start a longer training run with prefix weights gray=3,rust=1.5.

If 128/64 fails:
Try gray=3,rust=2 or gray=3,rust=1.5,brown=1.2 before moving to structure modules.
```

---

## 2026-06-09 Fast Screening Update: gray=2 Rejected, Keep gray=3 Signal

gray=2 was tested to reduce the rust loss seen with gray=3.

Discovery-tier two-seed average:

| Candidate | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Normal sampling | 36.50 | 25.51 | 0.04133 | 73.44 | 44.83 | 0.00 | 1.63 | baseline |
| gray=3 sampling | 37.08 | 26.59 | 0.02553 | 79.69 | 38.89 | 8.14 | 6.65 | keep as gray-recovery signal |
| gray=2 sampling | 36.62 | 25.73 | 0.03197 | 78.12 | 44.37 | 0.00 | 2.66 | reject |

Decision:

```text
Reject gray=2 because it does not recover gray_spot.

Keep gray=3 as the active discovery signal, but do not promote it directly because rust drops too much.

Next experiment:
Use strong gray exposure with rust protection:
1. gray=3,rust=1.5 or gray=3,rust=2 prefix weights.
2. If needed, combine gray=3 with class weights that protect rust/alternaria.
3. Promotion still requires 128/64 two-seed average gains in mIoU and foreground mIoU.
```

---

## 2026-06-09 Fast Screening Update: Gray-Focused Sampling

Implemented prefix-weighted fast subset sampling:

```text
scripts/make_fast_voc_subset.py --prefix-weights gray=3
scripts/run_fast_deeplabv3plus_screen.ps1 -PrefixWeights "gray=3"
```

Discovery-tier two-seed average:

| Candidate | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Normal sampling | 36.50 | 25.51 | 0.04133 | 73.44 | 44.83 | 0.00 | 0.00 | 1.63 | 96.27 | baseline |
| gray=3 sampling | 37.08 | 26.59 | 0.02553 | 79.69 | 38.89 | 0.00 | 8.14 | 6.65 | 92.10 | discovery signal |
| Delta | +0.57 | +1.08 | -0.01580 | +6.25 | -5.95 | +0.00 | +8.14 | +5.03 | -4.17 | tune |

Decision:

```text
Keep gray-focused sampling as the active data-side path.

Do not promote gray=3 directly to 128/64 because rust drops too much and gray recovery is not stable across seeds.

Immediate next experiment:
Run gray=2 at 64/32, seeds 11 and 23.

Promotion rule:
Promote to 128/64 only if gray=2 keeps mIoU/FG/severity gains while reducing rust loss versus gray=3.
```

---

## 2026-06-09 Fast Screening Update: Boundary1-384 Not Promoted After 128/64

Boundary1-384 was expanded from the discovery tier (`64 train / 32 val`) to the retention tier (`128 train / 64 val`).

Retention-tier two-seed average:

| Candidate | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FLOPs | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boundary1-256 128/64 | 49.83 | 40.85 | 0.02484 | 84.38 | 57.02 | 28.88 | 0.00 | 30.99 | 15.294G | 83.43 | baseline |
| Boundary1-384 128/64 | 49.61 | 40.74 | 0.03208 | 82.03 | 56.50 | 28.89 | 0.00 | 33.07 | 34.407G | 77.26 | reject promotion |
| Delta | -0.22 | -0.11 | +0.00724 | -2.34 | -0.52 | +0.00 | +0.00 | +2.08 | +19.113G | -6.17 | unstable |

Decision:

```text
Do not promote Boundary1-384 to formal training yet.

Updated screening policy:
1. 64/32 is a discovery tier only.
2. 128/64 two-seed average is the retention gate.
3. A candidate must improve mIoU and foreground mIoU at the retention gate before formal training.
4. Secondary improvements, such as brown IoU, are not enough if severity and main segmentation metrics fall.

Next plan:
Focus on the class that remains unsolved: gray_spot.

Preferred next experiments:
1. Build a gray-focused fast subset or class-balanced hard subset.
2. Re-test Boundary1-256 on that subset to create a gray-sensitive baseline.
3. Only then try task-specific modules: zero-init component feedback, gray/boundary confidence gating, or light high-pass gating.
```

Failed or deprioritized attempts:

```text
B4 backbone = 65.59 mIoU
SCLP 0.7 = 68.97 mIoU
SCLP 0.3 = 69.90 mIoU
PConv = 71.76 mIoU
PConv + LBSB = 71.68 mIoU
LCAF = 72.68 mIoU
LGLC = 72.31 mIoU
Severity loss = 72.12 mIoU
```

New working hypothesis:

```text
The failure pattern says ordinary modules are not the main bottleneck.
The likely bottleneck is small lesion preservation, especially alternaria_leaf_spot, gray_spot, and brown_spot.
All major successful and failed runs used 256x256 input, so the next clean test is 384x384 without changing architecture.
```

Rules for this plan:

```text
Do not continue PConv, LCAF, LGLC, SCLP, B4, generic CAA, or LBFTLoss as next-step routes.
Change only one experimental factor per run.
Run full 150 epochs before comparing final best_miou reports.
Use actual report JSON values when updating summaries.
Do not claim final superiority from one seed if repeat runs disagree.
```

Success gates:

```text
Strong success:
  mIoU >= 73.20 and FG mIoU >= 68.10.

Practical success:
  mIoU > 72.86 and at least two of alternaria_leaf_spot, gray_spot, brown_spot improve over Boundary1.

Negative result:
  mIoU <= 72.86 with no clear small-lesion-class improvement.
```

Boundary1 per-class IoU to beat:

```text
background = 97.71
leaf = 94.05
rust = 81.47
alternaria_leaf_spot = 51.14
gray_spot = 60.20
brown_spot = 52.61
```

## 2026 Innovation Guide Upgrade

This plan was upgraded after reading:

```text
C:\Users\Administrator\Desktop\论文创新指南2026：手把手带你发论文.pdf
D:\Code\ATLDSD\seg\ATLDSD论文创新指南2026借鉴记录.md
```

Borrowed points:

```text
1. Token Mixer + FFN:
   Keep Boundary1 as the CNN anchor and change only targeted mixer/refinement modules.

2. A+B / A+B+C branches:
   Local CNN boundary features + low-resolution Mamba global context + frequency/high-frequency detail.

3. Low-resolution global modeling:
   Run Mamba only on ASPP 1/16 features, not as a full second backbone.

4. High-frequency tiny-object refinement:
   After LGLC failed, prioritize high-frequency lesion detail before another context-only module.

5. Adaptive fusion:
   Use lesion / boundary / center auxiliary predictions as gates instead of raw concat.
```

GitHub references checked:

```text
VMamba:
https://github.com/MzeroMiko/VMamba
Reference: classification/models/vmamba.py, SS2D and VSSBlock.

FreqConvMamba:
https://github.com/ccode-Rookie/FreqConvMamba
Reference: FreqConvMamba/FrequencyBlock.py and FGMM channel-split pattern.

SegMAN:
https://github.com/yunxiangfu2001/SegMAN
Reference: segmentation/mmseg/models/tmp.py, low-resolution query and local+pooled query.
```

Updated priority:

```text
Boundary1-384 remains first because it tests whether 256x256 loses small lesions.
CHFR-256 replaces CFR-256 as the first structural experiment.
CFR-only is downgraded to a component-gate ablation.
CMF-256 is a high-risk candidate after CHFR, not an immediate full dual-branch backbone.
```

## 2026-06-09 Fast Screening Addendum

After the user requested very fast module stitching, a small ATLDSD_FAST protocol was added before full 150-epoch runs:

```text
train = 64
val = 32
seeds = 11 and 23
epochs = 12
optimizer = adam
init_lr = 0.0005
class weights = 1.0 1.0 2.0 3.0 3.0 4.0
anchor = Boundary1 = Mainline1 + LBSB
```

New files:

```text
scripts/make_fast_voc_subset.py
scripts/run_fast_deeplabv3plus_screen.ps1
seg/ATLDSD快速模块筛选记录_2026-06-09.md
```

Screened PlugNPlay candidates:

```text
LSK-ASPP:
  seed11 mIoU = 37.52, FG mIoU = 26.71
  seed23 mIoU = 38.13, FG mIoU = 26.92
  two-seed average mIoU = 37.83, +1.33 over baseline average
  two-seed average FG mIoU = 26.82, +1.30 over baseline average
  Decision: promote to medium fast screen.

CPCA-ASPP:
  two-seed average mIoU = 36.97, +0.47 over baseline average
  seed23 regressed below baseline.
  Decision: weak candidate only.

EMA-ASPP:
  seed11 mIoU = 36.05, FG mIoU = 24.70
  Decision: stop.

SCSA-ASPP:
  seed11 mIoU = 35.36, FG mIoU = 23.96
  Decision: stop.
```

Plan override from this evidence:

```text
Before implementing CHFR / CMF, run LSK-ASPP medium screening:
  64/32, seed11 and seed23, 24 or 32 epochs.

If LSK-ASPP remains above the same-seed baseline:
  expand to 128/64 or 192/64 screening.

Only if expanded screening remains positive:
  run the full 150-epoch Boundary1 + LSK-ASPP experiment.

If LSK only improves rust and not alternaria_leaf_spot / gray_spot / brown_spot:
  keep it as a negative ablation, not as the final model.
```

### 2026-06-09 Fast Screening Update

Later fast screens invalidated the generic PlugNPlay route:

```text
LSK-ASPP:
  12ep two-seed positive, but 24ep two-seed average failed.

SP-ASPP:
  12ep two-seed positive, but 24ep two-seed average failed.

SP-Decoder:
  64/32 24ep passed, but 128/64 12ep failed.

SimAM/ECA/SCSE-Decoder:
  64/32 12ep failed or did not beat baseline on two-seed averages.

CHFR-v0:
  Implemented component-gated high-frequency refinement after LBSB.
  alpha=0.20 and alpha=0.05 both failed on 64/32 seed11, so the direct high-frequency residual injection design is not promoted.
```

Updated plan override:

```text
Stop ordinary attention/plugin stacking as the next route.
Keep CHFR-v0 code default-off as a negative ablation and implementation scaffold only.
Return to the PDF-informed priority order:
  1. Boundary1-384 to test the resolution bottleneck.
  2. CHFR-v1 only if redesigned with zero-initialized gamma and stricter component confidence gating.
  3. CMF only after the resolution test and conservative CHFR-v1 fail to move small-lesion classes.
```

## File Structure

Create or modify these files:

```text
scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1
  Windows launcher for Boundary1-384.

scripts/run_ubuntu_component_aux_lbsb_384_v3.sh
  Ubuntu launcher for Boundary1-384.

scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_seeded_150.ps1
  Windows repeat-run launcher for Boundary1 seed checks.

scripts/run_ubuntu_component_aux_lbsb_seeded_v3.sh
  Ubuntu repeat-run launcher for Boundary1 seed checks.

scripts/run_ubuntu.sh
  Add aliases for boundary1_384 and boundary1_seeded.

src/models/deeplabv3plus/nets/deeplabv3_plus.py
  Add ComponentFeedbackRefinementBlock and wire it after LBSB and before cls_conv.

src/models/deeplabv3plus/train.py
  Add CLI flags and pass CFR options to training and auto-report export.

src/models/deeplabv3plus/deeplab.py
  Add CFR checkpoint auto-detection for inference.

scripts/export_segmentation_report.py
  Add CFR args and pass them into the report model.

scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150.ps1
  Windows launcher for CFR-only ablation if CHFR / CMF need it.

scripts/run_ubuntu_component_aux_lbsb_cfr_v3.sh
  Ubuntu launcher for CFR-only ablation if CHFR / CMF need it.

scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_chfr_150.ps1
  Windows launcher for CHFR-256.

scripts/run_ubuntu_component_aux_lbsb_chfr_v3.sh
  Ubuntu launcher for CHFR-256.

scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cmf_150.ps1
  Windows launcher for CMF-256.

scripts/run_ubuntu_component_aux_lbsb_cmf_v3.sh
  Ubuntu launcher for CMF-256.

figures/gen_fig_training_results_summary.py
  Add completed experiment rows after reports exist.

seg/ATLDSD项目进度.md
  Record every completed run and decision.
```

---

### Task 1: Boundary1-384 Launchers

**Files:**
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_384_v3.sh`
- Modify: `D:\Code\ATLDSD\scripts\run_ubuntu.sh`

- [ ] **Step 1: Create the Windows 384 launcher**

Create `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$root = "D:\Code\ATLDSD"
$env:PYTHONPATH = "$root\src;$root\src\models\deeplabv3plus;$root\src\modules"
$python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$out = "$root\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$cmd = @(
    "$root\src\models\deeplabv3plus\train.py",
    "--cuda", "true",
    "--seed", "11",
    "--num-classes", "6",
    "--backbone", "mobilenetv3_large",
    "--pretrained", "true",
    "--downsample-factor", "16",
    "--attention-type", "none",
    "--decoder-conv-type", "standard",
    "--use-ppm", "false",
    "--input-shape", "384", "384",
    "--init-epoch", "0",
    "--freeze-epoch", "50",
    "--freeze-batch-size", "4",
    "--unfreeze-epoch", "150",
    "--unfreeze-batch-size", "2",
    "--freeze-train", "true",
    "--init-lr", "0.002",
    "--optimizer-type", "sgd",
    "--lr-decay-type", "cos",
    "--save-period", "10",
    "--eval-period", "10",
    "--dataset-name", "ATLDSD",
    "--vocdevkit-path", "D:\dataset\ATLDSD\VOCdevkit",
    "--dice-loss", "true",
    "--focal-loss", "false",
    "--component-aux", "true",
    "--component-lesion-weight", "0.4",
    "--component-boundary-weight", "0.2",
    "--component-center-weight", "0.2",
    "--lesion-boundary-sharpen", "true",
    "--lesion-boundary-sharpen-alpha", "0.25",
    "--num-workers", "0",
    "--auto-export-report", "true",
    "--report-dir", "$out\reports\best_miou",
    "--report-checkpoint", "best_miou",
    "--report-split", "val",
    "--report-fps-interval", "100",
    "--save-dir", "$out\weights",
    "--log-dir", "$out\logs",
    "--class-names", "background", "leaf", "rust", "alternaria_leaf_spot", "gray_spot", "brown_spot"
)

Set-Content -LiteralPath "$out\train_command.txt" -Value "$python $($cmd -join ' ')" -Encoding UTF8
$process = Start-Process -FilePath $python -ArgumentList $cmd -WorkingDirectory $root -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log" -PassThru -WindowStyle Hidden
Set-Content -LiteralPath "$out\train_pid.txt" -Value $process.Id -Encoding ASCII
Write-Host "Started Boundary1-384 training. PID=$($process.Id)"
Write-Host "Output: $out"
```

- [ ] **Step 2: Create the Ubuntu 384 launcher**

Create `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_384_v3.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

resolve_vocdevkit() {
  if [[ -n "${ATLDSD_VOCDEVKIT_PATH:-}" ]]; then
    echo "${ATLDSD_VOCDEVKIT_PATH}"
  elif [[ -d "${PROJECT_ROOT}/VOCdevkit" ]]; then
    echo "${PROJECT_ROOT}/VOCdevkit"
  elif [[ -d "${PROJECT_ROOT}/data/VOCdevkit" ]]; then
    echo "${PROJECT_ROOT}/data/VOCdevkit"
  elif [[ -d "/home/liuzhe/dataset/ATLDSD/VOCdevkit" ]]; then
    echo "/home/liuzhe/dataset/ATLDSD/VOCdevkit"
  else
    echo "${PROJECT_ROOT}/VOCdevkit"
  fi
}

VOCDEVKIT_PATH="$(resolve_vocdevkit)"
SPLIT_FILE="${VOCDEVKIT_PATH}/VOC2007/ImageSets/Segmentation/train.txt"
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "[error] Cannot find VOC split file: ${SPLIT_FILE}" >&2
  exit 1
fi

export ATLDSD_PROJECT_ROOT="${PROJECT_ROOT}"
export ATLDSD_VOCDEVKIT_PATH="${VOCDEVKIT_PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/models/deeplabv3plus:${PROJECT_ROOT}/src/modules${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ROOT="${PROJECT_ROOT}/outputs/atldsd/deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150"
mkdir -p "${RUN_ROOT}/weights" "${RUN_ROOT}/logs" "${RUN_ROOT}/reports/best_miou"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/models/deeplabv3plus/train.py" \
  --cuda true \
  --seed 11 \
  --num-classes 6 \
  --backbone mobilenetv3_large \
  --pretrained true \
  --downsample-factor 16 \
  --attention-type none \
  --decoder-conv-type standard \
  --use-ppm false \
  --input-shape 384 384 \
  --init-epoch 0 \
  --freeze-epoch 50 \
  --freeze-batch-size "${FREEZE_BATCH_SIZE:-4}" \
  --unfreeze-epoch "${EPOCHS:-150}" \
  --unfreeze-batch-size "${BATCH_SIZE:-2}" \
  --freeze-train true \
  --init-lr "${INIT_LR:-0.002}" \
  --optimizer-type sgd \
  --lr-decay-type cos \
  --save-period 10 \
  --eval-period 10 \
  --dataset-name ATLDSD \
  --vocdevkit-path "${VOCDEVKIT_PATH}" \
  --dice-loss true \
  --focal-loss false \
  --component-aux true \
  --component-lesion-weight 0.4 \
  --component-boundary-weight 0.2 \
  --component-center-weight 0.2 \
  --lesion-boundary-sharpen true \
  --lesion-boundary-sharpen-alpha "${LBSB_ALPHA:-0.25}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --auto-export-report true \
  --report-dir "${RUN_ROOT}/reports/best_miou" \
  --report-checkpoint best_miou \
  --report-split val \
  --report-fps-interval 100 \
  --save-dir "${RUN_ROOT}/weights" \
  --log-dir "${RUN_ROOT}/logs" \
  --class-names background leaf rust alternaria_leaf_spot gray_spot brown_spot
```

- [ ] **Step 3: Add Ubuntu aliases**

Modify `D:\Code\ATLDSD\scripts\run_ubuntu.sh` by adding this case before `component_aux_severity`:

```bash
  component_aux_lbsb_384|lbsb_384|boundary1_384)
    exec "${SCRIPT_DIR}/run_ubuntu_component_aux_lbsb_384_v3.sh"
    ;;
```

Update the usage line to include:

```text
component_aux_lbsb_384
```

- [ ] **Step 4: Validate launcher syntax**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1
```

Expected:

```text
Started Boundary1-384 training. PID is an integer process id.
Output: D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150
```

After the process starts, do not start another training job on the same GPU.

- [ ] **Step 5: Commit the launchers**

Run:

```powershell
git add D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1 D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_384_v3.sh D:\Code\ATLDSD\scripts\run_ubuntu.sh
git commit -m "Add Boundary1 384 ATLDSD launchers"
```

Expected:

```text
[<branch> <hash>] Add Boundary1 384 ATLDSD launchers
```

---

### Task 2: Boundary1-384 Evaluation Gate

**Files:**
- Read: `D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150\reports\best_miou\metrics_summary.json`
- Read: `D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150\reports\best_miou\severity_metrics.json`
- Modify: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Modify: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`

- [ ] **Step 1: Confirm the run finished**

Run:

```powershell
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150\train_stdout.log' -Tail 120
```

Expected:

```text
Epoch:150/150
[AutoReport]
```

- [ ] **Step 2: Read the report**

Run:

```powershell
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150\reports\best_miou\metrics_summary.json' -Encoding UTF8
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150\reports\best_miou\severity_metrics.json' -Encoding UTF8
```

Expected:

```text
metrics_summary.json contains miou, foreground_miou, pixel_accuracy, per_class.
severity_metrics.json contains severity_mae and grade_accuracy.
```

- [ ] **Step 3: Apply the 384 decision gate**

Use this decision exactly:

```text
If mIoU >= 73.20 and FG mIoU >= 68.10:
  Promote Boundary1-384 as the new best.
  Next structural test is CFR-384, not CFR-256.

If 72.86 < mIoU < 73.20:
  Treat 384 as promising.
  Run Boundary1 seed repeats before claiming superiority.
  Then run CFR-384.

If mIoU <= 72.86 but at least two of alternaria_leaf_spot, gray_spot, brown_spot improve:
  Treat 384 as a small-lesion diagnostic result.
  Run CFR-256 first, then CFR-384 only if CFR-256 is close.

If mIoU <= 72.86 and small-lesion classes do not improve:
  Do not keep 384 as the main route.
  Run CFR-256 next.
```

- [ ] **Step 4: Generate the summary row from report files**

Run this command after the report exists:

```powershell
$report = "D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150\reports\best_miou"
D:\soft\Anaconda\envs\Pytorch\python.exe -c "import json, pathlib; r=pathlib.Path(r'$report'); m=json.loads((r/'metrics_summary.json').read_text(encoding='utf-8')); s=json.loads((r/'severity_metrics.json').read_text(encoding='utf-8')); c=json.loads((r/'complexity.json').read_text(encoding='utf-8')); row={'id':'Boundary1-384','method':'Mainline1 + LBSB','change':'384 input resolution','status':'done','miou':round(m['miou_all']*100,2),'fg_miou':round(m['miou_foreground']*100,2),'acc':round(m['pixel_accuracy']*100,2),'severity_mae':round(s['severity_mae'],5),'grade_acc':round(s['severity_grade_accuracy']*100,2),'params_m':round(c['params']/1_000_000,2),'flops_g':round(c['flops_estimate_2x_macs']/1_000_000_000,2),'fps':round(c['fps'],2),'decision':'Boundary1-384 report imported; gate decision recorded in seg notes'}; print(row)"
```

Expected:

```text
The command prints a Python dictionary with exact metric values from the Boundary1-384 report.
```

Add the printed dictionary to `ROWS` in `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`. If the Step 3 gate produces a stronger conclusion, replace the `decision` text with one of these exact strings:

```text
new best; test CFR-384 next
promising 384 result; run seed repeats before claim
small-lesion diagnostic; run CFR-256 next
negative 384 result; run CFR-256 next
```

- [ ] **Step 5: Regenerate summary artifacts**

Run:

```powershell
python D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py
```

Expected:

```text
Wrote D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.csv
Wrote D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.md
Wrote D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_results_table.png
Wrote D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_miou_comparison.png
Wrote D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_model_tradeoff.png
```

- [ ] **Step 6: Commit the result update**

Run:

```powershell
git add D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py D:\Code\ATLDSD\outputs\atldsd\summary D:\Code\ATLDSD\seg\ATLDSD项目进度.md
git commit -m "Record Boundary1 384 ATLDSD result"
```

Expected:

```text
[<branch> <hash>] Record Boundary1 384 ATLDSD result
```

---

### Task 3: Boundary1 Seed Stability

**Files:**
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_seeded_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_seeded_v3.sh`
- Modify: `D:\Code\ATLDSD\scripts\run_ubuntu.sh`
- Modify after runs: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Modify after runs: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`

- [ ] **Step 1: Create the Windows seeded launcher**

Create `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_seeded_150.ps1`:

```powershell
param(
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$root = "D:\Code\ATLDSD"
$env:PYTHONPATH = "$root\src;$root\src\models\deeplabv3plus;$root\src\modules"
$python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$out = "$root\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed$Seed"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$cmd = @(
    "$root\src\models\deeplabv3plus\train.py",
    "--cuda", "true",
    "--seed", "$Seed",
    "--num-classes", "6",
    "--backbone", "mobilenetv3_large",
    "--pretrained", "true",
    "--downsample-factor", "16",
    "--attention-type", "none",
    "--decoder-conv-type", "standard",
    "--use-ppm", "false",
    "--input-shape", "256", "256",
    "--init-epoch", "0",
    "--freeze-epoch", "50",
    "--freeze-batch-size", "8",
    "--unfreeze-epoch", "150",
    "--unfreeze-batch-size", "4",
    "--freeze-train", "true",
    "--init-lr", "0.003",
    "--optimizer-type", "sgd",
    "--lr-decay-type", "cos",
    "--save-period", "10",
    "--eval-period", "10",
    "--dataset-name", "ATLDSD",
    "--vocdevkit-path", "D:\dataset\ATLDSD\VOCdevkit",
    "--dice-loss", "true",
    "--focal-loss", "false",
    "--component-aux", "true",
    "--component-lesion-weight", "0.4",
    "--component-boundary-weight", "0.2",
    "--component-center-weight", "0.2",
    "--lesion-boundary-sharpen", "true",
    "--lesion-boundary-sharpen-alpha", "0.25",
    "--num-workers", "0",
    "--auto-export-report", "true",
    "--report-dir", "$out\reports\best_miou",
    "--report-checkpoint", "best_miou",
    "--report-split", "val",
    "--report-fps-interval", "100",
    "--save-dir", "$out\weights",
    "--log-dir", "$out\logs",
    "--class-names", "background", "leaf", "rust", "alternaria_leaf_spot", "gray_spot", "brown_spot"
)

Set-Content -LiteralPath "$out\train_command.txt" -Value "$python $($cmd -join ' ')" -Encoding UTF8
$process = Start-Process -FilePath $python -ArgumentList $cmd -WorkingDirectory $root -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log" -PassThru -WindowStyle Hidden
Set-Content -LiteralPath "$out\train_pid.txt" -Value $process.Id -Encoding ASCII
Write-Host "Started Boundary1 seed $Seed training. PID=$($process.Id)"
Write-Host "Output: $out"
```

- [ ] **Step 2: Run seed 42**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_seeded_150.ps1 -Seed 42
```

Expected:

```text
Started Boundary1 seed 42 training. PID is an integer process id.
Output: D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed42
```

- [ ] **Step 3: Run seed 2026**

Run after seed 42 completes:

```powershell
powershell -ExecutionPolicy Bypass -File D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_seeded_150.ps1 -Seed 2026
```

Expected:

```text
Started Boundary1 seed 2026 training. PID is an integer process id.
Output: D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed2026
```

- [ ] **Step 4: Apply the stability gate**

Use this decision exactly:

```text
If seed 11, 42, and 2026 all have mIoU >= 72.50:
  Boundary1 is stable. New models must beat the mean, not just 72.86.

If one repeat is below 72.30:
  Boundary1 has high variance. Final paper must report mean ± std for Boundary1 and the final candidate.

If both repeats are below 72.30:
  Treat 72.86 as an optimistic seed. Continue CFR, but do not use single-run peak as the main claim.
```

- [ ] **Step 5: Commit the stability scripts and results**

Run after both reports are recorded:

```powershell
git add D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_seeded_150.ps1 D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_seeded_v3.sh D:\Code\ATLDSD\scripts\run_ubuntu.sh D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py D:\Code\ATLDSD\outputs\atldsd\summary D:\Code\ATLDSD\seg\ATLDSD项目进度.md
git commit -m "Record Boundary1 seed stability"
```

Expected:

```text
[<branch> <hash>] Record Boundary1 seed stability
```

---

### Task 4: Implement CHFR

**Files:**
- Modify: `D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py`
- Modify: `D:\Code\ATLDSD\src\models\deeplabv3plus\train.py`
- Modify: `D:\Code\ATLDSD\src\models\deeplabv3plus\deeplab.py`
- Modify: `D:\Code\ATLDSD\scripts\export_segmentation_report.py`
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_chfr_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_chfr_v3.sh`

Purpose:

```text
CHFR = Component-guided High-Frequency Refinement.
This replaces CFR-only as the first structural experiment.
It directly targets small lesion boundary/texture loss after LGLC showed that context-only modules do not beat Boundary1.
```

Borrowed points:

```text
PDF:
  HS-FPN high-frequency tiny-object refinement.
  FMambaIR / FreqConvMamba frequency-domain enhancement.
  alpha * A + (1 - alpha) * B adaptive fusion.

GitHub:
  D:\Code\ATLDSD\outputs\github_refs\FreqConvMamba\FreqConvMamba\FrequencyBlock.py
```

- [ ] **Step 1: Add `ComponentGuidedHighFrequencyRefinementBlock`**

In `D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py`, add this block after `LesionBoundarySharpeningBlock`:

```python
class ComponentGuidedHighFrequencyRefinementBlock(nn.Module):
    """Refine decoder features with component-gated high-frequency cues."""

    def __init__(self, channels=256, alpha=0.15, use_fft=False, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        self.use_fft = use_fft
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.component_gate = nn.Sequential(
            nn.Conv2d(3, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature, lesion_logits, boundary_logits, center_logits):
        cues = []
        for logits in (lesion_logits, boundary_logits, center_logits):
            if logits.shape[-2:] != feature.shape[-2:]:
                logits = F.interpolate(logits, size=feature.shape[-2:], mode="bilinear", align_corners=True)
            cues.append(torch.sigmoid(logits))
        gate = self.component_gate(torch.cat(cues, dim=1))
        high = feature - self.smooth(feature)
        high = self.high_proj(high)
        return feature + self.alpha * gate * high
```

Do not enable FFT in the first implementation. The first CHFR run should isolate whether component-gated high-frequency refinement helps.

- [ ] **Step 2: Wire CHFR into `DeepLab`**

Add constructor args:

```python
use_chfr=False,
chfr_alpha=0.15,
```

Add the module after `self.lbsb`:

```python
self.use_chfr = use_chfr
self.chfr = ComponentGuidedHighFrequencyRefinementBlock(256, alpha=chfr_alpha) if use_chfr else nn.Identity()
```

Update validation:

```python
elif self.use_lbsb or self.use_chfr:
    raise ValueError("LBSB and CHFR require use_component_aux=True so component logits are available.")
```

In `forward`, compute component logits before CHFR and apply:

```python
lesion_feature_logits = None
boundary_feature_logits = None
center_feature_logits = None
if self.use_lbsb:
    boundary_gate_logits = self.boundary_aux_head(x)
    x = self.lbsb(x, boundary_gate_logits)
if self.use_component_aux:
    lesion_feature_logits = self.lesion_aux_head(x)
    boundary_feature_logits = self.boundary_aux_head(x)
    center_feature_logits = self.center_aux_head(x)
if self.use_chfr:
    x = self.chfr(x, lesion_feature_logits, boundary_feature_logits, center_feature_logits)
```

- [ ] **Step 3: Add CLI/report flags**

In `train.py`, `deeplab.py`, and `export_segmentation_report.py`, add:

```text
--component-high-frequency-refine true/false
--component-high-frequency-refine-alpha 0.15
```

Python names:

```python
component_high_frequency_refine
component_high_frequency_refine_alpha
```

Pass them into `DeepLab` as:

```python
use_chfr=args.component_high_frequency_refine,
chfr_alpha=args.component_high_frequency_refine_alpha,
```

- [ ] **Step 4: Add CHFR launchers**

Create Windows and Ubuntu launchers by copying Boundary1 and changing:

```text
output directory = deeplabv3plus_mobilenetv3_large_component_aux_lbsb_chfr_150
add --component-high-frequency-refine true
add --component-high-frequency-refine-alpha 0.15
```

- [ ] **Step 5: Run CHFR-256 gate**

Decision:

```text
If CHFR-256 mIoU >= 73.20:
  CHFR becomes the main structural innovation.

If CHFR-256 mIoU > 72.86 and at least two small-lesion classes improve:
  Run CHFR-384 or seed repeat.

If CHFR-256 mIoU >= 72.60 and severity MAE < 0.01169:
  Keep CHFR as severity-friendly ablation.

If CHFR-256 mIoU < 72.60:
  Stop CHFR. Do not tune alpha repeatedly.
```

---

### Task 5: CMF-256 High-Risk Candidate

**Files:**
- Modify only if CHFR does not clearly solve the problem: `D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py`
- Create only if enabled: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cmf_150.ps1`
- Create only if enabled: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_cmf_v3.sh`

Purpose:

```text
CMF = Component-guided CNN-Mamba-Frequency Fusion.
This is the controlled version of the user's CNN+Mamba idea.
It is allowed only as a lightweight ASPP-level branch, not a full CNN+Mamba dual backbone.
```

Borrowed points:

```text
PDF:
  CNN+Mamba A+B branch.
  FFT+Mamba branch.
  A+B+C adaptive fusion.

GitHub:
  VMamba SS2D / VSSBlock.
  FreqConvMamba FrequencyBlock.
  SegMAN low-resolution query pattern.
```

- [ ] **Step 1: Enforce CMF constraints**

CMF must satisfy:

```text
Input feature: ASPP output only.
Spatial size: 1/16 resolution only.
Mamba depth: 1 or 2 blocks only.
No replacement of MobileNetV3-Large.
No second full encoder.
No CMF-384 unless CMF-256 beats Boundary1.
```

- [ ] **Step 2: Implement only after CHFR gate**

Do not implement CMF if:

```text
Boundary1-384 already strongly succeeds.
CHFR-256 strongly succeeds.
Compute budget is not available for one high-risk run.
```

CMF decision:

```text
If CMF-256 <= 72.86:
  Stop. Do not tune Mamba depth, alpha, or frequency groups.

If CMF-256 > 72.86:
  Run CMF-384 or seed repeat.
```

---

### Task 6: Conditional CFR-Only Ablation

Run this task only if CHFR or CMF needs a clean ablation proving that component feedback alone is weaker than component-guided high-frequency / Mamba-frequency fusion.

- [ ] **Step 1: Add the CFR block**

In `D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py`, insert this class after `LesionBoundarySharpeningBlock`:

```python
class ComponentFeedbackRefinementBlock(nn.Module):
    """Use auxiliary component predictions to refine decoder features."""

    def __init__(self, channels=256, alpha=0.2, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        hidden = max(channels // 4, 16)
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature, lesion_logits, boundary_logits, center_logits):
        cues = []
        for logits in (lesion_logits, boundary_logits, center_logits):
            if logits.shape[-2:] != feature.shape[-2:]:
                logits = F.interpolate(logits, size=feature.shape[-2:], mode="bilinear", align_corners=True)
            cues.append(torch.sigmoid(logits))
        gate = self.feedback_gate(torch.cat(cues, dim=1))
        return feature + self.alpha * self.refine(feature) * gate
```

- [ ] **Step 2: Add CFR constructor arguments**

In `DeepLab.__init__`, add:

```python
use_cfr=False,
cfr_alpha=0.2,
```

After the existing flags, add:

```python
self.use_cfr = use_cfr
```

After `self.lbsb = ...`, add:

```python
self.cfr = ComponentFeedbackRefinementBlock(256, alpha=cfr_alpha) if use_cfr else nn.Identity()
```

Replace the current component-aux validation block with:

```python
if self.use_component_aux:
    self.lesion_aux_head = nn.Conv2d(256, 1, 1, stride=1)
    self.boundary_aux_head = nn.Conv2d(256, 1, 1, stride=1)
    self.center_aux_head = nn.Conv2d(256, 1, 1, stride=1)
elif self.use_lbsb or self.use_cfr:
    raise ValueError("LBSB and CFR require use_component_aux=True so component logits are available.")
```

- [ ] **Step 3: Replace the forward tail**

In `DeepLab.forward`, replace lines from the current `if self.use_lbsb:` block through the returned dict with:

```python
lesion_feature_logits = None
boundary_feature_logits = None
center_feature_logits = None
if self.use_lbsb:
    boundary_gate_logits = self.boundary_aux_head(x)
    x = self.lbsb(x, boundary_gate_logits)
if self.use_component_aux:
    lesion_feature_logits = self.lesion_aux_head(x)
    boundary_feature_logits = self.boundary_aux_head(x)
    center_feature_logits = self.center_aux_head(x)
if self.use_cfr:
    x = self.cfr(x, lesion_feature_logits, boundary_feature_logits, center_feature_logits)
logits = self.cls_conv(x)
logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
if not self.use_component_aux:
    return logits
return {
    "logits": logits,
    "lesion_logits": F.interpolate(lesion_feature_logits, size=(height, width), mode="bilinear", align_corners=True),
    "boundary_logits": F.interpolate(boundary_feature_logits, size=(height, width), mode="bilinear", align_corners=True),
    "center_logits": F.interpolate(center_feature_logits, size=(height, width), mode="bilinear", align_corners=True),
}
```

- [ ] **Step 4: Add train CLI flags**

In `D:\Code\ATLDSD\src\models\deeplabv3plus\train.py`, after the LGLC args, add:

```python
parser.add_argument("--component-feedback-refine", type=str2bool, default=False, help="Use component feedback refinement before the final segmentation head.")
parser.add_argument("--component-feedback-refine-alpha", type=float, default=0.2)
```

In the `DeepLab(...)` call, add:

```python
use_cfr=args.component_feedback_refine,
cfr_alpha=args.component_feedback_refine_alpha,
```

In `auto_export_report`, add these command args after `--lesion-local-global-context-alpha`:

```python
"--component-feedback-refine",
str(args.component_feedback_refine).lower(),
"--component-feedback-refine-alpha",
str(args.component_feedback_refine_alpha),
```

In `show_config`, add:

```python
component_feedback_refine=args.component_feedback_refine,
component_feedback_refine_alpha=args.component_feedback_refine_alpha,
```

- [ ] **Step 5: Add inference support**

In `D:\Code\ATLDSD\src\models\deeplabv3plus\deeplab.py`, add defaults:

```python
"component_feedback_refine": "auto",
"component_feedback_refine_alpha": 0.2,
```

Add this resolver after `_resolve_use_lglc`:

```python
def _resolve_use_cfr(self, checkpoint):
    if isinstance(self.component_feedback_refine, bool):
        return self.component_feedback_refine
    if isinstance(self.component_feedback_refine, str) and self.component_feedback_refine.lower() != "auto":
        return self.component_feedback_refine.lower() in {"true", "1", "yes", "y"}
    return any(key.startswith("cfr.") for key in checkpoint.keys())
```

In `generate`, add:

```python
use_cfr = self._resolve_use_cfr(checkpoint)
```

Pass to `DeepLab`:

```python
use_cfr=use_cfr,
cfr_alpha=float(self.component_feedback_refine_alpha),
```

Update the printed format string so it includes:

```text
cfr={}
```

and pass `use_cfr` as the matching value.

- [ ] **Step 6: Add report support**

In `D:\Code\ATLDSD\scripts\export_segmentation_report.py`, after the LGLC args, add:

```python
parser.add_argument("--component-feedback-refine", type=str2bool, default=False)
parser.add_argument("--component-feedback-refine-alpha", type=float, default=0.2)
```

Pass these values to `Model(...)`:

```python
use_cfr=args.component_feedback_refine,
cfr_alpha=args.component_feedback_refine_alpha,
```

Add both fields to the saved config dictionary:

```python
"component_feedback_refine": args.component_feedback_refine,
"component_feedback_refine_alpha": args.component_feedback_refine_alpha,
```

- [ ] **Step 7: Run syntax checks**

Run:

```powershell
D:\soft\Anaconda\envs\Pytorch\python.exe -m py_compile D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py D:\Code\ATLDSD\src\models\deeplabv3plus\train.py D:\Code\ATLDSD\src\models\deeplabv3plus\deeplab.py D:\Code\ATLDSD\scripts\export_segmentation_report.py
```

Expected:

```text
No output and exit code 0.
```

- [ ] **Step 8: Run a CFR forward check**

Run:

```powershell
$env:PYTHONPATH = "D:\Code\ATLDSD\src;D:\Code\ATLDSD\src\models\deeplabv3plus;D:\Code\ATLDSD\src\modules"
D:\soft\Anaconda\envs\Pytorch\python.exe -c "import torch; from nets.deeplabv3_plus import DeepLab; model=DeepLab(num_classes=6, backbone='mobilenetv3_large', pretrained=False, use_component_aux=True, use_lbsb=True, use_cfr=True); out=model(torch.randn(1,3,256,256)); assert out['logits'].shape==(1,6,256,256); assert out['lesion_logits'].shape==(1,1,256,256); assert out['boundary_logits'].shape==(1,1,256,256); assert out['center_logits'].shape==(1,1,256,256); print('CFR forward test passed')"
```

Expected:

```text
CFR forward test passed
```

- [ ] **Step 9: Commit CFR implementation**

Run:

```powershell
git add D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py D:\Code\ATLDSD\src\models\deeplabv3plus\train.py D:\Code\ATLDSD\src\models\deeplabv3plus\deeplab.py D:\Code\ATLDSD\scripts\export_segmentation_report.py
git commit -m "Add component feedback refinement for ATLDSD"
```

Expected:

```text
[<branch> <hash>] Add component feedback refinement for ATLDSD
```

---

### Task 7: Conditional CFR-Only Training

**Files:**
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_cfr_v3.sh`
- Modify: `D:\Code\ATLDSD\scripts\run_ubuntu.sh`

- [ ] **Step 1: Create the Windows CFR-256 launcher**

Create `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150.ps1` by copying the Boundary1 launcher and changing only these values:

```powershell
$out = "$root\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150"
```

Add these args after `--lesion-boundary-sharpen-alpha`:

```powershell
"--component-feedback-refine", "true",
"--component-feedback-refine-alpha", "0.2",
```

Change the final message to:

```powershell
Write-Host "Started CFR-256 training. PID=$($process.Id)"
```

- [ ] **Step 2: Create the Ubuntu CFR-256 launcher**

Create `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_cfr_v3.sh` by copying `run_ubuntu_component_aux_lbsb_v3.sh` and changing:

```bash
RUN_ROOT="${PROJECT_ROOT}/outputs/atldsd/deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150"
```

Add:

```bash
  --component-feedback-refine true \
  --component-feedback-refine-alpha "${CFR_ALPHA:-0.2}" \
```

- [ ] **Step 3: Add Ubuntu aliases**

Modify `D:\Code\ATLDSD\scripts\run_ubuntu.sh`:

```bash
  component_aux_lbsb_cfr|lbsb_cfr|cfr|refine1)
    exec "${SCRIPT_DIR}/run_ubuntu_component_aux_lbsb_cfr_v3.sh"
    ;;
```

Update the usage line to include:

```text
component_aux_lbsb_cfr
```

- [ ] **Step 4: Run CFR-256**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150.ps1
```

Expected:

```text
Started CFR-256 training. PID is an integer process id.
Output: D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150
```

- [ ] **Step 5: Apply the CFR-256 gate**

Use this decision exactly:

```text
If mIoU >= 73.20:
  CFR is the main structural innovation.
  Run seed repeats for Boundary1 and CFR-256.

If 72.86 < mIoU < 73.20:
  CFR is promising.
  Run CFR-384 only if Boundary1-384 was not clearly negative.

If mIoU <= 72.86 but severity MAE < 0.01169 and mIoU >= 72.60:
  CFR is a severity-friendly ablation, not final segmentation model.

If mIoU < 72.60:
  Stop CFR at 256. Do not tune alpha repeatedly.
```

---

### Task 8: CFR-384 Conditional Training

**Files:**
- Create only if gate passes: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_384_150.ps1`
- Create only if gate passes: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_cfr_384_v3.sh`
- Modify only if gate passes: `D:\Code\ATLDSD\scripts\run_ubuntu.sh`

- [ ] **Step 1: Check whether CFR-384 is allowed**

Run CFR-384 only if one of these is true:

```text
Boundary1-384 mIoU > 72.86.
CFR-256 mIoU > 72.86.
Boundary1-384 improves at least two small-lesion classes while staying above 72.60 mIoU.
```

Do not run CFR-384 if both 384 and CFR-256 are negative.

- [ ] **Step 2: Create CFR-384 launcher**

Create it by copying the CFR-256 launcher and changing:

```text
--input-shape 384 384
--freeze-batch-size 4
--unfreeze-batch-size 2
--init-lr 0.002
output dir = deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_384_150
```

- [ ] **Step 3: Apply the final gate**

Use this decision exactly:

```text
If CFR-384 mIoU >= 73.20:
  Final candidate = Mainline1 + LBSB + CFR at 384.

If CFR-384 mIoU > 72.86 but < 73.20:
  Final candidate depends on seed mean.
  Repeat Boundary1 and CFR-384.

If CFR-384 mIoU <= 72.86:
  Final model remains Boundary1 unless Boundary1-384 had already won.
```

---

### Task 9: Final Reporting

**Files:**
- Modify: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.csv`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.md`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_results_table.png`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_miou_comparison.png`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_model_tradeoff.png`
- Modify: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`

- [ ] **Step 1: Update the paper story**

Use this story if 384 wins:

```text
Component-aware lesion segmentation benefits most from preserving small-lesion resolution.
Component heads and LBSB provide the structure; 384 input prevents small lesions from being lost before decoding.
```

Use this story if CFR wins:

```text
Auxiliary lesion, boundary, and center predictions are not only training regularizers.
CFR feeds component cues back into the final segmentation feature, improving task-specific lesion masks.
```

Use this story if nothing beats Boundary1:

```text
Boundary1 remains the final model.
The paper should report negative evidence honestly: generic context, PConv, LCAF, SCLP, B4, and severity loss do not beat component supervision plus boundary sharpening on ATLDSD.
```

- [ ] **Step 2: Sync the Obsidian notes**

Run:

```powershell
Copy-Item -LiteralPath "D:\Code\ATLDSD\seg\ATLDSD项目进度.md" -Destination "D:\soft\obsidian_notion\seg\ATLDSD项目进度.md" -Force
Copy-Item -LiteralPath "D:\Code\ATLDSD\docs\superpowers\plans\2026-06-08-beyond-boundary1-replan.md" -Destination "D:\soft\obsidian_notion\seg\ATLDSD超越Boundary1重规划_2026-06-08.md" -Force
```

- [ ] **Step 3: Commit the final reporting update**

Run:

```powershell
git add D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py D:\Code\ATLDSD\outputs\atldsd\summary D:\Code\ATLDSD\seg D:\Code\ATLDSD\docs\superpowers\plans\2026-06-08-beyond-boundary1-replan.md
git commit -m "Finalize ATLDSD beyond Boundary1 plan and results"
```

Expected:

```text
[<branch> <hash>] Finalize ATLDSD beyond Boundary1 plan and results
```

---

## Self-Review

Spec coverage:

```text
The plan addresses the failed attempts, stops unproductive branches, prioritizes the cleanest next experiment, and keeps one task-specific structure route for exceeding Boundary1.
```

Placeholder scan:

```text
No implementation step depends on undefined files or unnamed commands.
Metric-update steps require actual report values before commit.
```

Type consistency:

```text
The planned CFR flags are component_feedback_refine and component_feedback_refine_alpha in Python, exposed as --component-feedback-refine and --component-feedback-refine-alpha in CLI scripts.
```

---

## 2026-06-09 Fast Screening Update: Boundary1-384 Promoted

The fast 64/32, 12-epoch screen shows that increasing Boundary1 input size from 256 to 384 is currently the strongest positive path.

Two-seed average:

| Candidate | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FLOPs | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boundary1-256 | 36.50 | 25.51 | 0.04133 | 73.44 | 44.83 | 0.00 | 0.00 | 1.63 | 15.294G | 96.27 | baseline |
| Boundary1-384 | 42.54 | 32.61 | 0.03463 | 79.69 | 56.25 | 13.32 | 0.00 | 10.97 | 34.407G | 73.46 | promote |
| Delta | +6.03 | +7.10 | -0.00670 | +6.25 | +11.42 | +13.32 | +0.00 | +9.34 | +19.113G | -22.81 | clear gain |

Decision:

```text
Promote Boundary1-384 to the next fast tier.

Rationale:
1. It improves both seeds, not just one lucky split.
2. It improves segmentation and severity together.
3. It adds no parameters; the cost is compute/FPS only, acceptable for the current gain-chasing phase.
4. It supports the paper story that component-aware ATLDSD segmentation is resolution-limited: component heads and LBSB provide structure, while 384 input preserves small-lesion evidence before decoding.

Immediate next task:
Run Boundary1-384 with 128 train / 64 val, 12 epochs, seeds 11 and 23.

If it wins:
Move formal training candidate to 384 input and treat gray_spot recovery as the next targeted module/sampling problem.

If it fails:
Keep 384 as a promising small-subset signal but do not change formal training yet; try gray-focused balanced subset before new modules.
```

## 2026-06-09 Update: Seed Is Gate, SP-Decoder Is Module Candidate

Correction from the user: changing seed is not a module experiment. From this point on, seed is only a robustness gate. A candidate must be named by the actual change: module, insertion point, loss, resolution, or data strategy.

Fast decision:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg gray IoU | avg brown IoU | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boundary1 baseline | 128/64 | 12 | 49.83 | 40.85 | 0.02484 | 84.38 | 57.02 | 0.00 | 30.99 | 83.43 | baseline |
| SP-Decoder | 128/64 | 24 | 58.64 | 51.14 | 0.01822 | 90.62 | 71.71 | 44.75 | 29.81 | 83.61 | keep as current strongest module |
| Delta | - | - | +8.80 | +10.30 | -0.00662 | +6.25 | +14.69 | +44.75 | -1.18 | +0.18 | strong gain |

Decision:

```text
Keep SP-Decoder as the current main candidate.
Do not start long training yet.
Continue fast module/data combinations until no higher-gain route is available.
```

Immediate next experiments:

```text
1. SP-Decoder + 384 input, 128/64 fast screen.
2. SP-Decoder + gray/brown/alternaria prefix sampling, 64/32 or 128/64 depending on speed.
3. PlugNPlay search should prioritize segmentation context modules with directional pooling, boundary preservation, or small-object protection. Generic decoder attention is paused after ECA/SCSE/SimAM failed.
```
## 2026-06-09 Update: SP-Decoder + 384 Fast Combo

SP-Decoder was tested with 384 input on the 64/32 fast screen.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-256 | 64/32 | 24 | 49.88 | 40.90 | 0.03128 | 79.69 | 61.56 | 23.02 | 22.87 | 10.68 | 71.90 | current module baseline |
| SP-384 | 64/32 | 24 | 51.63 | 42.91 | 0.03074 | 81.25 | 66.03 | 25.84 | 19.15 | 15.97 | 71.16 | promote to 128/64 gate |
| Delta | - | - | +1.75 | +2.02 | -0.00054 | +1.56 | +4.47 | +2.82 | -3.72 | +5.30 | -0.74 | small stable gain |

Decision:

```text
Promote SP-384 to a 128/64 fast gate.
Keep SP-256 128/64 ep24 as the current strongest verified module until SP-384 also passes the larger gate.
```
## 2026-06-09 Update: SP-384 Passes 128/64 Gate

SP-Decoder with 384 input has now passed the 128/64 fast gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FLOPs | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boundary1 baseline | 128/64 | 12 | 49.83 | 40.85 | 0.02484 | 84.38 | 57.02 | 28.88 | 0.00 | 30.99 | 15.29G | baseline |
| SP-256 | 128/64 | 24 | 58.64 | 51.14 | 0.01822 | 90.62 | 71.71 | 19.28 | 44.75 | 29.81 | 16.80G | severity/gray backup |
| SP-384 | 128/64 | 24 | 60.34 | 53.15 | 0.01939 | 87.50 | 72.90 | 32.64 | 31.52 | 38.23 | 37.75G | current segmentation-main candidate |

Decision:

```text
Promote SP-384 as the current segmentation-main candidate.
Do not start long training yet because SP-384 trades away some gray IoU and Grade Acc relative to SP-256.
Next fast screen should target gray/severity compensation under SP-384.
```

Next:

```text
1. SP-384 + gray-focused sampling, 64/32 first.
2. If mIoU remains strong and gray/Grade recover, promote to 128/64.
3. If it hurts mIoU, keep vanilla SP-384 and search PlugNPlay/PDF-aligned boundary or frequency modules.
```

## 2026-06-09 Update: Seed Is Not The Module

User correction accepted: changing seed is only a stability check. The actual experimental unit must be a named module, insertion point, resolution, loss, or data strategy.

SP-384 + gray-focused sampling was tested as a data compensation strategy:

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 64/32 | 24 | 51.18 | 42.35 | 0.03155 | 81.25 | 64.97 | 21.43 | 26.01 | 11.54 | 70.27 | reference |
| SP-384 + gray=2 sampling | 11 | 64/32 | 24 | 48.73 | 39.74 | 0.02953 | 84.38 | 65.85 | 9.00 | 28.85 | 10.33 | 73.80 | reject |

Decision:

```text
Reject SP-384 + gray=2 sampling.
It slightly improves severity grade and gray IoU versus the same 64/32 SP-384 seed11 reference, but loses -2.45 mIoU, -2.61 FG mIoU, and collapses alternaria. Do not spend seed23 on this failed compensation route.
Next work returns to concrete modules: PlugNPlay/PDF-aligned boundary, high-frequency, and spatial context modules.
```

Next concrete module screen:

```text
Run CAA after ASPP + SP at decoder, input 384, train/val 64/32, seed11, 24 epochs.
Rationale: CAA is a context-anchor spatial module with long horizontal/vertical aggregation. It complements SP-Decoder instead of merely changing seed or sampling. If it beats vanilla SP-384 seed11 on mIoU/FG without destroying gray/Grade, then run seed23.
```

## 2026-06-09 Update: CAA-ASPP + SP Rejected

CAA after ASPP was tested as a concrete module combination with SP-Decoder and 384 input.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | FLOPs | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 64/32 | 24 | 51.18 | 42.35 | 0.03155 | 81.25 | 64.97 | 21.43 | 26.01 | 11.54 | 37.75G | 70.27 | reference |
| CAA-ASPP + SP-384 | 11 | 64/32 | 24 | 48.99 | 40.24 | 0.03541 | 78.12 | 71.66 | 19.76 | 26.58 | 0.02 | 37.91G | 64.45 | reject |

Decision:

```text
Reject CAA-ASPP + SP-384.
The CAA context anchor increases rust but hurts mIoU, FG mIoU, severity, FPS, and collapses brown_spot. Do not run seed23.
Next concrete candidate should target high-frequency/detail refinement more directly rather than broad context anchoring.
```

## 2026-06-09 Update: HFR + SP-384 Rejected

Component high-frequency refinement was tested as a concrete detail-recovery module on top of SP-384.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | FLOPs | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 64/32 | 24 | 51.18 | 42.35 | 0.03155 | 81.25 | 64.97 | 21.43 | 26.01 | 11.54 | 37.75G | 70.27 | reference |
| HFR + SP-384 | 11 | 64/32 | 24 | 49.20 | 40.03 | 0.03082 | 81.25 | 72.86 | 26.30 | 3.76 | 9.52 | 39.07G | 63.21 | reject |

Decision:

```text
Reject HFR + SP-384.
It improves rust and alternaria, but the main mIoU/FG drop and gray_spot collapse make it unsafe. Do not run seed23.
Next candidate: lesion cross-scale fusion with SP-384, because the current failure pattern suggests a need for fusion rather than sharper high-frequency suppression.
```

## 2026-06-09 Update: LCSF + SP-384 Passes Seed11 Gate

Lesion cross-scale fusion was tested as the next concrete fusion module on top of SP-384.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | FLOPs | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 64/32 | 24 | 51.18 | 42.35 | 0.03155 | 81.25 | 64.97 | 21.43 | 26.01 | 11.54 | 37.75G | 70.27 | reference |
| LCSF + SP-384 | 11 | 64/32 | 24 | 52.55 | 44.22 | 0.03515 | 81.25 | 69.35 | 28.64 | 23.49 | 13.85 | 38.28G | 59.06 | run seed23 |

Decision:

```text
LCSF + SP-384 passes the seed11 gate for segmentation.
It improves mIoU, FG mIoU, rust, alternaria, and brown versus vanilla SP-384 seed11. The cost is worse severity MAE, slightly lower gray, and slower FPS.
Run seed23 before promoting to 128/64.
```

## 2026-06-09 Update: LCSF + SP-384 Passes 64/32 Dual-Seed Gate

LCSF + SP-384 was confirmed on seed23 and passed the 64/32 robustness gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FLOPs | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 64/32 | 24 | 51.63 | 42.91 | 0.03074 | 81.25 | 66.03 | 25.84 | 19.15 | 15.97 | 37.75G | 71.16 | reference |
| LCSF + SP-384 | 64/32 | 24 | 54.31 | 46.27 | 0.02959 | 84.38 | 67.25 | 29.65 | 24.01 | 24.11 | 38.28G | 62.23 | promote to 128/64 |
| Delta | - | - | +2.68 | +3.36 | -0.00115 | +3.12 | +1.22 | +3.81 | +4.87 | +8.14 | +0.53G | -8.93 | stable gain |

Decision:

```text
Promote LCSF + SP-384 to a 128/64 gate.
This is the first concrete module combination after SP-384 that improves both seeds on average. It aligns with the PDF strategy of targeted fusion rather than random attention stacking.
Run seed11 at 128/64 first. If it beats SP-384 128/64 seed11 or clearly keeps the large baseline gain while improving weak classes, then run seed23.
```

## 2026-06-09 Update: LCSF + SP-384 Fails 128/64 Seed11 Gate

LCSF + SP-384 was promoted to 128/64 after passing the small dual-seed gate, but it did not beat the current SP-384 main candidate at the larger gate.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | leaf IoU | rust IoU | alternaria IoU | gray IoU | brown IoU | FLOPs | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 128/64 | 24 | 62.41 | 55.67 | 0.01949 | 87.50 | 90.48 | 78.07 | 36.27 | 47.53 | 26.01 | 37.75G | - | main reference |
| LCSF + SP-384 | 11 | 128/64 | 24 | 59.44 | 52.03 | 0.02093 | 85.94 | 90.79 | 73.67 | 35.10 | 34.56 | 26.04 | 38.28G | 68.94 | reject at 128/64 |

Decision:

```text
Do not run LCSF seed23 at 128/64.
Keep LCSF as a useful small-sample signal, but it is not the new mainline because it fails the larger gate against SP-384.
Next candidate should not be another broad fusion of the same kind. Try lesion local-global context with SP-384, which targets disease-local detail plus whole-leaf context more directly.
```

## 2026-06-09 Update: LGC + SP-384 Passes Seed11 Gate

Lesion local-global context was tested with SP-384 as a concrete context module rather than another seed-only change.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | FLOPs | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 64/32 | 24 | 51.18 | 42.35 | 0.03155 | 81.25 | 64.97 | 21.43 | 26.01 | 11.54 | 37.75G | 70.27 | reference |
| LGC + SP-384 | 11 | 64/32 | 24 | 57.53 | 50.18 | 0.02742 | 84.38 | 69.65 | 31.15 | 47.95 | 16.35 | 37.84G | 62.78 | run seed23 |

Decision:

```text
LGC + SP-384 passes seed11 strongly.
It improves mIoU, FG mIoU, severity MAE, Grade Acc, rust, alternaria, gray, and brown against SP-384 seed11. This directly fixes the gray weakness that remained after SP-384.
Run seed23 immediately.
```

## 2026-06-09 Update: LGC + SP-384 Passes 64/32 Dual-Seed Gate

LGC + SP-384 was confirmed with seed23 and is now the strongest small-gate module combination after SP-384.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | avg FLOPs | avg FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 64/32 | 24 | 51.63 | 42.91 | 0.03074 | 81.25 | 66.03 | 25.84 | 19.15 | 15.97 | 37.75G | 71.16 | reference |
| LGC + SP-384 | 64/32 | 24 | 56.42 | 48.71 | 0.02855 | 85.94 | 69.25 | 32.45 | 34.59 | 19.43 | 37.84G | 65.07 | promote to 128/64 |
| Delta | - | - | +4.80 | +5.80 | -0.00219 | +4.69 | +3.22 | +6.61 | +15.44 | +3.47 | +0.09G | -6.09 | strong stable gain |

Decision:

```text
Promote LGC + SP-384 to 128/64.
Compared with LCSF, LGC has stronger small-gate gains and directly repairs SP-384's gray weakness. It also keeps FLOPs almost unchanged relative to SP-384.
Run 128/64 seed11 first. If it beats SP-384 seed11 or keeps the segmentation gain with clear gray/severity recovery, run seed23.
```

## 2026-06-09 Update: LGC + SP-384 Passes 128/64 Seed11 Gate

LGC + SP-384 beat the current SP-384 main candidate on the larger 128/64 seed11 gate.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | leaf IoU | rust IoU | alternaria IoU | gray IoU | brown IoU | FLOPs | FPS | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 11 | 128/64 | 24 | 62.41 | 55.67 | 0.01949 | 87.50 | 90.48 | 78.07 | 36.27 | 47.53 | 26.01 | 37.75G | - | current main reference |
| LGC + SP-384 | 11 | 128/64 | 24 | 64.60 | 58.22 | 0.01980 | 85.94 | 90.80 | 76.07 | 42.21 | 49.15 | 32.89 | 37.84G | 68.34 | run seed23 |

Decision:

```text
Run LGC + SP-384 128/64 seed23.
This is now the highest single-seed fast result. It improves the main segmentation target and weak classes with almost no FLOPs increase. Watch severity grade on seed23 because Grade Acc is slightly below SP-384 seed11.
```

## 2026-06-09 Update: LGC + LCSF + SP-384 Becomes New Fast Main Candidate

After LGC passed the 128/64 seed11 gate, LCSF was re-tested as a complementary cross-scale lesion gate on top of the LGC + SP-384 candidate. The combined module passed both the 64/32 and 128/64 dual-seed gates.

64/32 gate:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 64/32 | 24 | 51.62 | 42.91 | 0.03074 | 81.25 | 66.03 | 25.84 | 19.14 | 15.97 | reference |
| LGC + SP-384 | 64/32 | 24 | 56.42 | 48.72 | 0.02855 | 85.94 | 69.24 | 32.97 | 34.80 | 19.44 | strong candidate |
| LGC + LCSF + SP-384 | 64/32 | 24 | 56.71 | 49.04 | 0.02672 | 84.38 | 71.74 | 30.04 | 34.41 | 21.72 | weak positive; promote |

128/64 gate:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 vanilla | 128/64 | 24 | 60.33 | 53.15 | 0.01940 | 87.50 | 72.90 | 32.64 | 31.52 | 38.23 | old mainline |
| LGC + SP-384 | 128/64 | 24 | 62.39 | 55.60 | 0.02458 | 86.72 | 73.96 | 39.14 | 35.53 | 38.68 | strong candidate |
| LGC + LCSF + SP-384 | 128/64 | 24 | 63.42 | 56.89 | 0.01942 | 89.84 | 72.26 | 43.56 | 34.98 | 43.32 | new main candidate |

Decision:

```text
Promote LGC + LCSF + SP-384 to the current fast-screen main candidate.
Relative to LGC + SP-384, it improves avg mIoU by +1.03, FG mIoU by +1.29, Severity MAE by -0.00516, Grade Acc by +3.12, alternaria IoU by +4.42, and brown IoU by +4.64.
The tradeoff is rust IoU -1.70 and gray IoU -0.55, but gray remains above the SP-384 baseline and the seed23 severity metrics are much stronger.

Next gate:
1. Test LGC + LCSF + SP-384 with gray=3,rust=1.5 prefix sampling on 128/64.
2. If the sampling variant keeps the segmentation gain without hurting rust/gray, promote it to a 192/96 or long-run candidate.
3. If sampling destabilizes class balance, keep the current non-weighted LGC + LCSF + SP-384 for long-run training.
```

## 2026-06-09 Update: gray=3,rust=1.5 Sampling Helps Severity/Gray but Does Not Replace Mainline

The current best fast candidate, LGC + LCSF + SP-384, was tested with prefix sampling `gray=3,rust=1.5` on the 128/64 dual-seed gate.

| Candidate | sampling | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC + LCSF + SP-384 | normal | 128/64 | 24 | 63.42 | 56.89 | 0.01942 | 89.84 | 72.26 | 43.56 | 34.98 | 43.32 | keep as mIoU mainline |
| LGC + LCSF + SP-384 | gray=3,rust=1.5 | 128/64 | 24 | 62.96 | 56.28 | 0.01638 | 91.40 | 77.12 | 27.54 | 51.76 | 34.18 | severity/gray side branch |

Decision:

```text
Do not replace the normal-sampling mainline.
gray=3,rust=1.5 strongly improves severity MAE, Grade Acc, rust, and gray, but it suppresses alternaria and brown enough to reduce mIoU/FG.
Keep this as a severity/gray branch for analysis or a possible class-balanced loss idea.
For further sampling, only try a milder balance such as gray=2,rust=1.5,alternaria=1.2,brown=1.2. Otherwise promote normal LGC + LCSF + SP-384 to a larger 192/96 gate or long-run training.
```

## 2026-06-09 Update: Balanced Prefix Sampling Becomes the New Fast Mainline

The milder prefix sampling variant `gray=2,rust=1.5,alternaria=1.2,brown=1.2` was tested on top of LGC + LCSF + SP-384 at 128/64 for seeds 11 and 23. It fixes the class-suppression problem seen in `gray=3,rust=1.5` and becomes the new fast-screen leader.

| Candidate | sampling | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC + LCSF + SP-384 | normal | 128/64 | 24 | 63.42 | 56.89 | 0.01942 | 89.84 | 72.26 | 43.56 | 34.98 | 43.32 | old mainline |
| LGC + LCSF + SP-384 | gray=3,rust=1.5 | 128/64 | 24 | 62.96 | 56.28 | 0.01638 | 91.40 | 77.12 | 27.54 | 51.76 | 34.18 | severity/gray side branch |
| LGC + LCSF + SP-384 | gray=2,rust=1.5,alternaria=1.2,brown=1.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | new mainline |

Decision:

```text
Promote LGC + LCSF + SP-384 with gray=2,rust=1.5,alternaria=1.2,brown=1.2 prefix sampling.
Relative to normal sampling: avg mIoU +2.36, FG mIoU +2.75, Severity MAE -0.00348, Grade Acc +1.56, rust IoU +3.32, gray IoU +12.93, brown IoU +0.17.
The only meaningful cost is alternaria IoU -3.10, but it remains acceptable and the total score is stable across both seeds.

Next gate:
1. Stop broad sampling sweeps for now.
2. Run a larger 192/96 fast gate for the balanced-prefix candidate.
3. If 192/96 holds or improves, start long-run training under the working name LGC-LCSF-SP384-BalancedPrefix.
```

## 2026-06-09 Update: BalancedPrefix Passes the 192/96 Gate

The current mainline, LGC-LCSF-SP384-BalancedPrefix, was promoted from the 128/64 gate to a 192/96 fast gate with seeds 11 and 23. It continues to improve the main segmentation metrics.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LGC-LCSF-SP384-BalancedPrefix | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | promote |

Decision:

```text
Keep LGC-LCSF-SP384-BalancedPrefix as the current first-choice candidate.
Relative to 128/64, the 192/96 gate improves avg mIoU by +1.69 and FG mIoU by +1.88, with rust +3.46, gray +4.69, and brown +2.77.
The tradeoff is Severity MAE +0.00144, Grade Acc -0.78, and alternaria -3.12.

Next:
1. Run a 256/128 fast gate if continuing the fast-screen ladder.
2. If time is prioritized, start long-run training from this candidate and monitor alternaria closely.
3. Long-run working name: LGC-LCSF-SP384-BalancedPrefix.
```

## 2026-06-09 Update: BalancedPrefix Passes the 256/128 Gate

The current mainline, LGC-LCSF-SP384-BalancedPrefix, was promoted again from 192/96 to a 256/128 fast gate with seeds 11 and 23. This is now the strongest fast-screen result.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LGC-LCSF-SP384-BalancedPrefix | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | passed |
| LGC-LCSF-SP384-BalancedPrefix | 256/128 | 24 | 70.00 | 64.50 | 0.01717 | 91.40 | 78.16 | 45.90 | 56.28 | 49.40 | current fast leader |

Decision:

```text
Keep LGC-LCSF-SP384-BalancedPrefix as the first-choice long-run candidate.
Relative to 192/96, the 256/128 gate improves avg mIoU by +2.53 and FG mIoU by +2.98, with Severity MAE -0.00021, Grade Acc +0.78, alternaria +8.56, gray +3.68, and brown +3.14.
The only tradeoff is rust -0.88, but rust remains stronger than the 128/64 reference.

Next:
1. Run one final larger fast gate, preferably 384/192 or 512/256.
2. If the trend holds, start the formal long-run training from this candidate.
3. Stop broad sampling sweeps unless the long run exposes a specific per-class collapse.
```

## 2026-06-09 Update: BalancedPrefix Keeps Improving at 384/192

The 384/192 dual-seed gate completed and continues the upward trend.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix | 256/128 | 24 | 70.00 | 64.50 | 0.01717 | 91.40 | 78.16 | 45.90 | 56.28 | 49.40 | reference |
| LGC-LCSF-SP384-BalancedPrefix | 384/192 | 24 | 71.65 | 66.41 | 0.01549 | 90.62 | 80.30 | 49.58 | 60.29 | 48.12 | current fast leader |

Decision:

```text
Keep LGC-LCSF-SP384-BalancedPrefix as the first-choice candidate.
Relative to 256/128, the 384/192 gate improves avg mIoU by +1.65 and FG mIoU by +1.91, with Severity MAE -0.00168, rust +2.14, alternaria +3.68, and gray +4.01.
The tradeoff is Grade Acc -0.78 and brown -1.28.

Next:
1. Run a 512/256 gate if continuing the fast-screen ladder.
2. If 512/256 improves or stays close, launch the formal long run from this candidate.
3. If 512/256 regresses, keep 384/192 as the strongest fast-screen evidence and launch long-run training from the same architecture/sampling recipe.
```

## 2026-06-09 Update: 512/256 Marginally Passes, Fast Gates Have Plateaued

The 512/256 dual-seed gate completed successfully after rerunning with `Start-Process -Wait` to avoid PowerShell treating Python progress stderr as a native command error.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix | 384/192 | 24 | 71.65 | 66.41 | 0.01549 | 90.62 | 80.30 | 49.58 | 60.29 | 48.12 | reference |
| LGC-LCSF-SP384-BalancedPrefix | 512/256 | 24 | 71.74 | 66.53 | 0.01166 | 92.88 | 82.06 | 46.20 | 58.82 | 51.48 | marginal pass |

Decision:

```text
Stop expanding the fast-gate subset size.
Relative to 384/192, 512/256 improves avg mIoU by only +0.09 and FG mIoU by +0.12, but improves Severity MAE by -0.00383, Grade Acc by +2.26, rust by +1.76, and brown by +3.36.
The tradeoff is alternaria -3.38 and gray -1.47.

Next:
1. Launch formal long-run training from LGC-LCSF-SP384-BalancedPrefix.
2. Monitor alternaria and gray closely.
3. If the long run shows a per-class collapse, tune prefix/loss weights before changing the architecture.
```

## 2026-06-09 Update: Full 80-Epoch Long Run Becomes the Formal Leader

The first formal long run completed with the selected recipe: DeepLabV3+ MobileNetV3-Large + component auxiliary heads + LBSB + SP decoder + 384 input + LGC + LCSF, using the balanced-prefix full-data copy. Seed 11 finished with ExitCode 0 and exported the `best_miou` report.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix fast | 11 | 512/246 | 24 | 72.45 | 67.39 | 0.01127 | 93.09 | 82.52 | 48.68 | 59.30 | 52.41 | reference |
| LGC-LCSF-SP384-BalancedPrefix long | 11 | 1148/246 | 80 | 77.23 | 72.97 | 0.00940 | 95.12 | 85.12 | 57.62 | 68.50 | 57.75 | formal leader |

Decision:

```text
Keep the full 80-epoch long run as the current formal result.
Relative to the 512/246 seed11 fast gate, it improves mIoU by +4.78, FG mIoU by +5.58, Severity MAE by -0.00187, Grade Acc by +2.03, rust by +2.60, alternaria by +8.94, gray by +9.20, and brown by +5.34.
No monitored weak class collapsed; alternaria and gray both improved strongly.

Next:
1. Run the same full 80-epoch long configuration with seed 23.
2. If seed 23 is stable, promote this recipe to the final experiment track.
3. Only resume module changes if the second long run reveals a specific regression.
```

## 2026-06-10 Update: Seed 23 Long Run Confirms Stability

The same full 80-epoch configuration was rerun with seed 23. It completed with ExitCode 0 and confirmed that the long-run gain is stable rather than a seed11 accident.

| Candidate | seed | train/val | epochs | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix long | 11 | 1148/246 | 80 | 77.23 | 72.97 | 0.00940 | 95.12 | 85.12 | 57.62 | 68.50 | 57.75 | formal leader |
| LGC-LCSF-SP384-BalancedPrefix long | 23 | 1148/246 | 80 | 75.96 | 71.48 | 0.00989 | 94.72 | 84.68 | 54.57 | 67.82 | 54.88 | confirmed |

Dual-seed average:

| Candidate | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix full e80 | 76.60 | 72.22 | 0.00965 | 94.92 | 84.90 | 56.10 | 68.16 | 56.32 | final-track mainline |

Decision:

```text
Promote LGC-LCSF-SP384-BalancedPrefix full e80 to the current final-track mainline.
Relative to the 512/246 e24 fast-gate average, it improves avg mIoU by +4.86, FG mIoU by +5.69, Severity MAE by -0.00201, Grade Acc by +2.04, rust by +2.84, alternaria by +9.90, gray by +9.34, and brown by +4.84.
Stop disruptive architecture recombination. Future experiments should be narrow refinements around this recipe.

Next:
1. Build final tables and qualitative visualizations from the two long-run reports.
2. If chasing more gain, test small severity-consistency or class-weight adjustments through 128/64 fast gates before any new long run.
3. Keep this recipe as the fallback formal result unless a refinement beats it on dual-seed evidence.
```

Generated assets:

```text
outputs/atldsd/summary/training_results_summary.csv
outputs/atldsd/summary/training_results_summary.md
outputs/atldsd/summary/fig_training_miou_comparison.png
outputs/atldsd/summary/fig_training_results_table.png
outputs/atldsd/summary/fig_training_model_tradeoff.png
outputs/atldsd/summary/final_report_figures_s11/final_lgc_lcsf_s11.png
outputs/atldsd/summary/final_report_figures_s23/final_lgc_lcsf_s23.png
outputs/atldsd/summary/long_full_e80_s11_curves/fig_long_full_e80_s11_metrics_overview.png
outputs/atldsd/summary/long_full_e80_s23_curves/fig_long_full_e80_s23_metrics_overview.png
```

Next refinement gate: test `severity_consistency_loss=0.05` on the selected mainline through a 128/64 dual-seed fast gate. It can only advance if mIoU/FG remain close to the BalancedPrefix reference while severity improves.

## 2026-06-10 Update: Severity Consistency 0.05 Fails the Fast Gate

The selected mainline was tested with `severity_consistency_loss=true` and `severity_consistency_weight=0.05` on the 128/64 dual-seed fast gate. The first wrapper attempt failed because `-ExtraArgs` was parsed incorrectly; the valid run was done by creating the subset and calling `train.py` directly.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + SC0.05 | 128/64 | 24 | 64.54 | 58.20 | 0.01761 | 89.84 | 75.72 | 38.72 | 46.68 | 39.47 | reject |

Decision:

```text
Reject severity consistency at weight 0.05.
Relative to the reference, it reduces avg mIoU by -1.24 and FG mIoU by -1.44, while making Severity MAE worse by +0.00167.
Do not scale it up or start a long run from it.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80.
```

## 2026-06-10 Update: Component Boundary Aux 0.3 Fails

The next mechanism-level refinement tuned the component auxiliary supervision rather than another class weight or block augmentation. Since LBSB uses `boundary_aux_head(x)` during inference, the boundary auxiliary loss was increased from 0.2 to 0.3 while keeping lesion=0.4 and center=0.2.

Implementation note: `scripts/run_fast_deeplabv3plus_screen.ps1` now exposes `-ComponentLesionWeight`, `-ComponentBoundaryWeight`, and `-ComponentCenterWeight`, defaulting to `0.4/0.2/0.2`.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| Boundary aux=0.3 | 128/64 | 24 | 64.23 | 57.81 | 0.01759 | 87.50 | 75.63 | 36.75 | 45.32 | 40.88 | reject |

Decision:

```text
Reject component boundary aux=0.3.
Relative to the reference, it reduces avg mIoU by -1.55 and FG mIoU by -1.83, worsens Severity MAE by +0.00165, and drops alternaria/gray/brown.
The only gain is rust +0.05, which is negligible.
Keep component auxiliary weights at lesion=0.4, boundary=0.2, center=0.2.
Do not prioritize stronger boundary auxiliary supervision next.
```

## 2026-06-10 Update: Component Boundary Aux Sweep Closes at 0.2

The symmetric weaker setting `component-boundary-weight=0.1` was tested after the 0.3 failure, keeping lesion=0.4 and center=0.2 unchanged. It also failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boundary aux=0.1 | 128/64 | 24 | 64.85 | 58.55 | 0.01677 | 89.84 | 76.06 | 36.30 | 48.10 | 41.72 | reject |
| Boundary aux=0.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | best |
| Boundary aux=0.3 | 128/64 | 24 | 64.23 | 57.81 | 0.01759 | 87.50 | 75.63 | 36.75 | 45.32 | 40.88 | reject |

Decision:

```text
Reject component boundary aux=0.1.
Relative to the 0.2 reference, it reduces avg mIoU by -0.93 and FG mIoU by -1.09, worsens Severity MAE by +0.00083, and drops alternaria by -4.16.
Together with the 0.3 failure, close the boundary auxiliary weight sweep. Keep component weights at lesion=0.4, boundary=0.2, center=0.2.
```

## 2026-06-10 Update: Component Center Aux 0.1 Fails

After the boundary auxiliary weight sweep closed, the center auxiliary weight was reduced from 0.2 to 0.1 while keeping lesion=0.4 and boundary=0.2 unchanged. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| Center aux=0.1 | 128/64 | 24 | 65.02 | 58.74 | 0.01802 | 89.06 | 75.43 | 38.36 | 49.62 | 39.64 | reject |

Decision:

```text
Reject component center aux=0.1.
Relative to the reference, it reduces avg mIoU by -0.76 and FG mIoU by -0.90, worsens Severity MAE by +0.00208, and drops alternaria and brown.
The only gain is gray +1.71.
Do not scale it up. Keep center weight at 0.2 unless the symmetric stronger setting gives a real dual-seed gain.
```

## 2026-06-10 Update: Component Center Aux Sweep Closes at 0.2

The symmetric stronger setting `component-center-weight=0.3` was tested after the 0.1 failure, keeping lesion=0.4 and boundary=0.2 unchanged. It also failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Center aux=0.1 | 128/64 | 24 | 65.02 | 58.74 | 0.01802 | 89.06 | 75.43 | 38.36 | 49.62 | 39.64 | reject |
| Center aux=0.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | best |
| Center aux=0.3 | 128/64 | 24 | 64.82 | 58.48 | 0.01749 | 91.40 | 75.59 | 39.55 | 46.01 | 40.44 | reject |

Decision:

```text
Reject component center aux=0.3.
Relative to the 0.2 reference, it reduces avg mIoU by -0.96 and FG mIoU by -1.16, worsens Severity MAE by +0.00155, and drops brown by -3.05.
Together with the 0.1 failure, close the center auxiliary weight sweep. Keep component weights at lesion=0.4, boundary=0.2, center=0.2.
```

## 2026-06-10 Update: Component Lesion Aux Sweep Closes at 0.4

The final unswept component auxiliary direction was lesion supervision. With boundary=0.2 and center=0.2 fixed, lesion weights 0.3 and 0.5 were tested against the 0.4 reference. Both failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lesion aux=0.3 | 128/64 | 24 | 65.19 | 58.94 | 0.01836 | 91.40 | 75.06 | 38.25 | 48.68 | 42.06 | reject |
| Lesion aux=0.4 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | best |
| Lesion aux=0.5 | 128/64 | 24 | 64.48 | 58.10 | 0.01668 | 90.62 | 75.21 | 35.82 | 46.55 | 42.27 | reject |

Decision:

```text
Reject component lesion aux=0.3 and 0.5.
The 0.3 setting reduces avg mIoU by -0.59 and FG mIoU by -0.70, while worsening Severity MAE by +0.00242.
The 0.5 setting reduces avg mIoU by -1.30 and FG mIoU by -1.54, with a severe alternaria drop of -4.64.
Close the component auxiliary weight sweep. Keep component weights at lesion=0.4, boundary=0.2, center=0.2.
```

## 2026-06-10 Update: Alternaria Class Weight 3.3 Fails

A low-intrusion loss-side refinement was tested near the selected mainline. The recipe kept `SP decoder + LGC alpha=0.5 + LCSF alpha=0.5 + BalancedPrefix(gray=2,rust=1.5,alternaria=1.2,brown=1.2)` unchanged and only changed CE class weights from `1.0 1.0 2.0 3.0 3.0 4.0` to `1.0 1.0 2.0 3.3 3.0 4.0`.

Infrastructure note: an initial launch was invalid because PowerShell `Start-Process -ArgumentList` split the space-containing `ClsWeights` string. `scripts/run_fast_deeplabv3plus_screen.ps1` now checks `$LASTEXITCODE` after each Python subprocess, and the valid rerun used `Start-Process + EncodedCommand` to preserve quoting.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + cls alt=3.3 | 128/64 | 24 | 65.10 | 58.84 | 0.01646 | 89.84 | 76.10 | 37.89 | 48.57 | 40.96 | reject |

Decision:

```text
Reject alternaria class weight 3.3.
Relative to the reference, it reduces avg mIoU by -0.68 and FG mIoU by -0.80, worsens Severity MAE by +0.00052, and lowers Grade Acc by -1.56.
The only gains are rust +0.52 and gray +0.66, while alternaria drops -2.57 and brown drops -2.53.
Do not scale it up. Keep the formal mainline class weights at 1.0 1.0 2.0 3.0 3.0 4.0.
Avoid further single-axis alternaria boosts through prefix sampling or CE class weights unless a new mechanism changes the tradeoff.
```

## 2026-06-10 Update: LBSB Alpha 0.30 Fails

The next mainline-internal refinement increased `lesion_boundary_sharpen_alpha` from 0.25 to 0.30 while keeping `SP decoder + LGC alpha=0.5 + LCSF alpha=0.5 + BalancedPrefix` and the default class weights unchanged.

Implementation note: `scripts/run_fast_deeplabv3plus_screen.ps1` now exposes `-LesionBoundarySharpenAlpha`, defaulting to 0.25, so historical commands keep the same behavior.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LBSB alpha=0.25 reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LBSB alpha=0.30 | 128/64 | 24 | 64.35 | 57.94 | 0.01684 | 90.62 | 76.47 | 38.22 | 46.17 | 38.17 | reject |

Decision:

```text
Reject LBSB alpha 0.30.
Relative to the 0.25 reference, it reduces avg mIoU by -1.43 and FG mIoU by -1.70, worsens Severity MAE by +0.00090, and drops brown by -5.32.
The only useful gain is rust +0.89.
Do not scale it up. Keep the formal mainline at LBSB alpha=0.25.
If LBSB is explored further, only test a weaker alpha such as 0.20; do not keep increasing boundary sharpening strength.
```

## 2026-06-10 Update: LBSB Alpha Sweep Closes at 0.25

The symmetric weaker setting `lesion_boundary_sharpen_alpha=0.20` was tested after the 0.30 failure, with the rest of the selected mainline unchanged. It did not pass the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LBSB alpha=0.20 | 128/64 | 24 | 65.00 | 58.72 | 0.01590 | 91.40 | 77.07 | 36.08 | 46.78 | 42.88 | reject |
| LBSB alpha=0.25 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | best |
| LBSB alpha=0.30 | 128/64 | 24 | 64.35 | 57.94 | 0.01684 | 90.62 | 76.47 | 38.22 | 46.17 | 38.17 | reject |

Decision:

```text
Reject LBSB alpha 0.20.
Relative to the 0.25 reference, it reduces avg mIoU by -0.78 and FG mIoU by -0.92.
It slightly improves Severity MAE by -0.00004 and rust by +1.49, but alternaria drops by -4.38.
Together with the 0.30 failure, this closes the LBSB alpha sweep. Keep alpha=0.25 in the formal mainline.
```

## 2026-06-10 Update: CutMix 0.5 Fails

After several module and scalar gate refinements failed, a non-axis data augmentation candidate was tested. The selected mainline was kept unchanged, and only `mix-mode=cutmix, mix-prob=0.5, cutmix-alpha=1.0` was enabled.

Implementation note: `scripts/run_fast_deeplabv3plus_screen.ps1` now exposes `-MixMode`, `-MixProb`, `-MixupAlpha`, and `-CutmixAlpha`, defaulting to disabled behavior.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| CutMix p=0.5 | 128/64 | 24 | 64.24 | 57.85 | 0.01873 | 87.50 | 73.77 | 39.07 | 44.73 | 41.50 | reject |

Decision:

```text
Reject CutMix p=0.5.
Relative to the reference, it reduces avg mIoU by -1.54 and FG mIoU by -1.79, worsens Severity MAE by +0.00279, drops Grade Acc by -3.90, and lowers all four disease-class IoUs.
Do not scale it up. Standard block-level CutMix likely disrupts small lesion boundaries and severity estimation.
If data augmentation is revisited, use much lower probability CutMix or lesion-level semantic augmentation rather than p=0.5 block mixing.
```

## 2026-06-10 Update: CutMix Direction Closes

The lighter sanity check `mix-mode=cutmix, mix-prob=0.1, cutmix-alpha=1.0` was tested after the p=0.5 failure. It still failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| CutMix p=0.1 | 128/64 | 24 | 64.20 | 57.81 | 0.01847 | 89.84 | 74.48 | 41.21 | 45.88 | 37.26 | reject |
| CutMix p=0.5 | 128/64 | 24 | 64.24 | 57.85 | 0.01873 | 87.50 | 73.77 | 39.07 | 44.73 | 41.50 | reject |

Decision:

```text
Reject CutMix p=0.1.
Relative to the reference, it reduces avg mIoU by -1.58 and FG mIoU by -1.83, worsens Severity MAE by +0.00253, and drops brown by -6.23.
The only gain is alternaria +0.75.
Together with the p=0.5 failure, close the block-level CutMix direction. Future augmentation work should be lesion-semantic rather than rectangular block mixing.
```

## 2026-06-10 Update: LCSF Alpha 0.7 Marginally Passes 128/64

The LCSF residual strength was increased from 0.5 to 0.7 while keeping LGC alpha at 0.5 and preserving SP decoder plus BalancedPrefix. It marginally passes the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LCSF alpha=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LCSF alpha=0.7 | 128/64 | 24 | 66.06 | 59.95 | 0.01572 | 89.84 | 76.27 | 40.83 | 47.82 | 43.72 | promote to 192/96 |

Decision:

```text
LCSF alpha=0.7 gives a small 128/64 gain: avg mIoU +0.28, FG +0.31, Severity MAE -0.00022.
The tradeoff is Grade Acc -1.56 and gray -0.09.
Run a 192/96 dual-seed gate before changing the formal recipe.
```

## 2026-06-10 Update: LCSF Alpha 0.7 Is Only a Marginal Side Branch at 192/96

The LCSF alpha=0.7 candidate was promoted to a 192/96 dual-seed gate. It remains marginally positive for mIoU/FG and improves gray/alternaria, but worsens severity, rust, and brown.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LCSF alpha=0.5 | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | reference |
| LCSF alpha=0.7 | 192/96 | 24 | 67.56 | 61.66 | 0.01841 | 90.62 | 78.44 | 38.34 | 54.24 | 45.18 | side branch |

Decision:

```text
Keep LCSF alpha=0.7 as a gray/alternaria side branch only.
Relative to alpha=0.5, it improves avg mIoU by +0.09, FG by +0.14, alternaria by +1.00, and gray by +1.64, but worsens Severity MAE by +0.00103, rust by -0.60, and brown by -1.08.
Do not launch a long run from alpha=0.7.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80 with LCSF alpha=0.5.
```

## 2026-06-10 Update: LGC Alpha 0.7 Fails the Fast Gate

The LGC residual strength was increased from 0.5 to 0.7 while keeping LCSF alpha at 0.5 and preserving SP decoder plus BalancedPrefix. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC alpha=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LGC alpha=0.7 | 128/64 | 24 | 65.38 | 59.16 | 0.01750 | 88.28 | 76.31 | 41.59 | 47.34 | 39.80 | reject |

Decision:

```text
Reject LGC alpha=0.7.
It improves rust by +0.73 and alternaria by +1.13, but reduces avg mIoU by -0.40 and worsens Severity MAE by +0.00156, Grade Acc by -3.12, and brown by -3.69.
Do not scale it up.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80 with LGC alpha=0.5.
```

## 2026-06-10 Update: LGC Alpha 0.3 Also Fails, Fix LGC Alpha at 0.5

The LGC residual strength was reduced from 0.5 to 0.3 while keeping LCSF alpha at 0.5 and preserving SP decoder plus BalancedPrefix. It failed the 128/64 dual-seed gate. Together with the alpha=0.7 failure, this closes the LGC alpha sweep.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC alpha=0.3 | 128/64 | 24 | 64.76 | 58.40 | 0.01662 | 89.06 | 75.44 | 36.92 | 46.78 | 41.92 | reject |
| LGC alpha=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | keep |
| LGC alpha=0.7 | 128/64 | 24 | 65.38 | 59.16 | 0.01750 | 88.28 | 76.31 | 41.59 | 47.34 | 39.80 | reject |

Decision:

```text
Reject LGC alpha=0.3 and close the LGC alpha sweep.
Relative to alpha=0.5, it reduces avg mIoU by -1.02 and FG mIoU by -1.24.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80 with LGC alpha=0.5.
```

## 2026-06-10 Update: LCSF Alpha 0.3 Fails the Fast Gate

The LCSF residual strength was reduced from 0.5 to 0.3 while keeping LGC alpha at 0.5 and preserving SP decoder plus BalancedPrefix. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix LCSF=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix LCSF=0.3 | 128/64 | 24 | 65.24 | 59.02 | 0.01766 | 89.84 | 75.61 | 43.10 | 46.96 | 39.12 | reject |

Decision:

```text
Reject LCSF alpha=0.3.
It improves alternaria by +2.64, but reduces avg mIoU by -0.54 and worsens Severity MAE by +0.00172, with brown -4.37.
Do not scale it up.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80.
```

## 2026-06-10 Update: Low-SCSE Fails the Fast Gate

SCSE was tested on the low-level feature path while keeping SP decoder, LGC, LCSF, and BalancedPrefix unchanged. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + Low-SCSE | 128/64 | 24 | 61.56 | 54.55 | 0.02214 | 89.84 | 74.55 | 29.46 | 41.14 | 36.78 | reject |

Decision:

```text
Reject Low-SCSE.
Relative to the reference, it reduces avg mIoU by -4.22 and FG mIoU by -5.09, with a severe alternaria drop of -10.99.
Do not scale it up.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80.
```

## 2026-06-10 Update: ASPP-ECA Fails the Fast Gate

ECA was tested as a lightweight PlugNPlay attention after ASPP while keeping the selected mainline unchanged otherwise. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + ASPP-ECA | 128/64 | 24 | 63.70 | 57.15 | 0.02002 | 88.28 | 72.06 | 38.91 | 45.44 | 38.78 | reject |

Decision:

```text
Reject ASPP-ECA.
Relative to the reference, it reduces avg mIoU by -2.08 and FG mIoU by -2.49, while worsening Severity MAE by +0.00408.
Do not scale it up.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80.
```

## 2026-06-10 Update: Alternaria 1.5 Prefix Tuning Fails

The next narrow refinement increased only the prefix sampling weight for alternaria from 1.2 to 1.5, leaving `gray=2,rust=1.5,brown=1.2` unchanged. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix alt=1.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix alt=1.5 | 128/64 | 24 | 64.56 | 58.22 | 0.01785 | 89.84 | 75.09 | 40.59 | 44.37 | 40.84 | reject |

Decision:

```text
Reject alternaria=1.5 prefix tuning.
It only improves alternaria by +0.13, while reducing avg mIoU by -1.22 and FG mIoU by -1.42.
Do not scale it up.
The formal mainline remains LGC-LCSF-SP384-BalancedPrefix full e80.
```

## 2026-06-10 Update: MixUp 0.1 Fails; Close Batch-Level Mixing

After CutMix 0.5 and 0.1 failed, a soft-label MixUp sanity check was run with the selected mainline unchanged and only `mix-mode=mixup, mix-prob=0.1, mixup-alpha=0.4` enabled. The code path was verified to use soft-label CE through `Softmax_CE_Loss`, so this was a valid run. It failed the 128/64 dual-seed gate.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| MixUp p=0.1 | 128/64 | 24 | 64.31 | 57.98 | 0.02042 | 89.84 | 72.94 | 43.54 | 48.03 | 35.46 | reject |

Decision:

```text
Reject MixUp p=0.1 and do not scale it up.
Relative to the reference, it reduces avg mIoU by -1.47 and FG mIoU by -1.66, worsens Severity MAE by +0.00448, and drops brown IoU by -8.03.
The only meaningful gain is alternaria +3.08, which is not worth the global damage.
Close conventional batch-level MixUp/CutMix augmentation.
Next candidates should return to structure-aware or semantic-local modules that preserve lesion area and boundary geometry.
```

## 2026-06-10 Update: RepConv Decoder Passes 128/64 and 192/96

A structure-level decoder candidate was tested after closing batch-level MixUp/CutMix. `decoder-conv-type=repconv` was enabled while keeping the selected mainline unchanged otherwise. The fast wrapper now exposes `-DecoderConvType`, with default `standard`, so old commands keep their behavior.

128/64 dual seed:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| RepConv decoder | 128/64 | 24 | 67.50 | 61.66 | 0.01737 | 90.62 | 77.18 | 45.03 | 51.96 | 42.79 | scale to 192/96 |

192/96 dual seed:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | reference |
| RepConv decoder | 192/96 | 24 | 69.46 | 63.94 | 0.01372 | 90.10 | 79.37 | 39.59 | 57.46 | 51.14 | promote to full e80 |

Decision:

```text
Promote RepConv decoder to full e80.
At 192/96, it improves avg mIoU by +1.99 and FG mIoU by +2.42, improves Severity MAE by -0.00366, and lifts gray/brown by +4.86/+4.88.
The only cost is Grade Acc -0.52.
Run full e80 seed11/23 against the current LGC-LCSF-SP384-BalancedPrefix full e80 reference before replacing the formal mainline.
```

## 2026-06-10 Update: RepConv Decoder Passes Full e80

The RepConv decoder full e80 run has completed for seeds 11 and 23. It beats the previous full-run mainline on both average mIoU and average foreground mIoU, so the formal mainline is promoted to `LGC-LCSF-SP384-RepConv-BalancedPrefix`.

| Candidate | seed | mIoU | FG mIoU | leaf IoU | rust IoU | alternaria IoU | gray IoU | brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC-LCSF-SP384-BalancedPrefix full e80 | 11 | 77.23 | 72.97 | 95.85 | 85.12 | 57.62 | 68.50 | 57.75 | old leader |
| LGC-LCSF-SP384-BalancedPrefix full e80 | 23 | 75.96 | 71.48 | 95.43 | 84.68 | 54.57 | 67.82 | 54.88 | old leader |
| RepConv decoder full e80 | 11 | 76.66 | 72.30 | 95.67 | 84.63 | 59.50 | 66.70 | 54.98 | lower seed11 |
| RepConv decoder full e80 | 23 | 77.21 | 72.96 | 95.69 | 84.57 | 57.63 | 69.99 | 56.89 | strong seed23 |
| LGC-LCSF-SP384-BalancedPrefix full e80 avg | - | 76.60 | 72.22 | 95.64 | 84.90 | 56.10 | 68.16 | 56.31 | reference |
| RepConv decoder full e80 avg | - | 76.94 | 72.63 | 95.68 | 84.60 | 58.57 | 68.34 | 55.93 | UPGRADE |

Decision:

```text
Promote RepConv decoder to the formal mainline.
Relative to the previous full e80 mainline, it improves avg mIoU by +0.34 and avg FG mIoU by +0.40.
The strongest per-class gain is alternaria +2.47, with gray +0.19 and leaf +0.04.
The costs are small rust/brown drops: rust -0.30 and brown -0.38.
Low-CA is no longer a fallback after RepConv failure. If tested next, it should be tested as an additive low-level CA screen on top of the new RepConv mainline.
```

## 2026-06-10 Update: RepConv + Low-CA Fails the 128/64 Gate

After promoting RepConv, low-level CoordAttention was retested as an additive module on top of the new mainline: `SP decoder + RepConv + LGC + LCSF + BalancedPrefix + attention_low_type=ca`.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + Low-CA | 128/64 | 24 | 66.78 | 60.80 | 91.19 | 75.70 | 43.90 | 53.38 | 39.85 | reject |

Decision:

```text
Reject RepConv + Low-CA.
It improves gray by +1.42, but reduces avg mIoU by -0.72 and FG mIoU by -0.85, with rust -1.47, alternaria -1.13, and brown -2.94.
Do not scale it up.
Do not prioritize more low-level attention sweeps; next candidates should focus on decoder-local refinement, class-conditional calibration, or lesion-semantic calibration with a hard guard against brown degradation.
```

## 2026-06-10 Update: RepConv + Decoder-CAA Fails the 128/64 Gate

`attention_decoder_type=caa` was tested as a decoder refinement candidate by replacing the selected SP decoder attention with Context Anchor Attention. This failed badly, indicating that the current SP decoder attention should be kept.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + Decoder-CAA | 128/64 | 24 | 64.13 | 57.65 | 90.89 | 74.24 | 39.41 | 48.84 | 34.86 | reject |

Decision:

```text
Reject RepConv + Decoder-CAA.
It reduces avg mIoU by -3.37 and FG mIoU by -4.01; all foreground classes decline, especially brown -7.93 and alternaria -5.62.
Do not scale it up.
Do not replace SP decoder attention; future decoder-side attempts must be additive lightweight refinement, or the search should shift back to training/class calibration.
```

## 2026-06-10 Update: RepConv + High-SimAM Nearly Ties but Fails

`attention_high_type=simam` was tested as a low-intrusion high-level attention module before ASPP while preserving `SP decoder + RepConv + LGC + LCSF + BalancedPrefix`.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-SimAM | 128/64 | 24 | 67.45 | 61.63 | 90.89 | 76.00 | 44.01 | 52.79 | 44.46 | reject / side clue |

Decision:

```text
Reject RepConv + High-SimAM for the mainline.
It misses the gate by a tiny margin: avg mIoU -0.06 and FG mIoU -0.02.
It is still useful as a brown/gray clue: brown +1.67 and gray +0.83.
Do not scale it up as-is; any next attention candidate should preserve the brown/gray gains without rust/alternaria loss.
```

## 2026-06-10 Update: RepConv + High-MLCA Fails

After High-SimAM nearly tied but missed the gate, MLCA was tested at the same high-level insertion point before ASPP. It did not preserve the brown/gray clue and instead damaged weak classes.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-MLCA | 128/64 | 24 | 64.10 | 57.60 | 90.98 | 75.18 | 35.64 | 49.50 | 36.68 | reject |

Decision:

```text
Reject RepConv + High-MLCA.
It reduces avg mIoU by -3.41 and FG mIoU by -4.06, with severe drops in alternaria -9.39 and brown -6.11.
Do not scale it up.
The formal mainline remains LGC-LCSF-SP384-RepConv-BalancedPrefix; High-SimAM remains only a brown/gray side clue.
```

## 2026-06-10 Update: RepConv + High-ECA Fails

High-level ECA was tested after High-SimAM nearly tied and High-MLCA failed. It was much worse than High-SimAM and heavily damaged brown.

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-ECA | 128/64 | 24 | 63.30 | 56.65 | 90.96 | 74.69 | 38.54 | 46.30 | 32.74 | reject |

Decision:

```text
Reject RepConv + High-ECA.
It reduces avg mIoU by -4.20 and FG mIoU by -5.00, with brown -10.05, alternaria -6.49, and gray -5.65.
Do not scale it up.
Keep only High-SimAM as a marginal brown/gray clue; close ECA/MLCA at this insertion point.
```

## 2026-06-10 Update: RepConv Brown=1.5 Passes 128/64 but Fails 192/96

Brown prefix sampling was increased from 1.2 to 1.5 on top of the formal RepConv mainline. It passed the 128/64 gate but failed the larger 192/96 confirmation gate, so it must not replace the mainline.

128/64 gate:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv brown=1.5 | 128/64 | 24 | 67.77 | 62.00 | 91.04 | 74.69 | 48.42 | 49.15 | 46.72 | promote to 192/96 |

192/96 confirmation:

| Candidate | train/val | epochs | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 192/96 | 24 | 69.46 | 63.94 | 92.15 | 79.37 | 39.59 | 57.47 | 51.13 | reference |
| RepConv brown=1.5 | 192/96 | 24 | 68.22 | 62.38 | 92.69 | 78.66 | 36.87 | 54.63 | 49.06 | reject |

Decision:

```text
Reject RepConv brown=1.5.
The 128/64 gain was not stable: at 192/96, avg mIoU drops by -1.24 and FG mIoU by -1.56.
Only leaf improves (+0.54); rust, alternaria, gray, and brown all decline.
Keep the formal mainline as LGC-LCSF-SP384-RepConv-BalancedPrefix with brown prefix sampling at 1.2.
Do not scale or long-run brown=1.5.
```
