# Literature-Informed ATLDSD Training Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert 2025/2026 related plant-disease segmentation and severity-estimation literature into a concrete ATLDSD training plan.

**Architecture:** Keep the current single-model DeepLabV3+ + MobileNetV3-Large line. Use component-aware heads as the biological structure prior, test PConv as the decoder-localization module, then test component-guided attention instead of generic attention stacking.

**Tech Stack:** Python, PyTorch, DeepLabV3+, VOC-style ATLDSD masks, PowerShell scripts, Ubuntu shell scripts, `scripts/export_segmentation_report.py`.

---

## Research Direction

The project direction is:

```text
Component-aware lightweight apple leaf disease lesion segmentation for severity estimation.
```

The paper should not be framed as:

```text
DeepLabV3+ + CAA + PConv + losses
```

It should be framed as:

```text
1. ATLDSD severity is lesion / leaf.
2. Ordinary 6-class softmax does not explicitly learn lesion component structure.
3. Component auxiliary heads teach lesion, boundary, and center localization.
4. PConv decoder tests whether local decoder structure improves small lesions and lesion boundaries.
5. Attention is only added if it is component-guided or experimentally useful.
6. Severity losses remain auxiliary ablations, not the main structural innovation.
```

## Literature Points To Borrow

| Paper direction | Borrowed point | Training-plan effect |
|---|---|---|
| 2025 ALDNet, apple leaf disease spot segmentation | Leaf-aware and spot-aware segmentation is stronger than blindly predicting disease pixels | Keep `主线1` component heads as the structural anchor |
| 2025 STAR-Net, tomato leaf disease segmentation | Multi-branch attention and loss balancing target complex leaf lesions | Do not add generic attention first; make `主线3` component-guided if generic CAA is weak |
| 2025 Sparse-MoE-SAM, plant disease segmentation | Foundation-model style local/global sparse attention helps heterogeneous lesion appearance | Add local/global attention only after PConv result is known |
| 2025 PDSNets, plant disease severity estimation | Severity papers care about speed and deployability, not just mIoU | Always report Params, FLOPs, FPS, severity MAE, and grade accuracy |
| 2026 severity classification work | Severity estimation commonly separates background/leaf/lesion reasoning | Keep lesion/leaf severity metrics as a core evaluation, but keep severity loss as an auxiliary ablation |

Primary literature links:

```text
ALDNet, Measurement 2025:
https://www.sciencedirect.com/science/article/pii/S0263224125010656

STAR-Net, Frontiers in Plant Science 2026:
https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1706072/full

Sparse-MoE-SAM, Plants 2025:
https://www.mdpi.com/3462676

Citrus multiclass semantic segmentation with severity, Scientific Reports 2025:
https://www.nature.com/articles/s41598-025-04758-y

Optimized Lightweight U-Net and YOLACT for pome leaf multi-disease severity, Scientific Reports 2026:
https://www.nature.com/articles/s41598-026-45947-7
```

## File Structure

Current files already available:

- `scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_pconv_150.ps1`: Windows mainline2 launcher.
- `scripts/run_ubuntu_component_aux_pconv_v3.sh`: Ubuntu mainline2 launcher.
- `scripts/run_ubuntu.sh`: Ubuntu dispatcher.
- `scripts/export_segmentation_report.py`: best-mIoU, per-class, severity, and complexity report exporter.
- `src/models/deeplabv3plus/train.py`: training CLI.
- `src/models/deeplabv3plus/nets/deeplabv3_plus.py`: model structure and decoder-conv selection.
- `src/modules/plugins/modules.py`: CAA module.
- `src/modules/plugins/factory.py`: attention factory.
- `seg/ATLDSD项目进度.md`: repository training note.

Files to create or modify later:

- Create: `scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_caa_150.ps1`
- Create: `scripts/run_ubuntu_component_aux_caa_v3.sh`
- Modify: `scripts/run_ubuntu.sh`
- Optional create: `scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_pconv_caa_150.ps1`
- Optional create: `scripts/run_ubuntu_component_aux_pconv_caa_v3.sh`
- Optional modify: `src/modules/plugins/modules.py` if a component-guided attention module is implemented.
- Modify after every run: `seg/ATLDSD项目进度.md`

## Task 1: Finish Current Mainline2 Evaluation

**Files:**
- Read: `D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\train_stdout.log`
- Read: `D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\reports\best_miou\metrics_summary.json`
- Read: `D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\reports\best_miou\severity_metrics.json`
- Modify: `seg/ATLDSD项目进度.md`

- [ ] **Step 1: Wait for mainline2 to finish**

Run:

```powershell
Get-Process -Id 41672 -ErrorAction SilentlyContinue
```

Expected while running: a `python` process is returned.  
Expected when complete: no process is returned.

- [ ] **Step 2: Read final log**

Run:

```powershell
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\train_stdout.log' -Tail 180
```

Expected: log reaches `Epoch:150/150` and `[AutoReport] Report saved`.

- [ ] **Step 3: Read best metrics**

Run:

```powershell
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\reports\best_miou\metrics_summary.json' -Encoding UTF8
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\reports\best_miou\severity_metrics.json' -Encoding UTF8
```

Expected: JSON contains `miou_all`, `miou_foreground`, `pixel_accuracy`, `severity_mae`, and `severity_grade_accuracy`.

- [ ] **Step 4: Decide whether PConv stays**

Use this rule:

```text
If mainline2 mIoU > 72.11:
  PConv stays.

If mainline2 mIoU is 71.90-72.11 but foreground or difficult lesion classes improve:
  PConv stays as a small-lesion / boundary module.

If mainline2 mIoU < 71.90 and no severity or difficult-class gain:
  PConv becomes a failed ablation.
```

- [ ] **Step 5: Update training note**

Add a short result block to:

```text
D:\Code\ATLDSD\seg\ATLDSD项目进度.md
D:\soft\obsidian_notion\seg\ATLDSD项目进度.md
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add seg/ATLDSD项目进度.md
git commit -m "Record mainline PConv result"
git push
```

Expected: commit is pushed to GitHub.

## Task 2: Run Mainline3 With Existing CAA As A Controlled Attention Test

**Files:**
- Create: `scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_caa_150.ps1`
- Create: `scripts/run_ubuntu_component_aux_caa_v3.sh`
- Modify: `scripts/run_ubuntu.sh`
- Modify: `seg/ATLDSD项目进度.md`

- [ ] **Step 1: Create the Windows launcher**

Create `scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_caa_150.ps1` using the mainline1 script and change only:

```text
--attention-type caa
output directory = deeplabv3plus_mobilenetv3_large_component_aux_caa_150
```

Keep:

```text
--decoder-conv-type standard
--component-aux true
--severity-consistency-loss false
--lbft-loss false
--sclp false
```

- [ ] **Step 2: Create the Ubuntu launcher**

Create `scripts/run_ubuntu_component_aux_caa_v3.sh` using the mainline1 Ubuntu script and change only:

```text
--attention-type caa
RUN_ROOT="${PROJECT_ROOT}/outputs/atldsd/deeplabv3plus_mobilenetv3_large_component_aux_caa_150"
```

- [ ] **Step 3: Add dispatcher alias**

Modify `scripts/run_ubuntu.sh`:

```bash
component_aux_caa|caa|mainline3)
  exec "${SCRIPT_DIR}/run_ubuntu_component_aux_caa_v3.sh"
  ;;
```

Expected usage:

```bash
./scripts/run_ubuntu.sh component_aux_caa
```

- [ ] **Step 4: Launch Windows training**

Run:

```powershell
& 'D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_caa_150.ps1'
```

Expected: script writes a PID to:

```text
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_caa_150\train_pid.txt
```

- [ ] **Step 5: Confirm configuration**

Run:

```powershell
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_caa_150\train_stdout.log' -Tail 80
```

Expected:

```text
attention_type = caa
decoder_conv_type = standard
component_aux = True
severity_consistency_loss = False
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_caa_150.ps1 scripts/run_ubuntu_component_aux_caa_v3.sh scripts/run_ubuntu.sh seg/ATLDSD项目进度.md
git update-index --chmod=+x scripts/run_ubuntu_component_aux_caa_v3.sh
git commit -m "Launch mainline CAA component auxiliary training"
git push
```

## Task 3: Only If CAA Is Weak, Implement Component-Guided Attention

**Files:**
- Modify: `src/modules/plugins/modules.py`
- Modify: `src/modules/plugins/factory.py`
- Modify: `src/models/deeplabv3plus/nets/deeplabv3_plus.py`
- Test: run a small import/config smoke test.

- [ ] **Step 1: Decide whether this task is needed**

Do this only if:

```text
Mainline3 generic CAA does not beat mainline1
AND qualitative masks show CAA attends broadly to the leaf instead of lesions.
```

- [ ] **Step 2: Implement a component-guided attention module**

The module should use component auxiliary logits or decoder features to gate decoder features. The intended behavior:

```text
lesion/boundary/center cues -> attention map -> decoder feature refinement
```

Do not add this before mainline2 and mainline3 are complete.

- [ ] **Step 3: Keep this as a new structural candidate**

If implemented, name it in notes as:

```text
Component-guided attention candidate
```

Do not rename it to a new chaotic experiment family.

## Task 4: Update Paper Narrative After Mainline2 And Mainline3

**Files:**
- Modify: `seg/ATLDSD项目进度.md`

- [ ] **Step 1: Write the final module story**

Use this template:

```text
Mainline0 proves the ordinary 6-class DeepLabV3+ baseline.
Mainline1 proves explicit lesion/boundary/center component learning.
Mainline2 tests whether decoder locality improves lesion boundary and small spots.
Mainline3 tests whether attention improves lesion response.
Auxiliary Experiment A shows severity MAE can improve without becoming the main structural innovation.
```

- [ ] **Step 2: Commit**

Run:

```powershell
git add seg/ATLDSD项目进度.md
git commit -m "Update literature-informed paper narrative"
git push
```

## Self-Review

Spec coverage:

```text
The plan summarizes the current research direction, identifies 2025/2026 literature-derived ideas, and converts them into the training sequence.
```

Placeholder scan:

```text
No TBD/TODO placeholders are used.
```

Type consistency:

```text
Experiment names remain mainline0-4 and auxiliary A-C. No new E/M/S/L names are introduced.
```
