# ATLDSD 超越 Boundary1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已经多次改进失败的基础上，重新制定一条最可能超过 `Boundary1 = Mainline1 + LBSB` 的训练路线。

**Architecture:** 当前最强不是复杂模块，而是 `DeepLabV3+ MobileNetV3-Large + component auxiliary heads + LBSB`。新计划先验证随机性和输入分辨率，再只保留一个真正结构创新：`CFR`，即把 lesion / boundary / center 辅助头反馈到主分割特征。

**Tech Stack:** PyTorch, DeepLabV3+, VOC-format ATLDSD, PowerShell training scripts, Ubuntu shell scripts, repository notes under `seg`.

---

## Current Facts

当前最强结果来自：

```text
Boundary1 = Mainline1 + LBSB
mIoU: 72.86
FG mIoU: 67.89
Accuracy: 97.97
Severity MAE: 0.01177
Grade Acc: 93.90
Params: 11.73M
FLOPs: 15.29G
FPS: 106.89
```

已经失败或不作为主线继续的方向：

```text
B4 backbone: 65.59 mIoU，过重且更差。
SCLP copy-paste: 68.97 / 69.90 mIoU，破坏真实分布。
PConv: 71.76 mIoU，轻量但不涨点。
PConv + LBSB: 71.68 mIoU，和边界锐化没有协同。
LCAF: 72.68 mIoU，接近但低于 Boundary1。
LGLC: 72.31 mIoU，补上下文没有解决核心短板。
Severity loss: 72.12 mIoU，严重度 MAE 好但主分割不强。
```

当前短板不是“模块不够多”，而是：

```text
1. 所有主要实验都使用 256x256 输入，小病斑可能被下采样抹掉。
2. component heads 目前主要通过 auxiliary loss 训练，推理时没有充分反馈主 mask。
3. Boundary1 只有一个 seed 结果，需要先确认 72.86 是否稳定。
```

## New Rule

```text
不再继续调 PConv / LCAF / LGLC / SCLP / B4。
不再一次混入两个以上新因素。
每次训练只回答一个问题。
每次完成后必须更新 summary 图表、seg 笔记、Obsidian，并提交推送 GitHub。
```

成功阈值：

```text
强成功: mIoU >= 73.20，且 FG mIoU >= 68.10。
弱成功: mIoU > 72.86，且小病斑类 alternaria / gray_spot / brown_spot 至少两个类别 IoU 提升。
失败: mIoU <= 72.86，或速度/复杂度恶化明显但精度没有超过。
```

---

### Task 1: Boundary1-384

**Files:**
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_384_v3.sh`
- Modify: `D:\Code\ATLDSD\scripts\run_ubuntu.sh`
- Modify after completion: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Modify after completion: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`

Purpose:

```text
这是下一步最优先实验。
它不引入新模块，只改变输入信息量，用来验证小病斑是否被 256x256 下采样压没。
如果 384 直接超过 Boundary1，说明之前很多模块失败不是因为结构不够，而是输入分辨率太低。
```

- [ ] **Step 1: Create Windows 384 script**

Create `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1` with:

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

- [ ] **Step 2: Run Boundary1-384**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150.ps1
```

Expected:

```text
Started Boundary1-384 training. PID=<number>
Output: D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_384_150
```

- [ ] **Step 3: Decide whether resolution is the new mainline**

Decision:

```text
If Boundary1-384 mIoU >= 73.20:
  Promote Boundary1-384 as new best. CFR should be tested on 384.

If Boundary1-384 mIoU is 72.86 to 73.19:
  Keep it as useful but not decisive. CFR should be tested on both 256 and 384 if time allows.

If Boundary1-384 mIoU < 72.86:
  Do not keep 384. Continue to CFR-256.
```

---

### Task 2: Boundary1-repeat

**Files:**
- Use: `D:\Code\ATLDSD\src\models\deeplabv3plus\train.py`
- Use: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150.ps1`
- Modify after completion: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Modify after completion: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`
- Modify after completion: `D:\Code\ATLDSD\seg\ATLDSD训练计划_2026-06-06.md`

Purpose:

```text
这是严谨性实验，不是超越实验。
它用于确认 Boundary1 的 72.86 是稳定水平，还是 seed=11 的偶然高点。
如果 seed=42 或 seed=2026 也能接近 72.8，说明上限真实存在。
如果差很多，后续必须用多 seed 均值写论文，不能只挑最高点。
```

- [ ] **Step 1: Run Boundary1 with seed 42**

Run in PowerShell:

```powershell
$root = "D:\Code\ATLDSD"
$python = "D:\soft\Anaconda\envs\Pytorch\python.exe"
$out = "$root\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed42"
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $python "$root\src\models\deeplabv3plus\train.py" `
  --cuda true --seed 42 --num-classes 6 `
  --backbone mobilenetv3_large --pretrained true --downsample-factor 16 `
  --attention-type none --decoder-conv-type standard --use-ppm false `
  --input-shape 256 256 `
  --init-epoch 0 --freeze-epoch 50 --freeze-batch-size 8 `
  --unfreeze-epoch 150 --unfreeze-batch-size 4 --freeze-train true `
  --init-lr 0.003 --optimizer-type sgd --lr-decay-type cos `
  --save-period 10 --eval-period 10 `
  --dataset-name ATLDSD --vocdevkit-path "D:\dataset\ATLDSD\VOCdevkit" `
  --dice-loss true --focal-loss false `
  --component-aux true --component-lesion-weight 0.4 --component-boundary-weight 0.2 --component-center-weight 0.2 `
  --lesion-boundary-sharpen true --lesion-boundary-sharpen-alpha 0.25 `
  --num-workers 0 --auto-export-report true `
  --report-dir "$out\reports\best_miou" --report-checkpoint best_miou --report-split val --report-fps-interval 100 `
  --save-dir "$out\weights" --log-dir "$out\logs" `
  --class-names background leaf rust alternaria_leaf_spot gray_spot brown_spot
```

Expected:

```text
Training finishes at 150 epochs.
Report exists at:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed42\reports\best_miou\metrics_summary.json
```

- [ ] **Step 2: Run Boundary1 with seed 2026**

Use the same command as Step 1, changing only:

```powershell
--seed 2026
$out = "$root\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed2026"
```

Expected:

```text
Training finishes at 150 epochs.
Report exists at:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150_seed2026\reports\best_miou\metrics_summary.json
```

- [ ] **Step 3: Decide whether Boundary1 is stable**

Decision:

```text
If both repeat runs are >= 72.50 mIoU:
  Boundary1 is a stable anchor. Continue to Task 2.

If either repeat run is < 72.30 mIoU:
  Stop claiming single-run superiority. Continue to Task 2 anyway, but final paper must report mean/std.
```

---

### Task 3: CFR-256

**Files:**
- Modify: `D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py`
- Modify: `D:\Code\ATLDSD\src\models\deeplabv3plus\train.py`
- Modify: `D:\Code\ATLDSD\src\models\deeplabv3plus\deeplab.py`
- Modify: `D:\Code\ATLDSD\scripts\export_segmentation_report.py`
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_cfr_v3.sh`
- Modify: `D:\Code\ATLDSD\scripts\run_ubuntu.sh`

Purpose:

```text
这是唯一保留的新结构创新。
component auxiliary heads 不能只当训练期 loss，它们必须参与推理期主 mask 修正。
```

- [ ] **Step 1: Add CFR block to `deeplabv3_plus.py`**

Insert this class after `LesionBoundarySharpeningBlock`:

```python
class ComponentFeedbackRefinementBlock(nn.Module):
    def __init__(self, channels, alpha=0.2, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        self.feedback = nn.Sequential(
            nn.Conv2d(3, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=True),
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
        feedback_logits = [lesion_logits, boundary_logits, center_logits]
        resized = []
        for item in feedback_logits:
            if item.shape[-2:] != feature.shape[-2:]:
                item = F.interpolate(item, size=feature.shape[-2:], mode="bilinear", align_corners=True)
            resized.append(torch.sigmoid(item))
        cue = torch.cat(resized, dim=1)
        gate = self.feedback(cue)
        return feature + self.alpha * self.refine(feature) * gate
```

- [ ] **Step 2: Wire CFR into `DeepLab.__init__`**

Add constructor args:

```python
use_cfr=False,
cfr_alpha=0.2,
```

Set flags:

```python
self.use_cfr = use_cfr
```

Create module after `self.lbsb`:

```python
self.cfr = ComponentFeedbackRefinementBlock(256, alpha=cfr_alpha) if use_cfr else nn.Identity()
if self.use_cfr and not self.use_component_aux:
    raise ValueError("CFR requires use_component_aux=True so lesion/boundary/center logits are available.")
```

- [ ] **Step 3: Wire CFR into `DeepLab.forward`**

Replace the current LBSB/logits section with:

```python
lesion_feature_logits = None
boundary_feature_logits = None
center_feature_logits = None
if self.use_component_aux:
    lesion_feature_logits = self.lesion_aux_head(x)
    boundary_feature_logits = self.boundary_aux_head(x)
    center_feature_logits = self.center_aux_head(x)
if self.use_lbsb:
    x = self.lbsb(x, boundary_feature_logits)
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

- [ ] **Step 4: Add CLI flags**

In `train.py`, add:

```python
parser.add_argument("--component-feedback-refine", type=str2bool, default=False, help="Use component feedback refinement before the final segmentation head.")
parser.add_argument("--component-feedback-refine-alpha", type=float, default=0.2)
```

Pass to `DeepLab`:

```python
use_cfr=args.component_feedback_refine,
cfr_alpha=args.component_feedback_refine_alpha,
```

Also add both flags to the auto-export command.

- [ ] **Step 5: Add inference/report support**

In `deeplab.py`, add defaults:

```python
"component_feedback_refine": "auto",
"component_feedback_refine_alpha": 0.2,
```

Add auto-detection:

```python
def _resolve_use_cfr(self, checkpoint):
    if isinstance(self.component_feedback_refine, bool):
        return self.component_feedback_refine
    if isinstance(self.component_feedback_refine, str) and self.component_feedback_refine.lower() != "auto":
        return self.component_feedback_refine.lower() in {"true", "1", "yes", "y"}
    return any(key.startswith("cfr.") for key in checkpoint.keys())
```

Pass:

```python
use_cfr=use_cfr,
cfr_alpha=float(self.component_feedback_refine_alpha),
```

In `export_segmentation_report.py`, add matching args and pass them into the model.

- [ ] **Step 6: Run syntax and forward tests**

Run:

```powershell
D:\soft\Anaconda\envs\Pytorch\python.exe -m py_compile `
  D:\Code\ATLDSD\src\models\deeplabv3plus\nets\deeplabv3_plus.py `
  D:\Code\ATLDSD\src\models\deeplabv3plus\train.py `
  D:\Code\ATLDSD\src\models\deeplabv3plus\deeplab.py `
  D:\Code\ATLDSD\scripts\export_segmentation_report.py
```

Expected:

```text
No output and exit code 0.
```

Run forward shape test:

```powershell
$env:PYTHONPATH="D:\Code\ATLDSD\src\models\deeplabv3plus;D:\Code\ATLDSD\src"
D:\soft\Anaconda\envs\Pytorch\python.exe - <<'PY'
import torch
from nets.deeplabv3_plus import DeepLab
model = DeepLab(
    num_classes=6,
    backbone="mobilenetv3_large",
    pretrained=False,
    use_component_aux=True,
    use_lbsb=True,
    use_cfr=True,
)
out = model(torch.randn(1, 3, 256, 256))
assert out["logits"].shape == (1, 6, 256, 256)
assert out["lesion_logits"].shape == (1, 1, 256, 256)
assert out["boundary_logits"].shape == (1, 1, 256, 256)
assert out["center_logits"].shape == (1, 1, 256, 256)
print("CFR forward test passed")
PY
```

Expected:

```text
CFR forward test passed
```

- [ ] **Step 7: Train CFR-256**

Use the same settings as Boundary1, adding:

```text
--component-feedback-refine true
--component-feedback-refine-alpha 0.2
```

Output directory:

```text
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_150
```

Decision:

```text
If CFR-256 mIoU >= 73.20:
  CFR becomes final structural innovation.

If CFR-256 mIoU is 72.86 to 73.19:
  Keep CFR as candidate and test CFR-384.

If CFR-256 mIoU < 72.86:
  CFR does not solve the problem at 256. Only try CFR-384 if Boundary1-384 improved over Boundary1.
```

---

### Task 4: CFR-384

**Files:**
- Create: `D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_384_150.ps1`
- Create: `D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_cfr_384_v3.sh`
- Modify after completion: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Modify after completion: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`

Purpose:

```text
只在 Boundary1-384 或 CFR-256 至少一个接近成功时执行。
验证“高分辨率 + 组件反馈”是否对小病斑最有效。
```

- [ ] **Step 1: Train CFR-384**

Use CFR-256 settings, changing:

```text
--input-shape 384 384
--freeze-batch-size 4
--unfreeze-batch-size 2
--init-lr 0.002
--save-dir outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_384_150\weights
--log-dir outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_384_150\logs
--report-dir outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_cfr_384_150\reports\best_miou
```

Decision:

```text
If CFR-384 mIoU >= 73.20:
  Final model = DeepLabV3+ MobileNetV3-Large + component auxiliary heads + LBSB + CFR at 384.

If CFR-384 mIoU <= Boundary1:
  Final model remains Boundary1. CFR is a negative ablation.
```

---

### Task 5: Documentation and Paper Framing

**Files:**
- Modify: `D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.md`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_results_table.png`
- Regenerate: `D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_model_tradeoff.png`
- Modify: `D:\Code\ATLDSD\seg\ATLDSD项目进度.md`
- Modify: `D:\Code\ATLDSD\seg\ATLDSD训练计划_2026-06-06.md`
- Copy to: `D:\soft\obsidian_notion\seg`

- [ ] **Step 1: Update summary table**

Add rows for every completed new run:

```python
{
    "id": "Boundary1-384",
    "method": "Boundary1 at 384 input",
    "change": "higher input resolution",
    "status": "done",
    "miou": <actual_miou>,
    "fg_miou": <actual_fg_miou>,
    "acc": <actual_acc>,
    "severity_mae": <actual_severity_mae>,
    "grade_acc": <actual_grade_acc>,
    "params_m": <actual_params_m>,
    "flops_g": <actual_flops_g>,
    "fps": <actual_fps>,
    "decision": "<actual decision>",
}
```

Use actual values from each report JSON; do not estimate.

- [ ] **Step 2: Regenerate figures**

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

- [ ] **Step 3: Sync notes to Obsidian**

Run:

```powershell
Copy-Item -LiteralPath "D:\Code\ATLDSD\seg\ATLDSD项目进度.md" -Destination "D:\soft\obsidian_notion\seg\ATLDSD项目进度.md" -Force
Copy-Item -LiteralPath "D:\Code\ATLDSD\seg\ATLDSD训练计划_2026-06-06.md" -Destination "D:\soft\obsidian_notion\seg\ATLDSD训练计划_2026-06-06.md" -Force
Copy-Item -LiteralPath "D:\Code\ATLDSD\seg\ATLDSD超越Boundary1训练计划_2026-06-08.md" -Destination "D:\soft\obsidian_notion\seg\ATLDSD超越Boundary1训练计划_2026-06-08.md" -Force
```

- [ ] **Step 4: Commit and push**

Run:

```powershell
git add D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py D:\Code\ATLDSD\outputs\atldsd\summary D:\Code\ATLDSD\seg
git commit -m "Revise ATLDSD training plan beyond Boundary1"
git push origin main
```

Expected:

```text
main -> main
```

---

## Final Paper Direction After This Plan

If `Boundary1-384` wins:

```text
Main story:
Compositional component supervision + lesion boundary sharpening + high-resolution lesion preservation.
```

If `CFR` wins:

```text
Main story:
Component-aware feedback segmentation: auxiliary lesion/boundary/center predictions refine the main semantic mask.
```

If no run beats Boundary1:

```text
Final model remains Boundary1.
Paper angle becomes: structured component supervision and boundary sharpening are effective; naive context, copy-paste, PConv, and cross-scale fusion fail on ATLDSD.
This is still publishable only if experiments are written honestly with strong ablation, per-class analysis, severity metrics, and visual evidence.
```

---

## 2026-06-09 快筛主线更新：LGC + LCSF + SP-384 暂列第一候选

新的快筛结果显示，`SP decoder + 384 input + lesion_local_global_context + lesion_cross_scale_fusion` 已经通过 128/64 双 seed 门槛，暂时替代单独 `LGC + SP-384` 成为当前主候选。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SP-384 | 128/64 | 24 | 60.33 | 53.15 | 0.01940 | 87.50 | 72.90 | 32.64 | 31.52 | 38.23 | 旧主线 |
| LGC + SP-384 | 128/64 | 24 | 62.39 | 55.60 | 0.02458 | 86.72 | 73.96 | 39.14 | 35.53 | 38.68 | 强候选 |
| LGC + LCSF + SP-384 | 128/64 | 24 | 63.42 | 56.89 | 0.01942 | 89.84 | 72.26 | 43.56 | 34.98 | 43.32 | 当前第一候选 |

训练计划调整：

```text
1. 不再单独推进 LCSF；LCSF 单模块 128/64 不过门，但和 LGC 组合后有效。
2. 当前主候选：DeepLabV3+ MobileNetV3-Large + component auxiliary heads + LBSB + SP decoder + 384 input + LGC + LCSF。
3. 下一轮快筛优先测试：在该组合上加入 gray=3,rust=1.5 前缀采样。
4. 如果采样版 128/64 双 seed 继续涨点，则升 192/96 或进入正式长轮。
5. 如果采样版伤 rust/gray，则保持非采样 LGC + LCSF + SP-384 进入长轮。
```

### 采样门补充：gray=3,rust=1.5 不替代主线

`LGC + LCSF + SP-384` 加 `gray=3,rust=1.5` 已跑完 128/64 双 seed。它显著改善 gray、rust、Severity MAE 和 Grade Acc，但会压低 alternaria/brown，导致 mIoU 与 FG mIoU 略低于 normal sampling 主线。

| 组别 | sampling | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC + LCSF + SP-384 | normal | 128/64 | 24 | 63.42 | 56.89 | 0.01942 | 89.84 | 72.26 | 43.56 | 34.98 | 43.32 | mIoU 主线 |
| LGC + LCSF + SP-384 | gray=3,rust=1.5 | 128/64 | 24 | 62.96 | 56.28 | 0.01638 | 91.40 | 77.12 | 27.54 | 51.76 | 34.18 | severity/gray 支线 |

```text
当前正式主线仍保持 normal sampling 的 LGC + LCSF + SP-384。
gray=3,rust=1.5 只作为 severity/gray 支线保留，不升长轮主线。
如果继续采样，只测更温和版本：gray=2,rust=1.5,alternaria=1.2,brown=1.2。
否则直接把 normal sampling 主线升 192/96 或正式长轮。
```

### 温和采样成为新主线

`gray=2,rust=1.5,alternaria=1.2,brown=1.2` 已经在 `LGC + LCSF + SP-384` 上通过 128/64 双 seed，并成为当前快筛最高方案。

| 组别 | sampling | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC + LCSF + SP-384 | normal | 128/64 | 24 | 63.42 | 56.89 | 0.01942 | 89.84 | 72.26 | 43.56 | 34.98 | 43.32 | 旧主线 |
| LGC + LCSF + SP-384 | gray=3,rust=1.5 | 128/64 | 24 | 62.96 | 56.28 | 0.01638 | 91.40 | 77.12 | 27.54 | 51.76 | 34.18 | severity/gray 支线 |
| LGC + LCSF + SP-384 | gray=2,rust=1.5,alternaria=1.2,brown=1.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | 当前新主线 |

```text
新主线命名：LGC-LCSF-SP384-BalancedPrefix。
下一步停止继续大范围采样，直接升 192/96 快门。
若 192/96 仍守住 128/64 的涨点趋势，则进入正式长轮训练。
```

### 192/96 中门通过

`LGC-LCSF-SP384-BalancedPrefix` 已从 128/64 升到 192/96 快门，并继续涨主指标。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | 继续保留 |

```text
192/96 相对 128/64：mIoU +1.69，FG +1.88，rust +3.46，gray +4.69，brown +2.77。
代价：MAE +0.00144，Grade Acc -0.78，alternaria -3.12。
结论：当前第一候选仍为 LGC-LCSF-SP384-BalancedPrefix。
下一步：继续 256/128 快门，或直接从该候选进入正式长轮并重点监控 alternaria。
```

### 256/128 快门通过，进入长轮前第一候选

`LGC-LCSF-SP384-BalancedPrefix` 继续升到 256/128 后，双 seed 平均 mIoU 达到 70.00，FG mIoU 达到 64.50，是目前所有快筛中最高结果。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | 已通过 |
| BalancedPrefix | 256/128 | 24 | 70.00 | 64.50 | 0.01717 | 91.40 | 78.16 | 45.90 | 56.28 | 49.40 | 长轮前第一候选 |

```text
256/128 相对 192/96：mIoU +2.53，FG +2.98，MAE -0.00021，Grade Acc +0.78，alternaria +8.56，gray +3.68，brown +3.14。
代价：rust -0.88，但总体仍稳。
结论：当前训练计划改为优先推进 LGC-LCSF-SP384-BalancedPrefix。
下一步：先做 384/192 或 512/256 最后一轮快门；若仍涨或基本持平，则启动正式长轮训练。
```

### 384/192 继续涨，长轮前主线再次上移

`LGC-LCSF-SP384-BalancedPrefix` 在 384/192 快门继续提升，说明 256/128 的涨点不是偶然峰值。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix | 256/128 | 24 | 70.00 | 64.50 | 0.01717 | 91.40 | 78.16 | 45.90 | 56.28 | 49.40 | reference |
| BalancedPrefix | 384/192 | 24 | 71.65 | 66.41 | 0.01549 | 90.62 | 80.30 | 49.58 | 60.29 | 48.12 | 长轮前第一候选 |

```text
384/192 相对 256/128：mIoU +1.65，FG +1.91，MAE -0.00168，rust +2.14，alternaria +3.68，gray +4.01。
代价：Grade Acc -0.78，brown -1.28。
训练计划：继续保留 BalancedPrefix，不回退；下一步跑 512/256 快门或直接进入长轮。若要继续榨快筛涨点，优先 512/256。
```

### 512/256 边际通过，停止继续样本快门并启动长轮

`LGC-LCSF-SP384-BalancedPrefix` 在 512/256 上仍小幅涨主指标，并显著改善 severity、rust、brown，但 alternaria/gray 比 384/192 回落。说明继续扩大快筛子集的收益已经接近平台期。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix | 384/192 | 24 | 71.65 | 66.41 | 0.01549 | 90.62 | 80.30 | 49.58 | 60.29 | 48.12 | reference |
| BalancedPrefix | 512/256 | 24 | 71.74 | 66.53 | 0.01166 | 92.88 | 82.06 | 46.20 | 58.82 | 51.48 | 边际通过 |

```text
512/256 相对 384/192：mIoU +0.09，FG +0.12，MAE -0.00383，Grade Acc +2.26，rust +1.76，brown +3.36。
代价：alternaria -3.38，gray -1.47。
训练计划：停止继续扩大快筛样本门，启动正式长轮。
正式长轮配置：DeepLabV3+ MobileNetV3-Large + component auxiliary heads + LBSB + SP decoder + input 384 + LGC + LCSF + BalancedPrefix sampling。
长轮重点监控：alternaria 和 gray，必要时只微调 prefix/loss 权重，不先改架构。
```

### 全量 80 epoch 长轮大涨，保留为当前正式主结果

长轮 seed11 已完成，报告位于 `outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s11/reports/best_miou`。结果显著超过 512/256 快筛和 384/192 快筛。

| 组别 | seed | train/val | epoch | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix fast | 11 | 512/246 | 24 | 72.45 | 67.39 | 0.01127 | 93.09 | 82.52 | 48.68 | 59.30 | 52.41 | reference |
| BalancedPrefix long | 11 | 1148/246 | 80 | 77.23 | 72.97 | 0.00940 | 95.12 | 85.12 | 57.62 | 68.50 | 57.75 | 当前正式主结果 |

```text
长轮相对 512/246 seed11：mIoU +4.78，FG +5.58，MAE -0.00187，Grade Acc +2.03，rust +2.60，alternaria +8.94，gray +9.20，brown +5.34。
训练计划更新：保留 LGC-LCSF-SP384-BalancedPrefix 长轮为当前正式主结果。
下一步：跑 seed23 全量 80 epoch 复核稳定性；如果双 seed 都稳，则停止架构拼接，进入最终长训/图表/论文结果整理。
```

### seed23 长轮复核通过，正式主线固定

同配置 seed23 全量 80 epoch 已完成并通过复核。当前正式结果不再依赖单 seed。

| 组别 | seed | train/val | epoch | mIoU | FG mIoU | Severity MAE | Grade Acc | rust IoU | alternaria IoU | gray IoU | brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix long | 11 | 1148/246 | 80 | 77.23 | 72.97 | 0.00940 | 95.12 | 85.12 | 57.62 | 68.50 | 57.75 | 正式主线 |
| BalancedPrefix long | 23 | 1148/246 | 80 | 75.96 | 71.48 | 0.00989 | 94.72 | 84.68 | 54.57 | 67.82 | 54.88 | 复核通过 |

双 seed 平均：

| 组别 | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix full e80 | 76.60 | 72.22 | 0.00965 | 94.92 | 84.90 | 56.10 | 68.16 | 56.32 | 当前正式主线 |

```text
训练计划更新：LGC-LCSF-SP384-BalancedPrefix 全量 80 epoch 固定为当前正式主线。
停止推翻式架构拼接；后续只做主线附近的小范围微调。
优先下一步：产出可视化和最终表格；若继续冲点，尝试 severity consistency 小权重或轻微 class/prefix 权重调节，并先走 128/64 快筛。
```

已生成可复现实验图表：

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

下一步冲点规则：只允许在正式主线附近快筛小改动。首个候选为 `severity_consistency_loss=0.05`，必须先通过 128/64 双 seed，再考虑放大或长轮。

### severity consistency 0.05 快筛失败，拒绝

在正式主线附近加入 `severity_consistency_loss=true, severity_consistency_weight=0.05` 后，128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + SC0.05 | 128/64 | 24 | 64.54 | 58.20 | 0.01761 | 89.84 | 75.72 | 38.72 | 46.68 | 39.47 | 拒绝 |

```text
SC0.05 相对 reference：mIoU -1.24，FG -1.44，MAE +0.00167，Grade Acc -1.56。
结论：不放大、不长轮，正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```

### alternaria=1.5 采样微调失败，拒绝

在正式主线附近把 prefix sampling 的 `alternaria` 从 1.2 提到 1.5 后，128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix alt=1.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix alt=1.5 | 128/64 | 24 | 64.56 | 58.22 | 0.01785 | 89.84 | 75.09 | 40.59 | 44.37 | 40.84 | 拒绝 |

```text
alt=1.5 只带来 alternaria +0.13，却造成 mIoU -1.22、FG -1.42、MAE +0.00191。
结论：不放大、不长轮，正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```

### ASPP-ECA 轻量注意力失败，拒绝

在正式主线附近加入 ASPP 后 `ECA` 注意力，128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + ASPP-ECA | 128/64 | 24 | 63.70 | 57.15 | 0.02002 | 88.28 | 72.06 | 38.91 | 45.44 | 38.78 | 拒绝 |

```text
ASPP-ECA 相对 reference：mIoU -2.08，FG -2.49，MAE +0.00408。
结论：不放大、不长轮，正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```

### Low-SCSE 边缘轻注意力失败，拒绝

在正式主线附近给 low-level feature 加 `SCSE` 注意力，128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + Low-SCSE | 128/64 | 24 | 61.56 | 54.55 | 0.02214 | 89.84 | 74.55 | 29.46 | 41.14 | 36.78 | 拒绝 |

```text
Low-SCSE 相对 reference：mIoU -4.22，FG -5.09，MAE +0.00620，alternaria -10.99。
结论：不放大、不长轮，正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```

### LCSF alpha=0.3 微调失败，拒绝

在正式主线附近把 `LCSF alpha` 从 0.5 降到 0.3，128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix LCSF=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix LCSF=0.3 | 128/64 | 24 | 65.24 | 59.02 | 0.01766 | 89.84 | 75.61 | 43.10 | 46.96 | 39.12 | 拒绝 |

```text
LCSF alpha=0.3 相对 reference：alternaria +2.64，但 mIoU -0.54，FG -0.62，MAE +0.00172，brown -4.37。
结论：只作为类补偿负例保留，不放大、不长轮，正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```

### LCSF alpha=0.7 边际通过，升 192/96

将 `LCSF alpha` 从 0.5 提到 0.7 后，128/64 双 seed 有小幅涨点，但 Grade Acc 回落，需要更大快门确认。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LCSF alpha=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LCSF alpha=0.7 | 128/64 | 24 | 66.06 | 59.95 | 0.01572 | 89.84 | 76.27 | 40.83 | 47.82 | 43.72 | 边际通过 |

```text
LCSF alpha=0.7 相对 reference：mIoU +0.28，FG +0.31，MAE -0.00022。
代价：Grade Acc -1.56。
计划：升 192/96 双 seed 复核；若不继续涨，则保持正式主线 alpha=0.5。
```

### LCSF alpha=0.7 192/96 复核：保留支线，不替换主线

`LCSF alpha=0.7` 在 192/96 上仍有极小 mIoU/FG 涨点，但 MAE、rust、brown 变差，因此不替换正式主线。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LCSF alpha=0.5 | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | reference |
| LCSF alpha=0.7 | 192/96 | 24 | 67.56 | 61.66 | 0.01841 | 90.62 | 78.44 | 38.34 | 54.24 | 45.18 | 类补偿支线 |

```text
LCSF alpha=0.7 相对 reference：mIoU +0.09，FG +0.14，alternaria +1.00，gray +1.64。
代价：MAE +0.00103，rust -0.60，brown -1.08。
训练计划：保留 alpha=0.7 为 gray/alternaria 支线；正式主线仍保持 alpha=0.5，不启动 alpha=0.7 长轮。
```

### LGC alpha=0.7 微调失败，拒绝

把 `LGC alpha` 从 0.5 提到 0.7 后，128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC alpha=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LGC alpha=0.7 | 128/64 | 24 | 65.38 | 59.16 | 0.01750 | 88.28 | 76.31 | 41.59 | 47.34 | 39.80 | 拒绝 |

```text
LGC alpha=0.7 相对 reference：rust +0.73，alternaria +1.13，但 mIoU -0.40，FG -0.48，MAE +0.00156，Grade Acc -3.12，brown -3.69。
结论：拒绝，不放大、不长轮，正式主线仍保持 LGC alpha=0.5。
```

### LGC alpha=0.3 微调失败，LGC alpha 固定为 0.5

把 `LGC alpha` 从 0.5 降到 0.3 后，128/64 双 seed 未通过。结合 `alpha=0.7` 的失败，LGC alpha 不再继续扫。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LGC alpha=0.3 | 128/64 | 24 | 64.76 | 58.40 | 0.01662 | 89.06 | 75.44 | 36.92 | 46.78 | 41.92 | 拒绝 |
| LGC alpha=0.5 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | 最佳 |
| LGC alpha=0.7 | 128/64 | 24 | 65.38 | 59.16 | 0.01750 | 88.28 | 76.31 | 41.59 | 47.34 | 39.80 | 拒绝 |

```text
LGC alpha=0.3 相对 reference：mIoU -1.02，FG -1.24，MAE +0.00068，Grade Acc -2.34。
结论：LGC alpha 固定为 0.5；不继续扫 LGC alpha。
```

### Class weight alternaria=3.3 失败，loss 侧单独补 alternaria 关掉

保持 `SP decoder + LGC alpha=0.5 + LCSF alpha=0.5 + BalancedPrefix(gray=2,rust=1.5,alternaria=1.2,brown=1.2)` 不变，只把 CE class weights 从 `1.0 1.0 2.0 3.0 3.0 4.0` 调为 `1.0 1.0 2.0 3.3 3.0 4.0`。128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| BalancedPrefix + cls alt=3.3 | 128/64 | 24 | 65.10 | 58.84 | 0.01646 | 89.84 | 76.10 | 37.89 | 48.57 | 40.96 | 拒绝 |

```text
cls alt=3.3 相对 reference：mIoU -0.68，FG -0.80，MAE +0.00052，Grade Acc -1.56，alternaria -2.57，brown -2.53。
收益只有 rust +0.52、gray +0.66，无法抵消主指标和弱类损失。
训练计划：拒绝，不放大、不长轮；正式主线 class weights 保持 1.0 1.0 2.0 3.0 3.0 4.0。
后续不再优先做“单独提高 alternaria”的 prefix 或 CE weight。
```

### LBSB alpha=0.30 失败，边界锐化强度保持 0.25

保持正式主线不变，只把 `lesion_boundary_sharpen_alpha` 从 0.25 提到 0.30。128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LBSB alpha=0.25 reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| LBSB alpha=0.30 | 128/64 | 24 | 64.35 | 57.94 | 0.01684 | 90.62 | 76.47 | 38.22 | 46.17 | 38.17 | 拒绝 |

```text
LBSB alpha=0.30 相对 reference：mIoU -1.43，FG -1.70，MAE +0.00090，Grade Acc -0.78，brown -5.32。
收益只有 rust +0.89。
训练计划：拒绝，不放大、不长轮；正式主线 LBSB alpha 保持 0.25。
若继续 LBSB 方向，只能尝试弱化 0.20；不继续加大边界锐化强度。
```

### LBSB alpha=0.20 也失败，LBSB alpha 固定为 0.25

把 `lesion_boundary_sharpen_alpha` 从 0.25 降到 0.20 后，128/64 双 seed 仍未通过。结合 `alpha=0.30` 的失败，LBSB alpha 不再继续扫。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LBSB alpha=0.20 | 128/64 | 24 | 65.00 | 58.72 | 0.01590 | 91.40 | 77.07 | 36.08 | 46.78 | 42.88 | 拒绝 |
| LBSB alpha=0.25 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | 最佳 |
| LBSB alpha=0.30 | 128/64 | 24 | 64.35 | 57.94 | 0.01684 | 90.62 | 76.47 | 38.22 | 46.17 | 38.17 | 拒绝 |

```text
LBSB alpha=0.20 相对 reference：mIoU -0.78，FG -0.92，MAE -0.00004，rust +1.49，但 alternaria -4.38。
结论：LBSB alpha 固定为 0.25；不继续扫 LBSB alpha。
正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```

### 2026-06-10 09:59 RepConv seed23 进度

```text
进程: PID 134368，仍在运行。
当前进度: stderr 已到 Epoch 25/80。
当前 best: best_miou.txt = 70.75767196516155，来自 Epoch 20。
报告状态: reports/best_miou/metrics_summary.json 尚未生成。
决策: 继续等待 RepConv seed23，暂不启动 Low-CA，避免抢占 GPU。
```

### 2026-06-10 训练接力：RepConv full e80 seed23 进行中

新对话接力后先检查了仓库、快筛记录、summary 和后台进程。当前正式 reference 仍是
`LGC-LCSF-SP384-BalancedPrefix full e80`，双 seed 平均已记录为 mIoU 76.60、FG 72.22。

RepConv decoder 在 128/64 与 192/96 快门均通过，因此已经进入 full e80 长轮：

| 组别 | seed | 当前状态 | best mIoU | FG mIoU | 备注 |
|---|---:|---|---:|---:|---|
| BalancedPrefix reference full e80 | 11 | done | 77.23 | 72.97 | 当前单 seed 最高 |
| BalancedPrefix reference full e80 | 23 | done | 75.96 | 71.48 | reference 双 seed 均值约 76.60/72.22 |
| RepConv decoder full e80 | 11 | done | 76.66 | 72.30 | 单 seed 低于 reference seed11，不能提前升级 |
| RepConv decoder full e80 | 23 | running | 20 epoch best 70.76 | - | PID 134368，2026-06-10 09:26 启动，需等最终 best_miou 报告 |

阶段判断：

```text
RepConv full e80 暂不升级主线。
原因：seed11 已经低于同 seed reference 约 0.57 mIoU / 0.67 FG。
只有当 seed23 明显高于 reference seed23，且双 seed 平均超过 76.60 / 72.22，才升级为正式主线。
否则 RepConv 写成“中门强、长轮未稳定”的结构负例，训练计划回到 LGC-LCSF-SP384-BalancedPrefix。
下一步：等待 seed23 跑完，导出 metrics_summary/per_class/severity 后立刻合并判断并更新 summary。
```

### 2026-06-10 RepConv 等待期间的下一枪候选：Low-CA

约束：RepConv seed23 正在运行，训练结束会自动调用当前源码导出报告。因此在它完成前不能改
`src/models/deeplabv3plus/nets/deeplabv3_plus.py`，避免 checkpoint 结构和导出代码不一致。

已整理过的失败/关闭方向：

```text
CHFR-v0 / HFR + SP384: 已失败，直接高频残差注入太粗。
低层 SCSE: 已失败。
ASPP ECA: 已失败。
CutMix / MixUp: 已失败，关闭常规 batch-level 混合增强。
LBSB / LGC / LCSF alpha: 已扫，当前 0.25 / 0.5 / 0.5 固定。
component aux weights: lesion=0.4, boundary=0.2, center=0.2 固定。
```

如果 RepConv full e80 不涨，下一轮快筛改为：

```text
Low-CA = LGC-LCSF-SP384-BalancedPrefix + attention_low_type=ca

动机:
1. 不改源码，只用已有 CoordAttention 插件，风险低。
2. 低层 skip 保存纹理、边界和空间坐标，比 ASPP 再堆全局注意力更贴近当前短板。
3. CoordAttention 的 H/W 方向编码可能帮助小病斑在叶片上的位置/方向感。
4. 只插 low-level，不碰 decoder SP / LGC / LCSF 主线，避免破坏已经涨点的结构。

快筛脚本:
D:\Code\ATLDSD\scripts\run_fast_lgc_lcsf_sp384_balanced_low_ca_screen.ps1

门槛:
128/64、24 epoch、seed11/23 双 seed 平均必须超过 reference
LGC-LCSF-SP384-BalancedPrefix 128/64 的 65.78 mIoU / 59.64 FG。
若未超过，Low-CA 直接拒绝，不放大、不长轮。
```

### 2026-06-10 09:58 RepConv seed23 进度

```text
进程: PID 134368，仍在运行。
当前进度: stderr 已到 Epoch 25/80。
当前 best: Epoch 20 保存 best_miou_weights，mIoU = 70.76。
报告状态: reports/best_miou/metrics_summary.json 尚未生成，说明最终 auto-export 还未执行。
决策: 继续等待，不启动 Low-CA，避免抢占 GPU 或影响 RepConv 长轮。
```

### MixUp prob=0.1 失败，常规 batch-level 混合增强路线关闭

在 CutMix p=0.5/p=0.1 均失败后，继续做不切块的软标签混合 sanity check。保持正式主线不变，只开启 `mix-mode=mixup, mix-prob=0.1, mixup-alpha=0.4`。代码路径确认：MixUp 进入 `Softmax_CE_Loss` 软标签 CE，不是空跑。128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| MixUp p=0.1 | 128/64 | 24 | 64.31 | 57.98 | 0.02042 | 89.84 | 72.94 | 43.54 | 48.03 | 35.46 | 拒绝 |

```text
MixUp p=0.1 相对 reference：mIoU -1.47，FG -1.66，Severity MAE +0.00448，Grade Acc -1.56，rust -2.64，brown -8.03。
收益只有 alternaria +3.08、gray +0.12，不能抵消主指标和 brown/severity 的损失。
训练计划：拒绝，不放大、不长轮。
结合 CutMix p=0.5/p=0.1 失败，常规 batch-level MixUp/CutMix 增强路线关闭。
下一步应回到结构/语义级模块：优先寻找能保留病斑面积与边界的局部语义融合、类别条件特征校准或轻量 decoder refinement，而不是继续扫整图混合增强。
```

### RepConv decoder 通过快筛与中门，晋级 full e80 长轮

回到结构级候选后，测试 `decoder-conv-type=repconv`。本轮只给 `scripts/run_fast_deeplabv3plus_screen.ps1` 增加 `-DecoderConvType` 参数，默认仍为 `standard`，正式主线其他设置完全不变。RepConv 不是 PConv 路线：它用 3x3 + 1x1 + identity 的结构化卷积增强 decoder 局部表达，和当前 `SP decoder + LGC + LCSF + LBSB` 互补。

128/64 双 seed：

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| RepConv decoder | 128/64 | 24 | 67.50 | 61.66 | 0.01737 | 90.62 | 77.18 | 45.03 | 51.96 | 42.79 | 升 192/96 |

192/96 双 seed：

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 192/96 | 24 | 67.47 | 61.52 | 0.01738 | 90.62 | 79.04 | 37.34 | 52.60 | 46.26 | reference |
| RepConv decoder | 192/96 | 24 | 69.46 | 63.94 | 0.01372 | 90.10 | 79.37 | 39.59 | 57.46 | 51.14 | 晋级 full e80 |

```text
RepConv decoder 通过 128/64 和 192/96 双门。
192/96 相对 reference：mIoU +1.99，FG +2.42，Severity MAE -0.00366，rust +0.33，alternaria +2.25，gray +4.86，brown +4.88；唯一代价是 Grade Acc -0.52。
训练计划：启动 RepConv decoder full e80 至少 seed11/23，与当前 LGC-LCSF-SP384-BalancedPrefix full e80 reference 对比。
若 full e80 仍保持 mIoU/FG 和 gray/brown 正增益且 severity 不劣化，则把正式主线升级为 LGC-LCSF-SP384-RepConv-BalancedPrefix。
```

### 2026-06-10 RepConv full e80 通过，正式主线升级

RepConv decoder full e80 seed23 已完成并导出 `reports/best_miou/metrics_summary.json`。结合 seed11/23 双 seed，RepConv 不再是候选，而是新的正式主线。

| 组别 | seed | mIoU | FG mIoU | leaf | rust | alternaria | gray | brown | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix full e80 | 11 | 77.23 | 72.97 | 95.12 | 85.12 | 57.62 | 68.50 | 57.75 | old leader |
| BalancedPrefix full e80 | 23 | 75.96 | 71.48 | 94.72 | 84.68 | 54.57 | 67.82 | 54.88 | old leader |
| RepConv decoder full e80 | 11 | 76.66 | 72.30 | 95.67 | 84.63 | 59.50 | 66.70 | 54.98 | lower seed11 |
| RepConv decoder full e80 | 23 | 77.21 | 72.96 | 95.69 | 84.57 | 57.63 | 69.99 | 56.89 | strong seed23 |
| BalancedPrefix full e80 avg | - | 76.60 | 72.22 | 94.92 | 84.90 | 56.10 | 68.16 | 56.31 | reference |
| RepConv decoder full e80 avg | - | 76.94 | 72.63 | 95.68 | 84.60 | 58.57 | 68.34 | 55.93 | UPGRADE |

```text
RepConv full e80 相对原正式主线：avg mIoU +0.34，avg FG +0.40。
分病害：alternaria +2.47，gray +0.19；rust -0.30，brown -0.38，leaf +0.04。
seed11 有小回撤，但 seed23 收益足以抵消，并且 final gate 判定为 UPGRADE。
训练计划：正式主线升级为 LGC-LCSF-SP384-RepConv-BalancedPrefix。
Low-CA 不再作为“RepConv 失败后的回退枪”，后续若继续试，需要改为 RepConv 主线上的 low-level CA 叠加快筛。
```

### 2026-06-10 RepConv+Low-CA 快筛失败，不放大

在 RepConv 成为正式主线后，重新测试 low-level CoordAttention 叠加，而不是旧的 standard decoder 回退线。配置保持 `SP decoder + RepConv + LGC + LCSF + BalancedPrefix`，只额外开启 `attention_low_type=ca`。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + Low-CA | 128/64 | 24 | 66.78 | 60.80 | 91.19 | 75.70 | 43.90 | 53.38 | 39.85 | reject |

```text
RepConv+Low-CA 相对 RepConv reference：mIoU -0.72，FG -0.85。
收益只集中在 gray +1.42，但 rust -1.47、alternaria -1.13、brown -2.94。
训练计划：拒绝，不放大、不长轮。
后续不再优先扫 low-level attention；下一枪应改为 decoder 局部 refinement、类别条件校准或病斑语义校准，硬门槛是不能再牺牲 brown。
```

### 2026-06-10 RepConv+Decoder-CAA 快筛失败，SP decoder 保留

测试 `attention_decoder_type=caa`，即用 Context Anchor Attention 替换当前主线的 SP decoder attention。该实验用于验证 decoder refinement 方向，但结果说明替换 SP 代价过高。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + Decoder-CAA | 128/64 | 24 | 64.13 | 57.65 | 90.89 | 74.24 | 39.41 | 48.84 | 34.86 | reject |

```text
RepConv+Decoder-CAA 相对 RepConv reference：mIoU -3.37，FG -4.01。
所有前景类都下降，尤其 brown -7.93、alternaria -5.62。
训练计划：拒绝，不放大、不长轮。
结论：不要替换 SP decoder attention；后续 decoder 方向必须是可叠加的轻量 refinement，或者回到训练/类别校准。
```

### 2026-06-10 RepConv+High-SimAM 边际失败，保留 brown/gray 支线线索

测试 `attention_high_type=simam`，保留 `SP decoder + RepConv + LGC + LCSF + BalancedPrefix` 不变，只在高层 backbone feature 进入 ASPP 前加入无参数 SimAM。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-SimAM | 128/64 | 24 | 67.45 | 61.63 | 90.89 | 76.00 | 44.01 | 52.79 | 44.46 | reject / side clue |

```text
RepConv+High-SimAM 相对 RepConv reference：mIoU -0.06，FG -0.02，主指标边际未过线。
收益：brown +1.67，gray +0.83。
代价：rust -1.17，alternaria -1.02，leaf -0.42。
训练计划：不升级主线、不放大、不长轮；但保留为 brown/gray 支线线索。
后续若继续 attention，应优先找不伤 rust/alternaria 的高层轻量模块，而不是 low-level CA 或替换 SP。
```

### 2026-06-10 RepConv+High-MLCA 快筛失败

在 High-SimAM 边际未过线后，同一 high-level 插入点测试 MLCA。该模块没有保留 SimAM 的 brown/gray 线索，反而大幅损伤弱类。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-MLCA | 128/64 | 24 | 64.10 | 57.60 | 90.98 | 75.18 | 35.64 | 49.50 | 36.68 | reject |

```text
RepConv+High-MLCA 相对 RepConv reference：mIoU -3.41，FG -4.06。
弱类严重下降：alternaria -9.39，brown -6.11，gray -2.46，rust -1.99。
训练计划：拒绝，不放大、不长轮。
当前正式主线仍是 LGC-LCSF-SP384-RepConv-BalancedPrefix；High-SimAM 只保留为 brown/gray 支线线索。
```

### 2026-06-10 RepConv+High-ECA 快筛失败

在 High-SimAM 边际未过线、High-MLCA 失败后，继续测试更轻量的 high-level ECA。结果显示普通高层通道注意力不适合当前 RepConv 主线。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-ECA | 128/64 | 24 | 63.30 | 56.65 | 90.96 | 74.69 | 38.54 | 46.30 | 32.74 | reject |

```text
RepConv+High-ECA 相对 RepConv reference：mIoU -4.20，FG -5.00。
brown -10.05，alternaria -6.49，gray -5.65，rust -2.49。
训练计划：拒绝，不放大、不长轮。
高层注意力方向只保留 High-SimAM 作为边际 brown/gray 线索；ECA/MLCA 关闭。
```

### 2026-06-10 RepConv brown=1.5 通过 128/64 快筛，晋级 192/96

为修复 RepConv full e80 的 brown 轻微回撤，保留结构不变，只把 prefix sampling 的 brown 从 1.2 提高到 1.5。该候选通过 128/64 双 seed 快筛。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv brown=1.5 | 128/64 | 24 | 67.77 | 62.00 | 91.04 | 74.69 | 48.42 | 49.15 | 46.72 | promote to 192/96 |

```text
RepConv brown=1.5 相对 RepConv reference：mIoU +0.26，FG +0.35。
收益：brown +3.93，alternaria +3.39。
代价：rust -2.48，gray -2.81，leaf -0.28。
训练计划：晋级 192/96 双 seed 中门复核；不能直接替换 full 主线。
中门必须继续超过 RepConv reference 的 mIoU/FG，且 rust/gray 代价不能扩大。
```

### 2026-06-10 RepConv brown=1.5 192/96 中门复核失败

128/64 的 brown=1.5 小涨没有在 192/96 复现。该结果说明单独提高 brown prefix sampling 会放大分布偏置，不适合作为 RepConv 主线的下一步。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 192/96 | 24 | 69.46 | 63.94 | 92.15 | 79.37 | 39.59 | 57.47 | 51.13 | reference |
| RepConv brown=1.5 | 192/96 | 24 | 68.22 | 62.38 | 92.69 | 78.66 | 36.87 | 54.63 | 49.06 | reject |

```text
RepConv brown=1.5 相对 RepConv 192/96 reference：mIoU -1.24，FG -1.56。
seed11: -0.78 mIoU；seed23: -1.69 mIoU。
只有 leaf +0.54；rust -0.71，alternaria -2.72，gray -2.84，brown -2.07。
训练计划：拒绝，不放大、不长轮；正式主线保持 LGC-LCSF-SP384-RepConv-BalancedPrefix，sampling 回到 brown=1.2。
后续不再单独提高 brown prefix；若继续救 brown/gray，优先找低侵入语义校准或更稳的轻量模块。
```

### 2026-06-10 RepConv+ASPP-ScSE 单 seed 快速拒绝

在 RepConv 主线基础上保留 decoder SP，只在 ASPP 输出后叠加 `attention_aspp_type=scse`。该候选 seed11 已明显失败，因此终止 seed23，不进入双 seed 完整 gate。

| 组别 | train/val | epoch | seed | mIoU | FG mIoU | leaf IoU | rust IoU | alternaria IoU | gray IoU | brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 11 | 66.82 | 60.88 | 90.78 | 78.15 | 49.29 | 54.96 | 31.20 | reference |
| RepConv + ASPP-ScSE | 128/64 | 24 | 11 | 63.73 | 57.21 | 90.44 | 76.41 | 36.48 | 49.55 | 33.17 | reject |

```text
RepConv+ASPP-ScSE seed11 相对 reference：mIoU -3.08，FG -3.66。
brown +1.97 不能抵消 alternaria -12.80、gray -5.41、rust -1.74。
训练计划：拒绝，不补 seed23、不放大、不长轮。
ASPP 后强 spatial+channel recalibration 暂不优先；下一步转向更弱模块或训练校准。
```

### 2026-06-10 RepConv+SeverityConsistency0.05 边际失败

保持 RepConv 主线结构和 sampling 不变，只增加弱 severity consistency loss：`--severity-consistency-loss true --severity-consistency-weight 0.05 --severity-loss-type smooth_l1`。该候选非常接近 gate，但严格未过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 0.01737 | 90.62 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + Severity0.05 | 128/64 | 24 | 67.46 | 61.64 | 0.01678 | 89.84 | 90.96 | 76.36 | 44.76 | 52.81 | 43.32 | reject / side clue |

```text
RepConv+Severity0.05 相对 RepConv reference：mIoU -0.05，FG -0.01，严格 gate 未过。
seed11 下降 -1.24 mIoU，seed23 上升 +1.15 mIoU。
gray +0.85、brown +0.53、Severity MAE 0.01737 -> 0.01678；但 leaf/rust/alternaria 下降，Grade Acc 90.62% -> 89.84%。
训练计划：拒绝，不升 192/96，不长轮。
保留为边际线索；若继续训练校准，优先试更弱 severity_weight=0.02，或在长轮中作为附加对照。
```

### 2026-06-10 RepConv+SeverityConsistency0.02 单 seed 快速拒绝

将 severity consistency 权重从 0.05 降到 0.02 后，seed11 反而更差，因此不补 seed23。

| 组别 | train/val | epoch | seed | mIoU | FG mIoU | leaf IoU | rust IoU | alternaria IoU | gray IoU | brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 11 | 66.82 | 60.88 | 90.78 | 78.15 | 49.29 | 54.96 | 31.20 | reference |
| RepConv + Severity0.02 | 128/64 | 24 | 11 | 64.81 | 58.49 | 90.52 | 77.83 | 40.68 | 52.12 | 31.33 | reject |

```text
RepConv+Severity0.02 seed11 相对 reference：mIoU -2.01，FG -2.38。
brown 只 +0.13，却造成 alternaria -8.61、gray -2.85。
训练计划：拒绝，不补 seed23、不放大、不长轮。
关闭 severity consistency sweep；0.05 仅保留为边际 gray/brown/MAE 线索。
```

### 2026-06-10 RepConv+High-GCT 通过 128/64 快筛，晋级 192/96

从 `external/PlugNPlay-Modules/GCTattention.py` 接入 Gated Channel Transformation，注册为 `attention_high_type=gct`。该模块初始近似 identity，作为比 ECA/MLCA 更温和的 high-level attention 候选。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 67.51 | 61.65 | 91.31 | 77.18 | 45.03 | 51.96 | 42.79 | reference |
| RepConv + High-GCT | 128/64 | 24 | 67.78 | 62.00 | 91.27 | 76.42 | 45.91 | 52.71 | 43.66 | promote to 192/96 |

```text
RepConv+High-GCT 相对 RepConv reference：mIoU +0.28，FG +0.34。
seed11 强涨 +1.68 mIoU / +2.00 FG，seed23 下滑 -1.13 mIoU / -1.32 FG。
per-class：alternaria +0.88，gray +0.76，brown +0.87；代价是 rust -0.76，leaf -0.05。
训练计划：晋级 192/96 双 seed 中门复核；中门若继续 mIoU/FG 双过线，再考虑 full e80。
```

### 2026-06-10 RepConv+High-GCT 192/96 中门失败

High-GCT 在 128/64 过 gate，但扩大到 192/96 后未能复现收益，因此不进入 full e80。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg leaf IoU | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 192/96 | 24 | 69.46 | 63.94 | 92.15 | 79.37 | 39.59 | 57.47 | 51.13 | reference |
| RepConv + High-GCT | 192/96 | 24 | 69.02 | 63.37 | 92.55 | 78.32 | 39.24 | 55.62 | 51.14 | reject |

```text
RepConv+High-GCT 相对 RepConv 192/96 reference：mIoU -0.43，FG -0.57。
seed11 只小涨 +0.04 mIoU 且 FG -0.07；seed23 下降 -0.91 mIoU / -1.06 FG。
per-class：leaf +0.40，brown +0.01；rust -1.05，alternaria -0.35，gray -1.85。
训练计划：拒绝，不升 full e80，不替换正式主线。
GCT 保留为已接入的边际模块线索；正式主线仍为 LGC-LCSF-SP384-RepConv-BalancedPrefix。
```

### 2026-06-10 RepConv+PPM 128/64 快速拒绝

保持 RepConv 主线不变，只开启 `--use-ppm true`，PPM 位于 ASPP/LGLC 后。seed11 完整失败，seed23 中途也没有翻盘迹象，因此止损。

| 组别 | train/val | epoch | seed | mIoU | FG mIoU | leaf IoU | rust IoU | alternaria IoU | gray IoU | brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RepConv mainline | 128/64 | 24 | 11 | 66.82 | 60.88 | 90.78 | 78.15 | 49.29 | 54.96 | 31.20 | reference |
| RepConv + PPM | 128/64 | 24 | 11 | 65.29 | 59.05 | 90.69 | 76.43 | 42.39 | 52.54 | 33.21 | reject |

```text
RepConv+PPM seed11 相对 reference：mIoU -1.53，FG -1.82。
brown +2.01，但 alternaria -6.90、gray -2.42、rust -1.71。
seed23 到 epoch20 best mIoU 64.47，低于 reference seed23 68.19，已无翻盘必要。
训练计划：拒绝，不补 seed23、不升 192/96、不长轮。
```

### Component boundary aux=0.3 失败，辅助头权重保持 0.4/0.2/0.2

保持正式主线不变，只把 `component-boundary-weight` 从 0.2 提到 0.3，`lesion=0.4, center=0.2` 不变。128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| Boundary aux=0.3 | 128/64 | 24 | 64.23 | 57.81 | 0.01759 | 87.50 | 75.63 | 36.75 | 45.32 | 40.88 | 拒绝 |

```text
Boundary aux=0.3 相对 reference：mIoU -1.55，FG -1.83，MAE +0.00165，Grade Acc -3.90，alternaria -3.71，gray -2.59，brown -2.61。
收益只有 rust +0.05。
训练计划：拒绝，不放大、不长轮；正式主线 component auxiliary 权重保持 lesion=0.4, boundary=0.2, center=0.2。
后续不要优先增强 boundary auxiliary supervision。
```

### Component boundary aux=0.1 也失败，boundary aux 权重固定为 0.2

把 `component-boundary-weight` 从 0.2 降到 0.1 后，128/64 双 seed 仍未通过。结合 `boundary=0.3` 的失败，boundary auxiliary weight 不再继续扫。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boundary aux=0.1 | 128/64 | 24 | 64.85 | 58.55 | 0.01677 | 89.84 | 76.06 | 36.30 | 48.10 | 41.72 | 拒绝 |
| Boundary aux=0.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | 最佳 |
| Boundary aux=0.3 | 128/64 | 24 | 64.23 | 57.81 | 0.01759 | 87.50 | 75.63 | 36.75 | 45.32 | 40.88 | 拒绝 |

```text
Boundary aux=0.1 相对 reference：mIoU -0.93，FG -1.09，MAE +0.00083，Grade Acc -1.56，alternaria -4.16。
结论：boundary auxiliary weight 固定为 0.2；不继续扫 boundary aux 权重。
正式主线 component auxiliary 权重仍保持 lesion=0.4, boundary=0.2, center=0.2。
```

### Component lesion aux sweep 失败，component auxiliary 权重全部固定

保持 `boundary=0.2, center=0.2`，分别测试 `component-lesion-weight=0.3` 和 `0.5`。128/64 双 seed 均未通过。结合 boundary/center sweep，component auxiliary 权重全部收口。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lesion aux=0.3 | 128/64 | 24 | 65.19 | 58.94 | 0.01836 | 91.40 | 75.06 | 38.25 | 48.68 | 42.06 | 拒绝 |
| Lesion aux=0.4 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | 最佳 |
| Lesion aux=0.5 | 128/64 | 24 | 64.48 | 58.10 | 0.01668 | 90.62 | 75.21 | 35.82 | 46.55 | 42.27 | 拒绝 |

```text
Lesion aux=0.3 相对 reference：mIoU -0.59，FG -0.70，MAE +0.00242，alternaria -2.21。
Lesion aux=0.5 相对 reference：mIoU -1.30，FG -1.54，MAE +0.00074，alternaria -4.64。
结论：lesion auxiliary weight 固定为 0.4；不继续扫 component auxiliary 权重。
正式主线 component auxiliary 权重固定为 lesion=0.4, boundary=0.2, center=0.2。
```

### Component center aux=0.1 失败，不放大

保持正式主线不变，只把 `component-center-weight` 从 0.2 降到 0.1，`lesion=0.4, boundary=0.2` 不变。128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| Center aux=0.1 | 128/64 | 24 | 65.02 | 58.74 | 0.01802 | 89.06 | 75.43 | 38.36 | 49.62 | 39.64 | 拒绝 |

```text
Center aux=0.1 相对 reference：mIoU -0.76，FG -0.90，MAE +0.00208，Grade Acc -2.34，alternaria -2.10，brown -3.85。
收益只有 gray +1.71。
训练计划：拒绝，不放大、不长轮；component center weight 暂不降到 0.1。
```

### Component center aux=0.3 也失败，center aux 权重固定为 0.2

把 `component-center-weight` 从 0.2 提到 0.3 后，128/64 双 seed 仍未通过。结合 `center=0.1` 的失败，center auxiliary weight 不再继续扫。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Center aux=0.1 | 128/64 | 24 | 65.02 | 58.74 | 0.01802 | 89.06 | 75.43 | 38.36 | 49.62 | 39.64 | 拒绝 |
| Center aux=0.2 | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | 最佳 |
| Center aux=0.3 | 128/64 | 24 | 64.82 | 58.48 | 0.01749 | 91.40 | 75.59 | 39.55 | 46.01 | 40.44 | 拒绝 |

```text
Center aux=0.3 相对 reference：mIoU -0.96，FG -1.16，MAE +0.00155，brown -3.05。
结合 center=0.1 失败，center auxiliary weight 固定为 0.2；不继续扫 center aux 权重。
正式主线 component auxiliary 权重仍保持 lesion=0.4, boundary=0.2, center=0.2。
```

### CutMix prob=0.5 失败，常规块级混合增强不作为下一主线

保持正式主线模块、采样和 class weights 不变，只开启 `mix-mode=cutmix, mix-prob=0.5, cutmix-alpha=1.0`。128/64 双 seed 未通过。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| CutMix p=0.5 | 128/64 | 24 | 64.24 | 57.85 | 0.01873 | 87.50 | 73.77 | 39.07 | 44.73 | 41.50 | 拒绝 |

```text
CutMix p=0.5 相对 reference：mIoU -1.54，FG -1.79，MAE +0.00279，Grade Acc -3.90，四个病害类全部下降。
训练计划：拒绝，不放大、不长轮。
常规块级 CutMix 会破坏小病斑边界与严重度估计；后续不把 CutMix p=0.5 作为主候选。
若继续数据增强方向，只考虑极低概率 CutMix 或病斑语义级增强。
```

### CutMix prob=0.1 也失败，CutMix 方向关闭

把 `mix-prob` 从 0.5 降到 0.1 后，128/64 双 seed 仍未通过。结合 p=0.5 失败，块级 CutMix 不再作为后续候选。

| 组别 | train/val | epoch | avg mIoU | avg FG mIoU | avg Severity MAE | avg Grade Acc | avg rust IoU | avg alternaria IoU | avg gray IoU | avg brown IoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedPrefix reference | 128/64 | 24 | 65.78 | 59.64 | 0.01594 | 91.40 | 75.58 | 40.46 | 47.91 | 43.49 | reference |
| CutMix p=0.1 | 128/64 | 24 | 64.20 | 57.81 | 0.01847 | 89.84 | 74.48 | 41.21 | 45.88 | 37.26 | 拒绝 |
| CutMix p=0.5 | 128/64 | 24 | 64.24 | 57.85 | 0.01873 | 87.50 | 73.77 | 39.07 | 44.73 | 41.50 | 拒绝 |

```text
CutMix p=0.1 相对 reference：mIoU -1.58，FG -1.83，MAE +0.00253，Grade Acc -1.56，brown -6.23。
唯一收益是 alternaria +0.75，不足以保留。
训练计划：关闭块级 CutMix 方向；后续若做增强，必须是病斑语义级增强，而不是整块 CutMix。
正式主线仍保持 LGC-LCSF-SP384-BalancedPrefix full e80。
```
