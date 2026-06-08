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
