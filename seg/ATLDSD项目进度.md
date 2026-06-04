# ATLDSD 语义分割项目交接笔记

更新时间：2026-06-04 23:07  
目的：让下一个完全不了解上下文的 AI 能直接接手当前项目。

## 1. 项目目标

用 ATLDSD 苹果叶病害数据集做语义分割，并进一步支持病害严重度判断。

数据标签结构：

```text
0 background
1 leaf
2 rust
3 alternaria_leaf_spot
4 gray_spot
5 brown_spot
```

严重度定义：

```text
severity = lesion pixels / leaf pixels
lesion classes = 2, 3, 4, 5
leaf area = class 1 + classes 2, 3, 4, 5
```

## 2. 路径与仓库

```text
项目目录:
D:\Code\ATLDSD

数据集:
D:\dataset\ATLDSD\VOCdevkit

Obsidian 笔记:
D:\soft\obsidian_notion\seg\ATLDSD项目进度.md

GitHub:
https://github.com/yezi56/SCLA-Net-ATLDSD
```

注意：以后修改项目进度时，同时更新：

```text
D:\Code\ATLDSD\seg\ATLDSD项目进度.md
D:\soft\obsidian_notion\seg\ATLDSD项目进度.md
```

并且每次代码或笔记变更后都要提交并推送 GitHub。

## 3. 当前统一编号

以后不要再混用 B0 / E / M / S / L。  
从现在开始只用：

```text
主线0、主线1、主线2、主线3、主线4
附加实验A、附加实验B、附加实验C
```

编号含义：

```text
主线0:
DeepLabV3+ + MobileNetV3-Large
作用: 最强普通 baseline
旧编号: B0-V3

主线1:
主线0 + Component Auxiliary Heads
作用: 第一个结构模块实验
旧编号: E2

主线2:
主线1 + PConv
作用: 下一步首选，真正改 decoder 结构

主线3:
主线1 + CAA
作用: 注意力模块实验

主线4:
主线1 + PConv + CAA
作用: 最终主模型候选

附加实验A:
主线1 + Severity Consistency Loss
作用: loss 辅助消融，不是主结构创新
旧编号: M2

附加实验B:
主线1 + LBFTLoss
作用: loss 对照

附加实验C:
EfficientNet-B4 + CAA + PConv + LBFTLoss
作用: 旧完整链路强对照，不作为主线
```

## 4. 已完成关键实验

### 主线0：普通强 baseline

```text
结构:
DeepLabV3+ + MobileNetV3-Large

结果:
mIoU = 71.72
FG mIoU = 66.58
Accuracy = 97.76
Severity MAE ≈ 0.0124
Severity grade accuracy ≈ 95.12%
Params = 11.73M
FLOPs = 15.28G
FPS = 98.80

结论:
这是当前最强普通 baseline。
后续主线实验必须和它比较。
```

### EfficientNet-B4 baseline

```text
结构:
DeepLabV3+ + EfficientNet-B4

结果:
mIoU = 65.59
FG mIoU = 59.59
Accuracy = 96.44
Params = 32.48M
FLOPs = 51.30G
FPS = 52.77

结论:
B4 更重、更慢、效果更差。
不要把 B4 当主 backbone。
它只保留到附加实验C作为旧链路对照。
```

### SCLP 病斑 copy-paste 增强

```text
SCLP 0.7:
mIoU = 68.97

SCLP 0.3:
mIoU = 69.90

结论:
均低于主线0的 71.72。
SCLP 不作为主创新，只能作为失败消融或辅助讨论。
```

### 主线1：Component Auxiliary Heads

```text
结构:
DeepLabV3+ + MobileNetV3-Large + Component Auxiliary Heads

新增模块:
lesion auxiliary head
boundary auxiliary head
center auxiliary head

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_150

best checkpoint:
weights\best_miou_weights.pth

结果:
mIoU = 72.11
FG mIoU = 67.03
Pixel Accuracy = 97.82
Severity MAE = 0.01212
Severity RMSE = 0.03583
Severity Pearson = 0.8691
Severity Spearman = 0.9141
Severity grade accuracy = 94.31%
Params = 11.73M
FLOPs = 15.29G
FPS = 101.10

对比主线0:
mIoU +0.39
Severity MAE 略好
Severity grade accuracy 略低

结论:
主线1有小幅正收益。
它是当前结构主线起点。
```

## 5. 当前正在训练

当前正在跑的是附加实验A，不是主线。

```text
附加实验A:
主线1 + Severity Consistency Loss

结构:
DeepLabV3+ + MobileNetV3-Large
+ Component Auxiliary Heads
+ Severity Consistency Loss

PID:
28052

启动时间:
2026-06-04 22:36

当前检查时间:
2026-06-04 23:07

当前进度:
约 epoch54/150

epoch50:
mIoU = 69.40
mPA = 86.04
Accuracy = 96.74

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_severity_150

日志:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_severity_150\train_stdout.log
```

定位：

```text
附加实验A只是 loss 辅助消融。
它不是主结构创新。
如果 GPU 资源冲突，优先跑主线2，而不是保附加实验A。
```

检查命令：

```powershell
Get-Process -Id 28052 -ErrorAction SilentlyContinue
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_severity_150\train_stdout.log' -Tail 80
```

## 6. 当前代码能力

已经实现：

```text
Component Auxiliary Heads:
src\models\deeplabv3plus\nets\deeplabv3_plus.py
src\models\deeplabv3plus\nets\deeplabv3_training.py
src\models\deeplabv3plus\utils\utils_fit.py

Severity Consistency Loss:
src\models\deeplabv3plus\nets\deeplabv3_training.py
src\models\deeplabv3plus\utils\utils_fit.py
src\models\deeplabv3plus\train.py

PConv:
src\models\deeplabv3plus\nets\deeplabv3_plus.py
参数: --decoder-conv-type pconv

CAA:
src\modules\plugins\modules.py
src\modules\plugins\factory.py
参数示例: --attention-type caa 或按具体插入点设置

LBFTLoss:
src\models\deeplabv3plus\train.py
src\models\deeplabv3plus\utils\utils_fit.py
参数: --lbft-loss true
```

已有脚本：

```text
主线1 Windows:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_150.ps1

主线1 Linux:
scripts\run_ubuntu_component_aux_v3.sh
./scripts/run_ubuntu.sh component_aux

附加实验A Windows:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_severity_150.ps1

附加实验A Linux:
scripts\run_ubuntu_component_aux_severity_v3.sh
./scripts/run_ubuntu.sh component_aux_severity
```

## 7. 下一步路线

下一步真正主线只做结构模块，不再先加 loss。

### 下一步首选：主线2

```text
主线2:
主线1 + PConv

目标:
验证 PConv decoder 是否改善小病斑、病斑边界和不规则局部纹理。

变量控制:
相对主线1，只改 decoder_conv_type:
standard -> pconv

不要同时加 CAA、LBFTLoss、Severity Loss。
```

需要新建：

```text
Windows 脚本:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_pconv_150.ps1

Linux 脚本:
scripts\run_ubuntu_component_aux_pconv_v3.sh

run_ubuntu.sh 入口:
./scripts/run_ubuntu.sh component_aux_pconv

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150
```

主线2判断标准：

```text
如果主线2 mIoU > 72.11:
  PConv 保留。

如果 mIoU 接近 72.11，但 FG mIoU 或 brown_spot / gray_spot IoU 提升:
  PConv 可作为边界/小病斑增强模块保留。

如果 mIoU 和小病斑类都下降:
  PConv 不进入主模型，只记录为失败消融。
```

### 主线2之后

```text
主线3:
主线1 + CAA
只验证注意力模块。

主线4:
主线1 + PConv + CAA
只有主线2或主线3至少有一个有效时再跑。
```

## 8. 论文逻辑

不要把论文写成“堆 CAA、PConv、LBFTLoss”。  
当前更合理的论文逻辑是：

```text
1. ATLDSD 的病害严重度来自 lesion / leaf。
2. 普通 6 类 softmax 没有显式学习病斑组件结构。
3. 主线1加入 lesion / boundary / center 辅助头，让模型学习病斑组件。
4. 主线2加入 PConv decoder，增强局部病斑边界和小目标建模。
5. 主线3/4再判断 CAA 是否作为注意力增强模块保留。
6. Severity Loss 和 LBFTLoss 只作为附加 loss 消融，不抢主结构创新。
```

论文表格命名建议：

```text
Baseline: 主线0
Ours-1: 主线1
Ours-2: 主线2
Ours-3: 主线3
Ours-Final: 主线4
Auxiliary Ablation: 附加实验A / 附加实验B
Legacy Comparator: 附加实验C
```

## 9. 给下一个 AI 的执行规则

```text
1. 不要再发明新的 E/M/S/L 编号。
2. 不要把 loss 当成主结构模块。
3. 每次改代码，Windows 脚本和 Linux 脚本都要一起更新。
4. 每次训练启动后，记录 PID、输出目录、脚本、关键参数。
5. 每次训练完成后，导出 best_miou report，并记录 mIoU、FG mIoU、severity MAE、grade accuracy、Params、FLOPs、FPS。
6. 每次写笔记，同时更新仓库笔记和 Obsidian 笔记。
7. 每次有代码或仓库笔记修改，都要 git commit 并 push。
```
