# ATLDSD 训练路线笔记

更新时间：2026-06-05 08:56  
用途：项目交接。下一个 AI 只读这份笔记，就能知道当前训练走到哪、为什么这么走、下一步该做什么。

## 0. 当前结论

```text
当前最强普通 baseline:
主线0 = DeepLabV3+ + MobileNetV3-Large
mIoU = 71.72

当前最强结构模型:
主线1 = 主线0 + Component Auxiliary Heads
mIoU = 72.11
相对主线0提升 +0.39

当前正在训练:
主线2 = 主线1 + PConv
只改 decoder_conv_type: standard -> pconv

刚完成:
附加实验A = 主线1 + Severity Consistency Loss
best mIoU = 72.12
Severity MAE = 0.01147
它是 loss 消融，不是主结构创新。
```

重要纠偏：

```text
不要再把 loss 当主创新。
不要再混用 E / M / S / L 编号。
以后只用“主线0-4”和“附加实验A-C”。
```

## 1. 数据集与任务

```text
任务:
ATLDSD 苹果叶病害语义分割 + 病害严重度估计

数据集:
D:\dataset\ATLDSD\VOCdevkit

类别:
0 background
1 leaf
2 rust
3 alternaria_leaf_spot
4 gray_spot
5 brown_spot

严重度:
severity = lesion pixels / leaf pixels
lesion = classes 2, 3, 4, 5
leaf = class 1 + classes 2, 3, 4, 5
```

项目路径：

```text
代码:
D:\Code\ATLDSD

GitHub:
https://github.com/yezi56/SCLA-Net-ATLDSD

仓库笔记:
D:\Code\ATLDSD\seg\ATLDSD项目进度.md

Obsidian 笔记:
D:\soft\obsidian_notion\seg\ATLDSD项目进度.md
```

## 2. 统一编号

### 主线实验

| 编号 | 结构 | 目的 | 状态 |
|---|---|---|---|
| 主线0 | DeepLabV3+ + MobileNetV3-Large | 最强普通 baseline | 已完成 |
| 主线1 | 主线0 + Component Auxiliary Heads | 第一个结构模块 | 已完成 |
| 主线2 | 主线1 + PConv | decoder 结构模块 | 下一步首选 |
| 主线3 | 主线1 + CAA | 注意力模块 | 待跑 |
| 主线4 | 主线1 + PConv + CAA | 最终组合候选 | 待跑 |

### 附加实验

| 编号 | 结构 | 定位 | 状态 |
|---|---|---|---|
| 附加实验A | 主线1 + Severity Consistency Loss | loss 消融 | 已完成 |
| 附加实验B | 主线1 + LBFTLoss | loss 对照 | 待跑 |
| 附加实验C | EfficientNet-B4 + CAA + PConv + LBFTLoss | 旧链路强对照 | 待跑 |

## 3. 已完成训练结果

| 实验 | 结构 | mIoU | FG mIoU | Acc | Severity MAE | Grade Acc | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 主线0 | DeepLabV3+ + MobileNetV3-Large | 71.72 | 66.58 | 97.76 | 约0.0124 | 约95.12 | 最强普通 baseline |
| 主线1 | 主线0 + Component Auxiliary Heads | 72.11 | 67.03 | 97.82 | 0.01212 | 94.31 | 当前最强结构起点 |
| 附加实验A | 主线1 + Severity Consistency Loss | 72.12 | 67.06 | 97.78 | 0.01147 | 93.90 | mIoU 基本持平，严重度 MAE 更好 |
| B4 baseline | DeepLabV3+ + EfficientNet-B4 | 65.59 | 59.59 | 96.44 | 未主用 | 未主用 | 更重更差，弃作主干 |
| SCLP 0.7 | 主线0 + 强 copy-paste | 68.97 | 未主用 | 未主用 | 未主用 | 未主用 | 失败增强 |
| SCLP 0.3 | 主线0 + 弱 copy-paste | 69.90 | 未主用 | 未主用 | 未主用 | 未主用 | 仍低于主线0 |

主线1细节：

```text
新增模块:
lesion auxiliary head
boundary auxiliary head
center auxiliary head

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_150

best checkpoint:
weights\best_miou_weights.pth

复杂度:
Params = 11.73M
FLOPs = 15.29G
FPS = 101.10
```

## 4. 当前正在训练

```text
当前实验:
主线2 = 主线1 + PConv

定位:
真正的结构模块实验。
相对主线1，只改 decoder_conv_type: standard -> pconv。

PID:
41672

启动时间:
2026-06-05 08:56

当前检查时间:
2026-06-05 08:56

当前进度:
epoch1/150 已开始，仍在运行

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150
```

当前判断：

```text
主线2是当前最重要的结构实验。
它只验证 PConv decoder 是否能改善小病斑和边界。
不要在这次训练中混入 CAA、LBFTLoss、Severity Loss 或 SCLP。
```

检查命令：

```powershell
Get-Process -Id 41672 -ErrorAction SilentlyContinue
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150\train_stdout.log' -Tail 80
```

## 5. 当前代码能力

| 能力 | 是否已有 | 入口或参数 |
|---|---|---|
| Component Auxiliary Heads | 已有 | `--component-aux true` |
| Severity Consistency Loss | 已有 | `--severity-consistency-loss true` |
| PConv decoder | 已有 | `--decoder-conv-type pconv` |
| CAA attention | 已有 | `--attention-type caa` 或分插入点参数 |
| LBFTLoss | 已有 | `--lbft-loss true` |
| best mIoU checkpoint | 已有 | `best_miou_weights.pth` |
| 自动导出 severity report | 已有 | `severity_metrics.json` 等 |

关键文件：

```text
DeepLabV3+ 模型:
src\models\deeplabv3plus\nets\deeplabv3_plus.py

训练 loss:
src\models\deeplabv3plus\nets\deeplabv3_training.py
src\models\deeplabv3plus\utils\utils_fit.py

训练入口:
src\models\deeplabv3plus\train.py

CAA / 注意力模块:
src\modules\plugins\modules.py
src\modules\plugins\factory.py
```

## 6. 已有启动脚本

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

主线2 Windows:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_pconv_150.ps1

主线2 Linux:
scripts\run_ubuntu_component_aux_pconv_v3.sh
./scripts/run_ubuntu.sh component_aux_pconv
```

## 7. 当前执行：主线2

当前正在跑：

```text
主线2 = 主线1 + PConv
```

核心原则：

```text
只改一个变量:
decoder_conv_type: standard -> pconv

不要同时加:
CAA
LBFTLoss
Severity Consistency Loss
SCLP
EfficientNet-B4
```

已新增脚本：

```text
Windows:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_pconv_150.ps1

Linux:
scripts\run_ubuntu_component_aux_pconv_v3.sh

Linux 总入口:
./scripts/run_ubuntu.sh component_aux_pconv

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_150
```

主线2判断标准：

```text
如果 mIoU > 72.11:
  PConv 保留，进入主模型候选。

如果 mIoU 接近 72.11，但 FG mIoU 或 brown_spot / gray_spot IoU 提升:
  PConv 可作为小病斑 / 边界增强模块保留。

如果 mIoU、小病斑类 IoU 都下降:
  PConv 只记录为失败消融。
```

## 8. 后续顺序

```text
第1步:
等待主线2 = 主线1 + PConv 跑完，并导出 best_miou report。

第2步:
跑主线3 = 主线1 + CAA

第3步:
如果主线2或主线3有效，再跑主线4 = 主线1 + PConv + CAA

第4步:
附加实验B = 主线1 + LBFTLoss
只做 loss 对照，不抢主线。

第5步:
附加实验C = EfficientNet-B4 + CAA + PConv + LBFTLoss
只做旧链路强对照。
```

## 9. 论文主线

建议论文逻辑：

```text
1. ATLDSD 的严重度本质是 lesion / leaf。
2. 普通 6 类 softmax 没有显式学习病斑组件结构。
3. 主线1通过 lesion / boundary / center 辅助头学习病斑组件。
4. 主线2通过 PConv decoder 改善小病斑边界和局部纹理。
5. 主线3/4再决定是否加入 CAA 注意力。
6. Severity Loss 和 LBFTLoss 是附加 loss 消融，不是主创新。
```

论文表格命名：

```text
Baseline: 主线0
Ours-1: 主线1
Ours-2: 主线2
Ours-3: 主线3
Ours-Final: 主线4
Auxiliary Ablation: 附加实验A / 附加实验B
Legacy Comparator: 附加实验C
```

## 10. 接手规则

```text
1. 不再发明新编号。
2. 不把 loss 写成主结构模块。
3. 每次改代码，Windows 和 Linux 脚本都一起改。
4. 每次启动训练，记录 PID、输出目录、参数、脚本。
5. 每次训练完成，记录 best mIoU、FG mIoU、severity MAE、grade accuracy、Params、FLOPs、FPS。
6. 每次写笔记，同时更新仓库笔记和 Obsidian 笔记。
7. 每次代码或仓库笔记变更后，git commit 并 push。
```
