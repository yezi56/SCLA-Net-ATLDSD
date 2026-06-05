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

## 8.5 2025/2026 文献反馈后的计划修正

研究方向重新压缩为：

```text
Component-aware lightweight apple leaf disease lesion segmentation for severity estimation.
```

### 重点借鉴了什么

不是照搬某一篇模型，而是抽取近两年高水平工作的共同有效设计，反馈到 ATLDSD 的训练计划里：

```text
借鉴点1: leaf / lesion / spot 分开建模
来源启发: ALDNet 等苹果叶病斑分割工作强调 leaf-aware、spot-aware。
落到本项目: 坚持主线1的 lesion / boundary / center 辅助头。
意义: 不让网络只做普通 6 类 softmax，而是显式学习病斑组件。
```

```text
借鉴点2: 小病斑和边界需要 decoder 局部增强
来源启发: 近年植物病害分割普遍强化多尺度、边界和局部纹理。
落到本项目: 当前主线2优先跑 PConv decoder。
意义: PConv 不是随便加模块，而是为小病斑、边界、不规则局部纹理服务。
```

```text
借鉴点3: 注意力不能泛泛堆，要看是否真正服务病斑区域
来源启发: STAR-Net、Sparse-MoE-SAM 类工作强调注意力/稀疏专家对复杂病斑区域的选择性响应。
落到本项目: 主线3先跑 CAA 作为受控实验；如果普通 CAA 无效，不继续堆注意力，转向 component-guided attention。
意义: 注意力必须被 lesion / boundary / center 组件信息约束，否则不作为主创新。
```

```text
借鉴点4: 严重度论文不只看 mIoU，还看部署和 severity 指标
来源启发: 2025/2026 严重度估计工作重视 severity、速度、轻量化。
落到本项目: 每个实验必须同时记录 mIoU、FG mIoU、severity MAE、grade accuracy、Params、FLOPs、FPS。
意义: 论文不只证明“分割准”，还要证明“严重度判断有用、模型足够轻”。
```

因此后续训练计划的重点不是：

```text
DeepLabV3+ + 一堆模块
```

而是：

```text
主线1: 组件建模
主线2: decoder 局部增强
主线3: 受控注意力验证
主线4: 组件 + 局部 + 注意力的最终组合
附加实验: severity / LBFTLoss 只做 loss 消融
```

近两年相关工作的共同点：

| 相关工作方向 | 可借鉴点 | 对当前训练计划的影响 |
|---|---|---|
| 2025 苹果叶病斑分割 ALDNet | leaf-aware / spot-aware 比盲目 6 类分割更有解释性 | 保留主线1的 lesion、boundary、center 组件辅助头 |
| 2025 STAR-Net 类番茄叶病分割 | 多分支注意力要服务复杂病斑，而不是泛泛加注意力 | 主线3不能只写“加 CAA”，要观察是否真正改善病斑区域 |
| 2025 Sparse-MoE-SAM 类植物病害分割 | local/global 稀疏注意力适合异质病斑 | 如果普通 CAA 无效，再考虑 component-guided attention |
| 2025 PDSNets 类严重度估计 | 严重度论文重视速度和部署，不只看 mIoU | 每个实验必须记录 Params、FLOPs、FPS、severity MAE |
| 2026 严重度分类相关工作 | 严重度判断通常需要区分 leaf / lesion / background | 保留 lesion / leaf 严重度指标，但 severity loss 仍是附加实验 |

参考链接：

```text
ALDNet, Measurement 2025:
https://www.sciencedirect.com/science/article/pii/S0263224125010656

STAR-Net, Frontiers in Plant Science 2026:
https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1706072/full

Sparse-MoE-SAM, Plants 2025:
https://www.mdpi.com/3462676

Citrus severity semantic segmentation, Scientific Reports 2025:
https://www.nature.com/articles/s41598-025-04758-y

Pome leaf multi-disease severity, Scientific Reports 2026:
https://www.nature.com/articles/s41598-026-45947-7
```

反馈到训练计划：

```text
1. 当前主线2继续跑完。
2. 主线2结束后，先判断 PConv 是否保留。
3. 主线3仍然可以跑 CAA，但它是受控注意力实验，不是主创新本身。
4. 如果普通 CAA 不明显提升，不继续堆注意力；改为设计 component-guided attention。
5. 附加实验A已经证明 severity MAE 改善，但它仍然是 loss 消融，不提升为主线。
6. 后续结果表必须同时看:
   mIoU
   FG mIoU
   difficult lesion class IoU
   severity MAE
   grade accuracy
   Params / FLOPs / FPS
```

正式执行计划已保存：

```text
docs\superpowers\plans\2026-06-05-literature-informed-training-plan.md
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
