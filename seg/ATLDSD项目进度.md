# ATLDSD 训练路线笔记

更新时间：2026-06-05 10:49  
用途：项目交接。下一个 AI 只读这份笔记，就能知道当前训练走到哪、为什么这么走、下一步该做什么。

## 0. 当前结论

```text
最新结论，2026-06-05 12:29:
边界1 = 主线1 + LBSB 已完成 150 epoch。
当前最强结构已经从“主线1”更新为“主线1 + LBSB”。

当前最高 mIoU = 72.86
FG mIoU = 67.89
Accuracy = 97.97
Severity MAE = 0.01177
Grade Acc = 93.90

最新完成:
边界2 = 主线1 + PConv + LBSB 已完成 150 epoch。

结果:
mIoU = 71.68
FG mIoU = 66.54
Accuracy = 97.72
Severity MAE = 0.01281
Grade Acc = 93.50
Params = 10.65M
FLOPs = 6.52G
FPS = 39.81

结论:
PConv 与 LBSB 没有形成互补。
边界2低于边界1，也低于主线1。
后续最终模型不保留 PConv。
当前最强仍然是:
主线1 + LBSB

最新完成:
融合1 = 主线1 + LBSB + LCAF 已完成 150 epoch。

结果:
mIoU = 72.68
FG mIoU = 67.70
Accuracy = 97.86
Severity MAE = 0.01169
Grade Acc = 93.90
Params = 11.76M
FLOPs = 15.53G
FPS = 88.16

结论:
LCAF 没有超过边界1的 mIoU 72.86。
rust IoU 略高于边界1，但 gray_spot / brown_spot 未超过边界1。
LCAF 可以作为候选消融结果保留，但当前不替代 LBSB-only。
当前最强仍然是:
主线1 + LBSB

完成时间:
2026-06-05 23:46

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_lcaf_150

完成时间:
2026-06-05 19:00

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_pconv_lbsb_150

Windows 启动脚本:
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_pconv_lbsb_150.ps1

Linux 启动脚本:
D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_pconv_lbsb_v3.sh
或:
./scripts/run_ubuntu.sh boundary2
```

# 2026-06-08 Context1 完成记录

```text
实验:
Context1 = 主线1 + LBSB + LGLC

训练状态:
150 / 150 已完成

最终 best_miou 验证集指标:
mIoU: 72.31
FG mIoU: 67.26
Accuracy: 97.87
Severity MAE: 0.01170
Severity Grade Acc: 93.50
Params: 11.84M
FLOPs: 15.33G
FPS: 42.85

关键类别 IoU:
leaf: 93.77
rust: 81.39
alternaria_leaf_spot: 53.45
gray_spot: 57.50
brown_spot: 50.20

和当前最强 Boundary1 对比:
Boundary1 mIoU: 72.86
Context1 mIoU: 72.31
差值: -0.55

结论:
LGLC 没有超过 Boundary1。
它说明单纯在 ASPP 后补 local-global context 不能稳定解决 ATLDSD 的小病斑问题。
后续不继续围绕 LGLC 调参。

下一步:
进入 Refine1 = 主线1 + LBSB + CFR。
CFR 的目标不是再堆上下文，而是让 component auxiliary heads 反向反馈主分割，
重点解决 alternaria_leaf_spot / gray_spot / brown_spot 这些小病斑类别。
```

```text
当前最强普通 baseline:
主线0 = DeepLabV3+ + MobileNetV3-Large
mIoU = 71.72

当前最强结构模型:
主线1 = 主线0 + Component Auxiliary Heads
mIoU = 72.11
相对主线0提升 +0.39

当前正在训练:
边界1 = 主线1 + LBSB
只加 lesion boundary sharpening block，不加 PConv。

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

统一汇总表和图表已经生成，后续每一次训练完成都必须更新。

```text
CSV:
D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.csv

Markdown:
D:\Code\ATLDSD\outputs\atldsd\summary\training_results_summary.md

表格图:
D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_results_table.png

mIoU 对比图:
D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_miou_comparison.png

模型效率-精度对比图:
D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_model_tradeoff.png

生成脚本:
D:\Code\ATLDSD\figures\gen_fig_training_results_summary.py
```

更新规则：

```text
每次训练完成后，必须执行:
1. 把新实验结果加入 figures\gen_fig_training_results_summary.py 的 ROWS。
2. 运行:
   python figures\gen_fig_training_results_summary.py
3. 更新本笔记中的关键结论。
4. 同步 Obsidian 笔记。
5. git add / commit / push。
```

模型对比图固定样式：

```text
以后所有模型对比图优先采用 Params(M) 为横轴、mIoU 为纵轴的 accuracy-efficiency trade-off 样式。
图中用点线连接同一模型族/同一改进链路，例如 Mainline0 -> Mainline1 -> Aux-A 或 Mainline1 -> Mainline2。
每个点直接标注模型名。
图内嵌入一个小表，至少包含 mIoU、Params、FLOPs、FPS。
参考风格类似 SegFormer 论文中的 ADE20K mIoU vs Params 图，而不是只画普通柱状图。
当前生成文件:
D:\Code\ATLDSD\outputs\atldsd\summary\fig_training_model_tradeoff.png
```

最新完成结果：

```text
2026-06-05 12:29
边界1 = 主线1 + LBSB 已完成 150 epoch。

验证集:
split = val
num_images = 246

结果:
mIoU = 72.86
FG mIoU = 67.89
Accuracy = 97.97
Severity MAE = 0.01177
Grade Acc = 93.90
Params = 11.73M
FLOPs = 15.29G
FPS = 106.89

结论:
边界1超过主线1的 mIoU 72.11，当前最强结构更新为:
主线1 + LBSB

相对主线1:
mIoU: 72.11 -> 72.86, +0.75
FG mIoU: 67.03 -> 67.89, +0.86
Severity MAE: 0.01212 -> 0.01177, 下降 0.00035

类别 IoU:
background = 97.71
leaf = 94.05
rust = 81.47
alternaria_leaf_spot = 51.14
gray_spot = 60.20
brown_spot = 52.61

报告目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150\reports\best_miou
```

| 实验 | 结构 | mIoU | FG mIoU | Acc | Severity MAE | Grade Acc | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 主线0 | DeepLabV3+ + MobileNetV3-Large | 71.72 | 66.58 | 97.76 | 约0.0124 | 约95.12 | 最强普通 baseline |
| 主线1 | 主线0 + Component Auxiliary Heads | 72.11 | 67.03 | 97.82 | 0.01212 | 94.31 | 当前最强结构起点 |
| 附加实验A | 主线1 + Severity Consistency Loss | 72.12 | 67.06 | 97.78 | 0.01147 | 93.90 | mIoU 基本持平，严重度 MAE 更好 |
| 主线2 | 主线1 + PConv | 71.76 | 66.62 | 97.80 | 0.01373 | 93.09 | 更轻但精度下降，需测试 PConv+LBSB 互补 |
| 边界1 | 主线1 + LBSB | 72.86 | 67.89 | 97.97 | 0.01177 | 93.90 | 当前最高 mIoU，LBSB 单独有效 |
| 边界2 | 主线1 + PConv + LBSB | 71.68 | 66.54 | 97.72 | 0.01281 | 93.50 | 未超过边界1，PConv 不保留 |
| 融合1 | 主线1 + LBSB + LCAF | 72.68 | 67.70 | 97.86 | 0.01169 | 93.90 | 接近但未超过边界1，不替代 LBSB-only |
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
边界1 = 主线1 + LBSB

定位:
病斑边界锐化结构实验。
相对主线1，只加 LBSB，不加 PConv。

PID:
36844

启动时间:
2026-06-05 10:49

当前检查时间:
2026-06-05 10:49

当前进度:
epoch1/150 已开始，仍在运行

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150
```

当前判断：

```text
边界1用于验证 LBSB 是否单独改善病斑边界和小病斑。
不要在这次训练中混入 PConv、CAA、LBFTLoss、Severity Loss 或 SCLP。
```

检查命令：

```powershell
Get-Process -Id 36844 -ErrorAction SilentlyContinue
Get-Content -LiteralPath 'D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150\train_stdout.log' -Tail 80
```

## 5. 当前代码能力

| 能力 | 是否已有 | 入口或参数 |
|---|---|---|
| Component Auxiliary Heads | 已有 | `--component-aux true` |
| Severity Consistency Loss | 已有 | `--severity-consistency-loss true` |
| PConv decoder | 已有 | `--decoder-conv-type pconv` |
| LBSB 边界锐化 | 已有 | `--lesion-boundary-sharpen true` |
| LCAF 跨尺度融合 | 已有 | `--lesion-cross-scale-fusion true` |
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

边界1 Windows:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150.ps1

边界1 Linux:
scripts\run_ubuntu_component_aux_lbsb_v3.sh
./scripts/run_ubuntu.sh component_aux_lbsb
```

## 7. 当前执行：边界1

当前正在跑：

```text
边界1 = 主线1 + LBSB
```

核心原则：

```text
只改一个变量:
lesion_boundary_sharpen: false -> true

不要同时加:
PConv
CAA
LBFTLoss
Severity Consistency Loss
SCLP
EfficientNet-B4
```

已新增脚本：

```text
Windows:
scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150.ps1

Linux:
scripts\run_ubuntu_component_aux_lbsb_v3.sh

Linux 总入口:
./scripts/run_ubuntu.sh component_aux_lbsb

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_150
```

边界1判断标准：

```text
如果 mIoU > 72.11:
  LBSB 单独有效，进入主模型候选。

如果 mIoU 接近 72.11，但 FG mIoU 或 brown_spot / gray_spot IoU 提升:
  LBSB 可作为边界 / 小病斑增强模块保留。

如果 mIoU、小病斑类 IoU 都下降:
  LBSB 单独不保留，但仍需跑 PConv+LBSB 判断互补性。
```

## 8. 后续顺序

```text
第1步:
等待边界1 = 主线1 + LBSB 跑完，并导出 best_miou report。

第2步:
跑边界2 = 主线1 + PConv + LBSB

第3步:
对比主线1、主线2、边界1、边界2，判断 PConv 和 LBSB 是否互补。

第4步:
跑融合1 = 主线1 + LCAF

第5步:
附加实验B = 主线1 + LBFTLoss
只做 loss 对照，不抢主线。

第6步:
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

### 顶会小创新点：直接反馈到训练计划

下面这些不是“泛泛借鉴”，而是可以拿到当前代码里做的小模块。优先级从高到低。

| 优先级 | 顶会来源 | 可借鉴小创新 | 放到本项目哪里 | 对应训练 |
|---|---|---|---|---|
| 1 | WACV 2025 COSNet | Boundary Feature Sharpening | decoder 融合后或 boundary auxiliary head 前 | 主线2之后的边界增强候选 |
| 2 | WACV 2025 CCASeg | Convolutional Cross-Attention Decoder | low-level / high-level feature fusion 处 | 主线3优先替代普通 CAA |
| 3 | CVPR 2025 SegMAN | Local attention + global context 的轻量组合 | ASPP 后或 decoder fusion 后 | 主线3的第二候选 |
| 4 | CVPR 2025 biomarker decoder | Depth-to-Space / PixelShuffle restoration | 替代 bilinear upsample | 如果主线2边界提升弱，再跑 |
| 5 | CVPR 2025 ResCLIP | Semantic Feedback Refinement | 用 lesion/boundary/center 概率反向调制 decoder feature | 主线4或 component-guided attention |

#### 可直接实现的候选1：Boundary Feature Sharpening

来源启发：

```text
WACV 2025 COSNet 使用 feature sharpening 和 boundary enhancement 强化边界。
它的核心思想很适合 ATLDSD:
病斑小、边界不规则、容易被叶片纹理淹没。
```

本项目怎么用：

```text
模块名:
Lesion Boundary Sharpening Block, LBSB

插入位置:
DeepLabV3+ decoder 融合后的特征上。

实现方式:
blur = avg_pool(feature)
edge = feature - blur
gate = sigmoid(conv(boundary_aux_logits))
feature_refined = feature + alpha * edge * gate

训练标签:
不用新增人工标注，直接复用现有 boundary auxiliary target。
```

进入训练计划：

```text
如果主线2 PConv 没有超过主线1:
  下一步不要急着跑普通 CAA。
  优先跑: 主线1 + LBSB

如果主线2 PConv 超过主线1:
  跑: 主线2 + LBSB
```

为什么适合你：

```text
它比“加注意力”更像病斑任务创新。
因为它明确解决 lesion boundary，而不是泛泛增强特征。
```

#### 可直接实现的候选2：Convolutional Cross-Attention Decoder

来源启发：

```text
WACV 2025 CCASeg 的重点是 decoder 里用 convolutional cross-attention 融合多尺度上下文。
它不是普通 Transformer attention，而是用卷积核捕获不同尺度上下文，计算更轻。
```

本项目怎么用：

```text
模块名:
Lesion Cross-Attention Fusion, LCAF

插入位置:
DeepLabV3+ low-level feature 和 ASPP high-level feature 融合前。

实现方式:
query = conv1x1(low_level_feature)
key   = depthwise_conv3x3(high_level_feature)
value = depthwise_conv5x5(high_level_feature)
attention = sigmoid(query * key)
fused = concat(low_level_feature, attention * value)

可选组件引导:
attention = attention * sigmoid(lesion_aux_logits + boundary_aux_logits)
```

进入训练计划：

```text
主线3不要只写“主线1 + CAA”。
更具体地改成:
主线3 = 主线1 + LCAF

普通 CAA 降级为对照:
附加实验D = 主线1 + CAA
```

为什么适合你：

```text
ATLDSD 的关键是低层边缘纹理 + 高层病害语义对齐。
LCAF 正好针对这个融合点。
```

#### 可直接实现的候选3：Local-Global Context Block

来源启发：

```text
CVPR 2025 SegMAN 强调 segmentation 同时需要:
local detail
global context
multi-scale feature
```

本项目怎么用一个轻量版本：

```text
模块名:
Local-Global Lesion Context Block, LGLC

插入位置:
ASPP 输出后，进入 decoder 前。

实现方式:
local = depthwise_conv3x3(feature)
strip_h = depthwise_conv1x7(feature)
strip_w = depthwise_conv7x1(feature)
global = upsample(conv(global_avg_pool(feature)))
out = conv1x1(concat(local, strip_h, strip_w, global))
```

进入训练计划：

```text
如果主线2 PConv 有效:
  主线4候选 = 主线2 + LGLC

如果主线2 PConv 无效:
  用 LGLC 替代 PConv 作为新的 decoder/context 模块。
```

为什么适合你：

```text
小病斑需要 local detail。
严重度估计又需要整片叶子的 global context。
LGLC 比单纯 CAA 更贴合 lesion / leaf 比例任务。
```

#### 可直接实现的候选4：PixelShuffle Restoration Decoder

来源启发：

```text
CVPR 2025 biomarker segmentation 方向重新思考 decoder，
用 restoration-style decoder 保护小结构边界。
```

本项目怎么用：

```text
模块名:
PixelShuffle Lesion Restoration Decoder, PLRD

插入位置:
替换 decoder 中 bilinear upsample + conv 的局部恢复部分。

实现方式:
conv 输出 4C 通道
PixelShuffle(2) 上采样
再接 depthwise separable conv
```

进入训练计划：

```text
不要现在就跑。
当主线2 PConv 边界提升不明显时，再跑:
主线1 + PLRD
```

为什么适合你：

```text
病斑是小目标，普通双线性上采样容易糊边界。
PixelShuffle 更像恢复任务，可能更适合病斑边缘。
```

#### 可直接实现的候选5：Semantic Feedback Refinement

来源启发：

```text
CVPR 2025 ResCLIP 使用 semantic feedback refinement，
用已有语义预测反过来调整注意力/空间响应。
```

本项目怎么用：

```text
模块名:
Component Feedback Refinement, CFR

插入位置:
最终 segmentation logits 前。

实现方式:
component_map = sigmoid(lesion_logits) + sigmoid(boundary_logits) + sigmoid(center_logits)
refine_gate = sigmoid(conv(component_map))
seg_logits_refined = seg_logits + beta * refine_gate * conv(decoder_feature)
```

进入训练计划：

```text
主线4不要简单等于 PConv + CAA。
更好的最终候选:
主线4 = 最优结构 + CFR
```

为什么适合你：

```text
你已经有 lesion / boundary / center 三个辅助头。
CFR 可以把辅助头从“只算 loss”升级成“参与推理的反馈模块”。
这比单纯辅助监督更像模型结构创新。
```

#### 修正后的训练顺序

```text
当前正在跑:
边界1 = 主线1 + LBSB

边界1结束后:
无论 LBSB 是否单独超过主线1，只要没有严重崩掉，都跑:
  边界2 = 主线1 + PConv + LBSB

然后:
跑 主线1 + LCAF

最后:
把最好的 decoder/boundary 模块与 CFR 组合，形成最终模型。

普通 CAA:
降级为对照实验，不再作为主线3默认方案。
```

这次真正从顶会拿来的可用点是：

```text
1. 边界锐化: LBSB
2. 卷积交叉注意力融合: LCAF
3. 局部-全局上下文块: LGLC
4. PixelShuffle 恢复式 decoder: PLRD
5. 组件反馈细化: CFR
```

### 毒辣审稿人复审后的漏洞与修正

当前计划还不够完善。主要漏洞如下：

```text
漏洞1: 候选模块太多，像“顶会模块购物清单”。
问题:
LBSB、LCAF、LGLC、PLRD、CFR 都想做，会让论文主线发散。
修正:
先只保留两个最贴合 ATLDSD 的结构方向:
  A. 边界方向: LBSB
  B. 融合方向: LCAF
LGLC、PLRD、CFR 暂时降级为后备方案。
```

```text
漏洞2: 还没有做到“先诊断失败，再加模块”。
问题:
如果不知道主线2到底错在哪里，直接加 LBSB / LCAF 仍然可能被审稿人认为是堆模块。
修正:
主线2结束后，先做错误诊断:
  1. per-class IoU，重点看 rust / alternaria / gray_spot / brown_spot
  2. FG mIoU
  3. severity MAE
  4. boundary visual examples
  5. 轻症样本和小病斑样本表现
然后再决定是边界问题优先，还是多尺度融合问题优先。
```

```text
漏洞3: PConv 单独不强，不代表 PConv + LBSB 不强。
问题:
如果主线2低于主线1就直接丢掉 PConv，消融不完整。
修正:
只要主线2没有明显崩掉，就必须测试:
  主线1 + LBSB
  主线1 + PConv + LBSB
这样才能判断 PConv 是否和边界锐化互补。
```

```text
漏洞4: 普通 CAA 仍然太弱，不能作为主线。
问题:
“加 CAA”缺少任务专属性。
修正:
普通 CAA 只作为对照。
真正主线注意力应改为 LCAF:
  low-level 边缘纹理
  high-level 病害语义
  lesion / boundary 组件引导
```

```text
漏洞5: 缺少最终可信度实验。
问题:
单次训练提升 0.3-0.5 mIoU，可能只是随机波动。
修正:
最终只对最强两个模型做 3 seed 复跑。
不是所有实验都复跑，避免成本爆炸。
```

### 修正后的闸门式训练计划

从现在开始，计划不再是“有模块就跑”，而是每一步回答一个明确问题。

#### 第0步：主线2跑完并诊断

```text
当前正在跑:
边界1 = 主线1 + LBSB

必须等待 150 epoch 完成，并导出 best_miou report。

诊断内容:
1. mIoU 是否超过主线1的 72.11
2. FG mIoU 是否超过主线1的 67.03
3. brown_spot / gray_spot 是否提升
4. severity MAE 是否优于 0.01212
5. 视觉上病斑边界是否更清楚
```

#### 第1步：边界模块必须跑两组

不管 PConv 是否小幅低于主线1，只要主线2没有严重崩掉，都跑：

```text
边界1:
主线1 + LBSB

边界2:
主线1 + PConv + LBSB
```

原因：

```text
PConv 可能单独收益弱，但与 LBSB 互补。
如果不跑 PConv + LBSB，审稿人会问:
为什么你认定 PConv 没有价值？
```

判断：

```text
如果 主线1 + LBSB 最强:
  最终模型不保留 PConv。

如果 主线1 + PConv + LBSB 最强:
  保留 PConv，写成 decoder locality 与 boundary sharpening 互补。

如果 两者都不如主线1:
  LBSB 与 PConv 均不进最终模型。
```

#### 第2步：再跑融合模块 LCAF

只有完成边界模块后，再跑：

```text
融合1:
主线1 + LCAF
```

如果边界2最强，再补：

```text
融合2:
主线1 + PConv + LBSB + LCAF
```

如果边界1最强，则补：

```text
融合2:
主线1 + LBSB + LCAF
```

判断：

```text
如果 LCAF 提升小病斑类或 FG mIoU:
  LCAF 保留。

如果 LCAF 只增加复杂度，不提升关键指标:
  LCAF 不进最终模型。
```

#### 第3步：CFR 只在最终模型候选上做

```text
CFR 不再提前跑。
CFR 只用于最强结构模型的最后增强。
```

原因：

```text
CFR 依赖 lesion / boundary / center 辅助预测。
如果前面的边界/融合模块没确定，提前加 CFR 会让因果不清。
```

#### 第4步：普通 CAA 降级为对照

```text
附加实验D:
主线1 + CAA
```

作用：

```text
证明我们不是随便加注意力。
如果 LCAF > CAA，可以写:
component/lesion-aware fusion is better than generic attention.
```

#### 第5步：最终只复跑最强两个模型

```text
复跑对象:
1. 主线1
2. 最终候选模型

seed:
11, 22, 33
```

报告：

```text
mean ± std mIoU
mean ± std FG mIoU
mean ± std severity MAE
mean ± std grade accuracy
Params / FLOPs / FPS
```

### 最近顶会借鉴是否还在计划里

已经保留，但变得更严格：

| 顶会来源 | 原计划借鉴 | 复审后定位 |
|---|---|---|
| WACV 2025 COSNet | Boundary Feature Sharpening | 核心候选，必须跑 LBSB 与 PConv+LBSB |
| WACV 2025 CCASeg | Convolutional Cross-Attention Decoder | 核心候选，但放在边界诊断之后 |
| CVPR 2025 SegMAN | local + global + multi-scale context | 后备候选，只在 LBSB/LCAF 不足时启用 |
| CVPR 2025 ResCLIP | Semantic Feedback Refinement | 最终候选增强，不提前跑 |
| 普通 CAA | generic attention | 降级为对照，不作为主线 |

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
8. 每次训练完成后，更新训练结果总表和图表:
   outputs\atldsd\summary\training_results_summary.csv
   outputs\atldsd\summary\training_results_summary.md
   outputs\atldsd\summary\fig_training_results_table.png
   outputs\atldsd\summary\fig_training_miou_comparison.png
   outputs\atldsd\summary\fig_training_model_tradeoff.png
```
# 2026-06-06 训练计划修正版

本节是当前最新训练计划，优先级高于后面旧的“后续顺序”和旧顶会借鉴段落。

```text
当前实验事实:
1. 主线1 + LBSB = 72.86 mIoU，当前最强。
2. 主线1 + PConv + LBSB = 71.68 mIoU，PConv 不保留。
3. 主线1 + LBSB + LCAF = 72.68 mIoU，接近但低于 LBSB-only，LCAF 不作为当前主模块。

因此:
不要继续堆 PConv。
不要继续围绕 LCAF 调参。
不要再把普通 CAA 写成主创新。
下一步必须围绕“为什么 LBSB 有效、如何进一步服务小病斑和严重度”来训练。
```

## 新借鉴点

这次换掉旧的“泛注意力/融合模块”借鉴，改成更贴合当前结果的 3 个方向。

| 新优先级 | 借鉴来源 | 论文/方向要点 | 本项目可落地点 | 为什么现在适合 |
|---|---|---|---|---|
| 1 | CVPR 2025 SegMAN | 语义分割需要 global context、local detail、multi-scale 同时存在 | `LGLC`: ASPP 后加入轻量 local-global context block | LCAF 已经证明 concat 前交叉融合不够，下一步改在 ASPP 后补上下文 |
| 2 | Measurement 2025 ALDNet | 苹果叶病斑分割中，小病斑、模糊边界、复杂环境是主要难点；RA/浅层边界细化有效 | `CFR`: component-feedback refinement，用 lesion/boundary/center 概率反向细化 decoder feature | 我们已有三辅助头，应该利用它们反馈主分割，而不是只当 loss |
| 3 | 2025 AFR/高频细化类语义分割思路 | 用低分辨率语义先验 + 高频细节/不确定性来改善边界 | `UHF`: uncertainty-guided high-frequency refinement，只在预测不确定区域增强高频 | ATLDSD 错误集中在小病斑边界，适合做局部增强而不是全图加模块 |
| 4 | 2025 plant disease severity / PDSNets 类工作 | 严重度估计不只看 mIoU，还要看速度、Params、FLOPs、severity MAE | 最终模型后补 `severity calibration loss` 一轮 | 这是论文应用价值，不作为主结构创新 |

参考来源：

```text
SegMAN, CVPR 2025:
https://openaccess.thecvf.com/content/CVPR2025/html/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.html

ALDNet, Measurement 2025:
https://doi.org/10.1016/j.measurement.2025.117706

AFRDA, arXiv 2025:
https://arxiv.org/abs/2507.17957
```

## 新训练顺序

```text
第1步: Context1
结构:
  主线1 + LBSB + LGLC

模块位置:
  ASPP 后，decoder concat 前。

目的:
  借鉴 SegMAN 的 local-global-multi-scale 思路，但不换 backbone，不引入大 Transformer。
  验证“全局叶片上下文 + 局部病斑细节”是否能继续提升 Boundary1。

判断:
  如果 mIoU > 72.86:
    LGLC 进入最终候选。
  如果 mIoU 接近 72.86，但 alternaria / gray_spot / brown_spot 任意两类提升:
    LGLC 作为小病斑候选。
  否则:
    LGLC 不保留。
```

```text
第2步: Refine1
结构:
  主线1 + LBSB + CFR

模块位置:
  LBSB 后、cls_conv 前。

实现思想:
  用 lesion_aux_head / boundary_aux_head / center_aux_head 的概率图生成 feedback gate。
  gate 只调制 decoder feature，不改变标签、不做两阶段推理。

目的:
  借鉴 ALDNet 的 reverse-attention / shallow refinement 逻辑。
  但本项目保持单阶段推理，用 component feedback 解释为“病斑组件反馈细化”。

判断:
  如果 mIoU > 72.86:
    CFR 进入最终候选。
  如果 severity MAE < 0.01169 且 mIoU 不低于 72.6:
    CFR 可作为严重度友好模块。
  否则:
    CFR 不保留。
```

```text
第3步: UHF
结构:
  当前最强候选 + UHF

模块位置:
  decoder feature 上，利用 softmax entropy 或 lesion probability 找到不确定区域。

目的:
  只在边界/不确定区域增强高频，不全图加复杂 attention。

判断:
  只有 Context1 / Refine1 仍无法超过 Boundary1，才做 UHF。
  UHF 是后备，不抢论文主线。
```

```text
第4步: Severity-Calib
结构:
  最终结构 + severity consistency loss

目的:
  不再追 mIoU，而是专门看 severity MAE / grade accuracy 是否改善。

定位:
  只作为应用指标增强，不写成主创新。
```

## 当前论文主线改写

```text
旧主线:
Component heads + LBSB + LCAF / attention

新主线:
Component heads + LBSB 是核心。
后续只允许围绕两个问题补模块:
1. 小病斑是否需要更好的上下文？ -> LGLC
2. 已有组件头能否反馈主分割？ -> CFR

最终论文不要写成“加了很多模块”。
应写成:
结构化组件监督 + 边界锐化 + 可解释组件反馈/上下文补偿。
```

## 立即下一步

```text
下一步优先做:
Context1 = 主线1 + LBSB + LGLC

暂不做:
PConv
LCAF 调参
普通 CAA
LBFTLoss
SCLP
```

# 2026-06-08 Context1 启动记录

```text
当前正在跑:
Context1 = 主线1 + LBSB + LGLC

PID:
80716

启动时间:
2026-06-08 09:18

输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_lbsb_lglc_150

Windows 脚本:
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_lbsb_lglc_150.ps1

Linux 脚本:
D:\Code\ATLDSD\scripts\run_ubuntu_component_aux_lbsb_lglc_v3.sh
或:
./scripts/run_ubuntu.sh context1

本次只验证:
主线1 + LBSB + LGLC

不混入:
PConv
LCAF
普通 CAA
LBFTLoss
SCLP
severity consistency loss

判断:
如果 mIoU > 72.86，LGLC 进入最终候选。
如果 mIoU 接近 72.86，但 alternaria / gray_spot / brown_spot 任意两类提升，LGLC 可作为小病斑候选。
否则 LGLC 不保留。
```
