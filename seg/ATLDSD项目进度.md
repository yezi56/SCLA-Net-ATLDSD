# ATLDSD 项目进度

## 2026-06-02 Backbone 选择更正

### 为什么第一轮用了 MobileNetV2

这次 150 轮训练选择的是：

```text
DeepLabV3+ + MobileNetV2
```

原因不是因为 `D:\Code\all` 的主线默认就是 MobileNetV2，而是因为我当时优先复用了 `D:\Code\all\docs\RICESEG_BASELINE_PROGRESS.md` 里的轻量 baseline 命令。

这样做的工程理由是：

1. 本地已经有 `deeplab_mobilenetv2.pth`，不用联网下载权重。
2. 训练速度快，150 epoch 能较快闭环。
3. 参数量低，适合作为轻量 baseline 和 FPS/Params 对比基准。
4. 先验证 ATLDSD 的 VOC 数据、类别、mask、训练和报告导出流程是否完整可跑。

### 需要更正的地方

`D:\Code\all` 的主线并不是 MobileNetV2。

核对结果：

```text
D:\Code\all\src\models\deeplabv3plus\train.py 默认 backbone = efficientnet_b4
D:\Code\all\plan.md 的 E00 = DeepLabV3+ + EfficientNet-B4 + CE
```

所以目前这轮结果应该定义为：

```text
轻量级 sanity baseline / speed baseline
```

不能把它当作论文主 baseline。

### 当前已有结果

MobileNetV2 150 epoch 已完成：

```text
输出目录：D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenet_150
best_val 报告：reports\best_val
```

统一报告指标：

```text
mIoU all:          67.58%
foreground mIoU:   61.75%
mDice all:         78.59%
FPS:              123.23
Params:             5.81M
FLOPs:             13.23G
```

每类 IoU：

```text
background:             96.75%
leaf:                   92.10%
rust:                   77.50%
alternaria_leaf_spot:   45.84%
gray_spot:              49.42%
brown_spot:             43.87%
```

短板很清楚：`alternaria_leaf_spot / gray_spot / brown_spot` 较低，可以支撑后续注意力、边界监督和类别不均衡 loss 的改进故事。

### 下一步应该怎么改

为了和 `D:\Code\all` 的主线一致，下一条正式 baseline 应该跑：

```text
DeepLabV3+ + EfficientNet-B4
```

本项目里已经复制到了：

```text
D:\Code\ATLDSD\src\models\deeplabv3plus
D:\Code\ATLDSD\src\models\efficientnet
```

并且本地已经有 EfficientNet-B4 权重：

```text
D:\Code\ATLDSD\src\models\efficientnet\model_data\efficientnet_b4_rwightman-23ab8bcd.pth
```

建议实验顺序更新为：

1. 保留 `DeepLabV3+ + MobileNetV2` 作为轻量 baseline。
2. 跑 `DeepLabV3+ + EfficientNet-B4` 作为正式 plain baseline。
3. 在 EfficientNet-B4 baseline 上加 Weighted CE / Dice / Focal / Tversky。
4. 再做单模块注意力消融，例如 ECA、CAA、Coordinate Attention。
5. 再加入 SegNeXt 作为近年对比模型，而不是直接拿 SegNeXt 替代 DeepLabV3+ 主线。

### 当前项目代码状态

之前 ATLDSD 目录下没有 `src`，因为第一轮训练直接调用了 `D:\Code\all` 的代码。
现在已经把核心代码复制到本项目：

```text
D:\Code\ATLDSD\src\models\deeplabv3plus
D:\Code\ATLDSD\src\models\segnext
D:\Code\ATLDSD\src\modules\plugins
D:\Code\ATLDSD\scripts
```

复跑 MobileNetV2 baseline 的脚本：

```text
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenet_150.ps1
```

结论：MobileNetV2 是第一条轻量跑通线，不是最终论文主线。正式论文 baseline 应该补跑 EfficientNet-B4。

## 2026-06-02 Backbone ???? EfficientNet-B4

???? DeepLabV3+ baseline ?? MobileNetV2 ????

```text
DeepLabV3+ + EfficientNet-B4
```

???????

```text
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_efficientnet_b4_150
```

?????

```text
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_b4_150.ps1
```

?? PID ???

```text
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_efficientnet_b4_150\train_pid.txt
```

????????? `backbone = efficientnet_b4`?`pretrained = true`??? EfficientNet-B4 ???? `src\models\efficientnet\backbone.py` ?????

## 2026-06-02 CLCS-Net ???????

???????????? CLCS-Net ??????????????? B4 baseline?

?????

```text
D:\Code\ATLDSD\src\atldsd_seg\models\clcs_deeplabv3plus.py
D:\Code\ATLDSD\src\atldsd_seg\losses\compositional.py
D:\Code\ATLDSD\src\atldsd_seg\models\CLCS_NET.md
```

?????

```text
???? -> ?? DeepLabV3+ encoder/decoder -> ?????

leaf head:    background vs leaf
lesion head:  non-lesion vs lesion
disease head: rust / alternaria / gray / brown
```

?????

```text
background   = not leaf
healthy leaf = leaf and not lesion
disease      = leaf and lesion and disease type
```

??? 6 ? mask ?????????????????smoke test ????`final_logits=(B,6,H,W)`?`leaf_logits=(B,2,H,W)`?`lesion_logits=(B,2,H,W)`?`disease_logits=(B,4,H,W)`?

## 2026-06-02 ?????????????

### ??????

??????? baseline?

```text
DeepLabV3+ + EfficientNet-B4
?????150 epochs
?????? Epoch 75/150
?????D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_efficientnet_b4_150
```

?????? mIoU ??? Epoch 70?

```text
mIoU:     63.40
mPA:      78.47
Accuracy: 95.36
Train Loss: 0.677
Val Loss:   0.695
```

???`mIoU / mPA / Accuracy / Val Loss` ????? val?`Train Loss` ????? train???????????150 ?????? `reports/best_val` ??????

### ???????

?? `src` ???????????

```text
src/atldsd_seg/          ??????
src/models/              ? DeepLabV3+?SegNeXt?U-Net?PSPNet ?????
src/modules/plugins/     ???? loss ??
```

??????????

```text
D:\Code\ATLDSD\scripts\run_atldsd_experiment.ps1
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_b4_150.ps1
```

### ??? CLCS-Net ????

????????? CLCS-Net ???????? B4 baseline ???

?????

```text
D:\Code\ATLDSD\src\atldsd_seg\models\clcs_deeplabv3plus.py
D:\Code\ATLDSD\src\atldsd_seg\losses\compositional.py
D:\Code\ATLDSD\src\atldsd_seg\models\CLCS_NET.md
```

CLCS-Net ????

```text
???? -> ?? DeepLabV3+ encoder/decoder -> ?????

leaf head:    background vs leaf
lesion head:  non-lesion vs lesion
disease head: rust / alternaria / gray / brown
```

????? 6 ??

```text
background   = not leaf
healthy leaf = leaf and not lesion
disease      = leaf and lesion and disease type
```

???? 6 ? softmax ??????????????? 6 ??CLCS-Net ?????? leaf?lesion?disease type ?????????????? 6 ? mask???????????????????

### Zotero ????

??? Zotero ? `seg` ?? 3 ??????

```text
Tomato TransDeepLab
DS-DETR
Deep learning architectures for semantic segmentation and automatic estimation of severity of foliar symptoms
```

???related work ??????? -> ?????? -> ?????????????

## ????? CLCS-Net???????

### ?????? Baseline

?????????????????????????

????????

```text
B1 DeepLabV3+ + MobileNetV2
B2 DeepLabV3+ + EfficientNet-B4
B3 U-Net
B4 PSPNet
B5 SegNeXt
```

?? B2 ?????? baseline????? CLCS-Net ????? EfficientNet-B4 backbone??????

### ????????

?????????????????????

???????

```text
A0 DeepLabV3+ + EfficientNet-B4, ordinary 6-class softmax
A1 A0 + leaf head auxiliary loss
A2 A1 + lesion head auxiliary loss
A3 A2 + disease type head auxiliary loss
A4 CLCS-Net: leaf/lesion/disease ??????
A5 A4 + component-aware branch
A6 A5 + severity-controlled lesion copy-paste
```

???

```text
A0 ??????? baseline?
A1/A2/A3 ??????????????
A4 ????? compositional fusion ?????
A5 ????/??/???????????????
A6 ??????? copy-paste ?????????????
```

### ?????????

???????

```text
mIoU
mDice
pixel accuracy
mPA
?? IoU
?? Dice
```

????????

```text
rust
alternaria_leaf_spot
gray_spot
brown_spot
```

??????

```text
overall severity MAE = |pred lesion_ratio - gt lesion_ratio|
per-disease severity MAE
severity grade accuracy
severity grade quadratic weighted kappa???
```

??????

```text
Params
FLOPs
FPS
```

### ???????????

? 1???????

```text
Model | Backbone | mIoU | mDice | Params | FLOPs | FPS
```

? 2??? IoU

```text
Model | leaf | rust | alternaria | gray | brown | mIoU
```

? 3?CLCS-Net ??

```text
Setting | leaf head | lesion head | disease head | compositional fusion | mIoU | lesion mIoU | severity MAE
```

? 4??????

```text
Model | overall severity MAE | rust MAE | alternaria MAE | gray MAE | brown MAE | grade accuracy
```

? 1??????

```text
image -> encoder -> decoder -> leaf head / lesion head / disease head -> composition -> 6-class mask -> severity
```

? 2??????

```text
?? | GT | DeepLabV3+ B4 | CLCS-Net | error map
```

? 3???????

```text
???? alternaria / gray / brown ????????
```

### ????????

????????????????

```text
1. ? DeepLabV3+ + EfficientNet-B4 150 ??????? baseline?
2. ? CLCS-Net ??????? A4?
3. ?? A4 ? A0 ? lesion mIoU ? severity MAE ??????? A1/A2/A3 ????
4. ?? component-aware branch?
5. ??? severity-controlled copy-paste?
6. ? SegNeXt / U-Net / PSPNet ???
```

??????????????

```text
CLCS-Net ?? DeepLabV3+ B4?
lesion mIoU ?????
????? IoU ?????
severity MAE ?????
```

????? Accuracy?? mIoU??? IoU?severity MAE ????????????

## 2026-06-02 B4 baseline 跑完后的实验顺序

### 总原则

当前这轮 `DeepLabV3+ + EfficientNet-B4 + 普通 6 类 softmax` 是正式论文的普通 baseline。它跑完后，不是把所有模型都随便重跑一遍，而是按“先确认普通 baseline、再确认三头结构、最后在三头上加模块”的顺序推进。

核心逻辑：

```text
普通 6 类 baseline -> 三头组合 baseline -> 三头 + 组件/边界/增强模块 -> 与外部模型对比
```

### 第一阶段：整理当前 B4 baseline

跑完当前 150 轮后，先做这些事：

```text
1. 用 best_epoch_weights.pth 和 last_epoch_weights.pth 分别在 val/test 上评估
2. 记录 mIoU、mDice、mPA、pixel accuracy
3. 记录每一类 IoU：leaf、rust、alternaria、gray、brown
4. 计算严重度误差：pred lesion area / pred leaf area 与 GT 的差
5. 导出可视化：image / GT / prediction / error map
```

这个实验在论文中记为：

```text
B0: DeepLabV3+ + EfficientNet-B4 + 6-class softmax
```

它的作用是证明：普通语义分割在 ATLDSD 上能做到什么水平。

### 第二阶段：训练三头组合 baseline

下一步优先训练：

```text
Ours-A: DeepLabV3+ + EfficientNet-B4 + leaf / lesion / disease type 三头组合
```

这个模型不加注意力、不加 copy-paste、不加复杂模块。只改变输出建模方式：

```text
Head 1: leaf / not leaf
Head 2: lesion / not lesion
Head 3: rust / alternaria / gray / brown

组合：
background = not leaf
healthy leaf = leaf and not lesion
disease class = leaf and lesion and disease type
```

这一步最关键。如果 Ours-A 比 B0 的 lesion IoU、disease IoU、severity MAE 更好，说明“三头组合”本身有价值，论文主线就站住了。

### 第三阶段：三头内部消融

为了证明不是随便多加几个 head，而是结构化建模有效，需要跑：

```text
A0: DeepLabV3+ + B4 + 6-class softmax
A1: A0 + leaf auxiliary head
A2: A1 + lesion auxiliary head
A3: A2 + disease type auxiliary head
A4: leaf / lesion / disease type compositional fusion
```

重点看：

```text
lesion mIoU 是否提高
alternaria / gray / brown 这些小病斑类 IoU 是否提高
severity MAE 是否下降
边界是否更完整
```

如果 A1-A3 有提升，但 A4 提升更明显，可以写成：辅助监督有帮助，但真正有效的是把输出空间按 leaf-lesion-disease 层级关系组合起来。

### 第四阶段：在三头基础上加模块

后续模块优先加在三头模型上，而不是普通 6 类 baseline 上。

建议顺序：

```text
Ours-A: 三头组合 baseline
Ours-B: Ours-A + component-aware auxiliary branch
Ours-C: Ours-B + boundary / distance transform supervision
Ours-D: Ours-C + severity-controlled lesion copy-paste
Ours-E: Ours-D + attention 或轻量特征模块
```

不要一开始就把所有模块堆上去。每次只加一个模块，形成可解释消融。

### 第五阶段：外部模型对比

等 Ours-A/Ours-B/Ours-C 跑出结果后，再跑外部模型对比：

```text
U-Net
U-Net++
PSPNet
SegNeXt
DeepLabV3+ + MobileNetV2
DeepLabV3+ + EfficientNet-B4
```

SegNeXt 可以作为近年分割模型对比，但不要把它当主创新。它的作用是回答：即使和较新的分割骨干/结构相比，我们的结构化输出建模仍然有效。

### 推荐训练顺序

```text
1. 等当前 DeepLabV3+ + B4 150 轮结束，导出完整 report
2. 训练 CLCS / 三头组合 baseline：DeepLabV3+ + B4 + 三头组合
3. 跑 A1/A2/A3/A4 三头消融
4. 加 component-aware branch
5. 加 boundary / distance transform 辅助监督
6. 加 severity-controlled lesion copy-paste
7. 跑 SegNeXt、U-Net、PSPNet 外部模型对比
8. 统一 test set 复测所有最终模型
```

### 论文中最重要的表

表 1：主模型对比

```text
Model | Backbone | mIoU | lesion mIoU | disease mIoU | severity MAE | Params | FLOPs | FPS
```

表 2：三头消融

```text
Setting | leaf head | lesion head | disease head | compositional fusion | mIoU | lesion mIoU | severity MAE
```

表 3：每类 IoU

```text
Model | leaf | rust | alternaria | gray | brown | mean
```

表 4：严重度估计

```text
Model | overall MAE | rust MAE | alternaria MAE | gray MAE | brown MAE | severity grade accuracy
```

### 当前判断

以后不是“只训练三头 baseline”，而是：

```text
普通 baseline 用来做对照；
三头 baseline 用来证明主创新；
后续模块主要加在三头基础上；
外部模型用来证明不是只赢了一个弱 baseline。
```

最优先级是把 `B0` 和 `Ours-A` 做扎实。只要 `Ours-A` 在 lesion IoU 和 severity MAE 上明显优于 `B0`，这条 SCI 二区论文路线就有基础。
## 2026-06-02 启动 Ours-A 三头组合 baseline

当前 B0 普通 baseline 已完成后，下一步实验启动：

```text
Ours-A: CLCS-Net / DeepLabV3+ + EfficientNet-B4 + leaf / lesion / disease type 三头组合
```

训练设置：

```text
backbone: EfficientNet-B4
input shape: 256 x 256
epochs: 150
freeze epochs: 50
freeze batch size: 4
unfreeze batch size: 2
optimizer: SGD
initial lr: 0.001
split: train 1148 / val 246
```

输出目录：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_efficientnet_b4_150
```

关键文件：

```text
train_stdout.log
train_stderr.log
train_pid.txt
weights\last_epoch_weights.pth
weights\best_miou_epoch_weights.pth
weights\best_loss_epoch_weights.pth
```

这一步的目的不是加模块，而是先回答：

```text
只把普通 6 类 softmax 改成 leaf / lesion / disease type 三头组合，是否能提升小病斑 IoU 和严重度估计？
```

对照对象：

```text
B0: DeepLabV3+ + EfficientNet-B4 + 6-class softmax
Ours-A: DeepLabV3+ + EfficientNet-B4 + 三头组合
```

如果 Ours-A 在 `lesion mIoU`、`alternaria / gray / brown IoU`、`severity MAE` 上优于 B0，后续再加 component-aware branch、boundary/distance supervision 和 severity-controlled copy-paste。
## 2026-06-02 Ours-A 三头组合 baseline 训练完成

实验：

```text
Ours-A: CLCS-Net / DeepLabV3+ + EfficientNet-B4 + leaf / lesion / disease type 三头组合
```

训练已完成 150 epoch。

输出目录：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_efficientnet_b4_150
```

最优 mIoU 权重：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_efficientnet_b4_150\weights\best_miou_epoch_weights.pth
```

最优 mIoU 出现在：

```text
Epoch 120
mIoU: 65.68%
mDice: 77.27%
Foreground mIoU: 60.18%
Pixel Accuracy: 94.91%
Val Loss: 0.264
```

150 epoch 最终轮：

```text
Epoch 150
mIoU: 63.56%
mPA: 70.77%
Accuracy: 95.95%
Train Loss: 0.144
Val Loss: 0.241
```

注意：150 轮的 val loss 更低，但 mIoU 不如 120 轮。因此论文对比应该优先使用 `best_miou_epoch_weights.pth`，而不是最后一轮权重。

### 与 B0 普通 B4 baseline 对比

```text
B0: DeepLabV3+ + EfficientNet-B4 + 普通 6 类 softmax
Ours-A: DeepLabV3+ + EfficientNet-B4 + 三头组合
```

总体指标：

```text
B0 epoch150:
mIoU 65.59%
mDice 77.00%
Foreground mIoU 59.59%
Accuracy 96.44%

Ours-A best epoch120:
mIoU 65.68%
mDice 77.27%
Foreground mIoU 60.18%
Accuracy 94.91%
```

结论：

```text
Ours-A 比 B0 略高，但整体提升很小。
mIoU: +0.09%
mDice: +0.27%
Foreground mIoU: +0.59%
Accuracy: -1.53%
```

每类 IoU：

```text
Class                  B0       Ours-A    Change
leaf                   89.96    86.20     -3.76
rust                   76.34    79.13     +2.79
alternaria_leaf_spot   42.08    48.70     +6.62
gray_spot              45.19    49.06     +3.87
brown_spot             44.38    37.79     -6.59
```

解释：

```text
三头组合对 rust、alternaria、gray spot 有帮助，尤其 alternaria 提升明显。
但 brown spot 明显下降，leaf 也下降，导致总体 mIoU 只略微超过 B0。
```

这说明三头结构不是完全没用，但目前还不足以作为强论文贡献。下一步应该专门解决病斑组件和 brown spot 退化问题。

## 2026-06-02 下一步训练建议

### 不建议马上做的事

暂时不要直接做：

```text
三头 + 随便一个注意力模块
三头 + 多个模块一起堆
三头 + copy-paste
```

原因：

```text
Ours-A 现在只是略微超过 B0，核心结构还不够稳。
如果直接堆模块，就无法判断到底是三头有效，还是模块偶然有效。
```

### 下一步优先实验：Ours-B

建议下一步训练：

```text
Ours-B: CLCS-Net + component-aware auxiliary branch
```

目标：

```text
救 brown spot
继续提升 alternaria / gray spot
增强小病斑连通域和边界定位
```

Ours-B 不改变主输出，仍然是：

```text
leaf head
lesion head
disease type head
compositional fusion
```

只是在 decoder feature 上增加组件感知辅助监督：

```text
lesion boundary map
lesion distance transform / center heatmap
connected lesion region awareness
```

推荐先实现最稳的版本：

```text
Ours-B1: 三头组合 + lesion boundary auxiliary head
```

不要一开始就同时加 boundary、distance、copy-paste。先用 boundary，因为它最容易从 mask 自动生成，解释也清楚。

### Ours-B1 训练设计

模型：

```text
DeepLabV3+ + EfficientNet-B4
leaf / lesion / disease type 三头组合
+ boundary auxiliary head
```

loss：

```text
L = L_final
  + 0.4 * L_leaf
  + 0.8 * L_lesion
  + 0.6 * L_disease
  + 0.3 * L_boundary
```

其中 boundary target 由 mask 自动生成：

```text
lesion mask = class in {rust, alternaria, gray, brown}
boundary = lesion mask - eroded lesion mask
```

训练设置保持和 Ours-A 一致：

```text
backbone: EfficientNet-B4
input shape: 256 x 256
epochs: 150
freeze epochs: 50
freeze batch size: 4
unfreeze batch size: 2
optimizer: SGD
lr: 0.001
seed: 11
```

### Ours-B1 需要重点观察

如果 Ours-B1 成功，应该看到：

```text
overall mIoU > 65.68%
foreground mIoU > 60.18%
alternaria IoU > 48.70%
gray spot IoU > 49.06%
brown spot IoU 回升，最好 > 44.38%
```

如果 brown spot 仍然下降，则说明三头组合对 disease type head 的类别区分不够，需要下一步做：

```text
Ours-B2: 三头组合 + disease-type class-balanced loss
```

### 推荐后续顺序

```text
1. Ours-B1: 三头 + lesion boundary auxiliary head
2. 如果 brown spot 仍差，做 Ours-B2: 三头 + disease-type class-balanced loss
3. 如果小病斑边界仍差，做 Ours-C: 三头 + boundary + distance transform
4. 最后再做 Ours-D: severity-controlled lesion copy-paste
5. 等 Ours-B/Ours-C 稳定后，再跑 SegNeXt / U-Net / PSPNet 外部模型对比
```

当前最合理的下一步不是追求“复杂”，而是把 Ours-A 中暴露的问题修正：

```text
三头结构提高了部分病斑类，但 brown spot 掉了。
下一步要让模型更关注病斑组件边界和小区域完整性。
```
## 2026-06-02 启动 Ours-B1 边界辅助训练

在 Ours-A 三头组合基础上，已启动下一步：

```text
Ours-B1: CLCS-Net + lesion boundary auxiliary head
```

目的：

```text
1. 保持 leaf / lesion / disease type 三头组合不变
2. 在 decoder feature 上增加 boundary head
3. 用 lesion mask 自动生成 boundary target
4. 强化小病斑边界和连通区域学习
5. 重点观察 brown spot 是否回升
```

边界标签生成：

```text
lesion mask = class in {rust, alternaria_leaf_spot, gray_spot, brown_spot}
boundary = lesion mask - eroded(lesion mask)
```

loss：

```text
L = L_final
  + 0.4 * L_leaf
  + 0.8 * L_lesion
  + 0.6 * L_disease
  + 0.3 * L_boundary
```

训练设置：

```text
backbone: EfficientNet-B4
input shape: 256 x 256
epochs: 150
freeze epochs: 50
freeze batch size: 4
unfreeze batch size: 2
optimizer: SGD
initial lr: 0.001
seed: 11
boundary weight: 0.3
boundary positive weight: 5.0
```

输出目录：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_boundary_deeplabv3plus_efficientnet_b4_150
```

启动 PID：

```text
30280
```

需要超过的 Ours-A 指标：

```text
overall mIoU > 65.68%
foreground mIoU > 60.18%
alternaria IoU > 48.70%
gray spot IoU > 49.06%
brown spot IoU 回升，最好 > 44.38%
```

代码改动：

```text
D:\Code\ATLDSD\src\atldsd_seg\models\clcs_deeplabv3plus.py
D:\Code\ATLDSD\src\atldsd_seg\losses\compositional.py
D:\Code\ATLDSD\src\atldsd_seg\engine\train_clcs.py
D:\Code\ATLDSD\scripts\run_atldsd_clcs_boundary_b4_150.ps1
```
## 2026-06-03 启动 Ours-B2 疾病类别加权训练

在 Ours-A 三头组合基础上，已启动：

```text
Ours-B2: CLCS-Net + disease-type class-balanced loss
```

这次不叠加 boundary head，目的是单独验证：

```text
disease type head 加类别权重后，brown spot 是否能回升。
```

模型结构：

```text
DeepLabV3+ + EfficientNet-B4
leaf head
lesion head
disease type head
compositional fusion
```

loss：

```text
L = L_final
  + 0.4 * L_leaf
  + 0.8 * L_lesion
  + 0.6 * L_disease_balanced
```

disease class weights：

```text
rust:       1.0
alternaria: 1.5
gray:       1.5
brown:      2.0
```

训练设置：

```text
backbone: EfficientNet-B4
input shape: 256 x 256
epochs: 150
freeze epochs: 50
freeze batch size: 4
unfreeze batch size: 2
optimizer: SGD
initial lr: 0.001
seed: 11
```

输出目录：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_disease_balanced_deeplabv3plus_efficientnet_b4_150
```

启动 PID：

```text
29292
```

需要超过的目标：

```text
overall mIoU > 65.87%
foreground mIoU > 60.20%
brown spot IoU > 44.38%
alternaria IoU 不低于 49%
gray spot IoU 不低于 46%
```

当前判断：

```text
Ours-B1 边界辅助只把 brown spot 从 37.79% 拉到 39.69%，仍低于 B0 的 44.38%。
因此下一步重点不是边界，而是 disease type head 的类别不均衡和类别混淆。
```

代码改动：

```text
D:\Code\ATLDSD\src\atldsd_seg\losses\compositional.py
D:\Code\ATLDSD\src\atldsd_seg\engine\train_clcs.py
D:\Code\ATLDSD\scripts\run_atldsd_clcs_disease_balanced_b4_150.ps1
```
## 2026-06-03 Ours-B2 当前训练进度

实验：

```text
Ours-B2: CLCS-Net + disease-type class-balanced loss
```

当前状态：

```text
PID: 29292
状态: 正在训练
当前进度: Epoch 54 / 150
输出目录: D:\Code\ATLDSD\outputs\atldsd\clcs_disease_balanced_deeplabv3plus_efficientnet_b4_150
```

当前每 10 轮验证结果：

```text
Epoch 10:
Train Loss 0.452
Val Loss   0.411
mIoU       51.17
mPA        57.95
Accuracy   92.93

Epoch 20:
Train Loss 0.315
Val Loss   0.512
mIoU       53.57
mPA        59.63
Accuracy   93.21

Epoch 30:
Train Loss 0.274
Val Loss   0.418
mIoU       58.83
mPA        66.08
Accuracy   94.86

Epoch 40:
Train Loss 0.264
Val Loss   0.321
mIoU       54.75
mPA        61.73
Accuracy   94.52

Epoch 50:
Train Loss 0.217
Val Loss   0.361
```

当前判断：

```text
Ours-B2 目前还没有接近 Ours-A / Ours-B1 的 65% mIoU。
当前最好 mIoU 暂时是 58.83%，出现在 Epoch 30。
第 50 轮后刚进入 unfreeze 阶段，loss 和 mIoU 波动是正常现象。
现在不能下结论，需要继续观察 Epoch 80、100、120 后的表现。
```

这轮训练要验证的问题：

```text
disease type head 加类别权重后，brown spot 是否能从 Ours-A 的 37.79% / Ours-B1 的 39.69% 回升到 B0 的 44.38% 以上。
```

如果后续 mIoU 仍然上不去，说明简单 disease class weight 不够，下一步不应继续调小权重，而应考虑：

```text
1. disease head 使用 pixel-level class-balanced CE + lesion-only Dice
2. brown spot 采样增强
3. severity-controlled lesion copy-paste
```
## 2026-06-03 已完成训练总记录

本节统一记录目前已经跑完的 ATLDSD 语义分割训练，后续写论文表格、消融实验和对比实验时优先以这里为准。

### 数据集与统一设置

```text
数据集: ATLDSD / Apple Tree Leaf Disease Segmentation Dataset
格式: VOC
路径: D:\dataset\ATLDSD\VOCdevkit\VOC2012
类别:
0 background
1 leaf
2 rust
3 alternaria_leaf_spot
4 gray_spot
5 brown_spot

训练划分: train 1148
验证划分: val 246
测试划分: test 247
主要评价集: val
主要指标: mIoU, mDice, pixel accuracy, foreground mIoU, per-class IoU
```

### 已完成实验汇总

| 编号 | 实验 | Backbone | 结构 | 最佳 mIoU | FG mIoU | 结论 |
|---|---|---|---|---:|---:|---|
| B0-M | DeepLabV3+ MobileNetV2 | MobileNetV2 | 普通 6 类 softmax | 69.36 | 63.66 | 当前最强 baseline |
| B0-B4 | DeepLabV3+ EfficientNet-B4 | EfficientNet-B4 | 普通 6 类 softmax | 65.59 | 59.59 | B4 并没有超过 MobileNetV2 |
| Ours-A | CLCS 三头组合 | EfficientNet-B4 | leaf / lesion / disease 三头组合 | 65.68 | 60.18 | 相比 B4 baseline 小幅提升 |
| Ours-B1 | CLCS + boundary | EfficientNet-B4 | 三头组合 + 病斑边界辅助头 | 65.87 | 60.20 | B4 系列当前最好，但提升仍小 |
| Ours-B2 | CLCS + disease class-balanced loss | EfficientNet-B4 | 三头组合 + 病害类别加权 | 64.79 | 59.07 | 失败，不建议作为主模块 |

### B0-M: DeepLabV3+ + MobileNetV2

```text
输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenet_150

最终/最佳结果:
mIoU: 69.36%
mDice: 79.60%
Pixel Accuracy: 98.00%
Foreground mIoU: 63.66%
Params: 5.81M
FLOPs: 13.23G
FPS: 104.32
```

判断：

```text
这是目前所有实验里最强的一条线。
它说明 ATLDSD 上轻量 backbone 不一定弱，MobileNetV2 反而比 B4 更稳。
后续主方法如果不能超过 69.36%，论文主张会比较危险。
```

### B0-B4: DeepLabV3+ + EfficientNet-B4

```text
输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_efficientnet_b4_150

Epoch 150 val:
mIoU: 65.59%
mDice: 77.00%
Pixel Accuracy: 96.44%
Foreground mIoU: 59.59%
Params: 32.48M
FLOPs: 51.30G
FPS: 52.77
```

逐类 IoU：

```text
background: 95.57%
leaf: 89.96%
rust: 76.34%
alternaria_leaf_spot: 42.08%
gray_spot: 45.19%
brown_spot: 44.38%
```

判断：

```text
B4 计算量更大，但效果低于 MobileNetV2。
因此后续不能默认 B4 是更强 backbone。
B4 可以保留为消融线，但主线需要补 MobileNetV2 版本的 CLCS。
```

### Ours-A: CLCS 三头组合

```text
输出目录:
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_efficientnet_b4_150

结构:
Head 1: leaf mask
Head 2: lesion mask
Head 3: disease type mask
最后组合为 6 类输出

最佳权重:
weights\best_miou_epoch_weights.pth

最佳 mIoU epoch:
Epoch 120

最佳结果:
mIoU: 65.68%
mDice: 77.27%
Foreground mIoU: 60.18%
Pixel Accuracy: 94.91%
Val Loss: 0.264
```

逐类 IoU：

```text
background: 93.22%
leaf: 86.20%
rust: 79.13%
alternaria_leaf_spot: 48.70%
gray_spot: 49.06%
brown_spot: 37.79%
```

相对 B0-B4：

```text
mIoU: +0.09%
mDice: +0.27%
Foreground mIoU: +0.59%
rust: +2.79%
alternaria_leaf_spot: +6.62%
gray_spot: +3.87%
brown_spot: -6.59%
```

判断：

```text
三头组合有一点信号，尤其对 rust / alternaria / gray 有帮助。
但 brown_spot 明显下降，导致整体提升很小。
单独把 CLCS 作为主创新，目前证据还不够。
```

### Ours-B1: CLCS + 病斑边界辅助头

```text
输出目录:
D:\Code\ATLDSD\outputs\atldsd\clcs_boundary_deeplabv3plus_efficientnet_b4_150

结构:
CLCS 三头组合
+ lesion boundary auxiliary head

loss:
final loss
+ 0.4 leaf loss
+ 0.8 lesion loss
+ 0.6 disease loss
+ 0.3 boundary loss

boundary_pos_weight: 5.0

最佳权重:
weights\best_miou_epoch_weights.pth
```

最佳结果：

```text
mIoU: 65.87%
mDice: 77.38%
Foreground mIoU: 60.20%
Foreground mDice: 73.45%
Pixel Accuracy: 95.56%
Val Loss: 0.2788
```

逐类 IoU：

```text
background: 94.26%
leaf: 87.65%
rust: 77.98%
alternaria_leaf_spot: 49.33%
gray_spot: 46.33%
brown_spot: 39.69%
```

判断：

```text
这是 B4 系列目前最好的模型。
相比 Ours-A，mIoU 从 65.68% 到 65.87%，提升只有 0.19%。
brown_spot 从 37.79% 回升到 39.69%，但仍低于普通 B4 baseline 的 44.38%。
边界辅助头可以保留为消融模块，但不能单独支撑强创新。
```

### Ours-B2: CLCS + disease class-balanced loss

```text
输出目录:
D:\Code\ATLDSD\outputs\atldsd\clcs_disease_balanced_deeplabv3plus_efficientnet_b4_150

结构:
CLCS 三头组合
+ disease head 类别加权交叉熵

disease weights:
rust: 1.0
alternaria_leaf_spot: 1.5
gray_spot: 1.5
brown_spot: 2.0

训练状态:
150 / 150 已完成

最佳权重:
weights\best_miou_epoch_weights.pth
```

最佳结果：

```text
mIoU: 64.79%
mDice: 76.49%
Foreground mIoU: 59.07%
Pixel Accuracy: 95.03%
Val Loss: 0.2654
```

逐类 IoU：

```text
background: 93.41%
leaf: 86.47%
rust: 76.65%
alternaria_leaf_spot: 48.10%
gray_spot: 48.44%
brown_spot: 35.67%
```

Epoch 150 结果：

```text
mIoU: 62.36%
mPA: 69.31%
Accuracy: 96.08%
Train Loss: 0.148
Val Loss: 0.234
```

判断：

```text
类别加权 loss 没有解决 brown_spot 问题，反而使整体 mIoU 低于 Ours-A 和 Ours-B1。
这个实验可以写进消融表作为负结果，但不建议作为最终方法的一部分。
后续不应继续盲目调 disease class weight。
```

### 当前总体结论

```text
1. MobileNetV2 baseline 是当前最强模型，mIoU 69.36%。
2. B4 baseline 和 B4-CLCS 系列都没有超过 MobileNetV2 baseline。
3. CLCS 三头组合在 B4 上有弱正收益，但收益太小。
4. boundary head 有轻微提升，是目前 B4 系列最好的消融模块。
5. disease class-balanced loss 失败，不能作为主线。
6. 当前最大问题不是网络不够复杂，而是小病斑、轻症样本、brown_spot 类别难学。
```

### 重新制定后的训练策略

下一步优先训练：

```text
Ours-A-M:
DeepLabV3+ MobileNetV2 + CLCS 三头组合

目的:
判断 CLCS 三头结构能否在当前最强 baseline MobileNetV2 上继续提升。

关键对比:
B0-M MobileNetV2 baseline: 69.36% mIoU
```

如果 Ours-A-M 超过 69.36%：

```text
说明三头组合结构有效。
继续训练 Ours-B1-M: MobileNetV2 + CLCS + boundary head。
```

如果 Ours-A-M 仍低于 69.36%：

```text
说明三头结构单独不够强。
主创新需要转向 severity-controlled lesion copy-paste。
CLCS 作为结构辅助模块，而不是唯一主创新。
```

后续更值得做的实验：

```text
Ours-C:
CLCS + severity-controlled lesion copy-paste

原因:
ATLDSD 的主要难点是小病斑、轻症样本和 brown_spot 不稳定。
Copy-Paste 能直接增加病斑区域，尤其适合病斑连通域较小、类别不均衡的情况。
```
## 2026-06-03 MobileNetV3-Large baseline 完成

本次训练目的：

```text
验证 MobileNetV3-Large 是否可以替代 MobileNetV2，作为后续论文主线的轻量 backbone。
```

### 实验配置

```text
实验编号: B0-V3
模型: DeepLabV3+
Backbone: MobileNetV3-Large
输出类别: 6
输入尺寸: 256 x 256
训练轮数: 150
训练集: 1148
验证集: 246
预训练: torchvision ImageNet pretrained MobileNetV3-Large
输出目录:
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_150
```

相关代码改动：

```text
D:\Code\ATLDSD\src\models\deeplabv3plus\nets\backbone_registry.py
D:\Code\ATLDSD\src\atldsd_seg\configs\experiments.py
D:\Code\ATLDSD\scripts\run_atldsd_experiment.ps1
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_150.ps1
```

前向验证：

```text
输出: 1 x 6 x 256 x 256
低层特征: 1 x 24 x 64 x 64
高层特征: 1 x 960 x 16 x 16
```

### 最终结果

注意：

```text
auto-export 的 reports\best_val 是按 best val-loss 权重导出的，不是 best mIoU。
论文表格应使用 ep150_val 报告。
```

论文可用报告：

```text
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_150\reports\ep150_val
```

第 150 轮验证集结果：

```text
mIoU: 71.72%
mDice: 81.99%
Pixel Accuracy: 97.76%
Foreground mIoU: 66.58%
Params: 11.73M
FLOPs: 15.28G
FPS: 98.80
```

逐类 IoU：

```text
background: 97.43%
leaf: 93.50%
rust: 81.23%
alternaria_leaf_spot: 52.88%
gray_spot: 56.88%
brown_spot: 48.42%
```

### 与之前 baseline 对比

```text
DeepLabV3+ MobileNetV2:
mIoU 69.36%
FG mIoU 63.66%
Params 5.81M
FLOPs 13.23G
FPS 104.32

DeepLabV3+ MobileNetV3-Large:
mIoU 71.72%
FG mIoU 66.58%
Params 11.73M
FLOPs 15.28G
FPS 98.80
```

结论：

```text
MobileNetV3-Large 比 MobileNetV2 提升 2.36% mIoU。
虽然参数量增加到 11.73M，但仍然属于轻量 backbone，速度也接近 100 FPS。
因此后续论文主 baseline 应从 MobileNetV2 切换为 MobileNetV3-Large。
```

### 目前训练总排名

| 编号 | 实验 | Backbone | mIoU | FG mIoU | 结论 |
|---|---|---|---:|---:|---|
| B0-V3 | DeepLabV3+ | MobileNetV3-Large | 71.72 | 66.58 | 当前最强 baseline |
| B0-M | DeepLabV3+ | MobileNetV2 | 69.36 | 63.66 | 旧轻量 baseline |
| Ours-B1 | CLCS + boundary | EfficientNet-B4 | 65.87 | 60.20 | B4 系列最好 |
| Ours-A | CLCS 三头 | EfficientNet-B4 | 65.68 | 60.18 | 三头弱正收益 |
| B0-B4 | DeepLabV3+ | EfficientNet-B4 | 65.59 | 59.59 | 大 backbone 但效果一般 |
| Ours-B2 | CLCS + 类别加权 | EfficientNet-B4 | 64.79 | 59.07 | 失败消融 |

### 后续策略更新

下一步应训练：

```text
Ours-A-V3:
DeepLabV3+ MobileNetV3-Large + CLCS 三头组合
```

目的：

```text
验证 CLCS 三头组合在当前最强 backbone MobileNetV3-Large 上是否仍然有效。
关键对比线是 B0-V3 的 71.72% mIoU。
```

如果 Ours-A-V3 超过 71.72%：

```text
说明 leaf / lesion / disease type 三头组合在强 backbone 上也有效。
继续训练 Ours-B1-V3: CLCS + boundary head。
```

如果 Ours-A-V3 低于 71.72%：

```text
说明三头组合本身不是主要增益来源。
主创新应转向 severity-controlled lesion copy-paste 或更强的数据增强策略。
```

论文写法建议：

```text
MobileNetV3-Large was adopted as the lightweight encoder backbone due to its favorable balance between segmentation accuracy and computational efficiency.
```
## 2026-06-03 重新规划训练计划：以 MobileNetV3-Large 为主线

本节基于目前所有已完成训练重新制定后续实验路线。

### 当前事实判断

目前最重要的实验事实：

```text
B0-V3: DeepLabV3+ + MobileNetV3-Large
mIoU: 71.72%
FG mIoU: 66.58%
Params: 11.73M
FLOPs: 15.28G
FPS: 98.80

B0-M: DeepLabV3+ + MobileNetV2
mIoU: 69.36%
FG mIoU: 63.66%

B0-B4: DeepLabV3+ + EfficientNet-B4
mIoU: 65.59%
FG mIoU: 59.59%

Ours-B1: CLCS + boundary + EfficientNet-B4
mIoU: 65.87%
FG mIoU: 60.20%
```

结论：

```text
1. MobileNetV3-Large 已经成为当前最强 baseline。
2. B4 线整体不再适合作为论文主线，只适合作为 backbone 对比或历史消融。
3. MobileNetV2 虽然效果不错，但论文观感偏旧，后续只作为 lightweight historical baseline。
4. 后续所有主方法必须优先围绕 MobileNetV3-Large 做。
```

### 总体策略

新的论文训练主线：

```text
DeepLabV3+ MobileNetV3-Large baseline
-> CLCS 三头组合
-> CLCS + boundary auxiliary head
-> CLCS + severity-controlled lesion copy-paste
```

核心问题：

```text
CLCS 三头组合是否能在当前最强 backbone 上超过 71.72% mIoU？
```

如果不能超过，说明三头结构不是主要增益来源，论文主创新应转向：

```text
severity-controlled lesion synthesis / lesion copy-paste
```

### 第一阶段：补齐 MobileNetV3-Large 的 CLCS 消融

#### 实验 1: Ours-A-V3

```text
实验名:
clcs_deeplabv3plus_mobilenetv3_large_150

模型:
DeepLabV3+ + MobileNetV3-Large + CLCS 三头组合

结构:
Head 1: leaf mask
Head 2: lesion mask
Head 3: disease type mask
组合输出 6 类 segmentation

对比对象:
B0-V3: DeepLabV3+ + MobileNetV3-Large, mIoU 71.72%
```

目标：

```text
最低目标: mIoU >= 71.72%
可接受目标: mIoU >= 72.0%
理想目标: mIoU >= 72.5%

重点观察:
alternaria_leaf_spot 是否高于 52.88%
gray_spot 是否高于 56.88%
brown_spot 是否高于 48.42%
```

判断：

```text
如果 Ours-A-V3 超过 71.72%，说明 CLCS 三头组合在强 backbone 上成立。
如果 Ours-A-V3 低于 71.72%，CLCS 不能作为单独主创新，只能作为辅助结构或负消融。
```

#### 实验 2: Ours-B1-V3

```text
实验名:
clcs_boundary_deeplabv3plus_mobilenetv3_large_150

模型:
DeepLabV3+ + MobileNetV3-Large + CLCS 三头组合 + lesion boundary head

对比对象:
Ours-A-V3
B0-V3
```

目标：

```text
最低目标: 高于 Ours-A-V3
可接受目标: mIoU >= 72.2%
理想目标: mIoU >= 72.8%

重点观察:
brown_spot IoU 是否继续升高
病斑边界类的可视化是否更完整
```

判断：

```text
如果 boundary head 只提升小于 0.2%，它只能作为弱消融。
如果提升超过 0.5%，可以保留为组件感知辅助分支。
```

### 第二阶段：真正冲指标的病斑合成增强

#### 实验 3: Ours-C-V3

```text
实验名:
clcs_lesion_copypaste_deeplabv3plus_mobilenetv3_large_150

模型:
DeepLabV3+ + MobileNetV3-Large + CLCS + severity-controlled lesion copy-paste
```

增强逻辑：

```text
1. 从训练 mask 中提取 lesion connected components。
2. 保留 disease class。
3. 只允许贴到 leaf mask 内部。
4. 自动更新 6 类 mask。
5. 根据 lesion_area / leaf_area 控制 low / medium / high severity。
6. 优先增强 brown_spot、gray_spot、alternaria_leaf_spot。
```

为什么这一步最重要：

```text
目前 ATLDSD 的主要瓶颈不是 backbone，而是小病斑、轻症样本和少数病害类别。
单纯 disease class weight 已经失败，说明只改 loss 不够。
Copy-Paste 可以直接增加病斑像素和病斑形态多样性，更符合数据问题。
```

目标：

```text
最低目标: mIoU >= 72.5%
可接受目标: mIoU >= 73.0%
理想目标: mIoU >= 74.0%

重点目标:
brown_spot IoU > 52%
gray_spot IoU > 58%
alternaria_leaf_spot IoU > 55%
```

判断：

```text
如果 Copy-Paste 有明显收益，它应该成为论文主创新之一。
如果 Copy-Paste 对 mIoU 提升不大，但小病斑类别提升明显，也可以作为 disease severity estimation 的支撑实验。
```

### 第三阶段：组合最终模型

只有在前面模块分别有效时，才训练最终组合：

```text
Final-V3:
MobileNetV3-Large + CLCS + boundary head + severity-controlled lesion copy-paste
```

不要一开始就堆所有模块。

原因：

```text
如果直接堆模块，即使指标提升，也无法说明到底是哪一部分有效。
SCI 论文更需要清晰消融，而不是把所有东西混在一起。
```

最终模型目标：

```text
mIoU: 73% 以上
FG mIoU: 68% 以上
brown_spot IoU: 52% 以上
FPS: 尽量保持 80 以上
```

### 第四阶段：对比实验

在主方法稳定后再补对比模型，不要现在到处开训练。

推荐对比表：

```text
1. UNet
2. UNet++
3. PSPNet
4. DeepLabV3+ EfficientNet-B4
5. DeepLabV3+ MobileNetV2
6. DeepLabV3+ MobileNetV3-Large
7. SegNeXt
8. Ours
```

其中重点：

```text
SegNeXt 必须加入，因为它是近年较新的分割结构，对论文说服力有帮助。
MobileNetV3-Large baseline 必须作为核心 baseline，因为它当前最强。
B4 不再作为主 baseline，只作为大 backbone 对比。
```

### 第五阶段：最终测试集

目前所有主要数字都来自 val：

```text
val images: 246
```

论文最终结果应在方法确定后再跑 test：

```text
test images: 247
```

原则：

```text
val 用来调策略和做消融。
test 只在最终方法确定后使用，避免把 test 当验证集反复调。
```

最终需要输出：

```text
val ablation table
test final comparison table
per-class IoU table
complexity table: Params / FLOPs / FPS
severity estimation error table: lesion_area / leaf_area
visualization figure: image / GT / baseline / ours
```

### 当前下一步执行顺序

立即下一步：

```text
1. 新建 MobileNetV3-Large 版 CLCS 训练脚本。
2. 跑 Ours-A-V3 150 epoch。
3. 导出 ep150_val 和 best_miou_val 两套报告。
4. 与 B0-V3 的 71.72% 做对比。
```

如果 Ours-A-V3 成功：

```text
继续 Ours-B1-V3: CLCS + boundary head。
```

如果 Ours-A-V3 失败：

```text
暂停 boundary-V3。
优先实现 severity-controlled lesion copy-paste。
```

当前不建议做：

```text
1. 不建议继续调 B4。
2. 不建议继续调 disease class weight。
3. 不建议马上堆注意力模块。
4. 不建议同时开太多对比模型，先把主线跑通。
```

### 论文主线暂定

如果后续实验成功，论文主线可以写成：

```text
CLCS-Net with Severity-Controlled Lesion Synthesis for Apple Leaf Disease Severity Segmentation
```

中文理解：

```text
不是单纯换 backbone。
不是简单加 attention。
而是：
1. 用 MobileNetV3-Large 建立强轻量 baseline；
2. 用 leaf / lesion / disease 三头组合表达叶片-病斑结构；
3. 用 boundary 或 component-aware 分支增强小病斑边界；
4. 用 severity-controlled copy-paste 解决轻症和小病斑不足。
```
## 2026-06-04 Ours-A-V3 训练完成：CLCS 三头在 MobileNetV3-Large 上失败

本次训练目的：

```text
验证 CLCS leaf / lesion / disease 三头组合结构，是否能在当前最强 backbone MobileNetV3-Large 上超过普通 6 类 softmax baseline。
```

### 实验配置

```text
实验编号: Ours-A-V3
模型: DeepLabV3+ + MobileNetV3-Large + CLCS 三头组合
Backbone: MobileNetV3-Large
输入尺寸: 256 x 256
训练轮数: 150
训练集: 1148
验证集: 246
输出目录:
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_mobilenetv3_large_150
```

新增训练脚本：

```text
D:\Code\ATLDSD\scripts\run_atldsd_clcs_mobilenetv3_large_150.ps1
```

前向验证：

```text
final_logits:   1 x 6 x 256 x 256
leaf_logits:    1 x 2 x 256 x 256
lesion_logits:  1 x 2 x 256 x 256
disease_logits: 1 x 4 x 256 x 256
```

### 最佳结果

最佳权重：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_mobilenetv3_large_150\weights\best_miou_epoch_weights.pth
```

最佳报告：

```text
D:\Code\ATLDSD\outputs\atldsd\clcs_deeplabv3plus_mobilenetv3_large_150\reports\best_miou_val
```

最佳验证集结果：

```text
mIoU: 70.50%
mDice: 80.70%
Pixel Accuracy: 97.97%
Foreground mIoU: 65.05%
Val Loss: 0.126
```

逐类 IoU：

```text
background: 97.76%
leaf: 93.98%
rust: 82.28%
alternaria_leaf_spot: 51.17%
gray_spot: 57.66%
brown_spot: 40.17%
```

第 150 轮结果：

```text
mIoU: 68.75%
mDice: 78.86%
Pixel Accuracy: 97.92%
Foreground mIoU: 62.95%
Val Loss: 0.156
```

第 150 轮逐类 IoU：

```text
background: 97.72%
leaf: 93.86%
rust: 82.58%
alternaria_leaf_spot: 47.68%
gray_spot: 58.54%
brown_spot: 32.11%
```

### 与 B0-V3 baseline 对比

当前最强 baseline：

```text
B0-V3:
DeepLabV3+ + MobileNetV3-Large
mIoU: 71.72%
Foreground mIoU: 66.58%
brown_spot IoU: 48.42%
```

Ours-A-V3：

```text
mIoU: 70.50%
Foreground mIoU: 65.05%
brown_spot IoU: 40.17%
```

差值：

```text
mIoU: -1.22%
Foreground mIoU: -1.53%
brown_spot IoU: -8.25%
```

### 结论

```text
CLCS 三头组合在 MobileNetV3-Large 上没有超过普通 6 类 softmax baseline。
主要失败点是 brown_spot，IoU 从 B0-V3 的 48.42% 降到 40.17%。
因此 CLCS 三头组合不能作为当前论文主创新继续硬推。
```

具体判断：

```text
1. CLCS 在 B4 上只有很弱正收益。
2. CLCS 在更强的 MobileNetV3-Large 上反而下降。
3. 这说明三头组合不是当前数据集上最主要的增益来源。
4. 后续不建议马上继续训练 Ours-B1-V3: CLCS + boundary head。
5. 更应该转向数据层面的病斑增强。
```

### 后续策略调整

下一步优先做：

```text
Ours-C-V3:
DeepLabV3+ + MobileNetV3-Large + severity-controlled lesion copy-paste
```

目的：

```text
解决 ATLDSD 中小病斑、轻症样本和 brown_spot 表现弱的问题。
```

为什么不继续 boundary-V3：

```text
CLCS-A-V3 已经低于 baseline。
如果继续在失败的三头结构上加 boundary，很可能只是继续补救结构缺陷，而不是解决主问题。
```

为什么转向 copy-paste：

```text
disease class-balanced loss 已经失败，说明单纯调 loss 不够。
当前 brown_spot 的问题更像样本不足、病斑形态不足、小区域学习不足。
Severity-controlled lesion copy-paste 能直接增加病斑像素和轻症/中症样本。
```

新的短期训练路线：

```text
1. 实现 severity-controlled lesion copy-paste 数据增强。
2. 先跑普通 DeepLabV3+ MobileNetV3-Large + copy-paste。
3. 与 B0-V3 的 71.72% 对比。
4. 如果 copy-paste 有收益，再考虑是否把 CLCS 作为辅助结构重新加入。
```

当前不建议：

```text
1. 不建议继续跑 CLCS + boundary + MobileNetV3-Large。
2. 不建议继续调 disease class weight。
3. 不建议继续堆注意力模块。
4. 不建议用 B4 作为主线。
```
## 2026-06-04 新整体训练计划：SCLA-Net 主线

本节重新规划后续完整训练路线。原因是当前实验已经证明：

```text
1. DeepLabV3+ + MobileNetV3-Large 已经是当前最强 baseline，mIoU 71.72%。
2. CLCS 三头组合在 MobileNetV3-Large 上下降到 70.50%，不能继续作为主创新硬推。
3. MobileNetV3-Large 自带 SE，但 SE 是通用通道注意力，创新性弱，不足以支撑论文主张。
4. ATLDSD 的核心难点是小病斑、轻症样本、brown_spot 弱、病斑边界不完整。
```

因此新主线不再是：

```text
换 backbone
普通 SE / CBAM / ECA
CLCS 三头强组合
单纯 class weight
```

而是：

```text
SCLA-Net:
Severity-Controlled Lesion Augmentation and Component-guided Attention Network

中文:
严重度控制病斑增强与组件引导注意力网络
```

### 总体思想

新方案由四个层次组成：

```text
Base:
DeepLabV3+ + MobileNetV3-Large

Module 1:
Severity-controlled lesion copy-paste

Module 2:
Lesion component auxiliary supervision

Module 3:
Severity-aware component-guided attention, SCA

Module 4:
Severity consistency loss
```

最终目标不是简单提高 mIoU，而是围绕论文题目形成闭环：

```text
分割病斑
-> 计算病斑面积 / 叶片面积
-> 判断病害严重度
-> 用严重度反过来约束训练
```

### 当前基准线

当前必须超过的核心 baseline：

```text
B0-V3:
DeepLabV3+ + MobileNetV3-Large
mIoU: 71.72%
FG mIoU: 66.58%
Accuracy: 97.76%
brown_spot IoU: 48.42%
gray_spot IoU: 56.88%
alternaria_leaf_spot IoU: 52.88%
FPS: 98.80
Params: 11.73M
FLOPs: 15.28G
```

失败消融：

```text
Ours-A-V3:
MobileNetV3-Large + CLCS 三头组合
mIoU: 70.50%
brown_spot IoU: 40.17%

结论:
CLCS 三头组合不能作为主创新。
```

### 阶段 1：先解决数据问题

#### E1: B0-V3 + Severity-controlled Lesion Copy-Paste

实验名建议：

```text
deeplabv3plus_mobilenetv3_large_sclp_150
```

目的：

```text
解决小病斑、轻症样本不足、brown_spot 弱的问题。
```

方法：

```text
1. 从训练集 mask 中提取 lesion connected components。
2. 每个病斑组件记录:
   disease class
   component area
   bounding box
   source image id
   source severity = lesion_area / leaf_area
3. 训练时随机选取目标 leaf 区域。
4. 只允许把病斑贴到 leaf mask 内。
5. 自动更新 segmentation mask。
6. 按 low / medium / high severity 控制粘贴数量和面积。
7. 优先采样 brown_spot、gray_spot、alternaria_leaf_spot。
```

为什么先做这个：

```text
当前不是模型完全看不懂，而是 brown_spot 和小病斑样本不够稳定。
单纯 disease class weight 已失败，说明只改 loss 不能补数据分布。
Copy-Paste 直接改变训练分布，最可能提升 brown_spot。
```

成功标准：

```text
最低标准:
mIoU >= 72.20%
brown_spot IoU >= 50.00%

理想标准:
mIoU >= 73.00%
brown_spot IoU >= 52.00%
FG mIoU >= 67.50%
```

如果 E1 失败：

```text
不继续加注意力。
先检查 copy-paste 是否贴得太假、是否破坏叶片纹理、是否 severity 分布不合理。
```

### 阶段 2：再加入病斑组件辅助监督

#### E2: B0-V3 + SCLP + Component Auxiliary Heads

实验名建议：

```text
deeplabv3plus_mobilenetv3_large_sclp_component_150
```

目的：

```text
让模型显式学习病斑是小块、边界不规则、位于叶片内部的组件。
```

辅助监督不改变最终输出，最终推理仍然是普通 6 类 segmentation 主头。

辅助头：

```text
Head A: lesion binary mask
Head B: lesion boundary map
Head C: lesion center / distance transform heatmap
```

伪标签来源：

```text
全部从现有 mask 自动生成，不需要额外人工标注。
```

损失函数：

```text
L_total =
L_seg
+ lambda_lesion * L_lesion
+ lambda_boundary * L_boundary
+ lambda_center * L_center
```

建议初始权重：

```text
lambda_lesion = 0.3
lambda_boundary = 0.2
lambda_center = 0.1
```

成功标准：

```text
相比 E1:
mIoU 提升 >= 0.30%
或 brown_spot / gray_spot / alternaria 至少两个类别提升。
```

如果 E2 只提升很小：

```text
保留 component auxiliary 作为弱消融。
不要把它作为唯一主创新。
```

### 阶段 3：加入真正有针对性的注意力

#### E3: SCA Attention

模块名：

```text
SCA:
Severity-aware Component-guided Attention

中文:
严重度感知组件引导注意力
```

这个模块不是 SE、CBAM、ECA。

区别：

```text
SE:
只做通道重标定，缺少空间病斑定位。

CBAM:
通用空间+通道注意力，不知道 leaf / lesion / severity 结构。

SCA:
用 lesion component、boundary 和 severity 估计来引导 decoder 特征关注叶片内部小病斑区域。
```

输入：

```text
decoder feature F
lesion auxiliary prediction P_lesion
boundary auxiliary prediction P_boundary
predicted severity scalar or severity map P_sev
```

结构建议：

```text
1. Multi-scale depthwise spatial branch:
   DWConv 3x3
   DWConv 5x5
   DWConv 7x7 or dilated 3x3

2. Component guidance branch:
   concat(P_lesion, P_boundary, P_sev)
   1x1 conv -> sigmoid attention map

3. Fusion:
   A = sigmoid(MS-DWConv(F) + ComponentGuide)
   F_out = F + F * A
```

放置位置：

```text
DeepLabV3+ decoder 后、分类头前。
```

为什么这样放：

```text
decoder 特征已经融合了低层边界和高层语义。
在这里做 lesion-guided attention，比在 backbone 里加通用 attention 更贴近病斑分割任务。
```

实验名建议：

```text
deeplabv3plus_mobilenetv3_large_sclp_component_sca_150
```

成功标准：

```text
相比 E2:
mIoU 提升 >= 0.30%
FG mIoU 提升 >= 0.30%
brown_spot IoU 不下降
```

如果 SCA 有效：

```text
SCA 可以作为论文的核心结构创新。
```

如果 SCA 无效：

```text
保留 SCLP + component auxiliary，SCA 作为失败消融或不写入主方法。
```

### 阶段 4：加入严重度一致性约束

#### E4: Severity Consistency Loss

实验名建议：

```text
deeplabv3plus_mobilenetv3_large_sclp_component_sca_sevloss_150
```

目的：

```text
让模型不只分割类别，还要让预测的病害严重度与 GT 严重度一致。
```

严重度定义：

```text
GT severity = GT lesion pixels / GT leaf pixels
Pred severity = predicted lesion probability area / predicted leaf probability area
```

损失：

```text
L_severity = SmoothL1(Pred severity, GT severity)
```

总损失：

```text
L_total =
L_seg
+ lambda_lesion * L_lesion
+ lambda_boundary * L_boundary
+ lambda_center * L_center
+ lambda_severity * L_severity
```

建议初始权重：

```text
lambda_severity = 0.05
```

成功标准：

```text
mIoU 不下降超过 0.10%
severity MAE 明显下降
严重度分组 low / medium / high 的分类准确率提高
```

为什么最后再加：

```text
severity loss 太早加入可能干扰像素分割。
先把 segmentation 和 lesion attention 做稳，再用 severity loss 做约束更合理。
```

### 阶段 5：最终模型

最终模型暂定：

```text
SCLA-Net:
MobileNetV3-Large DeepLabV3+
+ Severity-controlled lesion copy-paste
+ Component auxiliary heads
+ Severity-aware component-guided attention
+ Severity consistency loss
```

最终目标：

```text
mIoU >= 73.00%
FG mIoU >= 68.00%
brown_spot IoU >= 52.00%
gray_spot IoU >= 58.00%
alternaria_leaf_spot IoU >= 55.00%
FPS >= 80
```

### 消融实验表设计

主消融表：

```text
B0-V3
B0-V3 + SCLP
B0-V3 + SCLP + ComponentAux
B0-V3 + SCLP + ComponentAux + SCA
B0-V3 + SCLP + ComponentAux + SCA + SeverityLoss
```

负消融表：

```text
CLCS 三头组合
disease class-balanced loss
普通 SE / CBAM，如果有时间可补
```

注意：

```text
普通 SE / CBAM 不是必须做。
如果要证明 SCA 不是普通注意力，可以补一个 CBAM 对比。
但不要把 CBAM 作为主方法。
```

### 对比实验表

最终对比模型：

```text
UNet
UNet++
PSPNet
DeepLabV3+ EfficientNet-B4
DeepLabV3+ MobileNetV2
DeepLabV3+ MobileNetV3-Large
SegNeXt
SCLA-Net
```

SegNeXt 必须加入：

```text
因为它是较新的分割结构，可以增强论文对比说服力。
```

### 执行顺序

立即执行：

```text
Step 1:
实现 severity-controlled lesion copy-paste。

Step 2:
跑 E1: B0-V3 + SCLP 150 epoch。

Step 3:
如果 E1 超过 72.20%，实现 Component Auxiliary Heads。

Step 4:
如果 E2 有提升，实现 SCA 注意力。

Step 5:
如果 E3 有提升，加 Severity Consistency Loss。

Step 6:
最终模型确定后，只在最终阶段跑 test set。
```

当前不要做：

```text
1. 不继续堆 CLCS。
2. 不把 SE 当创新。
3. 不直接上最终大杂烩模型。
4. 不先跑 test。
5. 不同时开太多对比模型。
```

### 论文创新表达

可以这样写创新点：

```text
1. A severity-controlled lesion copy-paste strategy is proposed to synthesize low-, medium-, and high-severity apple leaf disease samples while preserving leaf-region constraints.

2. A component-aware auxiliary learning branch is introduced to enhance small lesion localization, boundary integrity, and lesion component representation.

3. A severity-aware component-guided attention module is designed to focus decoder features on lesion-prone regions, avoiding generic channel-only attention such as SE.

4. A severity consistency loss is formulated to align segmentation predictions with disease severity estimation.
```

中文概括：

```text
不是简单加注意力。
而是围绕 ATLDSD 的叶片-病斑-严重度关系，做一个从数据增强、组件学习、注意力引导到严重度约束的完整方法。
```
## 2026-06-04 E1 训练启动：B0-V3 + SCLP

本次训练对应新整体方案 SCLA-Net 的第一阶段。

实验目的：

```text
验证 severity-controlled lesion copy-paste 是否能超过当前最强 baseline B0-V3。
重点解决小病斑、轻症样本不足、brown_spot 弱的问题。
```

实验配置：

```text
实验编号: E1
实验名: deeplabv3plus_mobilenetv3_large_sclp_150
模型: DeepLabV3+ + MobileNetV3-Large
增强: Severity-controlled lesion copy-paste, SCLP
训练轮数: 150
输入尺寸: 256 x 256
训练集: 1148
验证集: 246
```

SCLP 参数：

```text
sclp: true
sclp_prob: 0.7
sclp_max_components: 3
sclp_class_weights:
rust 1.0
alternaria_leaf_spot 2.0
gray_spot 2.0
brown_spot 3.0
```

输出目录：

```text
D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_sclp_150
```

启动状态：

```text
PID: 34000
状态: 正在训练
当前: Epoch 1 / 150
启动时间: 2026-06-04 10:39
```

新增/修改代码：

```text
D:\Code\ATLDSD\src\models\deeplabv3plus\utils\dataloader.py
D:\Code\ATLDSD\src\models\deeplabv3plus\train.py
D:\Code\ATLDSD\scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_sclp_150.ps1
```

本轮成功标准：

```text
最低标准:
mIoU >= 72.20%
brown_spot IoU >= 50.00%

理想标准:
mIoU >= 73.00%
brown_spot IoU >= 52.00%
FG mIoU >= 67.50%
```

关键对比 baseline：

```text
B0-V3:
DeepLabV3+ + MobileNetV3-Large
mIoU: 71.72%
FG mIoU: 66.58%
brown_spot IoU: 48.42%
```

## 2026-06-04 Ubuntu 服务器迁移与代码同步要求

本次已经把项目从“只适合 Windows 本机跑”的状态，改造成 Windows / Ubuntu 都能启动训练的状态。

### 当前项目状态

```text
GitHub 仓库:
https://github.com/yezi56/SCLA-Net-ATLDSD

当前主分支:
main

最新提交:
6a1e330 Add Ubuntu Python bootstrap script
56e3a63 Add portable Ubuntu training scripts
c3dc630 Initial SCLA-Net ATLDSD research code
```

当前最强 baseline：

```text
B0-V3
模型: DeepLabV3+ + MobileNetV3-Large
训练方式: 普通 6 类语义分割
增强: 常规增强，无 SCLP
mIoU: 71.72%
FG mIoU: 66.58%
Accuracy: 97.76%
brown_spot IoU: 48.42%
```

当前正在跑 / 当前下一步实验：

```text
E1: B0-V3 + SCLP
模型结构: DeepLabV3+ + MobileNetV3-Large
输出方式: 6 类 softmax
与最强 baseline 的区别: 只增加 SCLP 数据增强
SCLP: severity-controlled lesion copy-paste
实验目的: 验证严重度控制病斑复制粘贴是否能提升小病斑、轻症样本和 brown_spot 类别表现
```

重要判断：

```text
E1 不是新网络结构。
E1 是数据增强消融实验。
如果 E1 超过 B0-V3，说明 SCLP 有价值。
如果 E1 不超过 B0-V3，则后续需要降低 SCLP 概率，或改进粘贴真实性，再进入注意力/辅助头实验。
```

### Ubuntu 一键启动能力

新增文件：

```text
requirements-atldsd.txt
scripts/bootstrap_ubuntu_miniconda.sh
scripts/setup_ubuntu_env.sh
scripts/run_ubuntu.sh
scripts/run_ubuntu_baseline_v3.sh
scripts/run_ubuntu_sclp_v3.sh
.gitattributes
```

Ubuntu 推荐目录：

```text
/home/liuzhe/SCLA-Net-ATLDSD
/home/liuzhe/SCLA-Net-ATLDSD/VOCdevkit/VOC2007
/home/liuzhe/SCLA-Net-ATLDSD/VOCdevkit/VOC2012
```

如果数据集不放在项目根目录：

```bash
export ATLDSD_VOCDEVKIT_PATH=/absolute/path/to/VOCdevkit
```

有 Python 的服务器：

```bash
cd /home/liuzhe/SCLA-Net-ATLDSD
chmod +x scripts/*.sh
./scripts/setup_ubuntu_env.sh cu121
./scripts/run_ubuntu.sh sclp
```

没有 Python 的服务器：

```bash
cd /home/liuzhe/SCLA-Net-ATLDSD
chmod +x scripts/*.sh
./scripts/bootstrap_ubuntu_miniconda.sh
export PATH="$HOME/miniconda3/bin:$PATH"
./scripts/setup_ubuntu_env.sh cu121
./scripts/run_ubuntu.sh sclp
```

启动 baseline：

```bash
./scripts/run_ubuntu.sh baseline
```

启动当前 SCLP 实验：

```bash
./scripts/run_ubuntu.sh sclp
```

### 以后代码修改必须遵守的同步规则

后续所有代码修改，都必须同时考虑 Windows 本机和 Ubuntu 服务器两套入口。

必须检查：

```text
1. Windows 训练脚本是否需要同步修改:
   scripts/run_atldsd_*.ps1

2. Ubuntu 训练脚本是否需要同步修改:
   scripts/run_ubuntu*.sh

3. Python 默认路径是否仍然非硬编码:
   src/atldsd_seg/paths.py
   src/models/deeplabv3plus/train.py
   src/atldsd_seg/engine/train_clcs.py

4. README 是否需要同步说明:
   README.md

5. 变更后是否提交并推送到 GitHub:
   git commit
   git push
```

特别强调：

```text
以后不能只改 Windows 的 .ps1，不改 Linux 的 .sh。
也不能只改 Linux 的 .sh，不改 Windows 的 .ps1。
如果训练参数、模型结构、数据增强、输出目录、依赖项发生变化，Windows 和 Linux 两边都要同步。
每次 push 到 GitHub 后，必须在笔记里写清楚:
1. 提交哈希
2. 修改了哪些文件
3. 修改目的
4. 对应哪个实验
5. Windows / Linux 是否都已同步
```

### 最近两次 GitHub 上传记录

```text
提交: 56e3a63
标题: Add portable Ubuntu training scripts
目的: 增加 Ubuntu 服务器一键训练能力，去除项目对 D:\Code 和 D:\dataset 的强依赖。
主要修改:
- src/atldsd_seg/paths.py
- src/atldsd_seg/engine/train_clcs.py
- src/models/deeplabv3plus/train.py
- requirements-atldsd.txt
- scripts/setup_ubuntu_env.sh
- scripts/run_ubuntu.sh
- scripts/run_ubuntu_baseline_v3.sh
- scripts/run_ubuntu_sclp_v3.sh
- README.md
Windows / Linux 同步状态: 已同步。
```

## 2026-06-04 基于 Reviewer 批评后的纠偏版训练计划

本节用于修正原训练路线。核心原则：

```text
先修实验协议，再谈创新模块。
先证明严重度任务闭环，再谈 SCI 论文主线。
先让 SCLP 通过消融筛选，再决定是否保留为创新点。
```

### 当前风险判断

Reviewer 会抓住的三个硬伤：

```text
1. SCLP 0.7 目前是负收益，不能直接作为主创新。
2. best_epoch_weights.pth 按 val loss 保存，但论文核心指标是 mIoU，checkpoint 选择规则混乱。
3. 论文目标说“病害严重度判断”，但目前主要只评估语义分割，没有正式严重度 MAE / RMSE / 相关性 / 分级准确率。
```

因此，后续训练计划必须从“堆模块”改为“先补协议闭环”。

### 当前正在跑的实验

```text
实验编号: E1.1
实验名称: deeplabv3plus_mobilenetv3_large_sclp03_150
状态: 正在训练，不中断
当前进度: 约 epoch 59 / 150
已知中期结果:
  epoch50 mIoU = 69.90
对比:
  B0-V3 baseline mIoU = 71.72
判断:
  E1.1 比 E1 的 SCLP 0.7 更稳，但当前仍未超过 baseline。
```

最新滚动状态：

```text
epoch60 mIoU = 62.02
说明: epoch50 后骨干解冻，短期震荡明显。
决策: 不因 epoch60 单点中断，至少观察 epoch80 / epoch100 是否恢复。
```

E1.1 继续跑完，原因：

```text
1. 它是纠正 E1 过强增强后的必要消融。
2. epoch50 已经达到 69.90，比 E1 后期最高 68.97 更好。
3. 不能只看中期结果，需要等 epoch100 / epoch120 / epoch150。
```

### 阶段 0：立刻修正评价协议

目的：

```text
消除 Reviewer 对结果选择规则的质疑。
```

必须补的代码/流程：

```text
1. 训练时同时保存:
   - best_val_loss_weights.pth
   - best_miou_weights.pth
   - last_epoch_weights.pth

2. 所有论文表格统一使用:
   best_miou_weights.pth on validation set

3. 最终模型确定后，只在 test set 上评估一次。

4. 每个实验都导出:
   - metrics_summary.json
   - per_class_metrics.csv
   - confusion_matrix.csv
   - complexity.json
   - severity_metrics.json
```

同步要求：

```text
如果改训练保存逻辑:
Windows .ps1 不一定改，但 Linux .sh 和 README 要检查。
如果新增报告参数:
Windows .ps1 和 Linux .sh 必须同步。
修改后必须 push GitHub，并在本笔记记录提交哈希。
```

### 阶段 1：补严重度评估，不先加新模块

严重度定义：

```text
GT severity = GT lesion pixels / GT leaf pixels
Pred severity = Pred lesion pixels / Pred leaf pixels
lesion classes = rust + alternaria_leaf_spot + gray_spot + brown_spot
leaf region = leaf + lesion classes
```

必须新增指标：

```text
1. Severity MAE
2. Severity RMSE
3. Pearson correlation
4. Spearman correlation
5. Low / Medium / High severity classification accuracy
6. Severity confusion matrix
```

建议严重度分级：

```text
low:    severity < 5%
medium: 5% <= severity < 20%
high:   severity >= 20%
```

如果数据分布不适合 5% / 20%，则改为按训练集三分位数划分，但必须在论文中说明阈值来自 train set，不能用 test set 调阈值。

### 阶段 2：SCLP 强度消融，而不是盲目继续堆

已有：

```text
B0-V3: SCLP 0.0, mIoU 71.72
E1:    SCLP 0.7, best observed mIoU 68.97
E1.1:  SCLP 0.3, 正在训练
```

纠偏后的消融顺序：

```text
1. 等 E1.1 跑完。
2. 如果 E1.1 best mIoU >= 71.72:
   保留 SCLP，继续做 SCLP 0.1 / 0.5 补曲线。

3. 如果 E1.1 best mIoU 在 70.50 到 71.72:
   查看 per-class IoU 和 severity metrics。
   如果 brown_spot / gray_spot 或 severity MAE 明显改善，可以把 SCLP 写成 severity-oriented trade-off。

4. 如果 E1.1 best mIoU < 70.50:
   SCLP 不再作为主创新，只作为失败消融记录。
   后续不再跑 SCLP 0.5，最多补一个 SCLP 0.1 证明趋势。
```

SCLP 的保留阈值：

```text
强保留:
  mIoU >= 71.72
  且 brown_spot IoU >= 48.42

弱保留:
  mIoU 下降不超过 1.0
  但 severity MAE 下降 >= 5%
  或 brown_spot / gray_spot 明显提升

放弃:
  mIoU < 70.50
  且 per-class / severity 无明显收益
```

### 阶段 3：重新定义主创新路线

如果 SCLP 不能成为主创新，则论文主线改为：

```text
Component-guided Disease Severity Segmentation
```

优先创新顺序：

```text
1. Component-aware auxiliary learning
   预测 lesion mask / lesion boundary / lesion center or distance map。
   目的: 强化小病斑定位和边界完整性。

2. Severity consistency loss
   约束预测 lesion/leaf ratio 与 GT lesion/leaf ratio 一致。
   目的: 让分割结果服务于严重度判断。

3. Severity-aware component-guided attention
   注意力不能写成普通 SE/CBAM。
   必须由 lesion / boundary / severity cue 引导，解决“病斑小、边界碎、严重度依赖面积比例”的问题。

4. SCLP
   只有在 E1.1 或后续强度消融证明有效时，才作为辅助增强保留。
```

### 阶段 4：新的训练顺序

当前不要马上开新结构训练。顺序改为：

```text
Step 1:
等待 E1.1 跑完。

Step 2:
改代码，保存 best_miou_weights.pth，并新增 severity_metrics.json。

Step 3:
用统一评价脚本重新导出:
  B0-V3
  E1 SCLP 0.7
  E1.1 SCLP 0.3

Step 4:
根据 E1.1 结果决定:
  - 是否补 SCLP 0.1
  - 是否补 SCLP 0.5
  - 是否放弃 SCLP 主线

Step 5:
训练 Component-aware auxiliary head。

Step 6:
训练 Component-aware auxiliary head + severity consistency loss。

Step 7:
最后再训练 severity-aware component-guided attention。

Step 8:
选最终方法后，跑 3 seeds:
  seed 11
  seed 22
  seed 33

Step 9:
最终 test set 只评一次。
```

### 纠偏后的实验表格设计

主表：

```text
Method | Backbone | mIoU | FG mIoU | Dice | PA | Params | FLOPs | FPS | Severity MAE | Severity RMSE
```

类别表：

```text
Method | leaf | rust | alternaria | gray | brown
```

消融表：

```text
B0-V3
B0-V3 + SCLP 0.7
B0-V3 + SCLP 0.3
B0-V3 + Component Aux
B0-V3 + Component Aux + Severity Loss
B0-V3 + Component Aux + Severity Loss + SCA
```

严重度表：

```text
Method | MAE | RMSE | Pearson | Spearman | Low Acc | Medium Acc | High Acc
```

### 当前执行决策

```text
当前不停止 E1.1。
当前不马上开启注意力训练。
当前下一项代码任务应是:
  1. best_miou checkpoint 保存
  2. severity metrics 导出
  3. 统一报告脚本
```

一句话结论：

```text
先把实验做可信，再把方法做复杂。
否则后续所有模块都会被 Reviewer 认为是在负收益增强和混乱评价协议上堆东西。
```

## 2026-06-04 结合 2025/2026 文献趋势后的再次修正计划

### E1.1 最终结果

```text
实验编号: E1.1
实验名称: deeplabv3plus_mobilenetv3_large_sclp03_150
状态: 已完成 150 epoch
模型: DeepLabV3+ + MobileNetV3-Large
差异: 只增加低强度 SCLP 数据增强
SCLP 参数:
  sclp_prob = 0.3
  sclp_max_components = 2
```

曲线 mIoU：

```text
epoch50:  69.90
epoch60:  62.02
epoch70:  65.69
epoch80:  64.52
epoch90:  68.08
epoch100: 67.11
epoch110: 68.20
epoch120: 63.24
epoch130: 63.31
epoch140: 64.21
epoch150: 66.24
```

自动报告 best-val-loss 权重：

```text
mIoU: 68.03
FG mIoU: 62.43
Accuracy: 96.74

per-class IoU:
background: 96.03
leaf: 90.76
rust: 77.34
alternaria_leaf_spot: 50.67
gray_spot: 48.74
brown_spot: 44.62
```

与最强 baseline 对比：

```text
B0-V3 baseline:
mIoU: 71.72
FG mIoU: 66.58
brown_spot IoU: 48.42

E1.1 SCLP 0.3:
最高曲线 mIoU: 69.90
best-val-loss report mIoU: 68.03
brown_spot IoU: 44.62
```

结论：

```text
SCLP 0.3 没有超过 baseline。
SCLP 0.7 也没有超过 baseline。
因此，SCLP 不能作为当前论文主创新。
后续 SCLP 只保留为负结果/辅助消融，不继续作为主线推进。
```

### 2025/2026 文献趋势

近期强相关论文给出的方向：

```text
1. 显式 leaf / lesion 分离，再做严重度评估。
2. 不满足于普通 mIoU，还要报告 severity estimation 指标。
3. 强调复杂背景、小病斑、边界、类别不平衡。
4. 用动态损失、边界/区域协同、显式关系建模，而不是简单 copy-paste。
5. 部分工作已经直接做 severity grading，因此只做普通语义分割不够。
```

参考文献/方向：

```text
FLARE: Focused leaf-lesion awareness via explicit relational modeling for plant disease severity assessment.
Computers and Electronics in Agriculture, 2026.
核心启发: leaf-lesion 显式关系建模，直接服务 disease severity assessment。

Plant Disease Segmentation Networks for Fast Automatic Severity Estimation.
Agriculture, 2025.
核心启发: 分割结果必须落到 severity estimation，不能只停留在 mIoU。

LKCAFormer: a large-kernel cooperative attention transformer for multi-disease segmentation and grading in maize leaf diseases.
BMC Plant Biology, 2026.
核心启发: segmentation + grading 一起做，注意力设计围绕多病害和严重度分级。

STAR-Net: spatial-spectral cross-attention and dynamic pixel-wise loss for plant disease segmentation under complex greenhouse conditions.
Frontiers in Plant Science, 2026.
核心启发: 动态像素级损失用于处理小目标、边界和类别不平衡，比固定 copy-paste 更像可发表方法。
```

### 对当前方案的更正

原计划的问题：

```text
把 SCLP 当作早期核心创新，但实验连续失败。
如果继续围绕 SCLP 堆注意力，会被 Reviewer 认为是在负收益增强上堆模块。
```

更正后主线：

```text
CLSG-Net:
Component-guided Lesion Segmentation and Severity Grading Network

中文:
组件引导的病斑分割与病害严重度分级网络
```

主张改成：

```text
不是主打 copy-paste。
不是主打普通注意力。
而是把 ATLDSD 的结构关系编码进训练和评价:
leaf region -> lesion component -> disease class -> severity ratio / severity grade
```

### 下一步不再继续训练 SCLP 0.5

取消/暂停：

```text
暂停 SCLP 0.5
暂停 SCLP 0.1
暂停继续调 copy-paste 参数
```

原因：

```text
SCLP 0.7: 最高 mIoU 68.97
SCLP 0.3: 最高 mIoU 69.90
baseline: 71.72

两个 SCLP 都低于 baseline。
继续调 SCLP 的科研收益低，时间成本高。
```

SCLP 后续定位：

```text
保留为 ablation 中的 failed augmentation baseline。
如果后续主模型已经稳定，再考虑作为 minor augmentation 重新测试。
```

### 新训练计划

#### Step 0：修评价协议

马上做代码修改：

```text
1. 训练过程保存 best_miou_weights.pth。
2. 保留 best_epoch_weights.pth 但重命名语义:
   best_val_loss_weights.pth
3. 报告脚本支持指定 checkpoint:
   last / best_val_loss / best_miou / epXXX
4. 自动报告默认改为 best_miou。
```

目的：

```text
让所有实验比较都使用同一 checkpoint 选择规则。
避免 Reviewer 质疑 cherry-picking。
```

#### Step 1：补 severity metrics

新增评估：

```text
GT severity = GT lesion pixels / GT leaf pixels
Pred severity = Pred lesion pixels / Pred leaf pixels

lesion classes:
rust, alternaria_leaf_spot, gray_spot, brown_spot

leaf area:
leaf + lesion classes
```

输出指标：

```text
Severity MAE
Severity RMSE
Pearson correlation
Spearman correlation
Low / Medium / High severity accuracy
Severity confusion matrix
```

输出文件：

```text
severity_metrics.json
severity_per_image.csv
severity_confusion_matrix.csv
```

#### Step 2：重新导出已有实验

统一导出：

```text
B0-V3 baseline
E1 SCLP 0.7
E1.1 SCLP 0.3
```

每个都必须有：

```text
best_miou report
severity metrics
per-class metrics
complexity metrics
```

这一步完成后才能继续训练新模块。

#### Step 3：训练 Component Auxiliary Baseline

新实验：

```text
E2:
DeepLabV3+ MobileNetV3-Large + Component Auxiliary Heads
```

辅助头：

```text
1. lesion binary mask head
2. lesion boundary head
3. lesion distance / center heatmap head
```

主输出仍然是：

```text
6-class semantic segmentation
```

目标：

```text
提升小病斑、边界和病斑连通组件定位。
```

成功标准：

```text
mIoU >= 71.72
FG mIoU >= 66.58
brown_spot IoU >= 48.42
Severity MAE 低于 baseline
```

#### Step 4：加入 Severity Consistency Loss

新实验：

```text
E3:
E2 + Severity Consistency Loss
```

损失：

```text
L_sev = SmoothL1(pred_lesion_area / pred_leaf_area, gt_lesion_area / gt_leaf_area)
```

目标：

```text
不只提升分割 mIoU，还要让预测病斑面积比例更接近真实严重度。
```

成功标准：

```text
即使 mIoU 只小幅提升，也必须显著降低 Severity MAE / RMSE。
```

#### Step 5：最后才做 SCA 注意力

新实验：

```text
E4:
E3 + Severity-aware Component-guided Attention
```

注意力必须满足：

```text
由 lesion / boundary / severity cue 引导。
不能只是 SE / CBAM / 普通空间注意力。
```

目标：

```text
让注意力服务于病斑组件和严重度判断，而不是泛泛加模块。
```

### 修正后的实验顺序

```text
现在:
E1.1 已完成，SCLP 降级。

下一步:
1. 改 best_miou checkpoint 保存逻辑。
2. 改 severity metrics 导出。
3. 重导 B0-V3 / E1 / E1.1。
4. 开 E2 Component Auxiliary Heads。
5. 开 E3 Severity Consistency Loss。
6. 开 E4 Severity-aware Component-guided Attention。
7. 选最终模型后做 3 seeds。
8. 最终 test set 只评一次。
```

### 论文主线更新

旧主线：

```text
SCLA-Net:
Severity-Controlled Lesion Augmentation and Component-guided Attention Network
```

问题：

```text
augmentation 目前实验失败，不适合放在标题和主创新第一位。
```

新主线建议：

```text
CLSG-Net:
Component-guided Lesion Segmentation and Severity Grading Network
```

论文贡献改为：

```text
1. A component-guided auxiliary learning framework for lesion-aware apple leaf disease segmentation.
2. A severity consistency constraint that aligns segmentation with disease severity grading.
3. A severity-aware component-guided attention module for small lesion and boundary-sensitive representation.
4. A rigorous ATLDSD benchmark reporting both segmentation metrics and severity estimation metrics.
```

最终判断：

```text
现在不该继续赌 SCLP。
下一步的真正关键是把“语义分割”升级成“病斑组件分割 + 严重度估计”。
这样才更接近 2025/2026 相关论文的发表逻辑。
```

## 2026-06-04 Step 0/1 完成：best-mIoU checkpoint 与严重度评估

### 本次代码提交

```text
提交: 2b797ca
标题: Add best mIoU checkpoint and severity metrics
目的:
1. 修正 checkpoint 选择协议，新增 best_miou_weights.pth。
2. 报告脚本新增 severity metrics。
3. Windows / Linux 启动脚本同步改为默认导出 best_miou 报告。

主要修改:
- src/models/deeplabv3plus/utils/callbacks.py
- src/models/deeplabv3plus/utils/utils_fit.py
- src/models/deeplabv3plus/train.py
- scripts/export_segmentation_report.py
- src/atldsd_seg/configs/experiments.py
- src/atldsd_seg/engine/launch_deeplabv3plus.py
- scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_sclp_150.ps1
- scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_sclp03_150.ps1
- scripts/run_ubuntu_baseline_v3.sh
- scripts/run_ubuntu_sclp_v3.sh
- scripts/run_ubuntu_sclp03_v3.sh
- README.md

Windows / Linux 同步状态:
已同步。
```

### 新增输出文件

训练后新增：

```text
best_miou_weights.pth
best_miou.txt
best_val_loss_weights.pth
last_epoch_weights.pth
```

报告后新增：

```text
severity_metrics.json
severity_per_image.csv
severity_confusion_matrix.csv
```

severity 定义：

```text
GT severity = GT lesion pixels / GT leaf pixels
Pred severity = Pred lesion pixels / Pred leaf pixels
lesion classes = 2, 3, 4, 5
leaf class = 1
severity grade thresholds:
  low < 0.05
  0.05 <= medium < 0.20
  high >= 0.20
```

### 已用新协议重导已有实验

统一使用 val split，并选择旧训练中 mIoU 曲线对应的最佳 checkpoint：

```text
B0-V3:
checkpoint: ep150-loss0.491-val_loss0.469.pth
report_dir: outputs/atldsd/deeplabv3plus_mobilenetv3_large_150/reports/best_miou

E1 SCLP 0.7:
checkpoint: ep120-loss0.387-val_loss0.499.pth
report_dir: outputs/atldsd/deeplabv3plus_mobilenetv3_large_sclp_150/reports/best_miou

E1.1 SCLP 0.3:
checkpoint: ep050-loss0.426-val_loss0.415.pth
report_dir: outputs/atldsd/deeplabv3plus_mobilenetv3_large_sclp03_150/reports/best_miou
```

### 统一重导结果

```text
Method              mIoU    FG mIoU  Acc     Sev MAE   Sev RMSE  Pearson  Spearman  Grade Acc
B0-V3              71.72   66.58    97.76   0.0124    0.0338    0.8652   0.9134    95.12
SCLP 0.7           68.97   63.28    97.71   0.0154    0.0397    0.8306   0.8911    90.24
SCLP 0.3           69.90   64.67    96.76   0.0159    0.0393    0.8586   0.8840    90.65
```

关键结论：

```text
1. B0-V3 在 mIoU、FG mIoU、Severity MAE、Severity RMSE、Spearman、Grade Accuracy 上都优于两个 SCLP。
2. SCLP 0.3 虽然比 SCLP 0.7 的 mIoU 稍高，但严重度 MAE 更差。
3. SCLP 不能作为主创新，也不适合作为当前优先增强。
```

当前决策：

```text
SCLP 主线正式降级。
下一步进入 E2:
DeepLabV3+ MobileNetV3-Large + Component Auxiliary Heads

E2 目标:
1. 保持或超过 B0-V3 mIoU = 71.72。
2. 降低 severity MAE < 0.0124。
3. 提升 gray_spot / brown_spot 等小病斑类。
```

## 2026-06-04 E2 Component Auxiliary Heads 启动

### 本次代码提交

```text
提交: 58ff1b9
标题: Add component auxiliary training experiment
目的:
实现并启动 E2: DeepLabV3+ MobileNetV3-Large + Component Auxiliary Heads。

主要修改:
- src/models/deeplabv3plus/nets/deeplabv3_plus.py
- src/models/deeplabv3plus/nets/deeplabv3_training.py
- src/models/deeplabv3plus/utils/utils_fit.py
- src/models/deeplabv3plus/utils/utils_metrics.py
- src/models/deeplabv3plus/utils/callbacks.py
- src/models/deeplabv3plus/deeplab.py
- src/models/deeplabv3plus/train.py
- scripts/export_segmentation_report.py
- scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_150.ps1
- scripts/run_ubuntu_component_aux_v3.sh
- scripts/run_ubuntu.sh
- README.md

Windows / Linux 同步状态:
已同步。
```

### E2 结构

```text
主干: MobileNetV3-Large
分割头: DeepLabV3+ 6-class softmax
辅助头:
1. lesion binary mask head
2. lesion boundary head
3. lesion center heatmap head
```

辅助监督来自原 mask 自动生成，不需要额外标注：

```text
lesion target = class 2/3/4/5
boundary target = lesion eroded boundary
center target = lesion local-density heatmap
```

loss 权重：

```text
main segmentation loss: CE + Dice
component_lesion_weight: 0.4
component_boundary_weight: 0.2
component_center_weight: 0.2
```

### 启动状态

```text
实验编号: E2
实验名称: deeplabv3plus_mobilenetv3_large_component_aux_150
启动时间: 2026-06-04 17:23
PID: 26536
状态: 正在训练
输出目录:
  D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_150
```

### 成功标准

```text
必须超过或至少接近:
B0-V3 mIoU = 71.72
B0-V3 FG mIoU = 66.58
B0-V3 Severity MAE = 0.0124
B0-V3 Grade Acc = 95.12
```

判断规则：

```text
如果 E2 mIoU 提升，且 severity MAE 不变差:
  Component Auxiliary Heads 可作为第一正向创新。

如果 E2 mIoU 小幅下降，但 severity MAE 明显改善:
  可写成 segmentation-severity trade-off，进入 E3 Severity Consistency Loss。

如果 E2 mIoU 和 severity 都下降:
  辅助头设计需要调整权重或目标，不进入 E3。
```

```text
提交: 6a1e330
标题: Add Ubuntu Python bootstrap script
目的: 解决 Ubuntu 服务器没有 python3 时无法启动环境安装的问题。
主要修改:
- scripts/bootstrap_ubuntu_miniconda.sh
- README.md
Windows / Linux 同步状态: Linux 侧新增兜底脚本，Windows 侧不需要对应改动。
```

## 2026-06-04 E1 训练完成与 E1.1 低强度 SCLP 计划

### E1 训练完成

```text
实验编号: E1
实验名称: deeplabv3plus_mobilenetv3_large_sclp_150
模型: DeepLabV3+ + MobileNetV3-Large
与 B0-V3 baseline 的区别: 只增加 SCLP 数据增强
SCLP 参数:
  sclp_prob = 0.7
  sclp_max_components = 3
训练轮数: 150
训练状态: 已完成
```

E1 结果：

```text
最高训练评估 mIoU: 68.97%，约 epoch 120
epoch150 mIoU: 68.47%
epoch150 mPA: 87.65%
epoch150 Accuracy: 97.76%
epoch150 Train Loss: 0.384
epoch150 Val Loss: 0.490
```

和最强 baseline 对比：

```text
B0-V3 baseline:
mIoU = 71.72%
FG mIoU = 66.58%
brown_spot IoU = 48.42%

E1 SCLP 0.7:
最高 mIoU = 68.97%

差距:
-2.75 mIoU
```

结论：

```text
SCLP_PROB=0.7 太强，没有超过 baseline。
当前 E1 不能作为正向创新结果。
E1 暂时只能说明: 过强的病斑 copy-paste 会干扰模型学习。
```

注意：

```text
best_val 报告中的 mIoU=65.56 是 best_epoch_weights.pth 的结果。
best_epoch_weights.pth 当前按 val loss 保存，不是按 mIoU 保存。
论文对比时应统一导出指定 checkpoint 或改成按 best mIoU 保存。
```

### 下一步 E1.1

E1.1 目的：

```text
降低 SCLP 强度，验证温和病斑增强是否有效。
如果 E1.1 能接近或超过 71.72%，说明 SCLP 仍有保留价值。
如果 E1.1 仍明显低于 baseline，则 SCLP 不适合作为主创新，只能降级为辅助增强或弃用。
```

E1.1 参数：

```text
实验编号: E1.1
实验名称: deeplabv3plus_mobilenetv3_large_sclp03_150
模型: DeepLabV3+ + MobileNetV3-Large
训练方式: 6 类 softmax
SCLP:
  sclp_prob = 0.3
  sclp_max_components = 2
训练轮数: 150
输出目录:
  D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_sclp03_150
```

本次代码同步修改：

```text
Windows:
- scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_sclp03_150.ps1

Linux:
- scripts/run_ubuntu_sclp03_v3.sh
- scripts/run_ubuntu.sh

说明文档:
- README.md

同步状态:
Windows / Linux 已同步。
```

E1.1 启动状态：

```text
启动时间: 2026-06-04 14:54
本机 PID: 8884
状态: 正在训练
当前输出目录:
  D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_sclp03_150
```

本次 GitHub 上传记录：

```text
提交: 364733e
标题: Add low-intensity SCLP experiment
目的: 新增 E1.1 低强度 SCLP 消融实验，验证 sclp_prob=0.3 是否比 E1 的 0.7 更稳。
主要修改:
- scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_sclp03_150.ps1
- scripts/run_ubuntu_sclp03_v3.sh
- scripts/run_ubuntu.sh
- README.md
- seg/ATLDSD项目进度.md
对应实验: E1.1
Windows / Linux 同步状态: 已同步。
```

## 2026-06-04 旧链路复核：B4 + CAA + PConv + LBFTLoss

用户提出的旧研究链路：

```text
DeepLabV3+ + EfficientNet-B4 + CAA + PConv + LBFTLoss
```

复核结论：

```text
这条链路有价值，但不适合直接作为下一步主线创新。
原因是它一次改变了四个变量:
1. backbone: MobileNetV3-Large -> EfficientNet-B4
2. attention: 无 -> CAA
3. decoder convolution: standard conv -> PConv
4. loss: CE + Dice -> LBFTLoss

如果直接跑完整链路，即使结果提升，也很难说明提升来自哪里。
审稿人会认为实验因果不清，只是模块堆叠。
```

已有结果对 B4 不利：

```text
B0-V3:
DeepLabV3+ + MobileNetV3-Large
mIoU = 71.72
FG mIoU = 66.58
Accuracy = 97.76
Params = 11.73M
FLOPs = 15.28G
FPS = 98.80

B0-B4:
DeepLabV3+ + EfficientNet-B4
mIoU = 65.59
FG mIoU = 59.59
Accuracy = 96.44
Params = 32.48M
FLOPs = 51.30G
FPS = 52.77

结论:
B4 在 ATLDSD 当前设置下更重、更慢、效果更差。
因此 B4 不应作为主论文默认 backbone。
```

当前 E2 训练状态：

```text
实验编号: E2
模型: DeepLabV3+ + MobileNetV3-Large + Component Auxiliary Heads
当前训练仍在运行
epoch50 mIoU = 68.83
epoch50 mPA = 85.47
epoch50 Accuracy = 96.75

判断:
epoch50 低于 B0-V3 的 71.72。
但 epoch50 后刚进入解冻阶段，val loss 短期上升属于可观察现象。
不应现在中断，应继续看到 epoch80/100 再判断。
```

### 修改后的整体训练计划

主论文创新线：

```text
M0: DeepLabV3+ + MobileNetV3-Large
作用: 最强普通 6 类 softmax baseline。
状态: 已完成，当前最强 baseline。

M1: M0 + Component Auxiliary Heads
作用: 显式学习 lesion mask、boundary、center，解决小病斑定位和边界完整性。
状态: 正在训练，即 E2。

M2: M1 + Severity Consistency Loss
作用: 让分割结果不仅追求像素 mIoU，还约束 lesion / leaf 的严重度估计。
目标: 回应“病害严重度判断”这个任务核心。

M3: M2 + Severity-aware Component-guided Attention
作用: 只在结构化病斑分支基础上加入注意力，不做泛泛的注意力堆叠。
目标: 让注意力服务于小病斑、边界、严重度，而不是作为普通模块。
```

旧链路拆分消融线：

```text
L0: B0-V3 + PConv
目的: 单独验证 PConv 是否改善病斑边界和小目标。

L1: B0-V3 + LBFTLoss
目的: 单独验证 LBFTLoss 是否改善类别不均衡和小病斑 IoU。

L2: B0-V3 + PConv + LBFTLoss
目的: 验证 decoder 改造与 loss 改造是否互补。

L3: B0-V3 + CAA + PConv + LBFTLoss
目的: 验证 CAA 在轻量 V3 backbone 上是否仍有收益。

L4: EfficientNet-B4 + CAA + PConv + LBFTLoss
目的: 作为“旧完整链路 / strong engineering comparator”，用于证明新主线不是简单复刻旧模块堆叠。
```

和 E2 的结合策略：

```text
如果 E2 best mIoU < 70.5 且 severity MAE 没有改善:
  先调整 component auxiliary 权重，不加 PConv / LBFTLoss。

如果 E2 best mIoU 在 70.5 到 71.72 之间，或 severity MAE 明显改善:
  跑 E2 + PConv
  跑 E2 + LBFTLoss
  再判断是否跑 E2 + PConv + LBFTLoss。

如果 E2 超过 B0-V3:
  先进入 M2: Severity Consistency Loss。
  PConv / LBFTLoss 只作为后续增强或附加消融。
```

论文写法调整：

```text
主创新不写成 CAA、PConv、LBFTLoss。
主创新写成:
Component-aware and severity-consistent lesion segmentation for ATLDSD。

CAA、PConv、LBFTLoss 的定位:
1. 对比旧链路
2. 工程增强
3. 可插拔消融

这样更容易通过审稿:
不是“堆模块”，而是先提出面向病害严重度的结构化建模，再证明传统模块是否能进一步增强。
```

下一步优先级：

```text
1. 不打断 E2，等到至少 epoch80/100。
2. E2 完成后导出 best_miou checkpoint 的 segmentation + severity report。
3. 若 E2 不达标，先调 auxiliary weights。
4. 若 E2 有保留价值，再跑 L0 / L1 这种单变量实验。
5. 最后才跑完整旧链路 L4。
```

## 2026-06-04 E2 完成与 M2 启动

### E2 最终结果

```text
实验编号: E2
实验名称: deeplabv3plus_mobilenetv3_large_component_aux_150
模型: DeepLabV3+ + MobileNetV3-Large + Component Auxiliary Heads
训练轮数: 150
状态: 已完成
best mIoU checkpoint: best_miou_weights.pth
最佳点: 约 epoch120
```

E2 best_miou 报告：

```text
mIoU: 72.11
FG mIoU: 67.03
Pixel Accuracy: 97.82
Severity MAE: 0.01212
Severity RMSE: 0.03583
Severity Pearson: 0.8691
Severity Spearman: 0.9141
Severity grade accuracy: 94.31%
Params: 11.73M
FLOPs: 15.29G
FPS: 101.10
```

相对 B0-V3 baseline：

```text
B0-V3 mIoU: 71.72
E2 mIoU: 72.11
提升: +0.39

B0-V3 severity MAE: 约 0.0124
E2 severity MAE: 0.01212
结论: 连续严重度误差略有改善。

B0-V3 severity grade accuracy: 约 95.12%
E2 severity grade accuracy: 94.31%
结论: 分级准确率略降，后续需要 severity consistency loss 约束。
```

### M2 决策

由于 E2 已经超过 B0-V3，按训练计划进入 M2：

```text
M2: DeepLabV3+ + MobileNetV3-Large + Component Auxiliary Heads + Severity Consistency Loss
```

M2 要解决的问题：

```text
E2 只让模型学习 lesion、boundary、center 辅助结构。
它提高了 mIoU 和连续严重度 MAE，但没有直接约束最终 lesion / leaf 严重度比例。

M2 新增 Severity Consistency Loss:
用 softmax 概率计算 predicted lesion / predicted leaf，
再和 GT lesion / GT leaf 做 L1 约束。

目标:
让模型不仅像素分割更准，也让病害严重度估计更稳定。
```

M2 参数：

```text
实验编号: M2
实验名称: deeplabv3plus_mobilenetv3_large_component_aux_severity_150
模型: DeepLabV3+ + MobileNetV3-Large
训练方式: 6 类 softmax + Dice + Component Auxiliary Loss + Severity Consistency Loss
component_aux: true
component_lesion_weight: 0.4
component_boundary_weight: 0.2
component_center_weight: 0.2
severity_consistency_loss: true
severity_consistency_weight: 0.1
severity_loss_type: l1
训练轮数: 150
输出目录:
  D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_severity_150
```

M2 启动状态：

```text
启动时间: 2026-06-04 22:36
本机 PID: 28052
状态: 正在训练
日志:
  D:\Code\ATLDSD\outputs\atldsd\deeplabv3plus_mobilenetv3_large_component_aux_severity_150\train_stdout.log
```

本次代码同步修改：

```text
Windows:
- scripts/run_atldsd_deeplabv3plus_mobilenetv3_large_component_aux_severity_150.ps1

Linux:
- scripts/run_ubuntu_component_aux_severity_v3.sh
- scripts/run_ubuntu.sh

训练代码:
- src/models/deeplabv3plus/nets/deeplabv3_training.py
- src/models/deeplabv3plus/utils/utils_fit.py
- src/models/deeplabv3plus/train.py

说明文档:
- README.md

同步状态:
Windows / Linux 已同步。
```

M2 判断标准：

```text
如果 M2 mIoU >= 72.11 且 severity MAE < 0.01212:
  M2 可作为主线正向创新。

如果 M2 mIoU 小幅低于 72.11，但 severity MAE 和 grade accuracy 明显改善:
  可以写成 segmentation-severity trade-off，并保留 M2。

如果 M2 mIoU 和 severity 都低于 E2:
  severity weight=0.1 过强或设计不合适，下一步改为 0.05 或 smooth_l1。
```

## 2026-06-04 训练路线纠偏：主线必须回到结构模块

### 问题纠正

上一版计划把 M2 放得太靠前，逻辑不够硬：

```text
E2 = Component Auxiliary Heads
这是结构模块。

M2 = Severity Consistency Loss
这是 loss 约束，不是结构模块。
```

如果论文目标是做语义分割模型创新，主线不能一直围绕 loss 和指标约束转。  
M2 可以保留，但只能作为“严重度一致性辅助消融”，不能作为主结构创新。

### 新主线定位

当前论文主线重新定义为：

```text
Component-aware modular segmentation for ATLDSD disease severity assessment
```

核心不是“加一个 loss”，而是：

```text
在 DeepLabV3+ + MobileNetV3-Large 的强 baseline 上，
围绕小病斑、边界、组件结构和注意力响应，
逐步加入真实结构模块。
```

### 已完成实验重新归位

```text
B0-V3:
DeepLabV3+ + MobileNetV3-Large
作用: 最强普通 baseline
结果: mIoU 71.72

E2:
B0-V3 + Component Auxiliary Heads
作用: 第一个结构模块实验
模块: lesion head + boundary head + center head
结果: mIoU 72.11
结论: 有小幅正收益，保留为结构主线起点。

M2:
E2 + Severity Consistency Loss
作用: 严重度 loss 消融
模块属性: 不是结构模块
当前状态: 正在训练，PID 28052
结论定位: 降级为辅助实验，不作为主线创新。
```

### 重新规划后的训练顺序

第一阶段：确认结构起点

```text
S0: B0-V3
DeepLabV3+ + MobileNetV3-Large
目的: 普通 6 类 softmax baseline。
状态: 已完成。

S1: E2
B0-V3 + Component Auxiliary Heads
目的: 证明 lesion / boundary / center 辅助头是否有结构收益。
状态: 已完成，mIoU 72.11。
```

第二阶段：真正动模块

```text
S2: E2 + PConv
目的:
把 decoder 中的 standard conv 换为 PConv。
验证 PConv 是否提升小病斑边缘、局部纹理和不规则病斑区域。

为什么先跑它:
PConv 是真实结构改动。
它比 loss 更适合作为“模型模块创新”的第一步。
```

```text
S3: E2 + CAA
目的:
在 E2 结构基础上加入 CAA 注意力。
验证注意力是否能增强病斑区域、边界区域和小组件响应。

注意:
CAA 不单独作为主创新，因为普通注意力太常见。
它应当写成 component-aware framework 的增强模块。
```

```text
S4: E2 + PConv + CAA
目的:
验证 PConv 的局部卷积增强和 CAA 的注意力增强是否互补。

如果 S2 和 S3 都有效:
S4 是主模型候选。

如果只有 S2 有效:
主模型偏向 Component Auxiliary Heads + PConv。

如果只有 S3 有效:
主模型偏向 Component Auxiliary Heads + CAA。
```

第三阶段：loss 只做辅助，不抢主线

```text
A1: E2 + Severity Consistency Loss
当前正在跑，即 M2。
目的: 验证严重度 MAE / grade accuracy 是否改善。
定位: 辅助消融。

A2: S4 + Severity Consistency Loss
只有当 S4 成为主模型候选后再跑。
目的: 验证结构模块 + 严重度约束是否进一步改善 severity 指标。
```

第四阶段：旧链路对照

```text
L0: B0-V3 + PConv
目的: 单独验证 PConv 对普通 baseline 的贡献。

L1: B0-V3 + CAA
目的: 单独验证 CAA 对普通 baseline 的贡献。

L2: B0-V3 + PConv + CAA
目的: 证明 E2 的 component heads 是否仍然必要。

L3: B0-V3 + LBFTLoss
目的: loss 对照，不作为模块主线。

L4: EfficientNet-B4 + CAA + PConv + LBFTLoss
目的: 旧完整链路强对照。
定位: strong engineering comparator，不是主线。
```

### 下一步应该跑什么

下一步首选：

```text
S2: E2 + PConv
```

原因：

```text
1. 它是真正动结构模块。
2. 它直接来自旧链路中有价值的部分。
3. 它和当前 E2 的 component heads 关系清楚，只多一个 decoder 模块变量。
4. 如果 S2 有提升，论文可以写成:
   component auxiliary supervision + partial convolution decoder improves lesion boundary localization。
```

M2 当前处理：

```text
M2 已经启动，不擅自中断。
但从论文计划上，M2 不再是主线下一步，而是 loss 辅助消融。
如果后续 GPU 资源冲突，应优先保留 S2 / S3 / S4 结构模块实验。
```

### 新判断标准

```text
S2 相对 E2:
如果 mIoU > 72.11，且 FG mIoU 或 brown_spot / gray_spot IoU 提升:
  PConv 保留。

如果 mIoU 接近 72.11，但边界类、小病斑类提升:
  PConv 可作为边界增强模块保留。

如果 mIoU 和小病斑类都下降:
  PConv 不进入主模型，只保留为失败消融。

S3 相对 E2:
如果 CAA 提升小病斑类 IoU 或 FG mIoU:
  CAA 保留为注意力增强模块。

如果 CAA 只增加复杂度、不提升指标:
  不把 CAA 写进主模型。

S4 相对 S2 / S3:
只有同时优于单模块结果，才作为最终主模型候选。
```
