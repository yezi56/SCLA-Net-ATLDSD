# ATLDSD 语义分割做 SCI 二区的路线图

创建日期：2026-06-01  
数据集：`D:\dataset\ATLDSD\VOCdevkit\VOC2012`

## 一句话定位

不要把论文写成“我换了一个分割网络”。要写成：面向作物叶片病害严重度评估，构建一个能同时分割叶片、病斑和病害类型的轻量化语义分割框架，并用病斑面积占叶片面积的比例完成严重度量化。

这篇文章的核心卖点不是单纯 mIoU 高，而是：分割结果能服务严重度判断。这样比只做病害分类更像农业视觉论文，也比单纯病斑二分类更容易讲完整故事。

## 当前数据基础

已经整理成 VOC 格式：

- `JPEGImages`：1641 张 RGB 图像
- `SegmentationClass`：1641 张单通道 mask
- `train / val / test`：1148 / 246 / 247
- `classes.txt`：类别映射
- `severity.csv`：每张图的病斑比例和严重度标签

类别定义：

```text
0 background
1 leaf
2 rust
3 alternaria_leaf_spot
4 gray_spot
5 brown_spot
```

严重度由 `lesion_ratio = lesion_pixels / leaf_pixels` 得到：

```text
0 healthy:      ratio = 0
1 slight:       0 < ratio <= 0.05
2 moderate:     0.05 < ratio <= 0.15
3 severe:       0.15 < ratio <= 0.30
4 very_severe:  ratio > 0.30
```

## 论文故事线

题目不要写得太虚。推荐方向：

**A severity-aware lightweight semantic segmentation network for tomato leaf disease assessment**

中文意思：一个面向番茄叶片病害严重度评估的轻量级语义分割网络。

逻辑链：

1. 农业病害严重度不是只看有没有病，而是看病斑面积占叶片面积多少。
2. 分类模型只能说是哪种病，不能量化严重度。
3. 普通分割模型能分病斑，但对小病斑、边界、类别不均衡不够稳。
4. 所以提出一个“病斑感知 + 多尺度 + 边界增强 + 严重度辅助”的分割框架。
5. 最后用预测 mask 计算严重度，证明模型不仅分得准，而且能服务农业决策。

## 模型怎么缝合才像论文

主模型可以叫：`DSA-LeafNet`，Disease-Severity-Aware Leaf Segmentation Network。

建议结构：

1. 主干网络：选择 `DeepLabV3+` 或 `SegFormer-B0/B1` 作为 baseline。
2. 多尺度上下文模块：保留 ASPP，或加入 PPM / strip pooling，解决病斑尺度变化。
3. 病斑注意力模块：加入 Coordinate Attention、CBAM、ECA、EMA 之一，不要全都堆。推荐 Coordinate Attention 或 EMA，因为容易解释为关注细长叶片和局部病斑。
4. 边界辅助分支：从 ground truth mask 生成边界监督，让模型更重视叶片和病斑边缘。
5. 严重度辅助头：从分割特征或预测 mask 中回归 `lesion_ratio`，或者分类为 0-4 级严重度。

最稳的缝合法：

```text
Backbone + ASPP/Decoder + Attention Refinement + Boundary Auxiliary Loss + Severity Auxiliary Head
```

不要一口气塞 5 个注意力模块。论文里最怕像乱炖。每个模块必须对应一个真实问题：

```text
小病斑漏检 -> 多尺度上下文
叶片形态细长 -> 坐标注意力
病斑边界模糊 -> 边界辅助监督
类别严重不均衡 -> Dice/Focal/Tversky loss
严重度不是分类能解决 -> lesion ratio 量化
```

## 实验步骤

### Step 1：跑通基础分割

先用整理好的 VOC 数据跑 baseline：

- U-Net
- DeepLabV3+
- PSPNet
- SegFormer-B0
- SegFormer-B1

必须记录：

- mIoU
- mDice
- Pixel Accuracy
- mPA
- 每类 IoU
- 每类 Dice
- Params
- FLOPs
- FPS

第一篇 SCI 二区的底盘不是模型多花，而是实验表要完整。

### Step 2：确定主模型

如果 DeepLabV3+ 表现稳定，就以 DeepLabV3+ 为主体改。  
如果 SegFormer-B0/B1 更强，就以 SegFormer 为主体改。

建议优先顺序：

```text
DeepLabV3+ 跑通最稳
SegFormer 做强 baseline
最后提出轻量增强版模型
```

### Step 3：加一个核心模块

先只加一个注意力或多尺度模块，观察是否提升小病斑类别：

- Rust：class 2
- Alternaria leaf spot：class 3
- Gray spot：class 4
- Brown spot：class 5

重点看每类 IoU，不要只看总 mIoU。因为这个数据集病斑像素很少，总指标容易被背景和叶片掩盖。

### Step 4：加边界分支

用 mask 生成边缘标签，做 auxiliary supervision。

论文解释：病斑区域通常小、边缘不规则，普通 decoder 容易边界粗糙，所以增加 boundary-aware branch。

### Step 5：加严重度辅助任务

两种做法：

1. 回归 `lesion_ratio`，指标用 MAE、RMSE、R2。
2. 分类严重度等级，指标用 Accuracy、Macro-F1、Kappa。

更推荐同时报告：

```text
分割：mIoU / mDice / per-class IoU
严重度：MAE of lesion ratio / severity accuracy / macro-F1
```

这样文章就不是纯分割，而是“分割驱动的病害严重度评估”。

## 消融实验必须这样设计

表格结构：

```text
Baseline
Baseline + Attention
Baseline + Boundary Loss
Baseline + Severity Head
Baseline + Attention + Boundary Loss
Full Model
```

每一行至少放：

```text
mIoU, mDice, lesion IoU, Params, FLOPs, FPS, Severity MAE
```

重点不是每次都涨很多，而是能解释：

- Attention 提升小病斑召回
- Boundary Loss 改善边缘质量
- Severity Head 让模型更贴近严重度量化目标
- Full Model 在精度和速度之间更平衡

## 对比实验要有梯队

传统 CNN：

- U-Net
- Attention U-Net
- DeepLabV3+
- PSPNet

Transformer / 混合模型：

- SegFormer-B0/B1
- Swin-Unet 或 UNetFormer

轻量模型：

- BiSeNetV2
- Fast-SCNN
- MobileNetV3-DeepLabV3+

你的模型要在两类结果里找位置：

1. 比轻量模型准。
2. 比大模型轻。
3. 或者在严重度误差上最好。

SCI 二区更喜欢这种平衡叙事，而不是只说我 mIoU 第一。

## 图表清单

论文至少准备这些图：

1. 数据集样例图：原图、mask、病斑比例、严重度等级。
2. 类别像素分布图：说明类别不均衡。
3. 模型结构图：主干、注意力、多尺度、边界分支、严重度头。
4. 训练曲线：loss、mIoU、mDice。
5. 对比实验柱状图：mIoU、mDice、Params、FLOPs、FPS。
6. 可视化对比：GT、DeepLabV3+、SegFormer、你的模型。
7. 错误图：漏检、误检、边界粗糙、小病斑失败案例。
8. 严重度散点图：真实 lesion ratio vs 预测 lesion ratio。

## 文章贡献点写法

贡献点不要写“提出了一个新网络”这么空。推荐写成：

1. 构建了一个面向番茄叶片病害严重度评估的语义分割流程，将像素级病斑分割与严重度量化连接起来。
2. 提出一个轻量级病斑感知分割网络，通过多尺度上下文、注意力细化和边界监督提升小病斑与不规则边界的分割质量。
3. 设计严重度辅助学习分支，使模型在优化像素分割的同时关注病斑面积比例，降低严重度估计误差。
4. 在 ATLDSD 数据集上与 CNN、Transformer 和轻量化模型进行系统对比，并报告精度、参数量、FLOPs、FPS 和严重度指标。

## 最小可发论文版本

如果只追求先发出去，最低配置是：

```text
DeepLabV3+ baseline
+ Coordinate Attention
+ Boundary Loss
+ Dice + Focal loss
+ lesion ratio 严重度计算
+ 完整对比实验
+ 完整消融实验
+ 可视化和失败案例
```

这个版本实现难度可控，论文故事也完整。

## 更强版本

如果想冲更好一点：

```text
SegFormer-B0/B1 backbone
+ 多尺度特征融合 decoder
+ EMA 或 Coordinate Attention
+ Boundary-aware auxiliary supervision
+ Severity-aware multi-task learning
+ 轻量化分析 Params/FLOPs/FPS
```

这个版本更像近年的视觉论文，但实现和调参成本更高。

## 实验时间表

第 1 周：数据和 baseline

- 检查 VOC 数据加载
- 跑 U-Net、DeepLabV3+、SegFormer-B0
- 输出 mIoU、Dice、每类指标

第 2 周：模型改进

- 加 attention
- 加 boundary loss
- 调 loss 组合
- 保存所有消融结果

第 3 周：严重度任务

- 从预测 mask 计算 lesion ratio
- 做严重度等级分类
- 输出 MAE、RMSE、Macro-F1、Kappa

第 4 周：补实验和画图

- Params / FLOPs / FPS
- 可视化对比
- 错误案例分析
- 严重度散点图

第 5 周：写论文

- Introduction：分类不能解决严重度，分割能量化病斑面积
- Methods：模型结构和 loss
- Experiments：数据集、指标、对比、消融
- Discussion：失败案例、部署价值、局限性

## 投稿定位

不要一开始就盯死某个期刊。先把文章做成“农业视觉 + 轻量分割 + 严重度评估”的形态。投稿前再按当年 JCR 和中科院分区核验。

可考虑方向：

- 农业信息化 / 智慧农业
- 图像处理应用
- 计算机视觉应用
- 植物病害检测与表型分析

文章质量判断标准：

```text
只有分割指标：普通实验报告
分割 + 轻量化：像工程论文
分割 + 严重度估计：像农业应用论文
分割 + 严重度 + 消融 + 速度 + 失败案例：更接近 SCI 二区完整稿
```

## 现在立刻该做的事

1. 用当前 VOC 数据先跑 DeepLabV3+，建立第一条可靠 baseline。
2. 训练结束后自动输出每类 IoU、Dice、mIoU、mDice、Params、FLOPs、FPS。
3. 用 `severity.csv` 验证 ground truth 的严重度分布。
4. 写一个脚本从预测 mask 计算 lesion ratio 和 severity grade。
5. 再决定是走 DeepLabV3+ 改进线，还是 SegFormer 改进线。

核心原则：先让 baseline 和评估体系闭环，再缝模块。没有完整指标表，缝再多模块也不像论文。
