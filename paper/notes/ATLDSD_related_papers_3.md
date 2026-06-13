# ATLDSD 相近方向论文 3 篇

更新时间：2026-06-02

研究定位：ATLDSD 的核心路线应写成“番茄叶片多类别病害语义分割 + 基于病斑/叶片面积比例的严重度评估”。下面 3 篇是最值得放进 related work 和实验设计参考的论文。

## 1. Tomato TransDeepLab

**Citation**

Gangwar, A., Rani, G., & Dhaka, V. S. (2025). *Tomato TransDeepLab: A Robust Framework for Tomato Leaf Segmentation, Disease Severity Prediction, and Crop Loss Estimation*. IEEE Access, 13, 170147-170160. https://doi.org/10.1109/access.2025.3611307

**为什么和 ATLDSD 最像**

这篇最接近我们现在要做的方向：番茄叶片、病害分割、严重度预测、作物损失估计。它不是只做分类，而是把病斑区域分割出来，再用病斑面积比例推严重度。

**可借鉴点**

- 论文主线可以写成：segmentation first, severity second。
- 对比模型可以设置为 U-Net、ResUNet、DeepLabV3、DeepLabV3+、Transformer/TransDeepLab 类模型。
- 严重度部分不要只报 mIoU，要补充 lesion ratio 与 severity level 的映射。
- 我们的 ATLDSD 多病害类别比它更适合讲“多类别病害区域分割 + 严重度评估”。

**可对标指标**

- IoU
- Pixel accuracy
- 疾病严重度等级
- crop loss / severity scale，若没有真实产损标签，可先只做 severity grade。

**对我们论文的启发**

这篇可以作为番茄病害严重度方向的直接竞品。我们的差异点可以写成：ATLDSD 不是只做二分类病斑，而是同时分割 leaf、rust、alternaria leaf spot、gray spot、brown spot，再根据各病害像素占叶片区域比例做严重度估计。

## 2. DS-DETR

**Citation**

Wu, J., Wen, C., Chen, H., Ma, Z., Zhang, T., Su, H., & Yang, C. (2022). *DS-DETR: A Model for Tomato Leaf Disease Segmentation and Damage Evaluation*. Agronomy, 12(9), 2023. https://doi.org/10.3390/agronomy12092023

**为什么值得读**

这篇同样是番茄叶片病害分割和 damage evaluation。它构建了 Tomato leaf Disease Segmentation Dataset，标签包含 leaf、early blight spot、late blight spot，并用病斑面积比例做病害损伤评价。

**可借鉴点**

- 任务表述：disease spot segmentation and damage evaluation。
- 标签体系：background / leaf / disease spot，与 ATLDSD 的 background / leaf / multiple disease classes 高度相似。
- 方法侧用了 DETR、SMCA、relative position encoding，适合给我们后续“加入注意力/位置关系模块”提供文献支撑。
- 它报告 AP_box、AP_mask、参数量、推理时间和病害等级准确率。我们可以对应报告 mIoU、mDice、Params、FLOPs、FPS、severity accuracy 或 severity consistency。

**对我们论文的启发**

不要只讲“我换了一个 backbone”。要把问题包装成：番茄叶片病害区域边界模糊、病斑小且分散、不同病害纹理相似，导致多类别病害分割困难。后续模块要围绕小病斑、边界、类别不均衡、全局上下文展开。

## 3. Deep learning architectures for semantic segmentation and automatic estimation of severity of foliar symptoms

**Citation**

Gonçalves, J. P., Pinto, F. A. C., Queiroz, D. M., Villar, F. M. M., Barbedo, J. G. A., & Del Ponte, E. M. (2021). *Deep learning architectures for semantic segmentation and automatic estimation of severity of foliar symptoms caused by diseases or pests*. Biosystems Engineering, 210, 129-142. https://doi.org/10.1016/j.biosystemseng.2021.08.011

**为什么值得放 related work**

这篇不是番茄，但期刊和问题定义都很正：用 CNN 语义分割叶片病虫害症状，并自动估计病害严重度。它比较了 6 种语义分割架构，包含 FPN、U-Net、DeepLabv3+ 等，核心结论是分割结果可以稳定估计 percent severity。

**可借鉴点**

- 论文主线非常适合我们：semantic segmentation -> percent severity。
- 它把图像像素分成 background、healthy leaf、injured leaf；我们可以扩展为 background、leaf、4 类病害。
- DeepLabv3+ 是它表现较好的架构之一，可支撑我们以 DeepLabV3+ 为 baseline 的合理性。
- 它强调复杂光照和复杂背景下传统阈值方法不稳，这可以作为我们引言里的痛点。

**对我们论文的启发**

我们可以把 ATLDSD 的严重度定义写成：

```text
severity = disease_pixels / leaf_pixels
```

其中：

```text
disease_pixels = rust + alternaria_leaf_spot + gray_spot + brown_spot
leaf_pixels = leaf + all disease pixels
```

再进一步按病害类别分别计算：

```text
rust_severity
alternaria_leaf_spot_severity
gray_spot_severity
brown_spot_severity
overall_severity
```

## 推荐写法

related work 不要堆太多分类论文。建议按三段写：

1. **Plant disease severity estimation**：说明传统阈值、人工评分和纯分类的不足。
2. **Semantic segmentation for lesion quantification**：重点引用 Gonçalves 2021、DS-DETR 2022、Tomato TransDeepLab 2025。
3. **Efficient and attention-enhanced segmentation networks**：再引出 DeepLabV3+、SegNeXt、attention、boundary-aware loss、class imbalance loss。

## 对 ATLDSD 的实验设计建议

当前最稳的实验矩阵：

```text
Baseline:
1. DeepLabV3+ + MobileNetV2
2. DeepLabV3+ + EfficientNet-B4

Comparison:
3. U-Net
4. PSPNet
5. SegNeXt

Ablation:
6. DeepLabV3+ + EfficientNet-B4 + Dice/Focal/Tversky
7. DeepLabV3+ + EfficientNet-B4 + ECA/CA/CBAM
8. DeepLabV3+ + EfficientNet-B4 + boundary-aware loss

Severity:
9. lesion ratio
10. per-disease severity ratio
11. severity grade consistency
```

## 参考链接

- Tomato TransDeepLab: https://doi.org/10.1109/access.2025.3611307
- DS-DETR: https://doi.org/10.3390/agronomy12092023
- Biosystems Engineering severity segmentation: https://doi.org/10.1016/j.biosystemseng.2021.08.011
