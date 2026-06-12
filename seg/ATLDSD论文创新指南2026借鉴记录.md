# ATLDSD 论文创新指南 2026 借鉴记录

更新时间：2026-06-08

参考文件：

```text
C:\Users\Administrator\Desktop\论文创新指南2026：手把手带你发论文.pdf
```

## 1. PDF 中实际借鉴的点

这份 PDF 的核心不是让我们继续无序堆模块，而是把“模块创新”压缩成可解释的结构模板。结合 ATLDSD 已经失败的路线，真正可借鉴的是下面 6 点。

### 借鉴点 1：Token Mixer + FFN 范式

PDF 原则：

```text
xx-Former = Token Mixer + FFN。
主创新可以改 Token Mixer，次创新可以改 FFN。
不要换一个大网络就说创新，要说明模块解决哪个任务瓶颈。
```

落到 ATLDSD：

```text
不再把 DeepLabV3+ 全部推翻。
保留 Boundary1 的 CNN 主线：
DeepLabV3+ MobileNetV3-Large + Component Auxiliary Heads + LBSB。

新模块只插在 ASPP 后或 decoder feature 上：
Token Mixer 负责低分辨率全局建模。
FFN / refinement 负责高频、小病斑、边界。
```

### 借鉴点 2：A+B 双分支，不是大杂烩

PDF 反复使用：

```text
A+B 双分支：
CNN + Transformer
CNN + Mamba
FFT + Mamba
CA/SA + Mamba

再用融合模块 C 做自适应融合。
```

落到 ATLDSD：

```text
CNN 分支：保留 Boundary1 的局部纹理、边界锐化能力。
Mamba 分支：只在 1/16 低分辨率特征上建模叶片级全局上下文。
频域/高频分支：专门补小病斑边缘和纹理。
融合模块：用 lesion / boundary / center 辅助头生成 gate，避免无条件 concat。
```

### 借鉴点 3：低分辨率全局建模

PDF 的 LRFormer 案例强调：

```text
先降到低分辨率做全局建模，再上采样恢复。
```

落到 ATLDSD：

```text
不要在 256x256 或 384x384 原图级别跑 Mamba。
Mamba 只接 ASPP 后的 1/16 特征：
256 输入时约 16x16。
384 输入时约 24x24。

这样既符合分割全局上下文需求，也避免完整双主干过重。
```

### 借鉴点 4：高频/频域是小目标更直接的突破口

PDF 的 HS-FPN、FMambaIR、Frequency-Enhanced Lightweight Vision Mamba 等案例都强调：

```text
小目标 / 病灶 / 边界问题，不只靠上下文。
高频、频域、边界、空间注意力更直接。
```

落到 ATLDSD：

```text
LGLC 已经证明“只补上下文”不够。
下一轮不能只做 Mamba 全局分支。
必须同时引入高频/频域分支，并且让它被 component cues 约束。
```

### 借鉴点 5：自适应融合，而不是简单相加

PDF 公式：

```text
alpha * A + (1 - alpha) * B
```

落到 ATLDSD：

```text
fusion = x + gamma * gate * branch_feature

gate 来源：
1. lesion_aux probability
2. boundary_aux probability
3. center_aux probability
4. optional softmax entropy

这样能解释为 component-guided adaptive fusion。
```

### 借鉴点 6：代码借鉴要变成任务专属模块

PDF 说可以从 GitHub 找模块，但不能照搬成拼装感。

落到 ATLDSD：

```text
借代码骨架，不搬整网。
不换完整 VMamba / SegMAN / FreqConvMamba backbone。
只抽取低分辨率 SS2D、FrequencyBlock、低分辨率注意力/融合思想，改造成 ATLDSD 的 component-guided lesion refinement。
```

## 2. GitHub 代码参考

已经浅克隆到本地参考目录：

```text
D:\Code\ATLDSD\outputs\github_refs\VMamba
D:\Code\ATLDSD\outputs\github_refs\SegMAN
D:\Code\ATLDSD\outputs\github_refs\FreqConvMamba
```

这些目录只作为参考，不进入项目源码。

### 参考 1：VMamba

GitHub：

```text
https://github.com/MzeroMiko/VMamba
```

可借鉴代码：

```text
D:\Code\ATLDSD\outputs\github_refs\VMamba\classification\models\vmamba.py
class SS2D
class VSSBlock

D:\Code\ATLDSD\outputs\github_refs\VMamba\segmentation\configs\vssm
UPerNet + VSSM segmentation configs
```

借鉴方式：

```text
只借 SS2D / VSSBlock 的低分辨率 2D selective scan 思想。
不要整套 VSSM backbone。
如果 selective_scan CUDA 依赖安装麻烦，先做 torch fallback 或低分辨率简化实现。
```

### 参考 2：FreqConvMamba

GitHub：

```text
https://github.com/ccode-Rookie/FreqConvMamba
```

可借鉴代码：

```text
D:\Code\ATLDSD\outputs\github_refs\FreqConvMamba\FreqConvMamba\FrequencyBlock.py
class FrequencyBlock

D:\Code\ATLDSD\outputs\github_refs\FreqConvMamba\FreqConvMamba\models\FreqConvMamba\vmamba.py
class SS2D
FGMM-style channel split: one side SS2D, one side FrequencyBlock
```

借鉴方式：

```text
借 FrequencyBlock 的 rfft2 -> real/imag concat -> depthwise frequency enhancement -> irfft2 流程。
在 ATLDSD 中改成 lightweight high-frequency branch，接 decoder feature 或 ASPP feature。
```

### 参考 3：SegMAN

GitHub：

```text
https://github.com/yunxiangfu2001/SegMAN
```

可借鉴代码：

```text
D:\Code\ATLDSD\outputs\github_refs\SegMAN\segmentation\mmseg\models\tmp.py
LGQuery
Attention4DDownsample
```

借鉴方式：

```text
借低分辨率 query / local + pooled query 的思想。
不引入整套 mmseg SegMAN。
```

### 参考 4：LRFormer / CAF-YOLO

GitHub：

```text
https://github.com/yuhuan-wu/LRFormer
https://github.com/xiaochen925/CAF-YOLO
```

借鉴方式：

```text
LRFormer：低分辨率全局建模范式。
CAF-YOLO：局部/全局双分支 + 多尺度卷积 + gate 的病灶检测思路。
```

## 3. 升级后的后续计划

### 总原则

```text
不要再“一个模块接一个模块”试。
每个实验必须回答一个具体失败问题。

失败问题 A:
256x256 是否压没小病斑？

失败问题 B:
只补上下文为什么没用？

失败问题 C:
辅助头为什么只带来 0.39 mIoU？

失败问题 D:
CNN+Mamba 是否能补全局，但不破坏 Boundary1 的边界优势？
```

### 新优先级

#### Priority 1：Boundary1-384

```text
结构:
Boundary1 原结构不变，只把 input_shape 从 256x256 提到 384x384。

借鉴点:
PDF 的低分辨率全局建模和高频小目标思路都依赖一个前提：
输入里要先保住小病斑信息。

结论规则:
如果 384 直接超过 72.86，说明主要瓶颈是分辨率。
如果 384 不涨，再做结构模块。
```

#### Priority 2：CHFR-256

全名：

```text
Component-guided High-Frequency Refinement
```

结构：

```text
Boundary1 + CHFR
```

模块位置：

```text
LBSB 后、cls_conv 前。
```

设计：

```text
high = feature - avg_pool(feature)
freq = lightweight FFT / depthwise frequency branch
component_gate = sigmoid(conv([lesion_prob, boundary_prob, center_prob]))
uncertainty_gate = normalized_entropy(softmax(seg_logits_pre))
refined = feature + alpha * component_gate * (high + freq)
```

借鉴点：

```text
PDF: HS-FPN 高频小目标、FMambaIR/FreqConvMamba 频域增强、自适应融合。
GitHub: FreqConvMamba FrequencyBlock。
```

为什么它排在 CNN+Mamba 前面：

```text
LGLC 已经证明只补上下文不够。
小病斑失败更像高频/边界/纹理问题。
CHFR 比 Mamba 更轻、更贴合 Boundary1 的成功原因。
```

成功标准：

```text
强成功: mIoU >= 73.20 且 FG mIoU >= 68.10。
弱成功: mIoU > 72.86，且 alternaria / gray_spot / brown_spot 至少两类提升。
保留为严重度模块: mIoU >= 72.60 且 severity MAE < 0.01169。
```

#### Priority 3：CMF-256

全名：

```text
Component-guided CNN-Mamba-Frequency Fusion
```

结构：

```text
Boundary1 + lightweight CMF
```

模块位置：

```text
ASPP 后，decoder concat 前。
```

设计：

```text
cnn_local = depthwise 3x3 + strip conv
mamba_global = SS2D/VSSBlock on low-resolution ASPP feature
freq_global = FrequencyBlock on low-resolution ASPP feature
component_gate = sigmoid(conv(component cues downsampled to ASPP size))
out = x + gamma * component_gate * conv1x1(concat(cnn_local, mamba_global, freq_global))
```

借鉴点：

```text
PDF: CNN+Mamba 双分支、FFT+Mamba 双分支、A+B+C 融合、自适应 alpha。
GitHub: VMamba SS2D/VSSBlock, FreqConvMamba FrequencyBlock, SegMAN low-resolution attention.
```

硬限制：

```text
禁止完整 CNN backbone + 完整 Mamba backbone 双塔。
禁止替换 MobileNetV3 主干。
Mamba 只允许在 1/16 特征上跑 1-2 层。
如果 Params 或 FLOPs 明显爆炸，直接停止。
```

成功标准：

```text
如果 CMF-256 <= 72.86:
  不继续 CMF-384，不调参。

如果 CMF-256 > 72.86:
  再做 CMF-384 或 seed repeat。

如果 CMF 只提高 rust，但 gray/brown/alternaria 不涨:
  作为负消融，不进最终模型。
```

#### Priority 4：CFR 降级为融合门控

原计划把 CFR 当唯一结构创新。现在降级：

```text
CFR 不再单独优先跑。
CFR 的 component feedback 思想保留，但优先嵌入 CHFR / CMF 的 gate。
如果 CHFR 或 CMF 接近成功，再补 CFR-only 作为消融。
```

原因：

```text
PDF 提醒模块要 A+B+C 讲完整机制。
单独 CFR 只是 feature gate，可能解释力和涨点都不够。
```

## 4. 论文叙事升级

旧叙事：

```text
Component heads + LBSB + CFR
```

新叙事：

```text
Structured component supervision first learns lesion/leaf/boundary cues.
LBSB preserves irregular lesion boundaries.
CHFR restores high-frequency lesion details under component guidance.
CMF optionally adds low-resolution Mamba global context without replacing the lightweight CNN backbone.
```

中文：

```text
结构化组件监督学习病斑组成；
边界锐化保护不规则病斑边缘；
组件引导高频细化恢复小病斑纹理；
低分辨率 CNN-Mamba-频域融合补充叶片级上下文。
```

## 5. 立即执行顺序

```text
1. Boundary1-384
2. Boundary1-repeat
3. CHFR-256
4. 只有 CHFR 有接近成功迹象，才做 CHFR-384
5. CMF-256
6. 只有 CMF-256 超过 Boundary1，才做 CMF-384 或 seed repeat
```

停止方向：

```text
PConv
LCAF 调参
LGLC 调参
普通 CAA
SCLP
B4 backbone
完整双主干 CNN+Mamba
```

## 6. 2026-06-09 PlugNPlay 快筛反馈

记录文件：

```text
D:\Code\ATLDSD\seg\ATLDSD快速模块筛选记录_2026-06-09.md
```

### 新证据

按 PDF 的“Token Mixer + FFN”“A+B 局部/全局分支”“高频/小目标”“自适应融合”模板，先用 64 train / 32 val 的 ATLDSD_FAST 快筛协议测试 PlugNPlay 模块，而不是直接跑 150 epoch。

```text
快筛协议:
train = 64
val = 32
epoch = 12
optimizer = adam
init_lr = 0.0005
class weights = 1.0 1.0 2.0 3.0 3.0 4.0
seed = 11 / 23
```

### 模块判定

```text
LSK-ASPP:
  两个 seed 均超过 baseline。
  avg mIoU: 36.50 -> 37.83, +1.33
  avg FG mIoU: 25.52 -> 26.82, +1.30
  借鉴点: 大核选择性空间上下文，符合 PDF 的局部/多尺度 Token Mixer 和小目标纹理增强方向。
  判定: 升级为下一轮优先候选。

CPCA-ASPP:
  seed11 提升，seed23 反转。
  avg mIoU: 36.50 -> 36.97, +0.47
  借鉴点: channel prior + 多尺度局部分支。
  判定: 弱候选，不直接进入正式长训。

EMA-ASPP / SCSA-ASPP:
  mIoU 和 FG mIoU 均不优于 baseline。
  判定: 停止。
```

### 对原计划的修正

```text
CHFR / CMF 仍然是论文机制候选，但立即训练优先级后移。
下一步先验证 LSK-ASPP 的中程稳定性:
  64/32, seed11/23, 24 或 32 epoch。

如果 LSK 中程仍稳定:
  扩到 128/64 或 192/64 快筛。

如果扩大数据仍稳定:
  再升级到正式 150 epoch。

如果 LSK 只提升 rust 而不提升 alternaria/gray/brown:
  只作为 large selective kernel 负消融保留，不进最终模型。
```
