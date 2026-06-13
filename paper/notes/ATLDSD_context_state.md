# ATLDSD Context State

## 当前主线
- 正式主线: `LGC-LCSF-SP384-RepConv-BalancedPrefix-LesionDice2`
- 任务类型: 6 类语义分割；禁止 detection head、bbox loss、anchor、NMS、YOLO 分支或检测后处理。
- 类别: background / leaf / rust / alternaria_leaf_spot / gray_spot / brown_spot
- baseline: DeepLabV3+ + MobileNetV3-Large，mIoU 71.72，FG mIoU 66.58。
- 当前正式 full/e80 dual seed avg: mIoU 77.10，FG mIoU 72.83。
- LesionDice2 定义: CE 监督全部 6 类；Dice 只算 rust / alternaria_leaf_spot / gray_spot / brown_spot，即 `--dice-class-start 2`。

## 最高结果
- 最好单 seed: RepConv seed23，mIoU 77.21，FG mIoU 72.96；论文中只能标注为 best single-seed result。
- 最好正式 dual seed avg: LesionDice2 full/e80，mIoU 77.10，FG mIoU 72.83。
- LesionDice2 相对旧 RepConv 主线 dual avg 76.94 / 72.63: mIoU +0.16，FG mIoU +0.20。
- LesionDice2 dual seed lesion IoU avg: rust 84.44，alternaria 58.92，gray 68.27，brown 56.95。
- 当前主线训练态参数: 12,284,955，约 12.28M；FP32 约 49.14 MB / 46.86 MiB。
- RepConv deploy 融合后参数: 12,139,547，约 12.14M；FP32 约 48.56 MB / 46.31 MiB。

## 候选分支
- 当前无已通过 gate、可晋级的新候选。
- `LesionDice2-CEBrown4.4` 已拒绝: 128/64 seed11 mIoU 67.33，FG 61.49；弱于 LesionDice2 reference seed11 67.99 / 62.36，且 brown 从 35.03 降到 34.30。
- 新快筛连续多轮未能强于 LesionDice2 reference，优先转入 formal ablation 补证据链，而不是继续堆小 loss。

## 正在跑 / 刚跑完实验
- 正在跑: 无 ATLDSD 训练进程；存在无关 `LeafSeg` Python 训练进程，不属于本项目。
- 刚跑完: `LesionDice2-CEBrown4.4 128/64 seed11`。
- 输出目录: `outputs/atldsd_fast/combo_lgc_lcsf_sp384_repconv_balanced_lesiondice2_cls_b44_128_64_e24_s11`
- 结果: mIoU 67.33，FG mIoU 61.49；IoU background 96.54，leaf 91.17，rust 77.71，alternaria 48.62，gray 55.63，brown 34.30。
- Dice: background 98.24，leaf 95.38，rust 87.46，alternaria 65.43，gray 71.49，brown 51.08。
- Precision: background 98.08，leaf 95.78，rust 82.70，alternaria 78.98，gray 67.68，brown 53.13。
- Recall: background 98.40，leaf 94.99，rust 92.79，alternaria 55.85，gray 75.76，brown 49.18。
- Severity: MAE 0.01830，Grade Acc 85.94。
- Complexity: Params 12.28M，MACs 20.53G，FLOPs 41.07G，FPS 59.02，time/image 0.01694s，device cuda。
- 判定: 过旧 128/64 seed11 gate，但弱于当前 LesionDice2 reference，拒绝，不补 seed23，不进 192/96。

## 最近 5 个实验结论
- `LesionDice2-CEBrown4.4 128/64 seed11`: REJECT，mIoU 67.33 / FG 61.49；弱于 LesionDice2 reference 67.99 / 62.36，brown 未补强，停止 CE brown 单轴增强。
- `LesionDice2-DiceW1.25 128/64 seed11`: FAIL，mIoU 65.59 / FG 59.45；不过 gate，alternaria/gray 明显下降，关闭 Dice 权重方向。
- `LesionDice2-DiceW0.75 128/64 seed11`: FAIL，mIoU 66.60 / FG 60.69；不过 gate，rust/alternaria/gray 均低于 LesionDice2 reference。
- `LesionDice2+SeverityConsistency0.05 128/64 seed11`: REJECT，mIoU 67.62 / FG 61.86；过旧 gate 但低于 LesionDice2 reference，gray 与 severity grade 受损。
- `LesionDice2 full/e80 seed23 final report 核查`: PASS，mIoU 77.14 / FG 72.90；与 seed11 组成 dual avg 77.10 / 72.83，正式主线升级为 LesionDice2。

## 已拒绝方向
- SCLP、LBFTLoss、focal scalar tuning、DySample/learned upsampling、CHFR、强 ASPP、decoder attention stacking、non-residual high attention、大 backbone 替换、目标检测式设计。
- Low-CA、Decoder-CAA、High-MLCA、High-ECA、PPM、ASPP-ScSE、Severity0.02/0.05、MixUp/CutMix、e150 同配方长轮。
- High-SimAM 仅保留 brown/gray 边际线索，不放大为主线。
- LesionDice2 CE gray/brown down-weight `gray=2.7,brown=3.6`: 128/64 seed11 不过 gate，brown/alternaria 明显受损。
- LesionDice2 AuxBoundary0.3: 128/64 seed11 不过 gate，alternaria/gray/brown 低于 LesionDice2 reference。
- LesionDice2 CFR-a0.15: 128/64 seed11 不过 gate，alternaria/brown 明显受损。
- LesionDice2 LesionPrior0.03-cap3: 128/64 seed11 不过 gate，brown 小涨但 rust/gray/alternaria 与 severity 受损。
- LesionDice2+SeverityConsistency0.05: 弱于 LesionDice2 reference，gray 与 severity grade 受损，不补 seed23。
- LesionDice2-DiceW0.75 / DiceW1.25: 均弱于默认 1.0，Dice 权重微调方向关闭。
- LesionDice2-CEBrown4.4: brown 未补强且 rust/main metrics 下降，停止 CE brown 单轴增强方向。

## 下一步 gate
- 若继续新模块/微调实验，仍必须从 128/64 seed11 开始。
- 旧 128/64 seed11 gate: 66.82 / 60.88；当前 LesionDice2 reference seed11: 67.99 / 62.36。
- 只有 full/e80 dual seed avg 超过当前正式主线 77.10 / 72.83，才更新主线。
- 当前优先级: 暂停继续堆新小模块，补 `+LGC+LCSF full/e80 seed23`，完成 formal ablation dual-seed evidence。

## 关键命令
- 查训练进程: `Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell' -and ($_.CommandLine -match 'ATLDSD|deeplab|atldsd|train.py|run_long|run_fast') }`
- 查 GPU: `nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv`
- 读 LesionDice2 seed23 报告: `Get-Content -Raw -Encoding UTF8 outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23/reports/best_miou/metrics_summary.json`
- 读 CEBrown4.4 报告: `Get-Content -Raw -Encoding UTF8 outputs/atldsd_fast/combo_lgc_lcsf_sp384_repconv_balanced_lesiondice2_cls_b44_128_64_e24_s11/reports/best_miou/metrics_summary.json`
- 启动 `+LGC+LCSF formal full/e80 seed23`: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_long_formal_ablation_lgc_lcsf_full_e80.ps1 -Variant LGC_LCSF -Seeds 23`
- 生成 formal ablation audit: `D:\soft\Anaconda\envs\Pytorch\python.exe figures\gen_fig_atldsd_formal_ablation_audit.py`
- 刷新 summary: `D:\soft\Anaconda\envs\Pytorch\python.exe figures\gen_fig_training_results_summary.py`

## 输出目录
- LesionDice2-CEBrown4.4 128/64 seed11: `outputs/atldsd_fast/combo_lgc_lcsf_sp384_repconv_balanced_lesiondice2_cls_b44_128_64_e24_s11`
- LGC formal full/e80 seed11/23: `outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s11`，`outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s23`
- LGC+LCSF formal full/e80 seed11: `outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s11`
- Planned LGC+LCSF formal full/e80 seed23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s23`
- RepConv old official seed11/23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s11`，`outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s23`
- BalancedPrefix seed11/23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s11`，`outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s23`
- LesionDice2 full/e80 seed11/23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s11`，`outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23`
- 汇总表: `outputs/atldsd/summary`

## 失败模块
- Low-CA: 128/64 avg mIoU -0.72，FG -0.85；gray 涨但 brown/rust/alternaria 受损。
- Decoder-CAA: 128/64 avg mIoU -3.37，FG -4.01；替换 SP 代价过大。
- High-MLCA / High-ECA: 高层注意力明显损伤 alternaria/brown。
- DySample: seed11 mIoU -2.14，FG -2.50；learned upsampling 暂停。
- FocalDice / LabelSmoothing: 小病斑类不稳，尤其 alternaria/brown 代价大。
- LesionDice2 CE gray/brown down-weight `g27_b36`: 128/64 seed11 mIoU 66.48，FG 60.53，brown 30.90，alternaria 44.94。
- LesionDice2 AuxBoundary0.3: 128/64 seed11 mIoU 66.74，FG 60.81，rust 81.58，alternaria 44.85，gray 54.73，brown 32.30。
- LesionDice2 CFR-a0.15: 128/64 seed11 mIoU 65.95，FG 59.95，alternaria 41.16，brown 29.53。
- LesionDice2 LesionPrior0.03-cap3: 128/64 seed11 mIoU 65.69，FG 59.58，rust 77.71，alternaria 43.56，gray 52.95，brown 33.41。
- LesionDice2-CEBrown4.4: 128/64 seed11 mIoU 67.33，FG 61.49，brown 34.30；弱于 LesionDice2 reference。
