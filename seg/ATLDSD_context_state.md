# ATLDSD Context State

## 当前主线
- 正式主线: `LGC-LCSF-SP384-RepConv-BalancedPrefix-LesionDice2`
- 任务类型: 6 类语义分割；禁止 detection head、bbox loss、anchor、NMS、YOLO 分支或检测后处理。
- 类别: background / leaf / rust / alternaria_leaf_spot / gray_spot / brown_spot
- baseline: DeepLabV3+ + MobileNetV3-Large，mIoU 71.72，FG mIoU 66.58。
- 当前正式 full/e80 dual seed avg: mIoU 77.10，FG mIoU 72.83。

## 最高结果
- 最好单 seed: RepConv seed23，mIoU 77.21，FG mIoU 72.96；论文中只能标注为 best single-seed result。
- 最好正式 dual seed avg: LesionDice2 full/e80，mIoU 77.10，FG mIoU 72.83。
- LesionDice2 相对旧 RepConv 主线: mIoU +0.16，FG mIoU +0.21。
- LesionDice2 dual seed lesion IoU avg: rust 84.44，alternaria 58.92，gray 68.27，brown 56.95。

## 候选分支
- 当前无可晋级新候选。
- LesionDice2 定义: CE 监督全部 6 类；Dice 只算 rust / alternaria_leaf_spot / gray_spot / brown_spot，即 `--dice-class-start 2`。
- 已过 gate: LesionDice2 128/64 双 seed、192/96 双 seed、full/e80 seed11、full/e80 seed23、full/e80 dual seed avg。
- 当前工作重心: 停止继续堆小模块，转入 full/e80、多 seed、消融、复杂度/FPS、RepConv deploy 证据、可视化和论文写作。
- 论文正式消融缺口: `+LGC` 已补齐 full/e80 seed11/23，状态为 PASS；`+LGC+LCSF` 仍只有 128/64 fast-screen evidence，状态为 GAP。

## 正在跑/刚跑完实验
- 正在跑: 无 ATLDSD 训练进程；存在无关 `LeafSeg` Python 训练进程。用户已要求暂时不再启动 ATLDSD 训练，避免占用其他项目 GPU。
- 刚完成: `+LGC+LCSF formal full/e80 seed11`。
- 目标: 为论文逐模块消融补齐 `+LGC+LCSF` 的 full/e80 正式证据。
- 来源: `paper_formal_ablation_audit` 显示 `+LGC+LCSF` 仍为 fast-screen-only GAP；PDF/笔记强调任务专属的跨尺度/跨分支自适应融合，不支持检测式结构。
- 修改文件: 训练未修改模型、loss、训练流程或推理结构；更新 `figures/gen_fig_atldsd_paper_evidence_audit.py`、summary audit、本文件和接力日志。
- 插入位置: ASPP 后 LGC；decoder concat 前 LCSF；decoder 使用 SP；RepConv=false；BalancedPrefix=false；LesionDice2=false。
- 命令: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_long_formal_ablation_lgc_lcsf_full_e80.ps1 -Variant LGC_LCSF -Seeds 11`
- seed / 尺度 / epoch: seed11，full train/val 1148/246，input 384，epoch 80，freeze epoch 5。
- 输出目录: `outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s11`
- 结果: mIoU 76.69，FG mIoU 72.31；IoU background 98.60，leaf 95.91，rust 84.53，alternaria 58.50，gray 67.73，brown 54.91。
- Dice: background 99.29，leaf 97.91，rust 91.61，alternaria 73.82，gray 80.76，brown 70.89。
- Precision: background 99.44，leaf 97.79，rust 90.07，alternaria 83.19，gray 73.95，brown 64.08。
- Recall: background 99.14，leaf 98.04，rust 93.21，alternaria 66.34，gray 88.95，brown 79.32。
- 复杂度/FPS: Params 12.14M，MACs 19.18G，FLOPs 38.37G，FPS 25.12，time/image 0.03981s，device cuda；该 FPS 受当前环境/负载影响，只作本报告记录。
- gate: 不用于更新主线；本轮为 formal ablation 补跑。seed11 单 seed 已达到旧 RepConv seed11 gate 76.66~76.68 / 72.30 附近并略高。
- 保留/废弃/调整: 保留为 `+LGC+LCSF` partial full/e80 evidence；缺 seed23，不能写成 formal dual-seed row。
- 下一步: 暂停训练；待用户允许后再补 `+LGC+LCSF full/e80 seed23`，再刷新 summary/audit；若不补，论文中必须标注为 PARTIAL。

## 最近 5 个实验结论
- `+LGC+LCSF formal full/e80 seed11`: PASS as partial evidence，mIoU 76.69 / FG 72.31；缺 seed23，不能写成 formal dual-seed row。
- `+LGC+LCSF evidence audit wording sync`: PASS，`paper_evidence_audit` 已改为 partial full/e80 evidence caveat，不再误写为纯 fast-screen-only。
- `+LGC formal full/e80 seed23`: PASS，mIoU 76.06 / FG 71.54；与 seed11 组成 dual avg 76.17 / 71.68，保留为 formal ablation row。
- `+LGC paper_ablation/evidence audit sync`: PASS，`paper_ablation_chain` 中 `+LGC` 已更新为 full/e80 dual-seed。
- `+LGC formal full/e80 seed11`: PASS，mIoU 76.28 / FG 71.82，作为 `+LGC` dual-seed 的第一粒种子证据保留。

## 已拒绝方向
- SCLP、LBFTLoss、focal scalar tuning、DySample/learned upsampling、CHFR、强 ASPP、decoder attention stacking、non-residual high attention、大 backbone 替换、目标检测式设计。
- Low-CA、Decoder-CAA、High-MLCA、High-ECA、PPM、ASPP-ScSE、Severity0.02/0.05、MixUp/CutMix、e150 同配方长轮。
- High-SimAM 仅保留 brown/gray 边际线索，不放大为主线。
- LesionDice2 CE gray/brown down-weight `gray=2.7,brown=3.6`: 128/64 seed11 不过 gate，brown/alternaria 明显受损。
- LesionDice2 AuxBoundary0.3: 128/64 seed11 不过 gate，alternaria/gray/brown 低于 LesionDice2 reference。
- LesionDice2 CFR-a0.15: 128/64 seed11 不过 gate，alternaria/brown 明显受损；不继续同形态 decoder feedback gate 放大。
- LesionDice2 LesionPrior0.03-cap3: 128/64 seed11 不过 gate，brown 小涨但 rust/gray/alternaria 与 severity 受损。

## 下一步 gate
- 下一轮若继续新模块实验，必须从 128/64 seed11 开始，并以 LesionDice2 新主线为 reference。
- 128/64 seed11 gate: 66.82 / 60.88；LesionDice2 reference seed11: 67.99 / 62.36。
- 只有 dual seed avg 超过 77.10 / 72.83，才更新正式主线。
- 当前优先级不是新模块 gate，而是补齐 formal ablation: `+LGC+LCSF seed23`。

## 关键命令
- 查训练进程: `Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell' -and ($_.CommandLine -match 'ATLDSD|deeplab|atldsd|train.py|run_long|run_fast') }`
- 查 GPU: `nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv`
- 读 `+LGC seed11` 报告: `Get-Content -Raw -Encoding UTF8 outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s11/reports/best_miou/metrics_summary.json`
- 读 LesionDice2 seed23 报告: `Get-Content -Raw -Encoding UTF8 outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23/reports/best_miou/metrics_summary.json`
- 读 `+LGC seed23` 报告: `Get-Content -Raw -Encoding UTF8 outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s23/reports/best_miou/metrics_summary.json`
- 读 `+LGC+LCSF seed11` 报告: `Get-Content -Raw -Encoding UTF8 outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s11/reports/best_miou/metrics_summary.json`
- LGC+LCSF formal full/e80 seed23 启动: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_long_formal_ablation_lgc_lcsf_full_e80.ps1 -Variant LGC_LCSF -Seeds 23`
- 生成 formal ablation audit: `D:\soft\Anaconda\envs\Pytorch\python.exe figures\gen_fig_atldsd_formal_ablation_audit.py`
- 刷新 summary: `D:\soft\Anaconda\envs\Pytorch\python.exe figures\gen_fig_training_results_summary.py`

## 输出目录
- LGC formal full/e80 seed11: `outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s11`
- LGC formal full/e80 seed23: `outputs/atldsd_fast/long_lgc_sp384_formal_full_e80_s23`
- LGC+LCSF formal full/e80 seed11: `outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s11`
- Planned LGC+LCSF formal full/e80 seed23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_formal_full_e80_s23`
- RepConv 旧正式主线 seed11/23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s11`，`outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_gray2_rust15_alt12_brown12_full_e80_s23`
- BalancedPrefix full/e80 seed11/23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s11`，`outputs/atldsd_fast/long_lgc_lcsf_sp384_gray2_rust15_alt12_brown12_full_e80_s23`
- LesionDice2 full/e80 seed11/23: `outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s11`，`outputs/atldsd_fast/long_lgc_lcsf_sp384_repconv_balanced_lesiondice2_full_e80_s23`
- 汇总表: `outputs/atldsd/summary`
- formal ablation audit: `outputs/atldsd/summary/paper_formal_ablation_audit.csv`，`outputs/atldsd/summary/paper_formal_ablation_audit.md`
- deploy summary: `outputs/atldsd/summary/deploy_fused_summary.csv`，`outputs/atldsd/summary/deploy_fused_summary.md`
- 论文证据审计: `outputs/atldsd/summary/paper_evidence_audit.csv`，`outputs/atldsd/summary/paper_evidence_audit.md`
- 定性样例: `outputs/atldsd/summary/paper_qualitative_cases`
- 误差分析: `outputs/atldsd/summary/paper_error_analysis`

## 失败模块
- Low-CA: 128/64 avg mIoU -0.72，FG -0.85，gray 涨但 brown/rust/alternaria 受损。
- Decoder-CAA: 128/64 avg mIoU -3.37，FG -4.01，替换 SP 代价过大。
- High-MLCA / High-ECA: 高层注意力明显损伤 alternaria/brown。
- DySample: seed11 mIoU -2.14，FG -2.50，learned upsampling 暂停。
- FocalDice / LabelSmoothing: 小病斑类不稳，尤其 alternaria/brown 代价大。
- LesionDice2 CE gray/brown down-weight `g27_b36`: 128/64 seed11 mIoU 66.48，FG 60.53，brown 30.90，alternaria 44.94。
- LesionDice2 AuxBoundary0.3: 128/64 seed11 mIoU 66.74，FG 60.81，rust 81.58，alternaria 44.85，gray 54.73，brown 32.30。
- LesionDice2 CFR-a0.15: 128/64 seed11 mIoU 65.95，FG 59.95，alternaria 41.16，brown 29.53。
- LesionDice2 LesionPrior0.03-cap3: 128/64 seed11 mIoU 65.69，FG 59.58，rust 77.71，alternaria 43.56，gray 52.95，brown 33.41。
