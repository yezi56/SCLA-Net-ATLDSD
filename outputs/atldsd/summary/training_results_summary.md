# ATLDSD Training Results Summary

| ID | Method | Change | Status | mIoU | FG mIoU | Sev. MAE | Grade Acc | Params | FLOPs | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Old-MNet | DeepLabV3+ MobileNet | old backbone | done | 67.58 | 61.75 | - | - | - | - | old baseline |
| B4 | DeepLabV3+ EfficientNet-B4 | heavier backbone | done | 65.59 | 59.59 | - | - | 32.48 | 51.30 | discard as main backbone |
| CLCS-V3 | CLCS + MobileNetV3-Large | early compositional trial | done | 70.50 | 65.05 | - | - | - | - | archival trial |
| SCLP-0.7 | Mainline0 + SCLP | strong copy-paste | done | 68.97 | 63.28 | 0.01542 | 90.24 | - | - | failed augmentation |
| SCLP-0.3 | Mainline0 + weak SCLP | weak copy-paste | done | 69.90 | 64.67 | 0.01592 | 90.65 | - | - | failed augmentation |
| Mainline0 | DeepLabV3+ MobileNetV3-Large | strong baseline | done | 71.72 | 66.58 | 0.01241 | 95.12 | 11.73 | 15.28 | baseline to beat |
| Mainline1 | Mainline0 + component heads | lesion/boundary/center heads | done | 72.11 | 67.03 | 0.01212 | 94.31 | 11.73 | 15.29 | current structural anchor |
| Aux-A | Mainline1 + severity loss | loss ablation | done | 72.12 | 67.06 | 0.01147 | 93.90 | 11.73 | 15.29 | best severity MAE; not main structure |
| Mainline2 | Mainline1 + PConv | decoder locality | done | 71.76 | 66.62 | 0.01373 | 93.09 | 10.65 | 6.51 | lighter; test PConv+LBSB synergy |
| Boundary1 | Mainline1 + LBSB | boundary sharpening | done | 72.86 | 67.89 | 0.01177 | 93.90 | 11.73 | 15.29 | best mIoU; promote to current best |
| Boundary2 | Mainline1 + PConv + LBSB | decoder locality + boundary sharpening | done | 71.68 | 66.54 | 0.01281 | 93.50 | 10.65 | 6.52 | no PConv-LBSB synergy; do not keep PConv |
| Fusion1 | Mainline1 + LBSB + LCAF | lesion-aware cross-scale fusion | done | 72.68 | 67.70 | 0.01169 | 93.90 | 11.76 | 15.53 | close, but below Boundary1; do not replace LBSB-only |
| Context1 | Mainline1 + LBSB + LGLC | local-global lesion context | done | 72.31 | 67.26 | 0.01170 | 93.50 | 11.84 | 15.33 | below Boundary1; keep as negative context ablation |
| Final-LGC-LCSF | Boundary1 + SP decoder + LGC + LCSF | balanced-prefix full e80, dual-seed avg | done | 76.60 | 72.22 | 0.00965 | 94.92 | 12.14 | 38.37 | previous pre-RepConv mainline |
| Final-RepConv | Final-LGC-LCSF + RepConv decoder | full e80, dual-seed avg | done | 76.94 | 72.63 | 0.01030 | 93.29 | 12.28 | 41.07 | previous official mainline |
| Final-LesionDice2 | Final-RepConv + lesion-only Dice | full e80, dual-seed avg | done | 77.10 | 72.83 | 0.01012 | 94.51 | 12.28 | 41.07 | current official mainline |
