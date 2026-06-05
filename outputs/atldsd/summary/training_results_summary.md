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
| Boundary1 | Mainline1 + LBSB | boundary sharpening | running | - | - | - | - | - | - | running; update after completion |
