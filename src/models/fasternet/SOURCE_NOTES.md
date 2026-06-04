# FasterNet Source Notes

Imported from the official FasterNet repository:

- Repository: `https://github.com/JierunChen/FasterNet`
- Local commit: `e8fba44`
- Paper: FasterNet: Rethinking PConv for Faster Neural Networks
- Reason for selection: official PyTorch implementation and the highest-star
  relevant FasterNet/FasterNet-T1/T2 repository checked for this workspace task.

## Relevant Variants

| Variant | Config | Official checkpoint |
|---|---|---|
| FasterNet-T1 | `cfg/fasternet_t1.yaml` | `https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t1-epoch.291-val_acc1.76.2180.pth` |
| FasterNet-T2 | `cfg/fasternet_t2.yaml` | `https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t2-epoch.289-val_acc1.78.8860.pth` |

## Local Role

This directory is currently a reference implementation, not yet wired into the
main DeepLabV3+ backbone registry. To use FasterNet as a segmentation backbone,
add a wrapper that returns `(low_level_features, high_level_features)` and then
register it in:

```text
src/models/deeplabv3plus/nets/backbone_registry.py
```
