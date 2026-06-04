# EfficientNet Weight Cache

This directory stores local pretrained EfficientNet weights used by the
segmentation backbone wrappers. Checkpoint binaries are ignored by Git.

## Current File

| File | Source | SHA256 |
|---|---|---|
| `efficientnet_b4_rwightman-23ab8bcd.pth` | `https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth` | `23AB8BCD5BDBEF61A7A43B91ADCAD81F622FD7F36FB4935A569828D77888C44E` |

Selection note: for EfficientNet-style PyTorch image backbones, the highest-star
reference checked was `huggingface/pytorch-image-models`. The active code uses
`torchvision.models.efficientnet_b4`, so the local cache stores the compatible
PyTorch/TorchVision checkpoint instead of a structurally different checkpoint.
