# U-Net

This is the main ordinary U-Net baseline for segmentation experiments.

It is based on the previous VOC-style U-Net implementation and now serves as the
single place for plain U-Net experiments. Attention modules are hot-pluggable
through `attention_type`, using the stable factory in `src/modules/plugins`.

## Role

- Plain U-Net baseline.
- Supports `vgg` and `resnet50` backbones.
- Supports optional plug-in modules such as `cbam`, `se`, `eca`, `bam`, `ca`,
  `caa`, `cpam`, `cpca`, `ela`, `ema`, `gam`, `gc`, `lsk`, `mlca`, `simam`, `sk`,
  `scse`, `dsam`, `shsa`, `ghost`, `triplet`, `shuffle`, and `emcam`.
- Uses VOC-style dataset utilities and mIoU evaluation.

Use `src/models/unetpp` when the experiment needs U-Net++ / Nested U-Net.

## Example

```powershell
cd D:\Code\all\src\models\unet
python train.py
```

To enable a stable attention module, set `attention_type` in `train.py` or pass
it through any wrapper script that exposes the option. Use an empty string or
`none` for the plain baseline.
