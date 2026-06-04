# ATLDSD Semantic Segmentation Source Layout

This `src` directory is organized for semantic segmentation paper experiments.

```text
src/
  atldsd_seg/          Main research package for ATLDSD
    configs/           Named experiment configs
    datasets/          VOC-style dataset adapters
    engine/            Training/evaluation launchers
    metrics/           mIoU, Dice, pixel accuracy utilities
    models/            Model registry and stable paper names
    severity/          Disease severity estimation from masks
    utils/             Shared helpers
  models/              Inherited third-party/baseline implementations
    deeplabv3plus/     Current runnable DeepLabV3+ baseline
    segnext/           SegNeXt comparison candidate
    unet, pspnet...    Later comparison baselines
  modules/plugins/     Attention and loss modules used for ablations
```

The inherited folders under `src/models` are intentionally kept in place so
existing checkpoints, scripts, and imports remain reproducible. New paper code
should import from `atldsd_seg` and treat `src/models` as implementation
backends.

Typical commands:

```powershell
$env:PYTHONPATH="D:\Code\ATLDSD\src"
D:\soft\Anaconda\envs\Pytorch\python.exe -m atldsd_seg.engine.launch_deeplabv3plus deeplabv3plus_mobilenet_150 --dry-run
D:\soft\Anaconda\envs\Pytorch\python.exe -m atldsd_seg.engine.launch_deeplabv3plus deeplabv3plus_efficientnet_b4_150 --dry-run
```
