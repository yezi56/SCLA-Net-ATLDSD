# SCLA-Net-ATLDSD

Semantic segmentation research code for the Apple Tree Leaf Disease Segmentation Dataset (ATLDSD).

## Project Goal

This repository studies apple leaf disease severity segmentation with VOC-style masks:

- `0` background
- `1` leaf
- `2` rust
- `3` alternaria leaf spot
- `4` gray spot
- `5` brown spot

The current strongest baseline is:

```text
DeepLabV3+ + MobileNetV3-Large
mIoU: 71.72%
Foreground mIoU: 66.58%
Pixel Accuracy: 97.76%
```

The current research direction is SCLA-Net:

```text
Severity-Controlled Lesion Augmentation and Component-guided Attention Network
```

## Main Ideas

1. Use `MobileNetV3-Large` as a strong lightweight encoder.
2. Use severity-controlled lesion copy-paste to improve small lesion and weak disease classes.
3. Add component-aware auxiliary learning for lesion, boundary, and center/distance cues.
4. Add severity-aware component-guided attention instead of generic SE/CBAM attention.
5. Add severity consistency loss between predicted and ground-truth lesion/leaf ratios.

## Dataset

The dataset is not included in this repository.

Windows local layout:

```text
D:\dataset\ATLDSD\VOCdevkit\VOC2012
D:\dataset\ATLDSD\VOCdevkit\VOC2007
```

Ubuntu server layout:

```text
/home/liuzhe/SCLA-Net-ATLDSD
/home/liuzhe/SCLA-Net-ATLDSD/VOCdevkit/VOC2007
/home/liuzhe/SCLA-Net-ATLDSD/VOCdevkit/VOC2012
```

You can also keep the dataset elsewhere and set:

```bash
export ATLDSD_VOCDEVKIT_PATH=/absolute/path/to/VOCdevkit
```

The prepared split sizes are:

```text
train: 1148
val:   246
test:  247
```

## Important Scripts

### Ubuntu Quickstart

Put the repository under `/home/liuzhe`, put `VOCdevkit` in the repository root or set `ATLDSD_VOCDEVKIT_PATH`, then run:

```bash
cd /home/liuzhe/SCLA-Net-ATLDSD
chmod +x scripts/*.sh
./scripts/setup_ubuntu_env.sh cu121
./scripts/run_ubuntu.sh sclp
```

If the server does not have `python3`, bootstrap Miniconda first:

```bash
chmod +x scripts/*.sh
./scripts/bootstrap_ubuntu_miniconda.sh
export PATH="$HOME/miniconda3/bin:$PATH"
./scripts/setup_ubuntu_env.sh cu121
./scripts/run_ubuntu.sh sclp
```

Use `cpu`, `cu118`, `cu121`, or `cu124` for the setup script according to the server CUDA driver. If PyTorch is already installed, use:

```bash
./scripts/setup_ubuntu_env.sh skip
```

Run the current strongest baseline:

```bash
./scripts/run_ubuntu.sh baseline
```

Run the current SCLA-Net E1 experiment:

```bash
./scripts/run_ubuntu.sh sclp
```

Run the lower-intensity SCLP ablation:

```bash
./scripts/run_ubuntu.sh sclp03
```

Useful overrides:

```bash
EPOCHS=50 BATCH_SIZE=2 NUM_WORKERS=8 ./scripts/run_ubuntu.sh baseline
SCLP_PROB=0.5 SCLP_MAX_COMPONENTS=2 ./scripts/run_ubuntu.sh sclp
PYTHON_BIN=/usr/bin/python3 ./scripts/run_ubuntu.sh baseline
```

### Windows Scripts

Baseline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_150.ps1
```

Current E1 experiment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_atldsd_deeplabv3plus_mobilenetv3_large_sclp_150.ps1
```

CLCS negative ablation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_atldsd_clcs_mobilenetv3_large_150.ps1
```

## Notes

Training outputs, checkpoints, generated reports, and datasets are intentionally ignored by git.

See `seg/ATLDSD项目进度.md` for the running experiment log and research plan.
