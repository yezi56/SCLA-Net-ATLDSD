# DeepLabV3+ Baseline

This directory is the plain DeepLabV3+ baseline used for semantic segmentation experiments.

The default training path is intentionally conservative:

- backbone: `mobilenet`
- attention: disabled
- PPM: disabled
- Focal Loss: disabled
- MixUp / CutMix: disabled

Optional modules remain in the codebase for later ablation experiments, but they should be enabled explicitly and should not be mixed into the baseline result.

## Supported Backbones

- `mobilenet`
- `xception`
- `mobilenet_swin` for later ablation only
- `efficientnet_b4` for EfficientNet-B4 + DeepLabV3+ ablations

Backbone integration is centralized in:

```text
nets/backbone_registry.py
```

To add a new DeepLabV3+ backbone, register one `BackboneSpec` there with:

- `builder`
- `in_channels`
- `low_level_channels`
- checkpoint markers for auto inference
- optional pretrained URL and LR limits

The model, training CLI, pretrained download behavior, LR-limit special cases,
and checkpoint auto-detection all read from this registry.

## Plain RiceSeg Baseline

Prepare the VOC-style adapter:

```powershell
cd D:\Code\all
D:\soft\Anaconda\envs\Pytorch\python.exe scripts\prepare_riceseg_voc.py --source-root D:\dataset\RiceSeg\semseg_prepared --output-root D:\dataset\RiceSeg\RiceSegdevkit --copy-mode hardlink
```

Train plain DeepLabV3+ MobileNetV2 for 300 epochs:

```powershell
cd D:\Code\all\src\models\deeplabv3plus
D:\soft\Anaconda\envs\Pytorch\python.exe train.py --dataset-name RiceSeg --datasets-root D:\dataset\RiceSeg --num-classes 5 --backbone mobilenet --attention-type none --use-ppm false --model-path model_data\deeplab_mobilenetv2.pth --freeze-epoch 50 --unfreeze-epoch 300 --freeze-batch-size 8 --unfreeze-batch-size 4 --num-workers 0 --init-lr 0.0035 --dice-loss true --focal-loss false --save-period 10 --eval-period 10 --save-dir outputs\riceseg\deeplabv3plus_mobilenet_baseline\weights --log-dir outputs\riceseg\deeplabv3plus_mobilenet_baseline\logs
```

Continue from the latest checkpoint:

```powershell
cd D:\Code\all\src\models\deeplabv3plus
D:\soft\Anaconda\envs\Pytorch\python.exe train.py --dataset-name RiceSeg --datasets-root D:\dataset\RiceSeg --num-classes 5 --backbone mobilenet --attention-type none --use-ppm false --model-path outputs\riceseg\deeplabv3plus_mobilenet_baseline\weights\last_epoch_weights.pth --init-epoch 11 --freeze-epoch 50 --unfreeze-epoch 300 --freeze-batch-size 8 --unfreeze-batch-size 4 --num-workers 0 --init-lr 0.0035 --dice-loss true --focal-loss false --save-period 10 --eval-period 10 --save-dir outputs\riceseg\deeplabv3plus_mobilenet_baseline\weights --log-dir outputs\riceseg\deeplabv3plus_mobilenet_baseline\logs
```

## Optional Ablation Switches

These are not baseline settings.

```powershell
python train.py --attention-type cbam
python train.py --backbone efficientnet_b4 --pretrained true --model-path "" --attention-type none --attention-low-type ca --attention-high-type eca --attention-aspp-type eca --attention-decoder-type cbam
python train.py --use-ppm true
python train.py --focal-loss true --focal-alpha 0.5 --focal-gamma 2.0
python train.py --mix-mode mixup --mix-prob 0.5 --mixup-alpha 0.4
python train.py --mix-mode cutmix --mix-prob 0.5 --cutmix-alpha 1.0
python train.py --backbone mobilenet_swin
```

## Output Convention

For RiceSeg baseline experiments:

```text
outputs/riceseg/deeplabv3plus_mobilenet_baseline/weights
outputs/riceseg/deeplabv3plus_mobilenet_baseline/logs
outputs/riceseg/deeplabv3plus_mobilenet_baseline/val_visualizations
```
