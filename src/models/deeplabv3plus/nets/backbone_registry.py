from __future__ import annotations

import sys
import importlib.util
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from nets.lite_swin import LightweightSwinBranch
from nets.mobilenetv2 import mobilenetv2
from nets.xception import xception

for _parent in Path(__file__).resolve().parents:
    if (_parent / "efficientnet").is_dir() and str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
        break

from efficientnet import EfficientNetB4


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    builder: Callable[..., nn.Module]
    in_channels: int
    low_level_channels: int
    checkpoint_markers: Tuple[str, ...]
    pretrained_url: Optional[str] = None
    adam_lr_limit_max: float = 5e-4
    adam_lr_limit_min: float = 3e-4
    sgd_lr_limit_max: float = 1e-1
    sgd_lr_limit_min: float = 5e-4


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") == -1:
            return
        if not hasattr(m, "stride") or not hasattr(m, "kernel_size"):
            return
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate // 2, dilate // 2)
                m.padding = (dilate // 2, dilate // 2)
        elif m.kernel_size == (3, 3):
            m.dilation = (dilate, dilate)
            m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class MobileNetV3Large(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        except ImportError as exc:
            raise ImportError("torchvision is required for the MobileNetV3-Large backbone.") from exc

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        self.features = model.features

        if downsample_factor == 8:
            self.features[7].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(8, 13):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            self.features[13].apply(partial(self._nostride_dilate, dilate=4))
            for i in range(14, len(self.features)):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            self.features[13].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(14, len(self.features)):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        else:
            raise ValueError("MobileNetV3-Large supports downsample_factor 8 or 16.")

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") == -1:
            return
        if not hasattr(m, "stride") or not hasattr(m, "kernel_size"):
            return
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size in {(3, 3), (5, 5)}:
                dilation = max(dilate // 2, 1)
                m.dilation = (dilation, dilation)
                m.padding = (dilation * (m.kernel_size[0] // 2), dilation * (m.kernel_size[1] // 2))
        elif m.kernel_size in {(3, 3), (5, 5)}:
            m.dilation = (dilate, dilate)
            m.padding = (dilate * (m.kernel_size[0] // 2), dilate * (m.kernel_size[1] // 2))

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class MobileNetV2Swin(nn.Module):
    """
    Lightweight dual-backbone design:
    - MobileNetV2 is the main backbone.
    - Swin Transformer is an auxiliary branch built on top of shared low-level CNN features.
    - Window attention is used to limit token interaction cost and parameter growth.
    """

    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

        self.swin_branch = LightweightSwinBranch(
            in_channels=24,
            embed_dim=192,
            depth=4,
            num_heads=4,
            window_size=7,
            mlp_ratio=2.0,
            patch_stride=4,
            out_channels=128,
            dropout=0.0,
        )
        self.swin_fuse = nn.Sequential(
            nn.Conv2d(320 + 128, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") == -1:
            return
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate // 2, dilate // 2)
                m.padding = (dilate // 2, dilate // 2)
        elif m.kernel_size == (3, 3):
            m.dilation = (dilate, dilate)
            m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        mobile_high = self.features[4:](low_level_features)
        swin_high = self.swin_branch(low_level_features, target_size=mobile_high.shape[2:])
        fused_high = self.swin_fuse(torch.cat([mobile_high, swin_high], dim=1))
        return low_level_features, fused_high


FASTERNET_VARIANTS = {
    "t1": {
        "embed_dim": 64,
        "depths": (1, 2, 8, 2),
        "drop_path_rate": 0.02,
        "act_layer": "GELU",
        "weight": "fasternet_t1-epoch.291-val_acc1.76.2180.pth",
    },
    "t2": {
        "embed_dim": 96,
        "depths": (1, 2, 8, 2),
        "drop_path_rate": 0.05,
        "act_layer": "RELU",
        "weight": "fasternet_t2-epoch.289-val_acc1.78.8860.pth",
    },
}


def _find_fasternet_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "fasternet"
        if (candidate / "models" / "fasternet.py").is_file():
            return candidate
    raise FileNotFoundError("FasterNet source not found under src/models/fasternet.")


def _load_fasternet_class():
    root = _find_fasternet_root()
    module_path = root / "models" / "fasternet.py"
    spec = importlib.util.spec_from_file_location("local_fasternet_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import FasterNet from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FasterNet, root


class FasterNetBackbone(nn.Module):
    def __init__(self, variant="t1", downsample_factor=16, pretrained=True):
        super().__init__()
        if downsample_factor != 16:
            raise ValueError("FasterNet backbones currently expose the /16 feature map for DeepLabV3+. Use --downsample-factor 16.")
        cfg = FASTERNET_VARIANTS[variant]
        FasterNet, root = _load_fasternet_class()
        self.model = FasterNet(
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            mlp_ratio=2.0,
            n_div=4,
            drop_path_rate=cfg["drop_path_rate"],
            norm_layer="BN",
            act_layer=cfg["act_layer"],
            fork_feat=True,
        )
        self.high_index = 2
        if pretrained:
            self._load_pretrained(root / "model_data" / cfg["weight"])

    def _load_pretrained(self, weight_path: Path):
        if not weight_path.is_file():
            raise FileNotFoundError(f"FasterNet pretrained weight not found: {weight_path}")
        checkpoint = torch.load(str(weight_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        cleaned = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            cleaned[key] = value
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(f"[FasterNet] Loaded {weight_path.name}: missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x):
        outs = self.model(x)
        return outs[0], outs[self.high_index]


class FasterNetT1(FasterNetBackbone):
    def __init__(self, downsample_factor=16, pretrained=True):
        super().__init__("t1", downsample_factor=downsample_factor, pretrained=pretrained)


class FasterNetT2(FasterNetBackbone):
    def __init__(self, downsample_factor=16, pretrained=True):
        super().__init__("t2", downsample_factor=downsample_factor, pretrained=pretrained)


BACKBONE_REGISTRY: Dict[str, BackboneSpec] = OrderedDict(
    (
        (
            "mobilenet",
            BackboneSpec(
                name="mobilenet",
                builder=MobileNetV2,
                in_channels=320,
                low_level_channels=24,
                checkpoint_markers=("backbone.features.",),
                pretrained_url="https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar",
            ),
        ),
        (
            "mobilenet_swin",
            BackboneSpec(
                name="mobilenet_swin",
                builder=MobileNetV2Swin,
                in_channels=320,
                low_level_channels=24,
                checkpoint_markers=("backbone.swin_branch.", "backbone.swin_fuse."),
                pretrained_url="https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar",
            ),
        ),
        (
            "mobilenetv3_large",
            BackboneSpec(
                name="mobilenetv3_large",
                builder=MobileNetV3Large,
                in_channels=960,
                low_level_channels=24,
                checkpoint_markers=("backbone.features.13.", "backbone.features.16."),
                sgd_lr_limit_max=5e-2,
                sgd_lr_limit_min=5e-4,
            ),
        ),
        (
            "xception",
            BackboneSpec(
                name="xception",
                builder=xception,
                in_channels=2048,
                low_level_channels=256,
                checkpoint_markers=("backbone.conv1.", "backbone.block1.", "backbone.midflow."),
                pretrained_url="https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth",
                adam_lr_limit_max=1e-4,
                adam_lr_limit_min=1e-4,
            ),
        ),
        (
            "efficientnet_b4",
            BackboneSpec(
                name="efficientnet_b4",
                builder=EfficientNetB4,
                in_channels=1792,
                low_level_channels=32,
                checkpoint_markers=("backbone.low_level_features.", "backbone.high_level_features."),
            ),
        ),
        (
            "fasternet_t1",
            BackboneSpec(
                name="fasternet_t1",
                builder=FasterNetT1,
                in_channels=256,
                low_level_channels=64,
                checkpoint_markers=("backbone.model.patch_embed.", "backbone.model.stages."),
                sgd_lr_limit_max=5e-2,
                sgd_lr_limit_min=1e-4,
            ),
        ),
        (
            "fasternet_t2",
            BackboneSpec(
                name="fasternet_t2",
                builder=FasterNetT2,
                in_channels=384,
                low_level_channels=96,
                checkpoint_markers=("backbone.model.patch_embed.", "backbone.model.stages."),
                sgd_lr_limit_max=5e-2,
                sgd_lr_limit_min=1e-4,
            ),
        ),
    )
)


def get_backbone_names():
    return list(BACKBONE_REGISTRY.keys())


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(get_backbone_names())
        raise ValueError(f"Unsupported backbone `{name}`. Supported backbones: {supported}.") from exc


def build_backbone(name: str, downsample_factor: int, pretrained: bool):
    spec = get_backbone_spec(name)
    backbone = spec.builder(downsample_factor=downsample_factor, pretrained=pretrained)
    return backbone, spec.in_channels, spec.low_level_channels


def backbone_has_external_pretrained_url(name: str) -> bool:
    return bool(get_backbone_spec(name).pretrained_url)


def download_backbone_pretrained_weights(name: str, model_dir: str = "./model_data") -> None:
    spec = get_backbone_spec(name)
    if not spec.pretrained_url:
        return

    from torch.hub import load_state_dict_from_url

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    load_state_dict_from_url(spec.pretrained_url, model_dir)


def get_backbone_lr_limits(name: str, optimizer_type: str):
    spec = get_backbone_spec(name)
    if optimizer_type == "adam":
        return spec.adam_lr_limit_max, spec.adam_lr_limit_min
    return spec.sgd_lr_limit_max, spec.sgd_lr_limit_min


def infer_backbone_from_state_dict(checkpoint: Iterable[str], default: str = "mobilenet") -> str:
    keys = list(checkpoint)
    for name, spec in BACKBONE_REGISTRY.items():
        if name == default:
            continue
        if any(any(key.startswith(marker) for marker in spec.checkpoint_markers) for key in keys):
            return name
    return default
