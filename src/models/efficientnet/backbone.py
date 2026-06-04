from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

_LOCAL_WEIGHT_PATH = Path(__file__).resolve().parent / "model_data" / "efficientnet_b4_rwightman-23ab8bcd.pth"


class EfficientNetB4(nn.Module):
    def __init__(self, downsample_factor: int = 16, pretrained: bool = True) -> None:
        super().__init__()
        if downsample_factor not in {8, 16}:
            raise ValueError("EfficientNet-B4 supports downsample_factor 8 or 16.")

        self.features = self._build_features(pretrained=pretrained)
        self._apply_output_stride(downsample_factor)

        children = list(self.features.named_children())
        low_children, high_children = self._split_low_high_children(children)
        if not low_children or not high_children:
            raise ValueError("EfficientNet-B4 features are empty.")

        self.low_level_features = nn.Sequential(OrderedDict(low_children))
        self.high_level_features = nn.Sequential(OrderedDict(high_children))

    @staticmethod
    def _build_features(pretrained: bool) -> nn.Sequential:
        try:
            from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

            if pretrained and _LOCAL_WEIGHT_PATH.exists():
                model = efficientnet_b4(weights=None)
                state_dict = torch.load(_LOCAL_WEIGHT_PATH, map_location="cpu")
                model.load_state_dict(state_dict)
                return model.features

            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            return efficientnet_b4(weights=weights).features
        except Exception:
            from .model import efficientnet_b4

            if pretrained:
                print(
                    "Warning: torchvision EfficientNet-B4 weights unavailable; "
                    "using the local EfficientNet-B4 without pretrained weights."
            )
            return efficientnet_b4(num_classes=1000).features

    @staticmethod
    def _nostride_dilate(m: nn.Module, dilate: int) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") == -1:
            return
        if getattr(m, "stride", None) == (2, 2):
            m.stride = (1, 1)
            if getattr(m, "kernel_size", None) == (3, 3):
                m.dilation = (dilate // 2, dilate // 2)
                m.padding = (dilate // 2, dilate // 2)
        elif getattr(m, "kernel_size", None) == (3, 3):
            m.dilation = (dilate, dilate)
            m.padding = (dilate, dilate)

    @staticmethod
    def _stage_number(name: str) -> Optional[int]:
        if name.isdigit():
            return int(name)
        if name and name[0].isdigit():
            return int(name[0])
        return None

    def _split_low_high_children(
        self, children: List[Tuple[str, nn.Module]]
    ) -> Tuple[List[Tuple[str, nn.Module]], List[Tuple[str, nn.Module]]]:
        low_children = []
        high_children = []
        for name, module in children:
            stage = self._stage_number(name)
            if stage is None and name in {"stem", "stem_conv"}:
                low_children.append((name, module))
            elif stage is not None and stage <= 2:
                low_children.append((name, module))
            else:
                high_children.append((name, module))
        return low_children, high_children

    def _apply_to_stages(self, predicate: Callable[[int], bool], dilate: int) -> None:
        for name, module in self.features.named_children():
            stage = self._stage_number(name)
            if stage is not None and predicate(stage):
                module.apply(lambda m, d=dilate: self._nostride_dilate(m, d))

    def _apply_output_stride(self, downsample_factor: int) -> None:
        if downsample_factor == 16:
            self._apply_to_stages(lambda stage: stage >= 6, dilate=2)
        elif downsample_factor == 8:
            self._apply_to_stages(lambda stage: 4 <= stage < 6, dilate=2)
            self._apply_to_stages(lambda stage: stage >= 6, dilate=4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_level_features = self.low_level_features(x)
        x = self.high_level_features(low_level_features)
        return low_level_features, x
