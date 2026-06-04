"""CLCS-Net: compositional leaf-lesion semantic segmentation.

This model keeps the DeepLabV3+ encoder/decoder feature extractor but replaces
the single 6-class classifier with three structured heads:

1. leaf head: background vs leaf
2. lesion head: non-lesion vs lesion
3. disease head: rust / alternaria / gray / brown for lesion pixels

The three heads are composed into the final 6-class mask:

background = not leaf
healthy leaf = leaf and not lesion
disease class = leaf and lesion and disease type
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from atldsd_seg.constants import LESION_CLASS_IDS, NUM_CLASSES
from atldsd_seg.paths import DEEPLABV3PLUS_ROOT, SRC_ROOT

if str(DEEPLABV3PLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPLABV3PLUS_ROOT))
if str(SRC_ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(SRC_ROOT / "modules"))

from nets.backbone_registry import build_backbone  # noqa: E402
from nets.deeplabv3_plus import ASPP, DecoderConvBlock, PyramidPoolingModule, _resolve_attention_type  # noqa: E402
from plugins import build_attention  # noqa: E402


class CLCSDeepLabV3Plus(nn.Module):
    """DeepLabV3+ with compositional leaf/lesion/disease output heads."""

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        downsample_factor: int = 16,
        attention_type: str = "",
        attention_low_type: str | None = None,
        attention_high_type: str | None = None,
        attention_aspp_type: str | None = None,
        attention_decoder_type: str | None = None,
        decoder_conv_type: str = "standard",
        use_ppm: bool = False,
        ppm_bins: tuple[int, ...] = (1, 2, 3, 6),
        disease_classes: int = len(LESION_CLASS_IDS),
        use_boundary_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.disease_classes = disease_classes
        self.use_boundary_head = use_boundary_head

        self.backbone, in_channels, low_level_channels = build_backbone(
            backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
        )

        attention_low_type = _resolve_attention_type(attention_type, attention_low_type)
        attention_high_type = _resolve_attention_type(attention_type, attention_high_type)
        attention_aspp_type = _resolve_attention_type(attention_type, attention_aspp_type)
        attention_decoder_type = _resolve_attention_type(attention_type, attention_decoder_type)

        self.attention_low = build_attention(attention_low_type, low_level_channels)
        self.attention_high = build_attention(attention_high_type, in_channels)
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.attention_aspp = build_attention(attention_aspp_type, 256)
        self.ppm = PyramidPoolingModule(256, out_channels=256, pool_sizes=tuple(ppm_bins)) if use_ppm else nn.Identity()

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DecoderConvBlock(48 + 256, 256, decoder_conv_type),
            nn.Dropout(0.5),
            DecoderConvBlock(256, 256, decoder_conv_type),
            nn.Dropout(0.1),
        )
        self.attention_decoder = build_attention(attention_decoder_type, 256)

        self.leaf_head = nn.Conv2d(256, 2, 1)
        self.lesion_head = nn.Conv2d(256, 2, 1)
        self.disease_head = nn.Conv2d(256, disease_classes, 1)
        self.boundary_head = nn.Conv2d(256, 1, 1) if use_boundary_head else None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        low_level_features, high_level_features = self.backbone(x)
        low_level_features = self.attention_low(low_level_features)
        high_level_features = self.attention_high(high_level_features)
        high_level_features = self.aspp(high_level_features)
        high_level_features = self.attention_aspp(high_level_features)
        high_level_features = self.ppm(high_level_features)

        low_level_features = self.shortcut_conv(low_level_features)
        high_level_features = F.interpolate(
            high_level_features,
            size=low_level_features.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        features = self.decoder(torch.cat((high_level_features, low_level_features), dim=1))
        return self.attention_decoder(features)

    @staticmethod
    def compose_probabilities(
        leaf_logits: torch.Tensor,
        lesion_logits: torch.Tensor,
        disease_logits: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leaf_prob = torch.softmax(leaf_logits, dim=1)[:, 1:2]
        lesion_prob = torch.softmax(lesion_logits, dim=1)[:, 1:2]
        disease_prob = torch.softmax(disease_logits, dim=1)

        background_prob = 1.0 - leaf_prob
        healthy_leaf_prob = leaf_prob * (1.0 - lesion_prob)
        disease_region_prob = leaf_prob * lesion_prob * disease_prob
        final_prob = torch.cat([background_prob, healthy_leaf_prob, disease_region_prob], dim=1)
        final_prob = final_prob / final_prob.sum(dim=1, keepdim=True).clamp_min(eps)
        final_logits = torch.log(final_prob.clamp_min(eps))
        return final_prob, final_logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = x.shape[2:]
        features = self.extract_features(x)
        leaf_logits = F.interpolate(self.leaf_head(features), size=(height, width), mode="bilinear", align_corners=True)
        lesion_logits = F.interpolate(self.lesion_head(features), size=(height, width), mode="bilinear", align_corners=True)
        disease_logits = F.interpolate(self.disease_head(features), size=(height, width), mode="bilinear", align_corners=True)
        final_prob, final_logits = self.compose_probabilities(leaf_logits, lesion_logits, disease_logits)
        outputs = {
            "final_logits": final_logits,
            "final_prob": final_prob,
            "leaf_logits": leaf_logits,
            "lesion_logits": lesion_logits,
            "disease_logits": disease_logits,
        }
        if self.boundary_head is not None:
            outputs["boundary_logits"] = F.interpolate(
                self.boundary_head(features),
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
        return outputs
