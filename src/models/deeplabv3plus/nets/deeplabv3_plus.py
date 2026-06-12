import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbone_registry import build_backbone

for _parent in Path(__file__).resolve().parents:
    direct_plugins = _parent / "plugins"
    modules_plugins = _parent / "modules" / "plugins"
    if direct_plugins.is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
    if modules_plugins.is_dir():
        modules_root = _parent / "modules"
        if str(modules_root) not in sys.path:
            sys.path.insert(0, str(modules_root))
        break

from plugins import build_attention


def _resolve_attention_type(global_type, stage_type):
    return global_type if stage_type is None else stage_type


def _make_base_grid(scale, groups):
    h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
    grid_y, grid_x = torch.meshgrid(h, h, indexing="ij")
    return torch.stack([grid_x, grid_y]).transpose(1, 2).repeat(1, groups, 1).reshape(1, -1, 1, 1)


class DySample(nn.Module):
    """Lightweight learned upsampling initialized near bilinear sampling."""

    def __init__(self, channels, scale=2, groups=4):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"DySample channels ({channels}) must be divisible by groups ({groups}).")
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset = nn.Conv2d(channels, 2 * groups * self.scale * self.scale, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        self.register_buffer("init_pos", _make_base_grid(self.scale, groups))

    def forward(self, x):
        b, _, h, w = x.shape
        offset = self.offset(x) * 0.25 + self.init_pos
        offset = offset.view(b, 2, -1, h, w)

        coords_y = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_x = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
        coords = torch.stack([grid_x, grid_y]).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(b, -1, self.scale * h, self.scale * w)


def build_decoder_upsample(upsample_type, channels, scale):
    upsample_type = (upsample_type or "bilinear").lower().strip()
    if upsample_type in {"bilinear", "interpolate", "none"}:
        return None
    if upsample_type in {"dysample", "dy_sample"}:
        return DySample(channels, scale=scale)
    raise ValueError("Unsupported decoder upsample type: {}".format(upsample_type))


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, _, row, col = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, "bilinear", True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        return self.conv_cat(feature_cat)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels=256, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        # The structure follows the standard PSP-style PPM design. A clean
        # reference copy is tracked under:
        # Integrated as plugins.PyramidPoolingPlugin / attention_type="ppm".
        # Here we keep the final projection inside the module so it can be
        # inserted directly after ASPP in DeepLabV3+.
        branch_channels = max(in_channels // len(pool_sizes), 1)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + branch_channels * len(pool_sizes), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        pyramids.extend(
            [F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=True) for stage in self.stages]
        )
        return self.project(torch.cat(pyramids, dim=1))


class LocalGlobalLesionContextBlock(nn.Module):
    def __init__(self, channels=256, reduction=4, alpha=0.5, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        hidden = max(channels // reduction, 32)
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        local = self.local_context(x)
        global_gate = self.global_context(x)
        spatial_gate = self.spatial_gate(local)
        context = self.project(local * global_gate * (1.0 + spatial_gate))
        return x + self.alpha * context


class PartialConv3(nn.Module):
    def __init__(self, channels, n_div=4):
        super().__init__()
        self.dim_conv3 = max(channels // n_div, 1)
        self.dim_untouched = channels - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        if self.dim_untouched == 0:
            return x1
        return torch.cat((x1, x2), dim=1)


class RepConvBlock(nn.Module):
    """3x3 + 1x1 structural re-parameterization block for decoder features."""

    def __init__(self, in_channels, out_channels, stride=1, padding=1, bn_mom=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.groups = 1

        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
        )
        self.rbr_identity = (
            nn.BatchNorm2d(in_channels, momentum=bn_mom)
            if out_channels == in_channels and stride == 1
            else None
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(x))

        identity = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + identity)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "rbr_reparam.weight" in state_dict and not hasattr(self, "rbr_reparam"):
            self.rbr_reparam = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.stride,
                padding=self.padding,
                bias=True,
            )
            if hasattr(self, "rbr_dense"):
                del self.rbr_dense
            if hasattr(self, "rbr_1x1"):
                del self.rbr_1x1
            if hasattr(self, "rbr_identity"):
                del self.rbr_identity
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _pad_1x1_to_3x3(kernel):
        return F.pad(kernel, [1, 1, 1, 1]) if kernel is not None else 0

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv = branch[0]
            bn = branch[1]
            kernel = conv.weight
        else:
            bn = branch
            kernel_value = torch.zeros(
                self.out_channels,
                self.in_channels,
                3,
                3,
                dtype=bn.weight.dtype,
                device=bn.weight.device,
            )
            for channel in range(self.in_channels):
                kernel_value[channel, channel, 1, 1] = 1
            kernel = kernel_value

        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel * scale, bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def fuse_for_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        del self.rbr_dense
        del self.rbr_1x1
        if self.rbr_identity is not None:
            del self.rbr_identity


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="standard", bn_mom=0.1):
        super().__init__()
        conv_type = (conv_type or "standard").lower()
        if conv_type == "standard":
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
        elif conv_type == "pconv":
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_mom),
                nn.ReLU(inplace=True),
                PartialConv3(out_channels),
                nn.BatchNorm2d(out_channels, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
        elif conv_type == "repconv":
            self.block = RepConvBlock(in_channels, out_channels, bn_mom=bn_mom)
        else:
            raise ValueError(f"Unsupported decoder conv type `{conv_type}`. Use `standard`, `pconv`, or `repconv`.")

    def forward(self, x):
        return self.block(x)


class LesionBoundarySharpeningBlock(nn.Module):
    def __init__(self, channels, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(1, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, feature, boundary_logits):
        if boundary_logits.shape[-2:] != feature.shape[-2:]:
            boundary_logits = F.interpolate(
                boundary_logits,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        edge = feature - self.smooth(feature)
        gate = self.boundary_gate(boundary_logits)
        return feature + self.alpha * edge * gate


class ComponentGuidedHighFrequencyRefinementBlock(nn.Module):
    def __init__(self, channels, alpha=0.2, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = nn.Parameter(torch.zeros(1))
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.frequency_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
        )
        self.component_gate = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature, lesion_logits, boundary_logits, center_logits):
        component_logits = [lesion_logits, boundary_logits, center_logits]
        resized_logits = []
        for logits in component_logits:
            if logits.shape[-2:] != feature.shape[-2:]:
                logits = F.interpolate(logits, size=feature.shape[-2:], mode="bilinear", align_corners=True)
            resized_logits.append(torch.sigmoid(logits))

        high_frequency = feature - self.smooth(feature)
        refined = self.frequency_refine(high_frequency)
        component_probs = torch.cat(resized_logits, dim=1)
        gate = self.component_gate(component_probs)
        confidence = component_probs.max(dim=1, keepdim=True).values
        return feature + self.alpha * torch.tanh(self.gamma) * confidence * gate * refined


class ComponentFeedbackRefinementBlock(nn.Module):
    """Use lesion/boundary/center component cues to softly refine decoder features."""

    def __init__(self, channels, alpha=0.15, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_mom),
        )
        nn.init.zeros_(self.refine[1].weight)
        nn.init.zeros_(self.refine[1].bias)

    def forward(self, feature, lesion_logits, boundary_logits, center_logits):
        component_logits = [lesion_logits, boundary_logits, center_logits]
        resized_probs = []
        for logits in component_logits:
            if logits.shape[-2:] != feature.shape[-2:]:
                logits = F.interpolate(logits, size=feature.shape[-2:], mode="bilinear", align_corners=True)
            resized_probs.append(torch.sigmoid(logits))

        component_probs = torch.cat(resized_probs, dim=1)
        gate = self.feedback_gate(component_probs)
        confidence = component_probs.max(dim=1, keepdim=True).values
        return feature + self.alpha * confidence * gate * self.refine(feature)


class LesionAwareCrossScaleFusion(nn.Module):
    """Cross-scale gates between high-level lesion semantics and low-level edges."""

    def __init__(self, high_channels=256, low_channels=48, alpha=0.5, bn_mom=0.1):
        super().__init__()
        self.alpha = alpha
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, 1, bias=False),
            nn.BatchNorm2d(low_channels, momentum=bn_mom),
            nn.Sigmoid(),
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, 1, bias=False),
            nn.BatchNorm2d(high_channels, momentum=bn_mom),
            nn.Sigmoid(),
        )
        self.lesion_spatial = nn.Sequential(
            nn.Conv2d(high_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.low_edge = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 3, padding=1, groups=low_channels, bias=False),
            nn.BatchNorm2d(low_channels, momentum=bn_mom),
            nn.Conv2d(low_channels, low_channels, 1, bias=False),
            nn.BatchNorm2d(low_channels, momentum=bn_mom),
        )

    def forward(self, high_feature, low_feature):
        low_gate = self.high_to_low(high_feature)
        high_gate = self.low_to_high(low_feature)
        lesion_gate = self.lesion_spatial(high_feature)
        high_feature = high_feature * (1.0 + self.alpha * high_gate)
        low_feature = low_feature * (1.0 + self.alpha * low_gate) + self.alpha * lesion_gate * self.low_edge(low_feature)
        return high_feature, low_feature


class DeepLab(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="mobilenet",
        pretrained=True,
        downsample_factor=16,
        attention_type="",
        attention_low_type=None,
        attention_high_type=None,
        attention_aspp_type=None,
        attention_decoder_type=None,
        decoder_conv_type="standard",
        decoder_upsample_type="bilinear",
        use_ppm=False,
        ppm_bins=(1, 2, 3, 6),
        use_component_aux=False,
        use_lbsb=False,
        lbsb_alpha=0.25,
        use_lcaf=False,
        lcaf_alpha=0.5,
        use_lglc=False,
        lglc_alpha=0.5,
        use_chfr=False,
        chfr_alpha=0.2,
        use_cfr=False,
        cfr_alpha=0.15,
    ):
        super().__init__()
        self.use_component_aux = use_component_aux
        self.use_lbsb = use_lbsb
        self.use_lcaf = use_lcaf
        self.use_lglc = use_lglc
        self.use_chfr = use_chfr
        self.use_cfr = use_cfr

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
        self.lglc = LocalGlobalLesionContextBlock(256, alpha=lglc_alpha) if use_lglc else nn.Identity()
        self.use_ppm = use_ppm
        self.ppm = PyramidPoolingModule(256, out_channels=256, pool_sizes=tuple(ppm_bins)) if use_ppm else nn.Identity()
        upsample_scale = max(1, downsample_factor // 4)
        self.decoder_upsample = build_decoder_upsample(decoder_upsample_type, 256, upsample_scale)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.lcaf = LesionAwareCrossScaleFusion(256, 48, alpha=lcaf_alpha) if use_lcaf else nn.Identity()

        self.cat_conv = nn.Sequential(
            DecoderConvBlock(48 + 256, 256, decoder_conv_type),
            nn.Dropout(0.5),
            DecoderConvBlock(256, 256, decoder_conv_type),
            nn.Dropout(0.1),
        )
        self.attention_decoder = build_attention(attention_decoder_type, 256)
        self.lbsb = LesionBoundarySharpeningBlock(256, alpha=lbsb_alpha) if use_lbsb else nn.Identity()
        self.chfr = ComponentGuidedHighFrequencyRefinementBlock(256, alpha=chfr_alpha) if use_chfr else nn.Identity()
        self.cfr = ComponentFeedbackRefinementBlock(256, alpha=cfr_alpha) if use_cfr else nn.Identity()
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        if self.use_component_aux:
            self.lesion_aux_head = nn.Conv2d(256, 1, 1, stride=1)
            self.boundary_aux_head = nn.Conv2d(256, 1, 1, stride=1)
            self.center_aux_head = nn.Conv2d(256, 1, 1, stride=1)
        elif self.use_lbsb:
            raise ValueError("LBSB requires use_component_aux=True so boundary logits are available.")
        elif self.use_chfr:
            raise ValueError("CHFR requires use_component_aux=True so component gates are available.")
        elif self.use_cfr:
            raise ValueError("CFR requires use_component_aux=True so component logits are available.")

    def forward(self, x):
        height, width = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        low_level_features = self.attention_low(low_level_features)
        x = self.attention_high(x)
        x = self.aspp(x)
        x = self.attention_aspp(x)
        x = self.lglc(x)
        x = self.ppm(x)
        low_level_features = self.shortcut_conv(low_level_features)

        if self.decoder_upsample is None:
            x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode="bilinear", align_corners=True)
        else:
            x = self.decoder_upsample(x)
            if x.size(2) != low_level_features.size(2) or x.size(3) != low_level_features.size(3):
                x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode="bilinear", align_corners=True)
        if self.use_lcaf:
            x, low_level_features = self.lcaf(x, low_level_features)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.attention_decoder(x)
        lesion_feature_logits = None
        boundary_feature_logits = None
        center_feature_logits = None
        if self.use_component_aux:
            lesion_feature_logits = self.lesion_aux_head(x)
            boundary_feature_logits = self.boundary_aux_head(x)
            center_feature_logits = self.center_aux_head(x)
        if self.use_lbsb:
            x = self.lbsb(x, boundary_feature_logits)
        if self.use_chfr:
            x = self.chfr(x, lesion_feature_logits, boundary_feature_logits, center_feature_logits)
        if self.use_cfr:
            x = self.cfr(x, lesion_feature_logits, boundary_feature_logits, center_feature_logits)
        logits = self.cls_conv(x)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
        if not self.use_component_aux:
            return logits
        return {
            "logits": logits,
            "lesion_logits": F.interpolate(lesion_feature_logits, size=(height, width), mode="bilinear", align_corners=True),
            "boundary_logits": F.interpolate(boundary_feature_logits, size=(height, width), mode="bilinear", align_corners=True),
            "center_logits": F.interpolate(center_feature_logits, size=(height, width), mode="bilinear", align_corners=True),
        }
