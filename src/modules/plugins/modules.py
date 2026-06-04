from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisor(value: int, upper: int) -> int:
    value = max(1, min(value, upper))
    while value > 1 and upper % value != 0:
        value -= 1
    return value


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1,
                 padding: int | None = None, groups: int = 1, activation: nn.Module | None = None) -> None:
        if padding is None:
            padding = kernel_size // 2
        if activation is None:
            activation = nn.SiLU()
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.act(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.act(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CAA(nn.Module):
    def __init__(self, channels: int, h_kernel_size: int = 11, v_kernel_size: int = 11) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvBNAct(channels, channels)
        self.h_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, h_kernel_size),
            padding=(0, h_kernel_size // 2),
            groups=channels,
            bias=False,
        )
        self.v_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(v_kernel_size, 1),
            padding=(v_kernel_size // 2, 0),
            groups=channels,
            bias=False,
        )
        self.conv2 = ConvBNAct(channels, channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        return x * self.act(attn)


class CPCA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attention = SEAttention(channels, reduction=8)
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.branch5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.branch7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.project = ConvBNAct(channels, channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        spatial = self.local(x) + self.branch5(x) + self.branch7(x)
        spatial = self.project(spatial)
        return x * self.act(spatial)


class CPAMAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = EfficientChannelAttention(channels)
        hidden = max(channels // reduction, 4)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.attn_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.attn_w = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        fused = self.channel_attention(x) + x
        x_h = torch.mean(fused, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(fused, dim=2, keepdim=True)
        y = self.reduce(torch.cat([x_h, x_w], dim=3))
        y_h, y_w = y.split([h, w], dim=3)
        s_h = self.attn_h(y_h.permute(0, 1, 3, 2)).sigmoid()
        s_w = self.attn_w(y_w).sigmoid()
        return fused * s_h * s_w


class PyramidPoolingPlugin(nn.Module):
    def __init__(self, channels: int, bins: tuple[int, ...] = (1, 2, 3, 6), branch_channels: int | None = None) -> None:
        super().__init__()
        branch_channels = branch_channels or max(channels // len(bins), 1)
        self.pool_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for bin_size in bins
            ]
        )
        self.project = ConvBNAct(channels + len(bins) * branch_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pooled = [F.interpolate(layer(x), size=(h, w), mode="bilinear", align_corners=True) for layer in self.pool_layers]
        return self.project(torch.cat([x, *pooled], dim=1))


class ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)], dim=1)


class AttentionGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compress = ZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.compress(x))


class TripletAttention(nn.Module):
    def __init__(self, channels: int, no_spatial: bool = False) -> None:
        super().__init__()
        self.no_spatial = no_spatial
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm1 = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_perm2 = self.hc(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.no_spatial:
            return 0.5 * (x_perm1 + x_perm2)
        x_out = self.hw(x)
        return (x_out + x_perm1 + x_perm2) / 3.0


class ShuffleAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        groups = max(1, min(groups, channels // 2 if channels >= 2 else 1))
        self.groups = groups
        split_channels = channels // (2 * groups)
        split_channels = max(split_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, split_channels, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, split_channels, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, split_channels, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, split_channels, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.channels = channels

    def channel_shuffle(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % (2 * self.groups) != 0:
            return x
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.sigmoid(self.cweight * xn + self.cbias) * x_0
        xs = self.sigmoid(self.sweight * x_1 + self.sbias) * x_1
        out = torch.cat([xn, xs], dim=1).reshape(b, c, h, w)
        return self.channel_shuffle(out, 2)


class EMCAM(nn.Module):
    def __init__(
        self,
        channels: int | None = None,
        expansion: int = 2,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        channels = channels or out_channels or in_channels
        if channels is None:
            raise ValueError("EMCAM requires `channels` or `in_channels`/`out_channels`.")
        hidden = max(channels // expansion, 8)
        self.pre = ConvBNAct(channels, hidden, kernel_size=1)
        self.dw3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        self.dw5 = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False)
        self.dw7 = nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden, bias=False)
        self.mix = ConvBNAct(hidden, channels, kernel_size=1)
        self.channel_gate = EfficientChannelAttention(channels)
        self.spatial_gate = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.pre(x)
        fused = self.dw3(fused) + self.dw5(fused) + self.dw7(fused)
        fused = self.mix(fused)
        fused = self.channel_gate(fused)
        fused = self.spatial_gate(fused)
        return fused


class DoubleAttention(nn.Module):
    def __init__(self, channels: int, c_m: int | None = None, c_n: int | None = None) -> None:
        super().__init__()
        c_m = c_m or channels
        c_n = c_n or max(1, channels // 2)
        self.c_m = c_m
        self.c_n = c_n
        self.conv_a = nn.Conv2d(channels, c_m, 1)
        self.conv_b = nn.Conv2d(channels, c_n, 1)
        self.conv_v = nn.Conv2d(channels, c_n, 1)
        self.project = nn.Conv2d(c_m, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        a = self.conv_a(x).view(b, self.c_m, -1)
        b_map = F.softmax(self.conv_b(x).view(b, self.c_n, -1), dim=-1)
        v = F.softmax(self.conv_v(x).view(b, self.c_n, -1), dim=-1)
        descriptors = torch.bmm(a, b_map.transpose(1, 2))
        out = torch.bmm(descriptors, v).view(b, self.c_m, h, w)
        return self.project(out)


class BAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, dilation: int = 2) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = self.channel_gate(x).view(x.shape[0], x.shape[1], 1, 1)
        spatial = self.spatial_gate(x)
        return x * (1 + torch.sigmoid(channel + spatial))


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class ELAAttention(nn.Module):
    def __init__(self, channels: int, phi: str = "T") -> None:
        super().__init__()
        phi = phi.upper()
        kernel_size = {"T": 5, "B": 7, "S": 5, "L": 7}.get(phi, 5)
        groups = channels if phi in {"T", "B"} else _make_divisor(max(1, channels // 8), channels)
        norm_groups = _make_divisor({"T": 32, "B": 16, "S": 16, "L": 16}.get(phi, 32), channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=groups, bias=False)
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = torch.mean(x, dim=3).view(b, c, h)
        x_w = torch.mean(x, dim=2).view(b, c, w)
        x_h = self.act(self.norm(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.act(self.norm(self.conv(x_w))).view(b, c, 1, w)
        return x * x_h * x_w


class EMAAttention(nn.Module):
    def __init__(self, channels: int, factor: int = 32) -> None:
        super().__init__()
        self.groups = _make_divisor(factor, channels)
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class GAMAttention(nn.Module):
    def __init__(self, channels: int, rate: int = 4) -> None:
        super().__init__()
        hidden = max(channels // rate, 4)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, hidden, 7, padding=3),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 7, padding=3),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel = self.channel_attention(x.permute(0, 2, 3, 1).reshape(b, -1, c))
        channel = channel.reshape(b, h, w, c).permute(0, 3, 1, 2).sigmoid()
        x = x * channel
        return x * self.spatial_attention(x).sigmoid()


class GlobalContextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: float = 0.25,
        pooling_type: str = "att",
        fusion_types: tuple[str, ...] = ("channel_add",),
    ) -> None:
        super().__init__()
        hidden = max(int(channels * ratio), 1)
        self.pooling_type = pooling_type
        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(channels, 1, 1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_add_conv = None
        self.channel_mul_conv = None
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(channels, hidden, 1),
                nn.LayerNorm([hidden, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
            )
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(channels, hidden, 1),
                nn.LayerNorm([hidden, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
            )

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.pooling_type == "att":
            input_x = x.reshape(b, c, h * w).unsqueeze(1)
            mask = self.softmax(self.conv_mask(x).reshape(b, 1, h * w)).unsqueeze(-1)
            return torch.matmul(input_x, mask).reshape(b, c, 1, 1)
        return self.avg_pool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            out = out * torch.sigmoid(self.channel_mul_conv(context))
        if self.channel_add_conv is not None:
            out = out + self.channel_add_conv(context)
        return out


class LSKAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(channels, hidden, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        gates = self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)).sigmoid()
        attn = attn1 * gates[:, 0:1] + attn2 * gates[:, 1:2]
        return x * self.conv(attn)


class MLCAAttention(nn.Module):
    def __init__(self, channels: int, local_size: int = 5, gamma: int = 2, b: int = 1, local_weight: float = 0.5) -> None:
        super().__init__()
        self.local_size = local_size
        self.local_weight = local_weight
        t = int(abs(math.log2(channels) + b) / gamma)
        k = max(t if t % 2 else t + 1, 3)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)
        self.local_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        local = self.local_pool(x)
        global_ = self.global_pool(local)
        temp_local = local.view(b, c, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_.view(b, c, -1).transpose(-1, -2)
        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)
        y_local = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2)
        y_local = y_local.reshape(b, c, self.local_size, self.local_size).sigmoid()
        y_global = y_global.transpose(-1, -2).unsqueeze(-1).sigmoid()
        y_global = F.adaptive_avg_pool2d(y_global, [self.local_size, self.local_size])
        attn = F.adaptive_avg_pool2d(y_global * (1 - self.local_weight) + y_local * self.local_weight, [h, w])
        return x * attn


class SimAM(nn.Module):
    def __init__(self, channels: int | None = None, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        n = max(h * w - 1, 1)
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class SKAttention(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...] = (1, 3, 5, 7), reduction: int = 16, group: int = 1) -> None:
        super().__init__()
        hidden = max(32, channels // reduction)
        group = _make_divisor(group, channels)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=group),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernels
            ]
        )
        self.fc = nn.Linear(channels, hidden)
        self.fcs = nn.ModuleList([nn.Linear(hidden, channels) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.convs]
        fused = sum(feats)
        z = self.fc(fused.mean(dim=(-1, -2)))
        weights = [fc(z).view(x.shape[0], x.shape[1], 1, 1) for fc in self.fcs]
        weights = self.softmax(torch.stack(weights, dim=0))
        return (weights * torch.stack(feats, dim=0)).sum(dim=0)


class ScSEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x * self.cse(x), x * self.sse(x))


class SCSAAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: int = 7,
        group_kernel_sizes: tuple[int, int, int, int] = (3, 5, 7, 9),
        qkv_bias: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
    ) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("SCSAAttention requires channels to be divisible by 4.")
        if gate_layer not in {"sigmoid", "softmax"}:
            raise ValueError("gate_layer must be 'sigmoid' or 'softmax'.")
        if down_sample_mode not in {"avg_pool", "max_pool", "adaptive"}:
            raise ValueError("down_sample_mode must be 'avg_pool', 'max_pool', or 'adaptive'.")

        group_channels = channels // 4
        self.window_size = window_size
        self.down_sample_mode = down_sample_mode
        self.local_dwc = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_channels,
        )
        self.global_dwc_s = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_channels,
        )
        self.global_dwc_m = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_channels,
        )
        self.global_dwc_l = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_channels,
        )
        self.norm_h = nn.GroupNorm(4, channels)
        self.norm_w = nn.GroupNorm(4, channels)
        self.spatial_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()

        self.num_heads = _make_divisor(num_heads, channels)
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(1, channels)
        self.q = nn.Conv2d(channels, channels, 1, groups=channels, bias=qkv_bias)
        self.k = nn.Conv2d(channels, channels, 1, groups=channels, bias=qkv_bias)
        self.v = nn.Conv2d(channels, channels, 1, groups=channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.channel_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_sample_mode == "adaptive" or self.window_size <= 0:
            return F.adaptive_avg_pool2d(x, 1)
        h, w = x.shape[-2:]
        if h < self.window_size or w < self.window_size:
            return F.adaptive_avg_pool2d(x, 1)
        if self.down_sample_mode == "max_pool":
            return F.max_pool2d(x, self.window_size, self.window_size)
        return F.avg_pool2d(x, self.window_size, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        x_h = x.mean(dim=3)
        x_w = x.mean(dim=2)
        h_local, h_s, h_m, h_l = torch.chunk(x_h, 4, dim=1)
        w_local, w_s, w_m, w_l = torch.chunk(x_w, 4, dim=1)

        attn_h = torch.cat(
            [
                self.local_dwc(h_local),
                self.global_dwc_s(h_s),
                self.global_dwc_m(h_m),
                self.global_dwc_l(h_l),
            ],
            dim=1,
        )
        attn_w = torch.cat(
            [
                self.local_dwc(w_local),
                self.global_dwc_s(w_s),
                self.global_dwc_m(w_m),
                self.global_dwc_l(w_l),
            ],
            dim=1,
        )
        x = x * self.spatial_gate(self.norm_h(attn_h)).unsqueeze(-1)
        x = x * self.spatial_gate(self.norm_w(attn_w)).unsqueeze(-2)

        y = self.norm(self._downsample(x))
        _, _, h, w = y.shape
        q = self.q(y).reshape(b, self.num_heads, self.head_dim, h * w)
        k = self.k(y).reshape(b, self.num_heads, self.head_dim, h * w)
        v = self.v(y).reshape(b, self.num_heads, self.head_dim, h * w)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = (attn @ v).reshape(b, c, h, w).mean(dim=(2, 3), keepdim=True)
        return x * self.channel_gate(attn)


class StripPoolingAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        pool_size: tuple[int, int] = (20, 12),
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        inter_channels = max(1, channels // 4)

        def conv_bn_relu(
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            padding: int | tuple[int, int],
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
            )

        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1_1 = conv_bn_relu(channels, inter_channels, 1, 0)
        self.conv1_2 = conv_bn_relu(channels, inter_channels, 1, 0)
        self.conv2_0 = conv_bn_relu(inter_channels, inter_channels, 3, 1)
        self.conv2_1 = conv_bn_relu(inter_channels, inter_channels, 3, 1)
        self.conv2_2 = conv_bn_relu(inter_channels, inter_channels, 3, 1)
        self.conv2_3 = conv_bn_relu(inter_channels, inter_channels, (1, 3), (0, 1))
        self.conv2_4 = conv_bn_relu(inter_channels, inter_channels, (3, 1), (1, 0))
        self.conv2_5 = conv_bn_relu(inter_channels, inter_channels, 3, 1)
        self.conv2_6 = conv_bn_relu(inter_channels, inter_channels, 3, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels * 2, channels, 1, bias=False),
            norm_layer(channels),
        )
        self.align_corners = align_corners

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=self.align_corners)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = self._resize(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = self._resize(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = self._resize(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = self._resize(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        x2 = self.conv2_6(F.relu(x2_5 + x2_4, inplace=True))
        return F.relu(x + self.conv3(torch.cat([x1, x2], dim=1)), inplace=True)


class CrissCrossAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        inter_channels = max(1, channels // 8)
        self.query_conv = nn.Conv2d(channels, inter_channels, 1)
        self.key_conv = nn.Conv2d(channels, inter_channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=3)

    @staticmethod
    def _neg_inf(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        diagonal = torch.diag(torch.full((height,), float("-inf"), device=device, dtype=dtype))
        return diagonal.unsqueeze(0).repeat(batch * width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        proj_query = self.query_conv(x)
        proj_query_h = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        proj_query_w = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_h = proj_key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        proj_key_w = proj_key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        proj_value = self.value_conv(x)
        proj_value_h = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        proj_value_w = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        energy_h = torch.bmm(proj_query_h, proj_key_h)
        energy_h = energy_h + self._neg_inf(b, h, w, x.device, energy_h.dtype)
        energy_h = energy_h.view(b, w, h, h).permute(0, 2, 1, 3)
        energy_w = torch.bmm(proj_query_w, proj_key_w).view(b, h, w, w)
        concate = self.softmax(torch.cat([energy_h, energy_w], dim=3))

        attn_h = concate[:, :, :, :h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attn_w = concate[:, :, :, h : h + w].contiguous().view(b * h, w, w)
        out_h = torch.bmm(proj_value_h, attn_h.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(proj_value_w, attn_w.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
        return self.gamma * (out_h + out_w) + x


class DSAMAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool_att = _SpecAtte(channels)
        half = channels // 2
        self.cubic_11 = _CubicAttention(half, group=1, kernel=11)
        self.cubic_7 = _CubicAttention(channels - half, group=1, kernel=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool_att(x)
        out_a, out_b = torch.split(out, [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2], dim=1)
        return torch.cat([self.cubic_11(out_a), self.cubic_7(out_b)], dim=1)


class _CubicAttention(nn.Module):
    def __init__(self, channels: int, group: int, kernel: int) -> None:
        super().__init__()
        self.h_spatial_att = _SpatialStripAttention(channels, group=group, kernel=kernel)
        self.w_spatial_att = _SpatialStripAttention(channels, group=group, kernel=kernel, horizontal=False)
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w_spatial_att(self.h_spatial_att(x))
        return self.gamma * out + x * self.beta


class _SpatialStripAttention(nn.Module):
    def __init__(self, channels: int, kernel: int = 5, group: int = 1, horizontal: bool = True) -> None:
        super().__init__()
        self.k = kernel
        self.group = _make_divisor(group, channels)
        pad = kernel // 2
        self.kernel = (1, kernel) if horizontal else (kernel, 1)
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if horizontal else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(channels, self.group * kernel, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        filters = self.filter_act(self.conv(self.pool(x))).reshape(n, self.group, self.k, 1).unsqueeze(2)
        unfolded = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)
        return torch.sum(unfolded * filters, dim=3).reshape(n, c, h, w)


class _SpecAtte(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_att = _GlobalPoolStripAttention(channels)
        self.local_att_7 = _LocalPoolStripAttention(channels, kernel=7)
        self.local_att_11 = _LocalPoolStripAttention(channels, kernel=11)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.global_att(x) + self.local_att_7(x) + self.local_att_11(x))


class _GlobalPoolStripAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.vert_low = nn.Parameter(torch.zeros(channels, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(channels, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(channels, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(channels, 1, 1))
        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hori_l = self.hori_pool(x)
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.0) * hori_h
        vert_l = self.vert_pool(hori_out)
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.0) * vert_h
        return x * self.beta + vert_out * self.gamma


class _LocalPoolStripAttention(nn.Module):
    def __init__(self, channels: int, kernel: int = 7) -> None:
        super().__init__()
        self.vert_low = nn.Parameter(torch.zeros(channels, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(channels, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(channels, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(channels, 1, 1))
        self.vert_pool = nn.AvgPool2d((kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d((1, kernel), stride=1)
        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.0) * hori_h
        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.0) * vert_h
        return x * self.beta + vert_out * self.gamma


class SHSAAttention(nn.Module):
    def __init__(self, channels: int, qk_dim: int = 16, partial_channels: int = 32) -> None:
        super().__init__()
        self.partial_channels = max(1, min(partial_channels, channels))
        self.qk_dim = max(1, min(qk_dim, self.partial_channels))
        self.scale = self.qk_dim ** -0.5
        self.pre_norm = nn.GroupNorm(1, self.partial_channels)
        self.qkv = nn.Sequential(
            nn.Conv2d(self.partial_channels, self.qk_dim * 2 + self.partial_channels, 1, bias=False),
            nn.BatchNorm2d(self.qk_dim * 2 + self.partial_channels),
        )
        self.proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x1 = x[:, : self.partial_channels]
        x2 = x[:, self.partial_channels :]
        qkv = self.qkv(self.pre_norm(x1))
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.partial_channels], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(b, self.partial_channels, h, w)
        return self.proj(torch.cat([x1, x2], dim=1))


class GhostModule(nn.Module):
    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 1, dw_size: int = 3, relu: bool = True) -> None:
        super().__init__()
        init_channels = math.ceil(channels / ratio)
        new_channels = init_channels * (ratio - 1)
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        self.primary_conv = nn.Sequential(
            nn.Conv2d(channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            activation,
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, padding=dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            activation,
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, : self.channels]
