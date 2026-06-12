from __future__ import annotations

import torch
import torch.nn as nn


def _to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | tuple[bool, bool] = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class RectangularSelfCalibrationAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: int = 1,
        band_kernel_size: int = 11,
        square_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        hidden = max(1, channels // ratio)
        groups = hidden if channels % hidden == 0 else 1
        self.dwconv_hw = nn.Conv2d(
            channels,
            channels,
            square_kernel_size,
            padding=square_kernel_size // 2,
            groups=channels,
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.excite = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=groups),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=groups),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc = self.dwconv_hw(x)
        attn = self.excite(self.pool_h(x) + self.pool_w(x))
        return attn * loc


class RCMAttention(nn.Module):
    """Rectangular Self-Calibration Module for efficient semantic segmentation."""

    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 2,
        band_kernel_size: int = 11,
        square_kernel_size: int = 3,
        ratio: int = 1,
        ls_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.token_mixer = RectangularSelfCalibrationAttention(
            channels=channels,
            ratio=ratio,
            band_kernel_size=band_kernel_size,
            square_kernel_size=square_kernel_size,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.mlp = ConvMlp(channels, int(mlp_ratio * channels), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(channels)) if ls_init_value else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x + shortcut
