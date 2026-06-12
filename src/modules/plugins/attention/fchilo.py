from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FCHiLoAttention(nn.Module):
    """High-low frequency attention adapted from crack segmentation FCHiLo."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: int = 2,
        alpha: float = 0.5,
        gamma_init: float = 1e-3,
    ) -> None:
        super().__init__()
        num_heads = max(1, min(num_heads, channels))
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        head_dim = channels // num_heads
        low_heads = int(num_heads * alpha)
        low_heads = max(1, min(low_heads, num_heads - 1)) if num_heads > 1 else 1
        high_heads = num_heads - low_heads

        self.channels = channels
        self.num_heads = num_heads
        self.low_heads = low_heads
        self.high_heads = high_heads
        self.low_dim = low_heads * head_dim
        self.high_dim = high_heads * head_dim
        self.head_dim = head_dim
        self.window_size = max(1, window_size)
        self.scale = head_dim ** -0.5

        self.low_q = _DepthwiseSeparableConv(channels, self.low_dim)
        self.low_kv = _DepthwiseSeparableConv(channels, self.low_dim * 2)
        self.low_proj = _DepthwiseSeparableConv(self.low_dim, self.low_dim)

        if self.high_heads > 0:
            self.high_qkv = _DepthwiseSeparableConv(channels, self.high_dim * 3)
            self.high_proj = _DepthwiseSeparableConv(self.high_dim, self.high_dim)
        else:
            self.high_qkv = None
            self.high_proj = None

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))

    def _low_attention(self, x: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.low_q(x).flatten(2).transpose(1, 2)
        kv = self.low_kv(low).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.low_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(b, -1, 2, self.low_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, h, w, self.low_dim)
        out = out.permute(0, 3, 1, 2).contiguous()
        return self.low_proj(out)

    def _high_attention(self, high: torch.Tensor) -> torch.Tensor | None:
        if self.high_heads == 0 or self.high_qkv is None or self.high_proj is None:
            return None
        b, _, h, w = high.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            high = F.pad(high, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = high.shape[-2:]
        qkv = self.high_qkv(high)
        qkv = qkv.permute(0, 2, 3, 1).reshape(
            b,
            hp // ws,
            ws,
            wp // ws,
            ws,
            3,
            self.high_heads,
            self.head_dim,
        )
        qkv = qkv.permute(5, 0, 1, 3, 6, 2, 4, 7).reshape(
            3,
            b * (hp // ws) * (wp // ws),
            self.high_heads,
            ws * ws,
            self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).reshape(
            b,
            hp // ws,
            wp // ws,
            self.high_heads,
            ws,
            ws,
            self.head_dim,
        )
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(b, self.high_dim, hp, wp)
        out = out[:, :, :h, :w].contiguous()
        return self.high_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h < self.window_size or w < self.window_size:
            return x
        low = F.avg_pool2d(x, kernel_size=self.window_size, stride=self.window_size)
        high = x - F.interpolate(low, size=(h, w), mode="nearest")
        parts = [self._low_attention(x, low)]
        high_out = self._high_attention(high)
        if high_out is not None:
            parts.append(high_out)
        out = torch.cat(parts, dim=1)
        if out.shape[1] != c:
            out = F.pad(out, (0, 0, 0, 0, 0, c - out.shape[1]))[:, :c]
        return x + self.gamma * self.out_proj(out)
