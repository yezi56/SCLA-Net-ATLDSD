from __future__ import annotations

import torch
from torch import nn


class GCTAttention(nn.Module):
    """Gated Channel Transformation attention."""

    def __init__(self, channels: int, epsilon: float = 1e-5, mode: str = "l2", after_relu: bool = False) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "l2":
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == "l1":
            source = x if self.after_relu else torch.abs(x)
            embedding = source.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            raise ValueError(f"Unknown GCT mode: {self.mode}")

        gate = 1.0 + torch.tanh(embedding * norm + self.beta)
        return x * gate


__all__ = ["GCTAttention"]
