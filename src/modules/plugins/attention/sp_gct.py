from __future__ import annotations

import torch
from torch import nn

from .gct import GCTAttention
from .strip_pooling import StripPoolingAttention


class SPGCTAttention(nn.Module):
    """Strip-pooling decoder refinement followed by identity-initialized GCT."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.strip_pooling = StripPoolingAttention(channels)
        self.gct = GCTAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gct(self.strip_pooling(x))


__all__ = ["SPGCTAttention"]
