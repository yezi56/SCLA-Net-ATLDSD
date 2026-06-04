from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt).pow(self.gamma) * ce_loss
        if self.alpha is not None:
            valid = targets != self.ignore_index
            alpha = torch.zeros_like(loss)
            alpha[valid] = self.alpha[targets[valid]]
            loss = alpha * loss
        return loss.mean()


AVAILABLE_LOSSES = {
    "ce": nn.CrossEntropyLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "focal_loss": FocalLoss,
}


def build_loss(loss_type: str | None, **kwargs) -> nn.Module:
    if not loss_type:
        return nn.CrossEntropyLoss(**kwargs)
    loss_type = loss_type.lower().strip()
    if loss_type not in AVAILABLE_LOSSES:
        raise ValueError(f"Unsupported loss `{loss_type}`. Available: {', '.join(sorted(AVAILABLE_LOSSES))}")
    return AVAILABLE_LOSSES[loss_type](**kwargs)
