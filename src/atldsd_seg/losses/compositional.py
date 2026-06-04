"""Loss for leaf-lesion compositional segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from atldsd_seg.constants import IGNORE_INDEX, LESION_CLASS_IDS


@dataclass(frozen=True)
class CompositionalTargets:
    final: torch.Tensor
    leaf: torch.Tensor
    lesion: torch.Tensor
    disease: torch.Tensor
    boundary: torch.Tensor


def build_lesion_boundary_target(
    lesion: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Build a binary lesion-boundary target from the lesion mask."""

    valid = lesion != ignore_index
    lesion_float = ((lesion == 1) & valid).float().unsqueeze(1)
    pad = kernel_size // 2
    eroded = 1.0 - F.max_pool2d(1.0 - lesion_float, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (lesion_float - eroded).clamp_min(0.0).squeeze(1)
    boundary[~valid] = ignore_index
    return boundary


def build_compositional_targets(mask: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> CompositionalTargets:
    """Derive leaf, lesion, and disease-type targets from the 6-class mask.

    Original ATLDSD labels:
    0 background
    1 leaf
    2 rust
    3 alternaria_leaf_spot
    4 gray_spot
    5 brown_spot
    """

    valid = mask != ignore_index

    leaf = torch.zeros_like(mask)
    leaf[valid & (mask > 0)] = 1
    leaf[~valid] = ignore_index

    lesion = torch.zeros_like(mask)
    lesion_ids = torch.tensor(LESION_CLASS_IDS, device=mask.device, dtype=mask.dtype)
    is_lesion = valid & torch.isin(mask, lesion_ids)
    lesion[is_lesion] = 1
    lesion[~valid] = ignore_index

    disease = torch.full_like(mask, ignore_index)
    disease[is_lesion] = mask[is_lesion] - min(LESION_CLASS_IDS)
    boundary = build_lesion_boundary_target(lesion, ignore_index=ignore_index)

    return CompositionalTargets(final=mask, leaf=leaf, lesion=lesion, disease=disease, boundary=boundary)


class CompositionalSegmentationLoss(nn.Module):
    """Supervise final 6-class output plus the three structured heads."""

    def __init__(
        self,
        final_weight: float = 1.0,
        leaf_weight: float = 0.4,
        lesion_weight: float = 0.8,
        disease_weight: float = 0.6,
        boundary_weight: float = 0.0,
        boundary_pos_weight: float = 5.0,
        disease_class_weights: list[float] | tuple[float, ...] | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.final_weight = final_weight
        self.leaf_weight = leaf_weight
        self.lesion_weight = lesion_weight
        self.disease_weight = disease_weight
        self.boundary_weight = boundary_weight
        self.boundary_pos_weight = boundary_pos_weight
        self.ignore_index = ignore_index
        if disease_class_weights is None:
            self.register_buffer("disease_class_weights", None)
        else:
            self.register_buffer(
                "disease_class_weights",
                torch.tensor(disease_class_weights, dtype=torch.float32),
            )

    def forward(self, outputs: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        targets = build_compositional_targets(mask, ignore_index=self.ignore_index)
        final_loss = F.nll_loss(outputs["final_logits"], targets.final, ignore_index=self.ignore_index)
        leaf_loss = F.cross_entropy(outputs["leaf_logits"], targets.leaf, ignore_index=self.ignore_index)
        lesion_loss = F.cross_entropy(outputs["lesion_logits"], targets.lesion, ignore_index=self.ignore_index)

        disease_valid = targets.disease != self.ignore_index
        if disease_valid.any():
            disease_class_weights = None
            if self.disease_class_weights is not None:
                disease_class_weights = self.disease_class_weights.to(
                    device=outputs["disease_logits"].device,
                    dtype=outputs["disease_logits"].dtype,
                )
            disease_loss = F.cross_entropy(
                outputs["disease_logits"],
                targets.disease,
                weight=disease_class_weights,
                ignore_index=self.ignore_index,
            )
        else:
            disease_loss = outputs["disease_logits"].sum() * 0.0

        boundary_loss = outputs["final_logits"].sum() * 0.0
        if self.boundary_weight > 0:
            if "boundary_logits" not in outputs:
                raise KeyError("boundary_logits is required when boundary_weight > 0")
            boundary_valid = targets.boundary != self.ignore_index
            if boundary_valid.any():
                pos_weight = torch.tensor(
                    self.boundary_pos_weight,
                    device=outputs["boundary_logits"].device,
                    dtype=outputs["boundary_logits"].dtype,
                )
                boundary_loss = F.binary_cross_entropy_with_logits(
                    outputs["boundary_logits"].squeeze(1)[boundary_valid],
                    targets.boundary[boundary_valid].float(),
                    pos_weight=pos_weight,
                )

        total = (
            self.final_weight * final_loss
            + self.leaf_weight * leaf_loss
            + self.lesion_weight * lesion_loss
            + self.disease_weight * disease_loss
            + self.boundary_weight * boundary_loss
        )
        return {
            "loss": total,
            "final_loss": final_loss.detach(),
            "leaf_loss": leaf_loss.detach(),
            "lesion_loss": lesion_loss.detach(),
            "disease_loss": disease_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
        }
