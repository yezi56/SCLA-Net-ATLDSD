"""Loss functions for ATLDSD experiments."""

from .compositional import CompositionalSegmentationLoss, build_compositional_targets

__all__ = ["CompositionalSegmentationLoss", "build_compositional_targets"]
