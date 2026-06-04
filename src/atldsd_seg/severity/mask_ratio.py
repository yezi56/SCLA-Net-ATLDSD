"""Severity estimation from leaf and lesion pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from atldsd_seg.constants import LEAF_CLASS_ID, LESION_CLASS_IDS, SEVERITY_BINS


@dataclass(frozen=True)
class SeverityResult:
    leaf_pixels: int
    lesion_pixels: int
    lesion_ratio: float
    severity_level: str


def _label_ratio(ratio: float) -> str:
    for label, lower, upper in SEVERITY_BINS:
        if lower <= ratio < upper:
            return label
    return SEVERITY_BINS[-1][0]


def estimate_severity(
    mask: np.ndarray,
    leaf_class_id: int = LEAF_CLASS_ID,
    lesion_class_ids: Iterable[int] = LESION_CLASS_IDS,
) -> SeverityResult:
    lesion_ids = tuple(lesion_class_ids)
    leaf_pixels = int((mask == leaf_class_id).sum() + np.isin(mask, lesion_ids).sum())
    lesion_pixels = int(np.isin(mask, lesion_ids).sum())
    ratio = float(lesion_pixels / leaf_pixels) if leaf_pixels > 0 else 0.0
    return SeverityResult(
        leaf_pixels=leaf_pixels,
        lesion_pixels=lesion_pixels,
        lesion_ratio=ratio,
        severity_level=_label_ratio(ratio),
    )
