"""Reusable segmentation metrics for paper tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationMetrics:
    pixel_accuracy: float
    mean_iou_all: float
    mean_dice_all: float
    mean_iou_foreground: float
    mean_dice_foreground: float
    per_class_iou: np.ndarray
    per_class_dice: np.ndarray


def compute_confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def summarize_metrics(confusion_matrix: np.ndarray) -> SegmentationMetrics:
    hist = confusion_matrix.astype(np.float64)
    true_positive = np.diag(hist)
    gt_area = hist.sum(axis=1)
    pred_area = hist.sum(axis=0)
    union = gt_area + pred_area - true_positive

    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    dice_den = gt_area + pred_area
    dice = np.divide(2 * true_positive, dice_den, out=np.full_like(true_positive, np.nan), where=dice_den > 0)

    total = hist.sum()
    pixel_accuracy = float(true_positive.sum() / total) if total > 0 else 0.0
    foreground = slice(1, None)
    return SegmentationMetrics(
        pixel_accuracy=pixel_accuracy,
        mean_iou_all=float(np.nanmean(iou)),
        mean_dice_all=float(np.nanmean(dice)),
        mean_iou_foreground=float(np.nanmean(iou[foreground])) if len(iou) > 1 else 0.0,
        mean_dice_foreground=float(np.nanmean(dice[foreground])) if len(dice) > 1 else 0.0,
        per_class_iou=iou,
        per_class_dice=dice,
    )
