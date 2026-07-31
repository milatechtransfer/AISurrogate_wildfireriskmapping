"""Binned/ordinal metrics for hazard class maps (pure NumPy, no I/O)."""

from __future__ import annotations

import numpy as np


def _valid_class_mask(arr: np.ndarray, invalid_class: int, num_classes: int) -> np.ndarray:
    finite = np.isfinite(arr)
    return finite & (arr != invalid_class) & (arr >= 1) & (arr <= num_classes)


def calculate_hazard_class_metrics(
    pred_classes: np.ndarray,
    gt_classes: np.ndarray,
    *,
    invalid_class: int = 0,
    num_classes: int = 13,
) -> dict[str, float | np.ndarray]:
    """Compute ordinal + set metrics over 1-based hazard class maps.

    Pixels where either input is ``invalid_class``, non-finite, or outside
    ``[1, num_classes]`` are ignored. Confusion matrix rows are GT, cols are
    prediction, ordered class ``1..num_classes``.
    """
    pred = np.asarray(pred_classes, dtype=float)
    gt = np.asarray(gt_classes, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt class maps must share a shape, got {pred.shape} and {gt.shape}")

    valid = _valid_class_mask(pred, invalid_class, num_classes) & _valid_class_mask(gt, invalid_class, num_classes)
    if not np.any(valid):
        raise ValueError("no valid overlapping class pixels to compute hazard metrics")

    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    diff = np.abs(p - g)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (g - 1, p - 1), 1)

    tp = np.diag(confusion).astype(float)
    pred_totals = confusion.sum(axis=0).astype(float)
    gt_totals = confusion.sum(axis=1).astype(float)
    union = tp + (pred_totals - tp) + (gt_totals - tp)

    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_iou = np.where(union > 0, tp / union, np.nan)
        f1_denom = 2.0 * tp + (pred_totals - tp) + (gt_totals - tp)
        per_class_f1 = np.where(f1_denom > 0, (2.0 * tp) / f1_denom, np.nan)

    return {
        "exact_accuracy": float(np.mean(p == g)),
        "within_1_accuracy": float(np.mean(diff <= 1)),
        "within_2_accuracy": float(np.mean(diff <= 2)),
        "mean_absolute_class_error": float(np.mean(diff)),
        "macro_iou": float(np.nanmean(per_class_iou)),
        "macro_f1": float(np.nanmean(per_class_f1)),
        "confusion_matrix": confusion,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
    }


def flatten_hazard_class_metrics(metrics: dict[str, float | np.ndarray]) -> dict[str, float]:
    """Flatten a metrics dict into scalar columns for logging/CSV."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            flat[key] = float(arr)
        else:
            for idx, item in enumerate(arr, start=1):
                flat[f"{key}_{idx}"] = float(item)
    return flat
