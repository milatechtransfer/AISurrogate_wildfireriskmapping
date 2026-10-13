"""Tests for src/full_map/compute_hazard_confusion_matrix.py: confusion matrix / ordinal
accuracy stats accumulated over row-block windows from two hazard class rasters."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.full_map.compute_hazard_confusion_matrix import accumulate_confusion_matrix


def _write_class_tif(path: Path, array: np.ndarray, transform, crs="EPSG:3978", nodata: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "int32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("int32"), 1)


def test_accumulate_confusion_matrix_matches_direct_computation(tmp_path):
    from src.datasets.postprocessing.hazard_metrics import build_confusion_matrix

    transform = from_origin(0, 4, 1, 1)
    rng = np.random.default_rng(0)
    gt = rng.integers(1, 4, size=(4, 5)).astype("int32")
    pred = rng.integers(1, 4, size=(4, 5)).astype("int32")

    gt_path = tmp_path / "gt.tif"
    pred_path = tmp_path / "pred.tif"
    _write_class_tif(gt_path, gt, transform)
    _write_class_tif(pred_path, pred, transform)

    # block_rows=2 forces multiple windows, exercising the accumulation path.
    confusion = accumulate_confusion_matrix(str(pred_path), str(gt_path), num_classes=3, block_rows=2)
    expected = build_confusion_matrix(pred, gt, num_classes=3)
    np.testing.assert_array_equal(confusion, expected)


def test_accumulate_confusion_matrix_ignores_invalid_class(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    gt = np.array([[1, 2], [0, 1]], dtype="int32")
    pred = np.array([[1, 2], [2, 1]], dtype="int32")

    gt_path = tmp_path / "gt.tif"
    pred_path = tmp_path / "pred.tif"
    _write_class_tif(gt_path, gt, transform)
    _write_class_tif(pred_path, pred, transform)

    confusion = accumulate_confusion_matrix(str(pred_path), str(gt_path), invalid_class=0, num_classes=2, block_rows=1)
    assert confusion.sum() == 3  # the (0, ...) pixel is ignored


def test_accumulate_confusion_matrix_shape_mismatch_raises(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    gt_path = tmp_path / "gt.tif"
    pred_path = tmp_path / "pred.tif"
    _write_class_tif(gt_path, np.array([[1, 2]], dtype="int32"), transform)
    _write_class_tif(pred_path, np.array([[1, 2], [1, 2]], dtype="int32"), transform)

    with pytest.raises(ValueError, match="shapes must match"):
        accumulate_confusion_matrix(str(pred_path), str(gt_path))
