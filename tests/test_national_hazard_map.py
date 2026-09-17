"""Tests for src/full_map/generate_national_hazard_map.py: hazard = BP x FI computed directly
from already-mosaicked national bp/fi rasters (both predicted and ground truth)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.full_map.generate_national_hazard_map import (
    GROUND_TRUTH_HAZARD_FILENAME,
    PREDICTED_HAZARD_FILENAME,
    generate_national_hazard_maps,
)


def _write_tif(path: Path, array: np.ndarray, transform, crs="EPSG:3978", nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


def test_generate_national_hazard_maps_writes_binned_rasters(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    mosaic_dir = tmp_path / "mosaics"
    gt_dir = tmp_path / "gt"

    _write_tif(mosaic_dir / "bp_national_predicted_map.tif", np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"), transform)
    _write_tif(mosaic_dir / "fi_national_predicted_map.tif", np.array([[100.0, 200.0], [300.0, 400.0]], dtype="float32"), transform)
    _write_tif(gt_dir / "bp_gt.tif", np.array([[0.1, 0.2], [0.3, 0.5]], dtype="float32"), transform)
    _write_tif(gt_dir / "fi_gt.tif", np.array([[100.0, 200.0], [300.0, 500.0]], dtype="float32"), transform)

    output_dir = tmp_path / "hazard"
    summary = generate_national_hazard_maps(
        predicted_bp_path=str(mosaic_dir / "bp_national_predicted_map.tif"),
        predicted_fi_path=str(mosaic_dir / "fi_national_predicted_map.tif"),
        gt_bp_path=str(gt_dir / "bp_gt.tif"),
        gt_fi_path=str(gt_dir / "fi_gt.tif"),
        output_dir=str(output_dir),
        bin_thresholds=[0.01, 0.1, 1.0, 10.0],
    )

    pred_path = output_dir / PREDICTED_HAZARD_FILENAME
    gt_path = output_dir / GROUND_TRUTH_HAZARD_FILENAME
    assert pred_path.exists()
    assert gt_path.exists()

    with rasterio.open(gt_path) as src:
        gt_binned = src.read(1)
        assert src.nodata == 0
    # GT raw hazard: [[10, 40], [90, 250]] -> denominator = max = 250 -> scaled *= 100/250
    # scaled = [[4, 16], [36, 100]] -> classes with edges [0.01,0.1,1,10]: 4 -> class 4, rest -> class 5
    assert np.array_equal(gt_binned, np.array([[4, 5], [5, 5]]))

    with rasterio.open(pred_path) as src:
        pred_binned = src.read(1)
    assert pred_binned.shape == gt_binned.shape

    assert summary["denominator"] == pytest.approx(250.0)
    assert "exact_accuracy" in summary["metrics"]


def test_generate_national_hazard_maps_gt_only(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    gt_dir = tmp_path / "gt"

    _write_tif(gt_dir / "bp_gt.tif", np.array([[0.1, 0.2], [0.3, 0.5]], dtype="float32"), transform)
    _write_tif(gt_dir / "fi_gt.tif", np.array([[100.0, 200.0], [300.0, 500.0]], dtype="float32"), transform)

    output_dir = tmp_path / "hazard"
    summary = generate_national_hazard_maps(
        gt_bp_path=str(gt_dir / "bp_gt.tif"),
        gt_fi_path=str(gt_dir / "fi_gt.tif"),
        output_dir=str(output_dir),
        bin_thresholds=[0.01, 0.1, 1.0, 10.0],
    )

    gt_path = output_dir / GROUND_TRUTH_HAZARD_FILENAME
    pred_path = output_dir / PREDICTED_HAZARD_FILENAME
    assert gt_path.exists()
    assert not pred_path.exists()
    assert summary["metrics"] is None
    assert summary["denominator"] == pytest.approx(250.0)


def test_generate_national_hazard_maps_uses_explicit_denominator(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    mosaic_dir = tmp_path / "mosaics"
    gt_dir = tmp_path / "gt"

    _write_tif(mosaic_dir / "bp_national_predicted_map.tif", np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"), transform)
    _write_tif(mosaic_dir / "fi_national_predicted_map.tif", np.array([[10.0, 10.0], [10.0, 10.0]], dtype="float32"), transform)
    _write_tif(gt_dir / "bp_gt.tif", np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"), transform)
    _write_tif(gt_dir / "fi_gt.tif", np.array([[10.0, 10.0], [10.0, 10.0]], dtype="float32"), transform)

    output_dir = tmp_path / "hazard"
    summary = generate_national_hazard_maps(
        predicted_bp_path=str(mosaic_dir / "bp_national_predicted_map.tif"),
        predicted_fi_path=str(mosaic_dir / "fi_national_predicted_map.tif"),
        gt_bp_path=str(gt_dir / "bp_gt.tif"),
        gt_fi_path=str(gt_dir / "fi_gt.tif"),
        output_dir=str(output_dir),
        scale_denominator=100.0,
    )

    assert summary["denominator"] == pytest.approx(100.0)


def test_generate_national_hazard_maps_rejects_mismatched_grids(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    mosaic_dir = tmp_path / "mosaics"
    gt_dir = tmp_path / "gt"

    _write_tif(mosaic_dir / "bp_national_predicted_map.tif", np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"), transform)
    # fi predicted mosaic has a different shape -> should raise.
    _write_tif(mosaic_dir / "fi_national_predicted_map.tif", np.array([[10.0, 10.0, 10.0]], dtype="float32"), from_origin(0, 1, 1, 1))
    _write_tif(gt_dir / "bp_gt.tif", np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"), transform)
    _write_tif(gt_dir / "fi_gt.tif", np.array([[10.0, 10.0], [10.0, 10.0]], dtype="float32"), transform)

    with pytest.raises(ValueError, match="shape"):
        generate_national_hazard_maps(
            predicted_bp_path=str(mosaic_dir / "bp_national_predicted_map.tif"),
            predicted_fi_path=str(mosaic_dir / "fi_national_predicted_map.tif"),
            gt_bp_path=str(gt_dir / "bp_gt.tif"),
            gt_fi_path=str(gt_dir / "fi_gt.tif"),
            output_dir=str(tmp_path / "hazard"),
        )


def test_generate_national_hazard_maps_skip_existing_does_not_recompute(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    mosaic_dir = tmp_path / "mosaics"
    gt_dir = tmp_path / "gt"

    _write_tif(mosaic_dir / "bp_national_predicted_map.tif", np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"), transform)
    _write_tif(mosaic_dir / "fi_national_predicted_map.tif", np.array([[10.0, 10.0], [10.0, 10.0]], dtype="float32"), transform)
    _write_tif(gt_dir / "bp_gt.tif", np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"), transform)
    _write_tif(gt_dir / "fi_gt.tif", np.array([[10.0, 10.0], [10.0, 10.0]], dtype="float32"), transform)

    output_dir = tmp_path / "hazard"
    generate_national_hazard_maps(
        predicted_bp_path=str(mosaic_dir / "bp_national_predicted_map.tif"),
        predicted_fi_path=str(mosaic_dir / "fi_national_predicted_map.tif"),
        gt_bp_path=str(gt_dir / "bp_gt.tif"),
        gt_fi_path=str(gt_dir / "fi_gt.tif"),
        output_dir=str(output_dir),
        scale_denominator=100.0,
    )
    pred_path = output_dir / PREDICTED_HAZARD_FILENAME
    original_mtime = pred_path.stat().st_mtime_ns

    generate_national_hazard_maps(
        predicted_bp_path=str(mosaic_dir / "bp_national_predicted_map.tif"),
        predicted_fi_path=str(mosaic_dir / "fi_national_predicted_map.tif"),
        gt_bp_path=str(gt_dir / "bp_gt.tif"),
        gt_fi_path=str(gt_dir / "fi_gt.tif"),
        output_dir=str(output_dir),
        scale_denominator=999.0,
        skip_existing=True,
    )
    assert pred_path.stat().st_mtime_ns == original_mtime
