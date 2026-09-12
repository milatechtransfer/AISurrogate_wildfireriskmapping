"""Tests for src/full_map/: real-CRS mosaicking, national diff
comparison, and per-split wide-format hexel metrics reshaping."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.full_map.generate_full_hexel_diff_map import compute_national_diff, generate_national_diffs
from src.full_map.generate_full_hexel_map import generate_national_mosaics
from src.full_map.generate_predictions import reshape_hexel_metrics_to_wide
from src.full_map.utils import group_predicted_hexel_files_by_target, load_hexel_shapefile, mosaic_predicted_hexels


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


def test_reshape_hexel_metrics_to_wide_builds_one_row_per_hexel():
    hexel_metrics = {
        "hex12/mse": 0.1,
        "hex12/bp_mse": 0.2,
        "hex7/mse": 0.3,
        "hex7/bp_mse": 0.4,
        "all/mse": 0.2,  # aggregate row -- should be dropped
    }

    df = reshape_hexel_metrics_to_wide(hexel_metrics)

    assert set(df["hex_id"]) == {"12", "7"}
    assert list(df.columns) == ["hex_id", "bp_mse", "mse"]
    row12 = df[df["hex_id"] == "12"].iloc[0]
    assert row12["mse"] == pytest.approx(0.1)
    assert row12["bp_mse"] == pytest.approx(0.2)


def test_reshape_hexel_metrics_to_wide_handles_empty_input():
    df = reshape_hexel_metrics_to_wide({})
    assert list(df.columns) == ["hex_id"]
    assert len(df) == 0


def test_load_hexel_shapefile_requires_configured_id_column(tmp_path):
    shp_path = tmp_path / "hexels.shp"
    gdf = gpd.GeoDataFrame({"other_col": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3978")
    gdf.to_file(shp_path)

    with pytest.raises(KeyError):
        load_hexel_shapefile(str(shp_path), hexel_id_column="hex_id")


def test_load_hexel_shapefile_normalizes_id_column_to_string(tmp_path):
    shp_path = tmp_path / "hexels.shp"
    gdf = gpd.GeoDataFrame({"hex_id": [12, 7]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs="EPSG:3978")
    gdf.to_file(shp_path)

    loaded = load_hexel_shapefile(str(shp_path), hexel_id_column="hex_id")

    assert set(loaded["hex_id"]) == {"12", "7"}


def test_group_predicted_hexel_files_by_target_parses_target_suffix(tmp_path):
    pred_dir = tmp_path / "predictions"
    (pred_dir / "predicted_hexels").mkdir(parents=True)
    for name in ["hexel_12_bp_predicted.tif", "hexel_12_fi_predicted.tif", "hexel_7_bp_predicted.tif", "hexel_7_ros_predicted.tif"]:
        (pred_dir / "predicted_hexels" / name).touch()

    grouped = group_predicted_hexel_files_by_target(pred_dir, "*/hexel_*_predicted.tif")

    assert set(grouped.keys()) == {"bp", "fi", "ros"}
    assert set(grouped["bp"].keys()) == {12, 7}
    assert set(grouped["fi"].keys()) == {12}
    assert set(grouped["ros"].keys()) == {7}


def test_group_predicted_hexel_files_by_target_defaults_key_for_single_target_models(tmp_path):
    pred_dir = tmp_path / "predictions"
    (pred_dir / "predicted_hexels").mkdir(parents=True)
    (pred_dir / "predicted_hexels" / "hexel_12_predicted.tif").touch()

    grouped = group_predicted_hexel_files_by_target(pred_dir, "*/hexel_*_predicted.tif")

    assert set(grouped.keys()) == {"default"}
    assert set(grouped["default"].keys()) == {12}


def test_mosaic_predicted_hexels_pastes_hexels_onto_reference_grid(tmp_path):
    # Reference (national GT) raster: a 4x4 canvas, all nodata to start.
    ref_transform = from_origin(0, 4, 1, 1)
    ref_array = np.full((4, 4), -9999.0, dtype="float32")
    ref_path = tmp_path / "national_gt.tif"
    _write_tif(ref_path, ref_array, ref_transform)

    # Two predicted hexels, each covering a distinct 2x2 quadrant of the reference grid, on the
    # same CRS/resolution (so reprojection is a simple identity placement).
    pred_dir = tmp_path / "predictions"
    hex12_transform = from_origin(0, 4, 1, 1)  # top-left quadrant
    hex12_array = np.full((2, 2), 0.5, dtype="float32")
    hex12_path = pred_dir / "train" / "predicted_hexels" / "hexel_12_predicted.tif"
    _write_tif(hex12_path, hex12_array, hex12_transform)

    hex7_transform = from_origin(2, 2, 1, 1)  # bottom-right quadrant
    hex7_array = np.full((2, 2), 0.8, dtype="float32")
    hex7_path = pred_dir / "test" / "predicted_hexels" / "hexel_7_predicted.tif"
    _write_tif(hex7_path, hex7_array, hex7_transform)

    shapefile_gdf = gpd.GeoDataFrame(
        {"hex_id": ["12", "7"]},
        geometry=[box(0, 2, 2, 4), box(2, 0, 4, 2)],
        crs="EPSG:3978",
    )

    mosaic, profile = mosaic_predicted_hexels(
        file_map={12: hex12_path, 7: hex7_path},
        shapefile_gdf=shapefile_gdf,
        hexel_id_column="hex_id",
        reference_raster_path=str(ref_path),
    )

    assert mosaic.shape == (4, 4)
    assert profile["nodata"] == -9999.0
    # top-left quadrant pasted from hex 12
    assert np.allclose(mosaic[0:2, 0:2], 0.5)
    # bottom-right quadrant pasted from hex 7
    assert np.allclose(mosaic[2:4, 2:4], 0.8)
    # untouched region remains nodata
    assert np.allclose(mosaic[0:2, 2:4], -9999.0)


def test_mosaic_predicted_hexels_raises_when_file_map_empty(tmp_path):
    ref_path = tmp_path / "national_gt.tif"
    _write_tif(ref_path, np.full((2, 2), -9999.0, dtype="float32"), from_origin(0, 2, 1, 1))
    shapefile_gdf = gpd.GeoDataFrame({"hex_id": ["1"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3978")

    with pytest.raises(ValueError, match="empty"):
        mosaic_predicted_hexels(
            file_map={},
            shapefile_gdf=shapefile_gdf,
            hexel_id_column="hex_id",
            reference_raster_path=str(ref_path),
        )


def test_generate_national_mosaics_produces_one_tif_per_target(tmp_path):
    # Two independent 2x2 national reference grids, one per target.
    bp_ref_path = tmp_path / "bp_gt.tif"
    fi_ref_path = tmp_path / "fi_gt.tif"
    _write_tif(bp_ref_path, np.full((2, 2), -9999.0, dtype="float32"), from_origin(0, 2, 1, 1))
    _write_tif(fi_ref_path, np.full((2, 2), -9999.0, dtype="float32"), from_origin(0, 2, 1, 1))

    pred_dir = tmp_path / "predictions"
    _write_tif(
        pred_dir / "test" / "predicted_hexels" / "hexel_12_bp_predicted.tif", np.full((2, 2), 0.5, dtype="float32"), from_origin(0, 2, 1, 1)
    )
    _write_tif(
        pred_dir / "test" / "predicted_hexels" / "hexel_12_fi_predicted.tif",
        np.full((2, 2), 10.0, dtype="float32"),
        from_origin(0, 2, 1, 1),
    )

    shapefile_gdf_path = tmp_path / "hexels.shp"
    gpd.GeoDataFrame({"hex_id": ["12"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:3978").to_file(shapefile_gdf_path)

    output_dir = tmp_path / "national_mosaics"
    results = generate_national_mosaics(
        pred_root=str(pred_dir),
        pattern="*/predicted_hexels/hexel_*_predicted.tif",
        shapefile_path=str(shapefile_gdf_path),
        hexel_id_column="hex_id",
        reference_raster_paths={"bp": str(bp_ref_path), "fi": str(fi_ref_path)},
        output_dir=str(output_dir),
    )

    assert set(results.keys()) == {"bp", "fi"}
    assert (output_dir / "bp_national_predicted_map.tif").exists()
    assert (output_dir / "fi_national_predicted_map.tif").exists()
    bp_mosaic, _ = results["bp"]
    fi_mosaic, _ = results["fi"]
    assert np.allclose(bp_mosaic, 0.5)
    assert np.allclose(fi_mosaic, 10.0)


def test_mosaic_predicted_hexels_raises_when_no_files_found(tmp_path):
    ref_path = tmp_path / "national_gt.tif"
    _write_tif(ref_path, np.full((2, 2), -9999.0, dtype="float32"), from_origin(0, 2, 1, 1))
    shapefile_gdf_path = tmp_path / "hexels.shp"
    gpd.GeoDataFrame({"hex_id": ["1"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3978").to_file(shapefile_gdf_path)

    with pytest.raises(FileNotFoundError):
        generate_national_mosaics(
            pred_root=str(tmp_path / "empty"),
            pattern="*.tif",
            shapefile_path=str(shapefile_gdf_path),
            hexel_id_column="hex_id",
            reference_raster_paths={"bp": str(ref_path)},
            output_dir=str(tmp_path / "out"),
        )


def test_compute_national_diff_computes_pixelwise_error_over_valid_region(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    pred_array = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32")
    gt_array = np.array([[0.5, 1.5], [2.5, 4.0]], dtype="float32")

    pred_path = tmp_path / "pred_mosaic.tif"
    gt_path = tmp_path / "national_gt.tif"
    _write_tif(pred_path, pred_array, transform)
    _write_tif(gt_path, gt_array, transform)

    diff, profile, metrics = compute_national_diff(str(pred_path), str(gt_path))

    # only 3 of 4 pixels are valid in both rasters (the 4th is nodata in the prediction)
    assert metrics["n_valid_pixels"] == 3
    assert diff[0, 0] == pytest.approx(0.5)
    assert diff[0, 1] == pytest.approx(0.5)
    assert diff[1, 0] == pytest.approx(0.5)
    assert diff.mask[1, 1]
    assert metrics["normalized_mae"] == pytest.approx(0.5 / np.mean([0.5, 1.5, 2.5]))
    assert "ccc" in metrics
    assert "spearman" in metrics
    assert profile["nodata"] == -9999.0


def test_generate_national_diffs_produces_one_result_per_target(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    mosaic_dir = tmp_path / "mosaics"
    _write_tif(mosaic_dir / "bp_national_predicted_map.tif", np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), transform)
    _write_tif(mosaic_dir / "fi_national_predicted_map.tif", np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"), transform)

    gt_dir = tmp_path / "gt"
    bp_gt_path = gt_dir / "bp_gt.tif"
    fi_gt_path = gt_dir / "fi_gt.tif"
    _write_tif(bp_gt_path, np.array([[0.5, 1.5], [2.5, 3.5]], dtype="float32"), transform)
    _write_tif(fi_gt_path, np.array([[9.0, 19.0], [29.0, 39.0]], dtype="float32"), transform)

    output_dir = tmp_path / "diffs"
    all_metrics = generate_national_diffs(
        mosaic_dir=str(mosaic_dir),
        national_gt_raster_paths={"bp": str(bp_gt_path), "fi": str(fi_gt_path)},
        output_dir=str(output_dir),
    )

    assert set(all_metrics.keys()) == {"bp", "fi"}
    assert all_metrics["bp"]["normalized_mae"] == pytest.approx(0.5 / np.mean([0.5, 1.5, 2.5, 3.5]))
    assert all_metrics["fi"]["normalized_mae"] == pytest.approx(1.0 / np.mean([9.0, 19.0, 29.0, 39.0]))
    assert (output_dir / "bp_national_diff_map.tif").exists()
    assert (output_dir / "fi_national_diff_map.tif").exists()


def test_generate_national_diffs_skips_targets_with_missing_mosaic(tmp_path):
    transform = from_origin(0, 2, 1, 1)
    mosaic_dir = tmp_path / "mosaics"
    _write_tif(mosaic_dir / "bp_national_predicted_map.tif", np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), transform)

    gt_dir = tmp_path / "gt"
    bp_gt_path = gt_dir / "bp_gt.tif"
    fi_gt_path = gt_dir / "fi_gt.tif"
    _write_tif(bp_gt_path, np.array([[0.5, 1.5], [2.5, 3.5]], dtype="float32"), transform)
    _write_tif(fi_gt_path, np.array([[9.0, 19.0], [29.0, 39.0]], dtype="float32"), transform)

    all_metrics = generate_national_diffs(
        mosaic_dir=str(mosaic_dir),
        national_gt_raster_paths={"bp": str(bp_gt_path), "fi": str(fi_gt_path)},
        output_dir=str(tmp_path / "diffs"),
    )

    # "fi" has no predicted mosaic on disk, so it's skipped rather than raising.
    assert set(all_metrics.keys()) == {"bp"}
