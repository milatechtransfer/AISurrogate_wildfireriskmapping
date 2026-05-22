import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.utils import (
    get_hexel_binary_maps,
    get_modelling_approach_two_bp_ground_truth,
    get_predicted_hexel,
    get_stitched_windows,
)
from src.datasets.postprocessing.visualize_predictions import get_distribution_axis_limit
from src.datasets.utils import denormalize_output_target, output_target_norm


def test_get_stitched_windows_uses_target_channel_mask(tmp_path):
    patch = np.ones((2, 2, 7), dtype=np.float32)
    patch[:, :, 0] = 1.0
    patch[:, :, 5] = 1.0
    patch[0, 1, 5] = np.nan
    np.save(tmp_path / "sample.npy", patch)

    df = pd.DataFrame([["sample.npy", None, None, None, None, 0, 0]])
    predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    stitched = get_stitched_windows(
        base_dir=str(tmp_path),
        df=df,
        predictions=predictions,
        start_idx=0,
        gt_shape=(2, 2),
        target_channel_index=5,
        win_h=2,
        win_w=2,
    )

    assert stitched[0, 0] == 1.0
    assert np.isnan(stitched[0, 1])
    assert stitched[1, 0] == 3.0
    assert stitched[1, 1] == 4.0


def test_distribution_axis_limit_uses_non_probability_fallback():
    empty = np.array([], dtype=np.float32)

    assert get_distribution_axis_limit(empty, empty, probability_scale=True) == 0.15
    assert get_distribution_axis_limit(empty, empty, probability_scale=False) == 1.0


def test_distribution_axis_limit_only_caps_probability_scale():
    gt_vals = np.array([0.0, 2.0], dtype=np.float32)
    pred_vals = np.array([3.0], dtype=np.float32)

    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=True) == 1.0
    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=False) > 3.0


def test_log_standard_target_transform_roundtrip():
    raw = np.array([[0.0, 1.0, 9.0]], dtype=np.float32)
    mean = 1.25
    std = 0.5

    normalized = output_target_norm(
        output_arr=raw,
        target_max=10.0,
        target_min=0.0,
        out_norm="log_standard",
        target_log_mean=mean,
        target_log_std=std,
    )
    recovered = denormalize_output_target(
        data=normalized,
        target_min=0.0,
        target_max=10.0,
        out_norm="log_standard",
        target_log_mean=mean,
        target_log_std=std,
    )

    np.testing.assert_allclose(recovered, raw, rtol=1e-6, atol=1e-6)


def test_log_standard_target_transform_requires_stats():
    raw = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="target_log_mean"):
        output_target_norm(output_arr=raw, target_max=1.0, target_min=0.0, out_norm="log_standard")


def test_get_hexel_binary_maps_respects_masked_arrays(recwarn):
    pred = np.ma.array(
        [[1.0, 5.0], [100.0, 2.0]],
        mask=[[False, False], [True, False]],
        dtype=np.float32,
    )
    gt = np.ma.array(
        [[1.0, 4.0], [9.0, 3.0]],
        mask=[[False, False], [True, False]],
        dtype=np.float32,
    )

    pred_bin, gt_bin = get_hexel_binary_maps(pred_grid=pred, gt_grid=gt, percentile=0.5)

    assert not recwarn
    assert not pred_bin[1, 0]
    assert pred_bin[0, 1]
    assert pred_bin[1, 1]
    assert not gt_bin[1, 0]
    assert gt_bin[0, 1]
    assert gt_bin[1, 1]


def test_get_predicted_hexel_approach_two_groups_predictions_by_season(tmp_path, monkeypatch):
    patch = np.ones((2, 2, 1), dtype=np.float32)
    np.save(tmp_path / "season2.npy", patch)
    np.save(tmp_path / "season1.npy", patch)

    test_df = pd.DataFrame(
        {
            "filename": ["season2.npy", "season1.npy"],
            "season": [2, 1],
            "cause": ["all", "all"],
            "hex_id": ["01", "01"],
            "window_id": [1, 2],
            "row": [0, 0],
            "col": [0, 0],
            "valid_ratio": [1.0, 1.0],
        }
    )
    predictions = np.array(
        [
            [[2.0, 2.0], [2.0, 2.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        "src.datasets.postprocessing.utils.load_spatial_raster",
        lambda path, actual_mask_path=None, reference_profile=None: (
            np.ma.masked_array(np.zeros((2, 2), dtype=np.float32), mask=np.zeros((2, 2), dtype=bool)),
            {"dtype": "float32", "nodata": -9999, "height": 2, "width": 2},
        ),
    )
    monkeypatch.setattr(
        "src.datasets.postprocessing.utils.denormalize_burn_count",
        lambda data, min_val, max_val: data,
    )

    reconstructed, profile = get_predicted_hexel(
        base_dir=str(tmp_path),
        raw_data_dir=str(tmp_path),
        test_df=test_df,
        predictions=predictions,
        min_target_val=0.0,
        max_target_val=10.0,
        hex_id="01",
        modelling_approach="2",
        out_norm="total_iters",
        target_channel_index=0,
        win_h=2,
        win_w=2,
    )

    np.testing.assert_allclose(reconstructed, np.full((2, 2), 3, dtype=np.int32))
    assert profile["dtype"] == "int32"


def test_get_modelling_approach_two_bp_ground_truth_sums_seasonal_targets(monkeypatch):
    calls = []

    def fake_load_spatial_raster(path, actual_mask_path=None, reference_profile=None):
        calls.append(path.name)
        if path.name == "burnProbability-sn319.tif":
            return np.ma.masked_array(np.full((2, 2), 1.0, dtype=np.float32), mask=False), {"dtype": "float32"}
        if path.name == "burnProbability-sn320.tif":
            return np.ma.masked_array(np.full((2, 2), 2.0, dtype=np.float32), mask=False), {"dtype": "float32"}
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setattr("src.datasets.postprocessing.utils.load_spatial_raster", fake_load_spatial_raster)

    grid = get_modelling_approach_two_bp_ground_truth(
        raw_data_dir="/tmp/raw",
        hex_id="01",
        test_df=pd.DataFrame({"season": [1, 2, 1], "cause": ["all", "all", "all"]}),
        reference_profile={"dtype": "float32"},
    )

    np.testing.assert_allclose(grid.filled(np.nan), np.full((2, 2), 3.0, dtype=np.float32))
    assert calls == ["burnProbability-sn319.tif", "burnProbability-sn320.tif"]
