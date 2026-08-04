from __future__ import annotations

import numpy as np
import pytest

from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    abs_share_at,
    build_endpoint_response,
    cumulative_abs_share,
    downsample_for_display,
    finite_values,
    pixel_fraction_for_share,
    prediction_raster_path,
    restrict_to_support,
    symmetric_percentile_limit,
    values_and_valid,
)


def test_prediction_raster_path_prefers_target_suffixed_file_when_present(tmp_path) -> None:
    predicted_dir = tmp_path / "predicted_hexels"
    predicted_dir.mkdir()
    (predicted_dir / "hexel_01_bp_predicted.tif").touch()
    (predicted_dir / "hexel_01_fi_predicted.tif").touch()

    assert prediction_raster_path(tmp_path, "1", target_name="bp") == predicted_dir / "hexel_01_bp_predicted.tif"
    assert prediction_raster_path(tmp_path, "1", target_name="fi") == predicted_dir / "hexel_01_fi_predicted.tif"


def test_prediction_raster_path_falls_back_to_unsuffixed_file(tmp_path) -> None:
    predicted_dir = tmp_path / "predicted_hexels"
    predicted_dir.mkdir()
    (predicted_dir / "hexel_01_predicted.tif").touch()

    # A single-output model's directory has no target-suffixed file, so requesting a
    # target name still resolves to the legacy unsuffixed raster.
    assert prediction_raster_path(tmp_path, "1", target_name="bp") == predicted_dir / "hexel_01_predicted.tif"
    assert prediction_raster_path(tmp_path, "1") == predicted_dir / "hexel_01_predicted.tif"


def test_build_endpoint_response_handles_support_added_and_removed() -> None:
    baseline = np.array([5.0, 7.0, 9.0])
    scenario = np.array([6.0, 8.0, 10.0])
    response = build_endpoint_response(
        baseline,
        scenario,
        baseline_support=np.array([True, False, True]),
        scenario_support=np.array([True, True, False]),
    )

    np.testing.assert_allclose(response.baseline.filled(np.nan), [5.0, np.nan, 9.0], equal_nan=True)
    np.testing.assert_allclose(response.scenario.filled(np.nan), [6.0, 8.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(response.delta.filled(np.nan), [1.0, 8.0, -9.0])


def test_finite_values_drops_nan_and_masked() -> None:
    data = np.ma.masked_array(np.array([1.0, np.nan, 3.0]), mask=[False, False, True])
    assert finite_values(data).tolist() == [1.0]


def test_values_and_valid_reports_finite_mask() -> None:
    data = np.ma.masked_array(np.array([1.0, np.nan]), mask=[False, False])
    values, valid = values_and_valid(data)
    assert valid.tolist() == [True, False]
    assert values[0] == pytest.approx(1.0)


def test_restrict_to_support_masks_outside_support() -> None:
    data = np.ma.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    support = np.array([[True, False], [False, True]])
    restricted = restrict_to_support(data, support)
    assert restricted.mask.tolist() == [[False, True], [True, False]]
    assert restricted.filled(np.nan)[0, 0] == pytest.approx(1.0)


def test_symmetric_percentile_limit_uses_absolute_pooled_deltas() -> None:
    deltas = [
        np.ma.asarray(np.array([-1.0, 2.0, np.nan])),
        np.ma.asarray(np.array([4.0, -8.0])),
    ]
    assert symmetric_percentile_limit(deltas, percentile=100.0) == pytest.approx(8.0)


def test_symmetric_percentile_limit_defaults_when_empty() -> None:
    assert symmetric_percentile_limit([], percentile=99.5) == pytest.approx(1.0)


def test_downsample_for_display_preserves_raster_edges() -> None:
    raster = np.arange(20).reshape(4, 5)
    display = downsample_for_display(raster, 2)
    assert display.tolist() == [[0, 2, 4], [10, 12, 14], [15, 17, 19]]


def test_downsample_for_display_no_op_below_factor_two() -> None:
    raster = np.arange(6).reshape(2, 3)
    assert downsample_for_display(raster, 1).tolist() == raster.tolist()


def test_cumulative_abs_share_uniform_delta_tracks_diagonal() -> None:
    delta = np.ma.masked_array(np.array([2.0, -2.0, 2.0, -2.0]), mask=False)
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert pixel_fraction.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert cumulative.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cumulative_abs_share_concentrated_delta_bows_to_top_left() -> None:
    delta = np.ma.masked_array(np.array([100.0, 1.0, 1.0, 1.0, 1.0]), mask=False)
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert abs_share_at(pixel_fraction, cumulative, 0.2) == pytest.approx(100.0 / 104.0)


def test_cumulative_abs_share_ignores_masked_and_handles_empty() -> None:
    delta = np.ma.masked_array(np.array([5.0, np.nan, -3.0]), mask=[False, True, False])
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert cumulative[-1] == pytest.approx(1.0)
    assert abs_share_at(pixel_fraction, cumulative, 0.5) == pytest.approx(5.0 / 8.0)
    empty_fraction, empty_cumulative = cumulative_abs_share(np.ma.masked_array(np.array([np.nan]), mask=[True]))
    assert abs_share_at(empty_fraction, empty_cumulative, 0.1) == pytest.approx(0.0)


def test_cumulative_abs_share_handles_all_zero_delta() -> None:
    pixel_fraction, cumulative = cumulative_abs_share(np.zeros(3))
    assert cumulative.tolist() == [0.0, 0.0, 0.0]
    assert pixel_fraction_for_share(pixel_fraction, cumulative, 0.5) == pytest.approx(1.0)


def test_pixel_fraction_for_share_inverts_cumulative_abs_share() -> None:
    delta = np.array([9.0, 1.0, 0.0, 0.0])
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert pixel_fraction_for_share(pixel_fraction, cumulative, 0.80) == pytest.approx(0.25)
