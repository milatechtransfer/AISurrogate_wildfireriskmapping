from __future__ import annotations

import numpy as np
import pytest

from src.datasets.postprocessing.counterfactual.plotting.counterfactual_local_zoom_panels import (
    NeighborhoodWindow,
    _effective_mean_on_crop,
    _hazard_response,
    _response_direction,
    candidate_starts,
    integral_image,
    select_neighborhood_windows,
    window_sum,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import build_endpoint_response


def test_integral_image_and_window_sum_match_direct_sum() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    integral = integral_image(values)
    assert window_sum(integral, 0, 3, 0, 4) == values.sum()
    assert window_sum(integral, 1, 3, 1, 3) == values[1:3, 1:3].sum()


def test_candidate_starts_always_includes_final_position() -> None:
    starts = candidate_starts(length=20, crop_size=6, stride=5)
    assert starts[0] == 0
    assert starts[-1] == 14
    assert all(0 <= start <= 14 for start in starts)


def test_candidate_starts_returns_zero_when_crop_exceeds_length() -> None:
    assert candidate_starts(length=5, crop_size=10, stride=2) == [0]


def test_select_neighborhood_windows_uses_positive_direction_aligned_response() -> None:
    edit_mask = np.zeros((30, 30), dtype=bool)
    edit_mask[2:8, 2:8] = True
    edit_mask[16:22, 16:22] = True
    valid = np.ones_like(edit_mask, dtype=bool)
    delta_hazard = np.ones((30, 30), dtype=float)
    delta_fi = np.ones((30, 30), dtype=float)
    delta_hazard[15:25, 15:25] = 5.0
    delta_fi[15:25, 15:25] = 10.0

    windows = select_neighborhood_windows(
        edit_mask=edit_mask,
        valid_mask=valid,
        delta_hazard=delta_hazard,
        delta_fi=delta_fi,
        response_direction=1,
        n_windows=1,
        crop_size=10,
        stride=5,
        min_edit_pixels=10,
        min_edit_density=0.05,
        max_edit_density=0.8,
        exclude_border_pixels=0,
    )
    assert len(windows) == 1
    window = windows[0]
    assert window.row_min < 22
    assert window.row_max > 16
    assert window.col_min < 22
    assert window.col_max > 16
    assert window.delta_hazard_mean > 1.0


def test_select_neighborhood_windows_uses_negative_direction_aligned_response() -> None:
    edit_mask = np.zeros((20, 20), dtype=bool)
    edit_mask[5:15, 5:15] = True
    valid = np.ones_like(edit_mask, dtype=bool)
    delta = -np.ones((20, 20), dtype=float)

    windows = select_neighborhood_windows(
        edit_mask=edit_mask,
        valid_mask=valid,
        delta_hazard=delta,
        delta_fi=delta,
        response_direction=-1,
        n_windows=1,
        crop_size=10,
        stride=5,
        min_edit_pixels=10,
        max_edit_density=1.0,
        exclude_border_pixels=0,
    )

    assert windows[0].delta_hazard_mean < 0.0
    assert windows[0].score > 0.0


def test_select_neighborhood_windows_uses_absolute_response_without_expected_direction() -> None:
    edit_mask = np.zeros((30, 30), dtype=bool)
    edit_mask[2:8, 2:8] = True
    edit_mask[16:22, 16:22] = True
    valid = np.ones_like(edit_mask, dtype=bool)
    delta_hazard = np.ones((30, 30), dtype=float)
    delta_fi = np.ones((30, 30), dtype=float)
    delta_hazard[15:25, 15:25] = -5.0
    delta_fi[15:25, 15:25] = -10.0

    windows = select_neighborhood_windows(
        edit_mask=edit_mask,
        valid_mask=valid,
        delta_hazard=delta_hazard,
        delta_fi=delta_fi,
        response_direction=None,
        n_windows=1,
        crop_size=10,
        stride=5,
        min_edit_pixels=10,
        min_edit_density=0.05,
        max_edit_density=0.8,
        exclude_border_pixels=0,
    )

    assert windows[0].delta_hazard_mean < -1.0
    assert windows[0].score > 0.0


def test_select_neighborhood_windows_keeps_footprint_separate_from_response_support() -> None:
    edit_mask = np.zeros((20, 20), dtype=bool)
    edit_mask[5:15, 5:15] = True
    footprint = np.ones_like(edit_mask, dtype=bool)
    delta = np.full((20, 20), np.nan)
    delta[5:15, 5:15] = -1.0

    windows = select_neighborhood_windows(
        edit_mask=edit_mask,
        valid_mask=footprint,
        delta_hazard=delta,
        delta_fi=delta,
        response_direction=-1,
        n_windows=1,
        crop_size=10,
        stride=5,
        min_edit_pixels=10,
        max_edit_density=1.0,
        min_valid_fraction=0.95,
        exclude_border_pixels=0,
    )

    assert windows[0].valid_fraction == 1.0
    assert windows[0].delta_hazard_mean == -1.0


def test_select_neighborhood_windows_raises_without_candidates() -> None:
    edit_mask = np.zeros((10, 10), dtype=bool)
    valid = np.ones_like(edit_mask, dtype=bool)
    delta = np.ones((10, 10), dtype=float)
    with pytest.raises(ValueError, match="No neighborhood windows"):
        select_neighborhood_windows(
            edit_mask=edit_mask,
            valid_mask=valid,
            delta_hazard=delta,
            delta_fi=delta,
            response_direction=1,
            crop_size=5,
            stride=2,
            min_edit_pixels=1,
            exclude_border_pixels=0,
        )


def test_hazard_response_uses_zero_outside_each_scenarios_support() -> None:
    baseline_support = np.array([[True, False]])
    scenario_support = np.array([[True, True]])
    bp_response = build_endpoint_response(
        np.array([[2.0, 100.0]]),
        np.array([[3.0, 4.0]]),
        baseline_support=baseline_support,
        scenario_support=scenario_support,
    )
    fi_response = build_endpoint_response(
        np.array([[5.0, 100.0]]),
        np.array([[7.0, 6.0]]),
        baseline_support=baseline_support,
        scenario_support=scenario_support,
    )

    hazard_response = _hazard_response(bp_response, fi_response)

    np.testing.assert_allclose(hazard_response.baseline.filled(0.0), [[10.0, 0.0]])
    np.testing.assert_allclose(hazard_response.scenario.filled(0.0), [[21.0, 24.0]])
    np.testing.assert_allclose(hazard_response.delta.filled(0.0), [[11.0, 24.0]])


def test_response_direction_distinguishes_added_and_removed_support() -> None:
    edit_mask = np.array([[False, True]])
    assert _response_direction(edit_mask, np.array([[True, False]]), np.array([[True, True]])) == 1
    assert _response_direction(edit_mask, np.array([[True, True]]), np.array([[True, False]])) == -1


def test_response_direction_is_neutral_when_edit_preserves_support() -> None:
    edit_mask = np.array([[False, True]])
    support = np.array([[True, True]])
    assert _response_direction(edit_mask, support, support) is None


def test_effective_crop_means_share_response_support() -> None:
    baseline = np.ma.masked_array([[2.0, 99.0]], mask=[[False, True]])
    scenario = np.ma.masked_array([[4.0, 6.0]], mask=[[False, False]])
    response_support = np.array([[True, True]])
    window = NeighborhoodWindow(
        window_id=1,
        rank=1,
        row_min=0,
        row_max=1,
        col_min=0,
        col_max=2,
        edit_pixels=1,
        edit_density=0.5,
        valid_fraction=1.0,
        delta_hazard_mean=0.0,
        delta_fi_mean=0.0,
        score=1.0,
    )

    assert _effective_mean_on_crop(baseline, response_support, window) == 1.0
    assert _effective_mean_on_crop(scenario, response_support, window) == 5.0
