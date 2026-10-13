from __future__ import annotations

from pathlib import Path

import numpy as np

from src.counterfactual.plotting.counterfactual_response_maps import (
    ENDPOINT_SPECS,
    block_response,
    hotspot_centers,
    plot_patch_zoom,
)


def test_endpoint_specs_cover_bp_fi_ros() -> None:
    assert set(ENDPOINT_SPECS) == {"bp", "fi", "ros"}
    for spec in ENDPOINT_SPECS.values():
        assert spec.label and spec.units and spec.cmap


def test_block_response_pools_mean_absolute_delta() -> None:
    delta = np.ma.masked_array(
        np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        ),
        mask=False,
    )
    pooled = block_response(delta, block=2)
    assert pooled.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_block_response_ignores_masked_pixels() -> None:
    delta = np.ma.masked_array(
        np.array([[1.0, 1.0], [1.0, 5.0]]),
        mask=[[False, False], [False, True]],
    )
    pooled = block_response(delta, block=2)
    assert pooled.tolist() == [[1.0]]


def test_hotspot_centers_returns_distinct_suppressed_peaks() -> None:
    delta = np.ma.masked_array(np.zeros((20, 20)), mask=False)
    delta[2:4, 2:4] = 10.0
    delta[15:17, 15:17] = 8.0
    centers = hotspot_centers(delta, block=2, count=2, window=4)
    assert len(centers) == 2
    assert centers[0] != centers[1]


def test_hotspot_centers_stops_when_no_positive_response_left() -> None:
    delta = np.ma.masked_array(np.zeros((10, 10)), mask=False)
    centers = hotspot_centers(delta, block=2, count=3, window=4)
    assert centers == []


def test_plot_patch_zoom_skips_cleanly_when_no_positive_response(tmp_path: Path) -> None:
    ground_truth = np.ma.masked_array(np.zeros((10, 10)), mask=False)
    baseline = np.ma.masked_array(np.zeros((10, 10)), mask=False)
    scenario_values = np.ma.masked_array(np.zeros((10, 10)), mask=False)
    delta = np.ma.masked_array(np.zeros((10, 10)), mask=False)
    out_path = tmp_path / "patch_zoom.png"

    plot_patch_zoom(
        ground_truth,
        baseline,
        scenario_values,
        delta,
        endpoint="fi",
        out_path=out_path,
        scenario_label="scenario",
        suptitle="No response",
        window=4,
        hotspot_block=2,
        patch_count=3,
    )

    assert not out_path.exists()
