from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.counterfactual.fuel_counterfactual_transform import fuel_intervention_raster_path
from src.counterfactual.plotting.counterfactual_change_distribution import (
    distance_bin_summary,
    prediction_response_change,
    summarize_change,
)


def test_change_summary_reports_concentrated_absolute_change() -> None:
    delta = np.array([[9.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    support = np.ones(delta.shape, dtype=bool)
    edit_mask = np.array([[True, False], [False, False]])

    summary = summarize_change(
        delta,
        support,
        edit_mask,
        scenario="fuel",
        endpoint="bp",
        hex_id="16",
    )

    assert summary.edited_pixel_abs_change_share == 0.9
    assert summary.top_50pct_abs_change_share == 1.0
    assert summary.pixel_share_for_80pct_abs_change == 0.25


def test_distance_summary_uses_requested_edit_distance_bins() -> None:
    delta = np.arange(9, dtype=np.float32).reshape(3, 3)
    support = np.ones(delta.shape, dtype=bool)
    edit_mask = np.zeros(delta.shape, dtype=bool)
    edit_mask[1, 1] = True

    summary = distance_bin_summary(
        delta,
        support,
        edit_mask,
        pixel_height_m=100.0,
        pixel_width_m=100.0,
    )

    assert summary["distance_bin"].tolist() == ["100-250 m"]
    assert summary["n_pixels"].tolist() == [8]


def test_prediction_response_change_uses_symmetric_fuel_support(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_dir = experiment_dir / "predictions" / "baseline" / "fi"
    scenario_dir = experiment_dir / "predictions" / "fuel" / "fi"
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 3,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1, 1),
        "nodata": -9999.0,
    }

    def write(path: Path, values: list[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.asarray([values], dtype=np.float32), 1)

    write(baseline_dir / "predicted_hexels" / "hexel_16_predicted.tif", [5.0, 7.0, 9.0])
    write(scenario_dir / "predicted_hexels" / "hexel_16_predicted.tif", [6.0, 8.0, 10.0])
    write(fuel_intervention_raster_path(scenario_dir, "16", "baseline"), [1.0, 101.0, 2.0])
    write(fuel_intervention_raster_path(scenario_dir, "16", "scenario"), [1.0, 2.0, 101.0])
    pd.DataFrame(
        [
            {"scenario": "baseline", "endpoint": "fi", "prediction_dir": baseline_dir},
            {"scenario": "fuel", "endpoint": "fi", "prediction_dir": scenario_dir},
        ]
    ).to_csv(experiment_dir / "scenario_prediction_index.csv", index=False)

    delta, support, _ = prediction_response_change(
        experiment_dir,
        scenario="fuel",
        endpoint="fi",
        hex_id="16",
        nonfuel_ids=[101],
    )

    assert support.tolist() == [[True, True, True]]
    assert delta.tolist() == [[1.0, 8.0, -9.0]]
