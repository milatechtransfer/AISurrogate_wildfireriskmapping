from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from src.aggregate_counterfactual_multirun_results import (
    aggregate_counterfactual_runs,
    ensemble_mean_std,
    extents_match,
    summarize_response_rows,
)


def test_ensemble_mean_std_uses_available_seed_values() -> None:
    first = np.ma.masked_invalid(np.array([[1.0, 2.0], [np.nan, 4.0]]))
    second = np.ma.masked_invalid(np.array([[3.0, 4.0], [5.0, np.nan]]))
    third = np.ma.masked_invalid(np.array([[5.0, 6.0], [7.0, 8.0]]))

    mean, std = ensemble_mean_std([first, second, third])

    np.testing.assert_allclose(mean.filled(np.nan), np.array([[3.0, 4.0], [6.0, 6.0]]))
    np.testing.assert_allclose(std.filled(np.nan), np.array([[2.0, 2.0], [np.sqrt(2.0), np.sqrt(8.0)]]))


def test_ensemble_mean_std_masks_std_with_only_one_seed_value() -> None:
    first = np.ma.masked_invalid(np.array([[1.0, np.nan]]))
    second = np.ma.masked_invalid(np.array([[np.nan, 2.0]]))

    mean, std = ensemble_mean_std([first, second])

    np.testing.assert_allclose(mean.filled(np.nan), np.array([[1.0, 2.0]]))
    assert std.mask.tolist() == [[True, True]]


def test_extents_match_ignores_submicron_raster_metadata_drift() -> None:
    first = (1545886.05749042, 1231516.6157504604, 1985792.2120705012, 1619527.6694258542)
    second = (1545886.0574904198, 1231516.6157504607, 1985792.2120705007, 1619527.6694258542)

    assert extents_match(first, second)
    assert not extents_match(first, (first[0], first[1], first[2] + 0.01, first[3]))


def test_summarize_response_rows_reports_across_seed_std() -> None:
    rows = pd.DataFrame(
        [
            {"seed": 42, "scenario": "fuel", "endpoint": "bp", "mean_delta": 1.0, "mean_absolute_delta": 2.0},
            {"seed": 1337, "scenario": "fuel", "endpoint": "bp", "mean_delta": 2.0, "mean_absolute_delta": 3.0},
            {"seed": 2024, "scenario": "fuel", "endpoint": "bp", "mean_delta": 3.0, "mean_absolute_delta": 4.0},
        ]
    )

    summary = summarize_response_rows(rows).iloc[0]

    assert summary["seed_count"] == 3
    assert summary["mean_delta"] == pytest.approx(2.0)
    assert summary["std_delta_across_seeds"] == pytest.approx(1.0)
    assert summary["mean_absolute_delta"] == pytest.approx(3.0)
    assert summary["std_absolute_delta_across_seeds"] == pytest.approx(1.0)


def test_summarize_response_rows_requires_seed_column() -> None:
    rows = pd.DataFrame(
        [
            {
                "scenario": "fuel",
                "endpoint": "bp",
                "mean_delta": 1.0,
                "mean_absolute_delta": 2.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="seed"):
        summarize_response_rows(rows)


def test_aggregate_counterfactual_runs_writes_mean_std_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "experiment",
                "hex_ids": ["16"],
                "endpoints": {endpoint: {"config_path": "model.yaml"} for endpoint in ("bp", "fi", "ros")},
                "scenarios": [
                    {"name": "baseline", "kind": "baseline"},
                    {"name": "climate", "kind": "weather"},
                ],
            },
            handle,
        )

    for run_id, seed in enumerate((42, 1337, 2024)):
        run_dir = tmp_path / "experiment" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "scenario_prediction_index.csv").write_text("scenario,endpoint,prediction_dir\n")
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "seed": seed,
                    "scenario": "climate",
                    "endpoint": "bp",
                    "metric": "mae",
                    "value": float(run_id + 1),
                }
            ]
        ).to_csv(run_dir / "counterfactual_metrics.csv", index=False)

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32610",
        "transform": from_origin(0, 200, 100, 100),
        "nodata": -9999.0,
    }

    def _fake_load_seed_delta(*, run, **_):
        value = float(run.run_id + 1)
        return np.ma.array(np.full((2, 2), value)), (0.0, 200.0, 0.0, 200.0), profile

    monkeypatch.setattr(
        "src.aggregate_counterfactual_multirun_results._load_seed_delta",
        _fake_load_seed_delta,
    )

    output_dir = aggregate_counterfactual_runs(
        config_path,
        scenario="climate",
        run_ids=[0, 1, 2],
        project_root=tmp_path,
        downsample=1,
    )

    response_summary = pd.read_csv(output_dir / "counterfactual_response_summary.csv")
    assert set(response_summary["endpoint"]) == {"bp", "fi", "ros"}
    assert set(response_summary["mean_delta"]) == {2.0}
    assert set(response_summary["std_delta_across_seeds"]) == {1.0}

    mean_path = output_dir / "rasters" / "hex16" / "climate_bp_delta_mean.tif"
    std_path = output_dir / "rasters" / "hex16" / "climate_bp_delta_std.tif"
    with rasterio.open(mean_path) as src:
        np.testing.assert_allclose(src.read(1, masked=True), np.full((2, 2), 2.0))
    with rasterio.open(std_path) as src:
        np.testing.assert_allclose(src.read(1, masked=True), np.full((2, 2), 1.0))

    assert (output_dir / "figures" / "hex16_climate_response_mean_std.png").is_file()
    assert (output_dir / "figures" / "hex16_climate_response_mean_std.pdf").is_file()
