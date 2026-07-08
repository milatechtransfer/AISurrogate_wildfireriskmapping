from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual import encoded_components_from_from_bearing
from src.datasets.postprocessing.counterfactual_weather import (
    WindEncodingStats,
    raw_wind_features,
    recover_wind_encoding_stats,
)
from src.datasets.postprocessing.counterfactual_wind_direction import (
    apply_wind_direction_scenario,
    dominant_from_bearing,
    uniform_direction_edit,
    zone_uniform_direction_edit,
)


def _build_weather(speeds: list[float], directions: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, WindEncodingStats]:
    raw = pd.DataFrame(
        {
            "Order": range(len(speeds)),
            "Season": "summer",
            "WeatherZone": 1,
            "WindSpeed": [float(s) for s in speeds],
            "WindDirection": [float(d) for d in directions],
        }
    )
    raw_features = raw_wind_features(raw)
    processed = pd.DataFrame(
        {
            "Order": raw["Order"].to_numpy(),
            "Season": raw["Season"].to_numpy(),
            "WeatherZone": raw["WeatherZone"].to_numpy(),
            "WindSpeed": (raw_features["WindSpeed"].to_numpy() - 8.0) / 4.0,
            "wind_x": (raw_features["wind_x"].to_numpy() - 1.0) / 3.0,
            "wind_y": (raw_features["wind_y"].to_numpy() + 2.0) / 5.0,
            "WindDirection": raw["WindDirection"].to_numpy(),
        }
    )
    stats = recover_wind_encoding_stats(raw_features, processed)
    return raw, processed, stats


def _reconstruct_raw(processed_edited: pd.DataFrame, stats: WindEncodingStats, column: str) -> np.ndarray:
    stat = stats.by_name()[column]
    return processed_edited[column].to_numpy(dtype=np.float64) * stat.std + stat.mean


def test_dominant_from_bearing_returns_constant_direction() -> None:
    speed = np.array([2.0, 9.0, 30.0])
    direction = np.array([270.0, 270.0, 270.0])
    assert dominant_from_bearing(speed, direction) == pytest.approx(270.0)


def test_dominant_from_bearing_is_speed_weighted() -> None:
    speed = np.array([1.0, 100.0])
    direction = np.array([0.0, 90.0])
    assert dominant_from_bearing(speed, direction) == pytest.approx(89.4, abs=0.5)


def test_uniform_direction_sets_target_and_reencodes_components() -> None:
    raw, processed, stats = _build_weather([4.0, 10.0, 16.0, 22.0], [10.0, 120.0, 210.0, 300.0])
    edited, _ = uniform_direction_edit(raw, processed, stats, from_bearing_deg=123.0)

    assert np.allclose(edited["WindDirection"], 123.0)
    speed = raw["WindSpeed"].to_numpy(dtype=np.float64)
    expected_x, expected_y = encoded_components_from_from_bearing(speed, 123.0)
    assert np.allclose(_reconstruct_raw(edited, stats, "wind_x"), expected_x)
    assert np.allclose(_reconstruct_raw(edited, stats, "wind_y"), expected_y)


def test_uniform_direction_keeps_wind_speed() -> None:
    raw, processed, stats = _build_weather([4.0, 10.0, 16.0, 22.0], [10.0, 120.0, 210.0, 300.0])
    edited, _ = uniform_direction_edit(raw, processed, stats, from_bearing_deg=42.0)
    assert np.allclose(edited["WindSpeed"], processed["WindSpeed"])


def test_zone_uniform_direction_uses_zone_dominant_bearings() -> None:
    raw, processed, stats = _build_weather([1.0, 100.0, 50.0, 1.0], [0.0, 90.0, 180.0, 270.0])
    raw["WeatherZone"] = [1, 1, 2, 2]
    processed["WeatherZone"] = raw["WeatherZone"]

    edited, report = zone_uniform_direction_edit(raw, processed, stats)

    zone_1_bearing = float(report.loc[report["zone"] == 1, "from_bearing_deg"].iloc[0])
    zone_2_bearing = float(report.loc[report["zone"] == 2, "from_bearing_deg"].iloc[0])
    assert zone_1_bearing == pytest.approx(dominant_from_bearing(np.array([1.0, 100.0]), np.array([0.0, 90.0])))
    assert zone_2_bearing == pytest.approx(dominant_from_bearing(np.array([50.0, 1.0]), np.array([180.0, 270.0])))
    assert np.allclose(edited.loc[raw["WeatherZone"] == 1, "WindDirection"], zone_1_bearing)
    assert np.allclose(edited.loc[raw["WeatherZone"] == 2, "WindDirection"], zone_2_bearing)
    assert np.allclose(edited["WindSpeed"], processed["WindSpeed"])


def test_zone_uniform_direction_applies_opposite_offset_per_zone() -> None:
    raw, processed, stats = _build_weather([1.0, 100.0, 50.0, 1.0], [0.0, 90.0, 180.0, 270.0])
    raw["WeatherZone"] = [1, 1, 2, 2]
    processed["WeatherZone"] = raw["WeatherZone"]

    dominant, dominant_report = zone_uniform_direction_edit(raw, processed, stats)
    opposite, opposite_report = zone_uniform_direction_edit(raw, processed, stats, offset_deg=180.0)

    merged = dominant_report.merge(opposite_report, on="zone", suffixes=("_dominant", "_opposite"))
    assert np.allclose((merged["from_bearing_deg_opposite"] - merged["from_bearing_deg_dominant"]) % 360.0, 180.0)
    assert np.allclose(_reconstruct_raw(opposite, stats, "wind_x"), -_reconstruct_raw(dominant, stats, "wind_x"))
    assert np.allclose(_reconstruct_raw(opposite, stats, "wind_y"), -_reconstruct_raw(dominant, stats, "wind_y"))


def test_opposite_offset_flips_raw_components_and_bearing() -> None:
    raw, processed, stats = _build_weather([4.0, 10.0, 16.0, 22.0], [10.0, 120.0, 210.0, 300.0])
    base, base_report = uniform_direction_edit(raw, processed, stats, offset_deg=0.0)
    opposite, opposite_report = uniform_direction_edit(raw, processed, stats, offset_deg=180.0)

    base_bearing = float(base_report["from_bearing_deg"].iloc[0])
    opposite_bearing = float(opposite_report["from_bearing_deg"].iloc[0])
    assert (opposite_bearing - base_bearing) % 360.0 == pytest.approx(180.0)
    assert base_report["dominant_from_bearing_deg"].iloc[0] == pytest.approx(opposite_report["dominant_from_bearing_deg"].iloc[0])

    base_x = _reconstruct_raw(base, stats, "wind_x")
    opposite_x = _reconstruct_raw(opposite, stats, "wind_x")
    assert np.allclose(opposite_x, -base_x)
    assert np.allclose(_reconstruct_raw(opposite, stats, "wind_y"), -_reconstruct_raw(base, stats, "wind_y"))


def test_apply_wind_direction_scenario_dispatches_and_rejects_unknown_mode() -> None:
    raw, processed, stats = _build_weather([4.0, 10.0, 16.0], [10.0, 120.0, 210.0])
    edited, report = apply_wind_direction_scenario(raw, processed, stats, {"mode": "uniform_direction", "offset_deg": 180.0})
    assert not edited.empty
    assert not report.empty
    edited, report = apply_wind_direction_scenario(raw, processed, stats, {"mode": "zone_uniform_direction"})
    assert not edited.empty
    assert set(report["zone"]) == {1}
    with pytest.raises(ValueError):
        apply_wind_direction_scenario(raw, processed, stats, {"mode": "nope"})
