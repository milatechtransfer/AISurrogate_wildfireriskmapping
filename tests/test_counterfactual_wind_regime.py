from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual_weather import (
    WindEncodingStats,
    raw_wind_features,
    recover_wind_encoding_stats,
)
from src.datasets.postprocessing.counterfactual_wind_regime import (
    apply_wind_regime_scenario,
    zone_peak_wind_transplant,
)

DRIVER_COLUMNS = ("FireWeatherIndex", "Temperature", "WindSpeed", "wind_x", "wind_y", "WindDirection")


def _build_weather(zone_speeds: dict[int, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame, WindEncodingStats]:
    rows = []
    for zone, speeds in zone_speeds.items():
        for index, speed in enumerate(speeds):
            rows.append(
                {
                    "Order": index,
                    "Season": "summer",
                    "WeatherZone": zone,
                    "WindSpeed": float(speed),
                    "WindDirection": float((37 * (zone + 1) + 50 * index) % 360),
                    "FireWeatherIndex": 5.0 + 0.1 * index,
                    "Temperature": 12.0 + 0.2 * index,
                }
            )
    raw = pd.DataFrame(rows)
    raw_features = raw_wind_features(raw)
    processed = pd.DataFrame(
        {
            "Order": raw["Order"].to_numpy(),
            "Season": raw["Season"].to_numpy(),
            "WeatherZone": raw["WeatherZone"].to_numpy(),
            "FireWeatherIndex": (raw["FireWeatherIndex"].to_numpy() - 4.0) / 3.0,
            "Temperature": (raw["Temperature"].to_numpy() - 15.0) / 5.0,
            "WindSpeed": (raw_features["WindSpeed"].to_numpy() - 8.0) / 4.0,
            "wind_x": (raw_features["wind_x"].to_numpy() - 1.0) / 3.0,
            "wind_y": (raw_features["wind_y"].to_numpy() + 2.0) / 5.0,
            "WindDirection": raw["WindDirection"].to_numpy(),
        }
    )
    stats = recover_wind_encoding_stats(raw_features, processed)
    return raw, processed, stats


def test_transplant_makes_every_zone_row_identical_to_its_peak_day() -> None:
    raw, processed, _ = _build_weather({0: [2.0, 4.0, 20.0, 8.0], 1: [30.0, 3.0, 5.0, 7.0]})
    edited, _ = zone_peak_wind_transplant(raw, processed)
    for zone, donor_local in {0: 2, 1: 0}.items():
        zone_mask = processed["WeatherZone"].to_numpy() == zone
        donor_pos = int(np.flatnonzero(zone_mask)[donor_local])
        for column in DRIVER_COLUMNS:
            assert np.allclose(edited.loc[zone_mask, column], processed.loc[donor_pos, column])


def test_transplant_reports_peak_speed_as_scenario_mean() -> None:
    raw, processed, _ = _build_weather({0: [2.0, 4.0, 20.0, 8.0], 1: [30.0, 3.0, 5.0, 7.0]})
    _, report = zone_peak_wind_transplant(raw, processed)
    by_zone = report.set_index("zone")
    assert by_zone.loc[0, "scenario_speed_mean"] == pytest.approx(20.0)
    assert by_zone.loc[1, "scenario_speed_mean"] == pytest.approx(30.0)
    assert by_zone.loc[0, "baseline_speed_mean"] == pytest.approx(8.5)
    assert (report["scenario_speed_mean"] >= report["baseline_speed_mean"]).all()


def test_transplant_leaves_structural_columns_untouched() -> None:
    raw, processed, _ = _build_weather({0: [2.0, 4.0, 20.0, 8.0]})
    edited, _ = zone_peak_wind_transplant(raw, processed)
    assert edited["Order"].tolist() == processed["Order"].tolist()
    assert edited["Season"].tolist() == processed["Season"].tolist()
    assert edited["WeatherZone"].tolist() == processed["WeatherZone"].tolist()


def test_transplant_is_deterministic() -> None:
    raw, processed, _ = _build_weather({0: [2.0, 4.0, 20.0, 8.0], 1: [30.0, 3.0, 5.0, 7.0]})
    first, _ = zone_peak_wind_transplant(raw, processed)
    second, _ = zone_peak_wind_transplant(raw, processed)
    pd.testing.assert_frame_equal(first, second)


def test_apply_wind_regime_scenario_dispatches_and_rejects_unknown_mode() -> None:
    raw, processed, stats = _build_weather({0: [2.0, 4.0, 20.0, 8.0]})
    edited, report = apply_wind_regime_scenario(raw, processed, stats, {"mode": "zone_peak_transplant"})
    assert not report.empty
    assert not edited.empty
    with pytest.raises(ValueError):
        apply_wind_regime_scenario(raw, processed, stats, {"mode": "nope"})
