from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual_fwi import (
    apply_fwi_scenario,
    daily_regime_swap,
    fwi_tercile_labels,
)
from src.datasets.postprocessing.counterfactual_weather import (
    WindEncodingStats,
    encode_raw_wind_features,
    raw_wind_features,
    recover_wind_encoding_stats,
)

SWAP_COLUMNS = ("FireWeatherIndex", "Temperature")


def _build_weather(zone_fwi: dict[int, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame, WindEncodingStats]:
    rows = []
    for zone, fwi_values in zone_fwi.items():
        for index, fwi in enumerate(fwi_values):
            rows.append(
                {
                    "WeatherZone": zone,
                    "WindSpeed": 2.0 + fwi + 0.5 * index,
                    "WindDirection": float((37 * (zone + 1) + 50 * index) % 360),
                    "FireWeatherIndex": float(fwi),
                    "Temperature": 10.0 + 2.0 * fwi,
                }
            )
    raw = pd.DataFrame(rows)
    raw_features = raw_wind_features(raw)
    processed = pd.DataFrame(
        {
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


def test_fwi_tercile_labels_partition_low_mid_high() -> None:
    labels = fwi_tercile_labels(np.array([1.0, 5.0, 9.0]), low_quantile=33.0, high_quantile=66.0)
    assert labels.tolist() == ["low", "mid", "high"]


def test_daily_low_to_high_raises_every_zone_mean_fwi() -> None:
    raw, processed, stats = _build_weather({0: [1.0, 5.0, 9.0], 1: [2.0, 6.0, 10.0]})
    edited, report = daily_regime_swap(raw, processed, stats, direction="low_to_high", thermo_columns=SWAP_COLUMNS, seed=7)
    for zone in (0, 1):
        baseline_mean = processed.loc[processed["WeatherZone"] == zone, "FireWeatherIndex"].mean()
        scenario_mean = edited.loc[edited["WeatherZone"] == zone, "FireWeatherIndex"].mean()
        assert scenario_mean > baseline_mean
    assert (report["scenario_fwi_mean"] >= report["baseline_fwi_mean"]).all()


def test_daily_high_to_low_lowers_every_zone_mean_fwi() -> None:
    raw, processed, stats = _build_weather({0: [1.0, 5.0, 9.0], 1: [2.0, 6.0, 10.0]})
    edited, _ = daily_regime_swap(raw, processed, stats, direction="high_to_low", thermo_columns=SWAP_COLUMNS, seed=7)
    for zone in (0, 1):
        baseline_mean = processed.loc[processed["WeatherZone"] == zone, "FireWeatherIndex"].mean()
        scenario_mean = edited.loc[edited["WeatherZone"] == zone, "FireWeatherIndex"].mean()
        assert scenario_mean < baseline_mean


def test_daily_swap_leaves_mid_rows_and_nonswapped_columns_untouched() -> None:
    raw, processed, stats = _build_weather({0: [1.0, 5.0, 9.0]})
    edited, _ = daily_regime_swap(raw, processed, stats, direction="low_to_high", thermo_columns=SWAP_COLUMNS, seed=1)
    # Row 1 is the only 'mid' row for the single zone and must be unchanged.
    assert edited.loc[1, "FireWeatherIndex"] == pytest.approx(processed.loc[1, "FireWeatherIndex"])
    assert edited.loc[1, "Temperature"] == pytest.approx(processed.loc[1, "Temperature"])
    # WindDirection is never modified.
    assert np.allclose(edited["WindDirection"], processed["WindDirection"])


def test_daily_swap_recomputes_recipient_wind_from_swapped_speed_and_kept_direction() -> None:
    raw, processed, stats = _build_weather({0: [1.0, 5.0, 9.0]})
    edited, _ = daily_regime_swap(raw, processed, stats, direction="low_to_high", thermo_columns=SWAP_COLUMNS, seed=1)

    # The single low row (index 0) takes the single high row's (index 2) wind speed,
    # but keeps its own wind direction; wind_x/wind_y are recomputed accordingly.
    expected_raw = raw.copy()
    expected_raw.loc[0, "WindSpeed"] = raw.loc[2, "WindSpeed"]
    expected_encoded = encode_raw_wind_features(raw_wind_features(expected_raw), stats)
    for column in ("WindSpeed", "wind_x", "wind_y"):
        assert edited.loc[0, column] == pytest.approx(expected_encoded.loc[0, column])
        # Non-recipient rows keep their baseline encoding untouched.
        assert edited.loc[1, column] == pytest.approx(processed.loc[1, column])
        assert edited.loc[2, column] == pytest.approx(processed.loc[2, column])


def test_daily_swap_is_deterministic_for_a_fixed_seed() -> None:
    raw, processed, stats = _build_weather({0: [1.0, 3.0, 5.0, 7.0, 9.0], 1: [2.0, 4.0, 6.0, 8.0, 10.0]})
    first, _ = daily_regime_swap(raw, processed, stats, direction="low_to_high", thermo_columns=SWAP_COLUMNS, seed=123)
    second, _ = daily_regime_swap(raw, processed, stats, direction="low_to_high", thermo_columns=SWAP_COLUMNS, seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_apply_fwi_scenario_dispatches_and_rejects_unknown_mode() -> None:
    raw, processed, stats = _build_weather({0: [1.0, 5.0, 9.0], 1: [2.0, 6.0, 10.0]})
    edited, _ = apply_fwi_scenario(
        raw,
        processed,
        stats,
        {"mode": "daily_regime_swap", "direction": "low_to_high", "swap_columns": list(SWAP_COLUMNS)},
        seed=3,
    )
    assert (
        edited.loc[edited["WeatherZone"] == 0, "FireWeatherIndex"].mean()
        > processed.loc[processed["WeatherZone"] == 0, "FireWeatherIndex"].mean()
    )
    with pytest.raises(ValueError, match="Unknown FWI scenario mode"):
        apply_fwi_scenario(raw, processed, stats, {"mode": "nope"})
