"""Uniform wind-direction counterfactual for fixed-model spatial-pattern probes.

Every day in the focus hex, or every day within each weather zone, is set to a
single from-bearing while per-day wind speeds are kept.  ``wind_x``/``wind_y``
are recomputed from the new direction and re-encoded with the recovered
baseline z-score stats.  Running the dominant from-bearing and its 180-degree
opposite as paired scenarios tests whether the model's spatial ROS response
flips with the wind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.datasets.postprocessing.counterfactual_weather import (
    RAW_WIND_COLUMNS,
    WIND_DIRECTION_COLUMN,
    WindEncodingStats,
    encode_raw_wind_features,
    raw_wind_features,
    validate_columns,
)

WIND_SPEED_COLUMN = "WindSpeed"


@dataclass(frozen=True)
class WindDirectionEditReport:
    """Summary of one uniform-direction edit applied to a hex."""

    n_rows: int
    from_bearing_deg: float
    dominant_from_bearing_deg: float
    offset_deg: float


def dominant_from_bearing(wind_speed: np.ndarray, wind_direction: np.ndarray) -> float:
    """Speed-weighted dominant meteorological from-bearing in [0, 360)."""

    finite = np.isfinite(wind_speed) & np.isfinite(wind_direction)
    if not finite.any():
        raise ValueError("Cannot compute dominant wind direction without finite rows.")
    radians = np.deg2rad(wind_direction[finite])
    weights = np.clip(wind_speed[finite], 0.0, None)
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    sum_x = float(np.sum(np.sin(radians) * weights))
    sum_y = float(np.sum(np.cos(radians) * weights))
    if np.isclose(sum_x, 0.0) and np.isclose(sum_y, 0.0):
        return float(np.mean(wind_direction[finite]) % 360.0)
    return float(np.rad2deg(np.arctan2(sum_x, sum_y)) % 360.0)


def uniform_direction_edit(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    stats: WindEncodingStats,
    *,
    offset_deg: float = 0.0,
    from_bearing_deg: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set every day to one from-bearing, recompute and re-encode wind components."""

    if len(raw_hex) != len(processed_hex):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_hex)} != {len(processed_hex)}.")
    validate_columns(raw_hex, (WIND_SPEED_COLUMN, WIND_DIRECTION_COLUMN), frame_name="raw weather")

    raw = raw_hex.reset_index(drop=True)
    speed = raw[WIND_SPEED_COLUMN].to_numpy(dtype=np.float64)
    dominant = dominant_from_bearing(speed, raw[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64))
    base = dominant if from_bearing_deg is None else float(from_bearing_deg)
    target = float((base + offset_deg) % 360.0)

    scenario_raw = raw.copy()
    scenario_raw[WIND_DIRECTION_COLUMN] = target
    encoded = encode_raw_wind_features(raw_wind_features(scenario_raw), stats)

    processed_edited = processed_hex.reset_index(drop=True).copy()
    for column in RAW_WIND_COLUMNS:
        processed_edited[column] = encoded[column].to_numpy(dtype=np.float64)
    if WIND_DIRECTION_COLUMN in processed_edited.columns:
        processed_edited[WIND_DIRECTION_COLUMN] = target

    report = WindDirectionEditReport(
        n_rows=int(len(raw)),
        from_bearing_deg=target,
        dominant_from_bearing_deg=dominant,
        offset_deg=float(offset_deg),
    )
    return processed_edited, pd.DataFrame([report.__dict__])


def zone_uniform_direction_edit(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    stats: WindEncodingStats,
    *,
    zone_column: str = "WeatherZone",
    offset_deg: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set each zone to its own dominant from-bearing, recomputing wind components."""

    if len(raw_hex) != len(processed_hex):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_hex)} != {len(processed_hex)}.")
    validate_columns(raw_hex, (WIND_SPEED_COLUMN, WIND_DIRECTION_COLUMN, zone_column), frame_name="raw weather")
    validate_columns(processed_hex, (zone_column,), frame_name="processed weather")

    raw = raw_hex.reset_index(drop=True)
    processed_edited = processed_hex.reset_index(drop=True).copy()
    scenario_raw = raw.copy()
    report_rows: list[dict[str, float | int]] = []

    for zone in sorted(raw[zone_column].dropna().unique()):
        zone_mask = raw[zone_column].to_numpy() == zone
        speed = raw.loc[zone_mask, WIND_SPEED_COLUMN].to_numpy(dtype=np.float64)
        direction = raw.loc[zone_mask, WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64)
        dominant = dominant_from_bearing(speed, direction)
        target = float((dominant + offset_deg) % 360.0)
        scenario_raw.loc[zone_mask, WIND_DIRECTION_COLUMN] = target
        report_rows.append(
            {
                "zone": int(zone),
                "n_rows": int(zone_mask.sum()),
                "from_bearing_deg": target,
                "dominant_from_bearing_deg": dominant,
                "offset_deg": float(offset_deg),
            }
        )

    encoded = encode_raw_wind_features(raw_wind_features(scenario_raw), stats)
    for column in RAW_WIND_COLUMNS:
        processed_edited[column] = encoded[column].to_numpy(dtype=np.float64)
    if WIND_DIRECTION_COLUMN in processed_edited.columns:
        processed_edited[WIND_DIRECTION_COLUMN] = scenario_raw[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64)

    return processed_edited, pd.DataFrame(report_rows)


def apply_wind_direction_scenario(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    stats: WindEncodingStats,
    params: dict,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch a wind-direction scenario to its uniform-direction implementation."""

    mode = str(params.get("mode", "uniform_direction"))
    if mode == "zone_uniform_direction":
        return zone_uniform_direction_edit(
            raw_hex,
            processed_hex,
            stats,
            zone_column=str(params.get("zone_column", "WeatherZone")),
            offset_deg=float(params.get("offset_deg", 0.0)),
        )
    if mode != "uniform_direction":
        raise ValueError(
            f"Unknown wind-direction scenario mode {mode!r}; expected one of " "{'uniform_direction', 'zone_uniform_direction'}."
        )
    return uniform_direction_edit(
        raw_hex,
        processed_hex,
        stats,
        offset_deg=float(params.get("offset_deg", 0.0)),
        from_bearing_deg=(float(params["from_bearing_deg"]) if "from_bearing_deg" in params else None),
    )
