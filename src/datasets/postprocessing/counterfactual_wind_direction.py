"""Uniform wind-direction counterfactual for fixed-model spatial-pattern probes.

Every day in the focus hex is set to a single from-bearing while per-day wind
speeds are kept.  ``wind_x``/``wind_y`` are recomputed from the new direction
and re-encoded with the recovered baseline z-score stats.  Running the hex's
speed-weighted dominant from-bearing and its 180-degree opposite as paired
scenarios tests whether the model's spatial ROS response flips with the wind.
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
    if mode != "uniform_direction":
        raise ValueError(f"Unknown wind-direction scenario mode {mode!r}; expected 'uniform_direction'.")
    return uniform_direction_edit(
        raw_hex,
        processed_hex,
        stats,
        offset_deg=float(params.get("offset_deg", 0.0)),
        from_bearing_deg=(float(params["from_bearing_deg"]) if "from_bearing_deg" in params else None),
    )
