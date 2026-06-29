"""FWI-regime weather counterfactuals for fixed-model analyses.

The intervention produces an edited ``weather_table_processed.csv`` whose zone
means encode a high- or low-FWI regime.  The spatialized-weather source
re-aggregates that table by WeatherZone at inference time, so editing the rows
is sufficient.

- ``daily_regime_swap`` (pre-aggregation): within each zone, bin daily rows by
  FireWeatherIndex terciles and replace low rows with sampled high rows (or the
  reverse).

The swapped feature set covers every FWI driver except WindDirection.  WindSpeed
is swapped, and because ``wind_x``/``wind_y`` are derived they are recomputed
from the swapped speed and the preserved direction using the recovered baseline
wind z-score stats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.datasets.postprocessing.counterfactual_weather import (
    RAW_WIND_COLUMNS,
    WindEncodingStats,
    encode_raw_wind_features,
    raw_wind_features,
    validate_columns,
)

THERMO_SWAP_COLUMNS: tuple[str, ...] = (
    "Temperature",
    "RelativeHumidity",
    "Precipitation",
    "FineFuelMoistureCode",
    "DuffMoistureCode",
    "DroughtCode",
    "InitialSpreadIndex",
    "BuildupIndex",
    "FireWeatherIndex",
)
FWI_COLUMN = "FireWeatherIndex"
RAW_WIND_SPEED_COLUMN = "WindSpeed"
SWAP_DIRECTIONS: tuple[str, ...] = ("low_to_high", "high_to_low")
SWAP_MODES: tuple[str, ...] = ("daily_regime_swap",)


@dataclass(frozen=True)
class FwiZoneEditReport:
    """Per-zone summary of one FWI counterfactual edit."""

    zone: int
    n_rows: int
    n_low: int
    n_mid: int
    n_high: int
    n_recipients: int
    donor_zone: int | None
    baseline_fwi_mean: float
    scenario_fwi_mean: float
    note: str


def fwi_tercile_labels(fwi_values: np.ndarray, *, low_quantile: float, high_quantile: float) -> np.ndarray:
    """Label each FWI value as 'low', 'mid', or 'high' by within-group terciles."""

    values = np.asarray(fwi_values, dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype="<U4")
    low_threshold = float(np.percentile(values, low_quantile))
    high_threshold = float(np.percentile(values, high_quantile))
    labels = np.full(values.shape, "mid", dtype="<U4")
    labels[values <= low_threshold] = "low"
    labels[values >= high_threshold] = "high"
    return labels


def _reencode_recipient_wind(
    raw_edited: pd.DataFrame,
    processed_edited: pd.DataFrame,
    stats: WindEncodingStats,
    recipient_positions: np.ndarray,
) -> pd.DataFrame:
    """Recompute processed wind columns for recipient rows from edited raw wind."""

    if recipient_positions.size == 0:
        return processed_edited
    encoded = encode_raw_wind_features(raw_wind_features(raw_edited), stats)
    for column in RAW_WIND_COLUMNS:
        updated = processed_edited[column].to_numpy(dtype=np.float64).copy()
        updated[recipient_positions] = encoded[column].to_numpy(dtype=np.float64)[recipient_positions]
        processed_edited[column] = updated
    return processed_edited


def daily_regime_swap(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    stats: WindEncodingStats,
    *,
    direction: str,
    low_quantile: float = 33.0,
    high_quantile: float = 66.0,
    zone_column: str = "WeatherZone",
    thermo_columns: tuple[str, ...] = THERMO_SWAP_COLUMNS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Swap each zone's low-FWI daily rows with high-FWI rows (or the reverse)."""

    if direction not in SWAP_DIRECTIONS:
        raise ValueError(f"direction={direction!r}; expected one of {SWAP_DIRECTIONS}.")
    if len(raw_hex) != len(processed_hex):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_hex)} != {len(processed_hex)}.")
    validate_columns(processed_hex, (zone_column, FWI_COLUMN, *thermo_columns, *RAW_WIND_COLUMNS), frame_name="processed weather")
    validate_columns(raw_hex, (RAW_WIND_SPEED_COLUMN, "WindDirection"), frame_name="raw weather")

    processed_edited = processed_hex.reset_index(drop=True).copy()
    raw_edited = raw_hex.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)

    zones = processed_edited[zone_column].to_numpy()
    fwi = processed_edited[FWI_COLUMN].to_numpy(dtype=np.float64)
    thermo_baseline = {column: processed_edited[column].to_numpy(dtype=np.float64) for column in thermo_columns}
    raw_speed_baseline = raw_edited[RAW_WIND_SPEED_COLUMN].to_numpy(dtype=np.float64)

    thermo_edited = {column: values.copy() for column, values in thermo_baseline.items()}
    raw_speed_edited = raw_speed_baseline.copy()
    recipient_positions: list[int] = []
    reports: list[FwiZoneEditReport] = []

    for zone in np.unique(zones):
        zone_positions = np.flatnonzero(zones == zone)
        labels = fwi_tercile_labels(fwi[zone_positions], low_quantile=low_quantile, high_quantile=high_quantile)
        low_positions = zone_positions[labels == "low"]
        high_positions = zone_positions[labels == "high"]
        recipients = low_positions if direction == "low_to_high" else high_positions
        donor_pool = high_positions if direction == "low_to_high" else low_positions

        note = "ok"
        if recipients.size == 0 or donor_pool.size == 0:
            note = "no recipients or donors; zone unchanged"
        else:
            donors = rng.choice(donor_pool, size=recipients.size, replace=True)
            for column in thermo_columns:
                thermo_edited[column][recipients] = thermo_baseline[column][donors]
            raw_speed_edited[recipients] = raw_speed_baseline[donors]
            recipient_positions.extend(int(position) for position in recipients)

        reports.append(
            FwiZoneEditReport(
                zone=int(zone),
                n_rows=int(zone_positions.size),
                n_low=int(low_positions.size),
                n_mid=int(np.sum(labels == "mid")),
                n_high=int(high_positions.size),
                n_recipients=int(recipients.size if note == "ok" else 0),
                donor_zone=None,
                baseline_fwi_mean=float(np.mean(thermo_baseline[FWI_COLUMN][zone_positions])),
                scenario_fwi_mean=float(np.mean(thermo_edited[FWI_COLUMN][zone_positions])),
                note=note,
            )
        )

    for column in thermo_columns:
        processed_edited[column] = thermo_edited[column]
    raw_edited[RAW_WIND_SPEED_COLUMN] = raw_speed_edited
    processed_edited = _reencode_recipient_wind(raw_edited, processed_edited, stats, np.asarray(recipient_positions, dtype=np.int64))
    return processed_edited, _report_frame(reports)


def apply_fwi_scenario(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    stats: WindEncodingStats,
    params: dict,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the daily FWI regime swap."""

    mode = str(params.get("mode", "daily_regime_swap"))
    if mode not in SWAP_MODES:
        raise ValueError(f"Unknown FWI scenario mode {mode!r}; expected one of {SWAP_MODES}.")
    swap_fn = daily_regime_swap
    thermo_columns = tuple(params["swap_columns"]) if "swap_columns" in params else THERMO_SWAP_COLUMNS
    return swap_fn(
        raw_hex,
        processed_hex,
        stats,
        direction=str(params.get("direction", "low_to_high")),
        low_quantile=float(params.get("low_quantile", 33.0)),
        high_quantile=float(params.get("high_quantile", 66.0)),
        zone_column=str(params.get("zone_column", "WeatherZone")),
        thermo_columns=thermo_columns,
        seed=seed,
    )


def _report_frame(reports: list[FwiZoneEditReport]) -> pd.DataFrame:
    return pd.DataFrame([report.__dict__ for report in reports])
