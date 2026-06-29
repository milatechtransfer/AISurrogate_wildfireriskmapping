"""Wind-speed regime counterfactual for fixed-model analyses.

Per ``WeatherZone``, the single day with the highest ``WindSpeed`` is the donor.
Its full weather regime -- every meteorological driver, including the wind
vector and thermo/FWI columns -- is transplanted onto every day in the zone, so
the whole zone adopts its empirical peak-wind day.  Structural columns
(``Order``, ``Season``, ``WeatherZone``) are left untouched; the donor's
already-encoded ``wind_x``/``wind_y`` values are copied directly, so no
re-encoding is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.datasets.postprocessing.counterfactual_weather import WindEncodingStats, validate_columns

WIND_SPEED_COLUMN = "WindSpeed"
STRUCTURAL_COLUMNS = ("Order", "Season")


@dataclass(frozen=True)
class WindZoneEditReport:
    """Per-zone summary of one peak-wind regime transplant."""

    zone: int
    n_rows: int
    baseline_speed_mean: float
    scenario_speed_mean: float
    note: str


def zone_peak_wind_transplant(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    *,
    zone_column: str = "WeatherZone",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace every day in a zone with that zone's highest-WindSpeed day."""

    if len(raw_hex) != len(processed_hex):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_hex)} != {len(processed_hex)}.")
    validate_columns(processed_hex, (zone_column,), frame_name="processed weather")
    validate_columns(raw_hex, (WIND_SPEED_COLUMN,), frame_name="raw weather")

    processed_edited = processed_hex.reset_index(drop=True).copy()
    raw_speed = raw_hex.reset_index(drop=True)[WIND_SPEED_COLUMN].to_numpy(dtype=np.float64)
    zones = processed_edited[zone_column].to_numpy()

    transplant_columns = [column for column in processed_edited.columns if column not in (*STRUCTURAL_COLUMNS, zone_column)]
    donor_source = {column: processed_edited[column].to_numpy().copy() for column in transplant_columns}
    edited = {column: values.copy() for column, values in donor_source.items()}
    reports: list[WindZoneEditReport] = []

    for zone in np.unique(zones):
        zone_positions = np.flatnonzero(zones == zone)
        donor_pos = int(zone_positions[int(np.argmax(raw_speed[zone_positions]))])
        for column in transplant_columns:
            edited[column][zone_positions] = donor_source[column][donor_pos]
        reports.append(
            WindZoneEditReport(
                zone=int(zone),
                n_rows=int(zone_positions.size),
                baseline_speed_mean=float(np.mean(raw_speed[zone_positions])),
                scenario_speed_mean=float(raw_speed[donor_pos]),
                note="ok",
            )
        )

    for column, values in edited.items():
        processed_edited[column] = values

    return processed_edited, pd.DataFrame([report.__dict__ for report in reports])


def apply_wind_regime_scenario(
    raw_hex: pd.DataFrame,
    processed_hex: pd.DataFrame,
    stats: WindEncodingStats,
    params: dict,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch a wind-regime scenario to its transplant implementation."""

    mode = str(params.get("mode", "zone_peak_transplant"))
    if mode != "zone_peak_transplant":
        raise ValueError(f"Unknown wind-regime scenario mode {mode!r}; expected 'zone_peak_transplant'.")
    return zone_peak_wind_transplant(
        raw_hex,
        processed_hex,
        zone_column=str(params.get("zone_column", "WeatherZone")),
    )
