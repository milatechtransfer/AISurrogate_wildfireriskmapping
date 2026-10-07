"""Weather-table editing helpers for counterfactual analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.tabular.weather import load_weather_list
from data_preparation.utils import aggregate_csv_by_pattern
from src.datasets.fuel_utils import normalize_hex_id

FWI_COLUMN = "FireWeatherIndex"
HEX_ID_COLUMN = "hex_id"
RAW_HEX_ID_COLUMN = "__hex_id"
WEATHER_ZONE_COLUMN = "WeatherZone"
STRUCTURAL_COLUMNS: tuple[str, ...] = ("Order", "Season", HEX_ID_COLUMN, WEATHER_ZONE_COLUMN)
NON_AVERAGE_COLUMNS: tuple[str, ...] = (*STRUCTURAL_COLUMNS, "WindDirection")
WEATHER_EDIT_MODES = ("external_mean_zone_transplant",)
RAW_WEATHER_GLOB_PATTERN = "hex*/tabular/hex*_DailyWeather.csv"


@dataclass(frozen=True)
class WeatherEditReport:
    """Summary of one weather counterfactual edit."""

    scenario_name: str
    mode: str
    recipient_hex_id: str
    n_recipient_rows: int
    n_recipient_zones: int
    donor_hex_ids: str
    n_donor_rows: int
    donor_fwi_mean: float
    baseline_fwi_mean: float
    scenario_fwi_mean: float


def _hex_id_from_weather_path(path: Path) -> str:
    return normalize_hex_id(path.parent.parent.name.removeprefix("hex"))


def _load_raw_weather_with_hex_id(path: Path) -> pd.DataFrame:
    frame = load_weather_list(str(path), normalize_weatherlist=False)
    frame.insert(0, RAW_HEX_ID_COLUMN, _hex_id_from_weather_path(path))
    return frame


def load_all_raw_weather_with_hex_ids(raw_data_dir: Path) -> pd.DataFrame:
    """Reconstruct the raw weather table in processed-table row order, tagged by hex."""
    return aggregate_csv_by_pattern(
        root_dir=raw_data_dir,
        pattern=RAW_WEATHER_GLOB_PATTERN,
        load_function=_load_raw_weather_with_hex_id,
    )


def _validated_processed_hex_ids(processed: pd.DataFrame) -> pd.Series:
    """Validate processed hex IDs and return their normalized string representation."""
    if HEX_ID_COLUMN not in processed.columns:
        raise ValueError(f"Processed weather table is missing column {HEX_ID_COLUMN!r}.")

    numeric_hex_ids = pd.to_numeric(processed[HEX_ID_COLUMN], errors="coerce")
    invalid_mask = processed[HEX_ID_COLUMN].notna() & numeric_hex_ids.isna()
    if invalid_mask.any():
        bad_value = processed.loc[invalid_mask, HEX_ID_COLUMN].iloc[0]
        raise ValueError(f"Processed weather table contains non-numeric hex_id {bad_value!r}.")
    if numeric_hex_ids.isna().any():
        raise ValueError("Processed weather table contains missing hex_id values.")

    values = numeric_hex_ids.to_numpy(dtype=np.float64)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-3):
        bad_value = float(values[np.argmax(np.abs(values - rounded))])
        raise ValueError(f"Processed weather table contains non-integer hex_id {bad_value}.")

    integer_hex_ids = pd.Series(rounded.astype(np.int64), index=processed.index)
    return integer_hex_ids.map(normalize_hex_id)


def _validate_raw_processed_alignment(raw_features: pd.DataFrame, processed: pd.DataFrame) -> pd.Series:
    """Verify that raw and processed rows refer to the same hex-zone observations."""
    if RAW_HEX_ID_COLUMN not in raw_features.columns:
        raise ValueError(f"raw_features is missing the {RAW_HEX_ID_COLUMN!r} tag column.")
    if len(raw_features) != len(processed):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_features)} != {len(processed)}.")
    if WEATHER_ZONE_COLUMN not in raw_features.columns or WEATHER_ZONE_COLUMN not in processed.columns:
        raise ValueError(f"Raw and processed weather tables must include {WEATHER_ZONE_COLUMN!r}.")

    processed_hex_ids = _validated_processed_hex_ids(processed)
    raw_hex_ids = raw_features[RAW_HEX_ID_COLUMN].map(normalize_hex_id)
    hex_mismatch = raw_hex_ids.to_numpy() != processed_hex_ids.to_numpy()
    if hex_mismatch.any():
        row = int(np.flatnonzero(hex_mismatch)[0])
        raise ValueError(f"Raw/processed hex_id mismatch at row {row}: " f"{raw_hex_ids.iloc[row]!r} != {processed_hex_ids.iloc[row]!r}.")

    raw_zones = pd.to_numeric(raw_features[WEATHER_ZONE_COLUMN], errors="coerce")
    processed_zones = pd.to_numeric(processed[WEATHER_ZONE_COLUMN], errors="coerce")
    zone_match = np.isclose(
        raw_zones.to_numpy(dtype=np.float64),
        processed_zones.to_numpy(dtype=np.float64),
        equal_nan=True,
    )
    if not zone_match.all():
        row = int(np.flatnonzero(~zone_match)[0])
        raise ValueError(
            f"Raw/processed {WEATHER_ZONE_COLUMN} mismatch at row {row}: "
            f"{raw_features[WEATHER_ZONE_COLUMN].iloc[row]!r} != {processed[WEATHER_ZONE_COLUMN].iloc[row]!r}."
        )
    return processed_hex_ids


def apply_external_mean_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the exact mean processed-weather vector of donor hexels.

    The returned table contains one row per ``(hex_id, WeatherZone)``. Baseline
    hex-zone means are retained everywhere except recipient hexels, whose zone rows
    receive the donor mean exactly.
    """
    if FWI_COLUMN not in raw_features.columns or FWI_COLUMN not in processed.columns:
        raise ValueError(f"Raw and processed weather tables must include {FWI_COLUMN!r}.")
    processed_hex_ids = _validate_raw_processed_alignment(raw_features, processed)

    normalized_recipient_ids = [normalize_hex_id(hex_id) for hex_id in recipient_hex_ids]
    if not normalized_recipient_ids:
        raise ValueError("recipient_hex_ids must be non-empty.")
    recipient_masks = {hex_id: (processed_hex_ids == hex_id).to_numpy() for hex_id in normalized_recipient_ids}
    missing_recipients = [hex_id for hex_id, mask in recipient_masks.items() if not mask.any()]
    if missing_recipients:
        raise ValueError(f"No weather rows found for recipient hex_id(s)={missing_recipients}.")

    normalized_donor_ids = [normalize_hex_id(hex_id) for hex_id in donor_hex_ids]
    if not normalized_donor_ids:
        raise ValueError("donor_hex_ids must be non-empty.")
    donor_mask = processed_hex_ids.isin(normalized_donor_ids).to_numpy()
    if not donor_mask.any():
        raise ValueError(f"No weather rows found for donor_hex_ids={normalized_donor_ids}.")

    mean_columns = [
        column for column in processed.columns if column not in NON_AVERAGE_COLUMNS and pd.api.types.is_numeric_dtype(processed[column])
    ]
    donor_mean = processed.loc[donor_mask, mean_columns].mean(axis=0)
    if not np.isfinite(donor_mean.to_numpy(dtype=np.float64)).all():
        raise ValueError("Donor mean weather vector contains non-finite values.")

    edited = processed.groupby([HEX_ID_COLUMN, WEATHER_ZONE_COLUMN], as_index=False)[mean_columns].mean()
    edited_hex_ids = _validated_processed_hex_ids(edited)
    edited_recipient_mask = edited_hex_ids.isin(normalized_recipient_ids)
    for column in mean_columns:
        edited.loc[edited_recipient_mask, column] = float(donor_mean[column])

    donor_hex_label = ",".join(normalized_donor_ids)
    donor_fwi_mean = float(raw_features.loc[donor_mask, FWI_COLUMN].mean())
    reports = []
    for recipient_hex_id in normalized_recipient_ids:
        recipient_mask = recipient_masks[recipient_hex_id]
        reports.append(
            WeatherEditReport(
                scenario_name=scenario_name,
                mode="external_mean_zone_transplant",
                recipient_hex_id=recipient_hex_id,
                n_recipient_rows=int(recipient_mask.sum()),
                n_recipient_zones=int(processed.loc[recipient_mask, WEATHER_ZONE_COLUMN].nunique()),
                donor_hex_ids=donor_hex_label,
                n_donor_rows=int(donor_mask.sum()),
                donor_fwi_mean=donor_fwi_mean,
                baseline_fwi_mean=float(raw_features.loc[recipient_mask, FWI_COLUMN].mean()),
                scenario_fwi_mean=donor_fwi_mean,
            )
        )
    return edited, reports


def apply_weather_edit(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    mode: str,
    scenario_name: str,
    recipient_hex_ids: list[str],
    params: dict,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Dispatch one configured weather edit."""
    if mode not in WEATHER_EDIT_MODES:
        raise ValueError(f"Unknown weather edit mode {mode!r}; expected one of {WEATHER_EDIT_MODES}.")
    donor_hex_ids = params.get("donor_hex_ids")
    if not isinstance(donor_hex_ids, list | tuple) or not donor_hex_ids:
        raise ValueError(f"Weather scenario {scenario_name!r} must define a non-empty donor_hex_ids list.")
    return apply_external_mean_zone_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=[str(value) for value in donor_hex_ids],
        scenario_name=scenario_name,
    )
