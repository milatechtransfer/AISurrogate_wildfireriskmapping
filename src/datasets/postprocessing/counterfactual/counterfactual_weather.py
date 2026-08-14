"""Weather-table editing helpers for counterfactual analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.tabular.weather import load_weather_list, load_weather_normalization_params, wind_to_components
from data_preparation.utils import aggregate_csv_by_pattern
from src.datasets.fuel_utils import normalize_hex_id

FWI_COLUMN = "FireWeatherIndex"
HEX_ID_COLUMN = "hex_id"
RAW_HEX_ID_COLUMN = "__hex_id"
WEATHER_ZONE_COLUMN = "WeatherZone"
WIND_SPEED_COLUMN = "WindSpeed"
WIND_DIRECTION_COLUMN = "WindDirection"
WIND_X_COLUMN = "wind_x"
WIND_Y_COLUMN = "wind_y"
WEATHER_NORM_PARAMS_FILENAME = "weather_norm_params.json"
STRUCTURAL_COLUMNS: tuple[str, ...] = ("Order", "Season", HEX_ID_COLUMN, WEATHER_ZONE_COLUMN)
NON_AVERAGE_COLUMNS: tuple[str, ...] = (*STRUCTURAL_COLUMNS, WIND_DIRECTION_COLUMN)
EXTERNAL_MEAN_ZONE_TRANSPLANT_MODE = "external_mean_zone_transplant"
WINDY_MEAN_ZONE_TRANSPLANT_MODE = "windy_mean_zone_transplant"
WIND_DIRECTION_ZONE_TRANSPLANT_MODE = "wind_direction_zone_transplant"
WEATHER_EDIT_MODES = (
    EXTERNAL_MEAN_ZONE_TRANSPLANT_MODE,
    WINDY_MEAN_ZONE_TRANSPLANT_MODE,
    WIND_DIRECTION_ZONE_TRANSPLANT_MODE,
)
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
    # Set only for wind-filtered donor means (`windy_mean_zone_transplant`,
    # `wind_direction_zone_transplant`); None otherwise.
    wind_speed_threshold: float | None = None
    # Set only for `wind_direction_zone_transplant`; None otherwise.
    direction_degrees: float | None = None


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


def _donor_mask_from_hex_ids(processed_hex_ids: pd.Series, donor_hex_ids: list[str]) -> tuple[list[str], np.ndarray]:
    normalized_donor_ids = [normalize_hex_id(hex_id) for hex_id in donor_hex_ids]
    if not normalized_donor_ids:
        raise ValueError("donor_hex_ids must be non-empty.")
    donor_mask = processed_hex_ids.isin(normalized_donor_ids).to_numpy()
    if not donor_mask.any():
        raise ValueError(f"No weather rows found for donor_hex_ids={normalized_donor_ids}.")
    return normalized_donor_ids, donor_mask


def _apply_mean_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    scenario_name: str,
    mode: str,
    wind_speed_threshold: float | None = None,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the exact mean processed-weather vector of donor hexels.

    The returned table contains one row per ``(hex_id, WeatherZone)``. Baseline
    hex-zone means are retained everywhere except recipient hexels, whose zone rows
    receive the donor mean exactly. `donor_hex_ids` may overlap or exactly match
    `recipient_hex_ids` (e.g. a hexel can donate its own filtered rows to itself).
    When `wind_speed_threshold` is set, the donor mean is restricted to donor rows
    whose raw `WindSpeed` is greater than or equal to the threshold.
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

    normalized_donor_ids, donor_mask = _donor_mask_from_hex_ids(processed_hex_ids, donor_hex_ids)

    if wind_speed_threshold is not None:
        if WIND_SPEED_COLUMN not in raw_features.columns:
            raise ValueError(f"raw_features is missing {WIND_SPEED_COLUMN!r}, required to apply a wind_speed_threshold.")
        wind_speed = pd.to_numeric(raw_features[WIND_SPEED_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
        windy_donor_mask = donor_mask & (wind_speed >= wind_speed_threshold)
        if not windy_donor_mask.any():
            raise ValueError(
                f"No donor weather rows with {WIND_SPEED_COLUMN} >= {wind_speed_threshold} "
                f"found among donor_hex_ids={normalized_donor_ids}."
            )
        donor_mask = windy_donor_mask

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
                mode=mode,
                recipient_hex_id=recipient_hex_id,
                n_recipient_rows=int(recipient_mask.sum()),
                n_recipient_zones=int(processed.loc[recipient_mask, WEATHER_ZONE_COLUMN].nunique()),
                donor_hex_ids=donor_hex_label,
                n_donor_rows=int(donor_mask.sum()),
                donor_fwi_mean=donor_fwi_mean,
                baseline_fwi_mean=float(raw_features.loc[recipient_mask, FWI_COLUMN].mean()),
                scenario_fwi_mean=donor_fwi_mean,
                wind_speed_threshold=wind_speed_threshold,
            )
        )
    return edited, reports


def apply_external_mean_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the exact mean processed-weather vector of donor hexels.

    See `_apply_mean_zone_transplant` for the shared implementation.
    """
    return _apply_mean_zone_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
        mode=EXTERNAL_MEAN_ZONE_TRANSPLANT_MODE,
    )


def apply_windy_mean_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    wind_speed_threshold: float,
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the mean processed-weather vector of donor rows with high wind.

    Identical to `apply_external_mean_zone_transplant`, except the donor mean is
    computed only over donor rows whose raw `WindSpeed` is >= `wind_speed_threshold`.
    `donor_hex_ids` may equal `recipient_hex_ids` to give a hexel the mean of its own
    windiest days instead of an external donor's.
    """
    return _apply_mean_zone_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
        mode=WINDY_MEAN_ZONE_TRANSPLANT_MODE,
        wind_speed_threshold=wind_speed_threshold,
    )


def apply_wind_direction_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    direction_degrees: float,
    wind_speed_threshold: float,
    norm_params_path: str | Path,
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the donor mean weather vector with wind forced to blow
    from a fixed compass direction.

    Donor rows are first filtered to raw `WindSpeed` >= `wind_speed_threshold` (set
    the threshold to 0 to include every donor row and use its ordinary/average wind
    speed instead of only its windiest days). Each surviving donor row keeps its own
    recorded `WindSpeed` magnitude but has its `WindDirection` overridden to
    `direction_degrees` before `wind_x`/`wind_y` are recomputed and re-normalized
    with the same z-score parameters fit at training time (loaded from
    `norm_params_path`, typically `weather_norm_params.json` next to the processed
    weather table). All other weather columns are averaged unchanged. Prefer a
    self-donor (`donor_hex_ids == recipient_hex_ids`) to isolate the pure direction
    effect; an external donor also imports that donor's non-wind climate, which
    conflates two interventions.
    """
    if FWI_COLUMN not in raw_features.columns or FWI_COLUMN not in processed.columns:
        raise ValueError(f"Raw and processed weather tables must include {FWI_COLUMN!r}.")
    if WIND_SPEED_COLUMN not in raw_features.columns:
        raise ValueError(f"raw_features is missing {WIND_SPEED_COLUMN!r}, required for {WIND_DIRECTION_ZONE_TRANSPLANT_MODE!r}.")
    if WIND_X_COLUMN not in processed.columns or WIND_Y_COLUMN not in processed.columns:
        raise ValueError(f"Processed weather table must include {WIND_X_COLUMN!r} and {WIND_Y_COLUMN!r}.")

    processed_hex_ids = _validate_raw_processed_alignment(raw_features, processed)

    normalized_recipient_ids = [normalize_hex_id(hex_id) for hex_id in recipient_hex_ids]
    if not normalized_recipient_ids:
        raise ValueError("recipient_hex_ids must be non-empty.")
    recipient_masks = {hex_id: (processed_hex_ids == hex_id).to_numpy() for hex_id in normalized_recipient_ids}
    missing_recipients = [hex_id for hex_id, mask in recipient_masks.items() if not mask.any()]
    if missing_recipients:
        raise ValueError(f"No weather rows found for recipient hex_id(s)={missing_recipients}.")

    normalized_donor_ids, donor_mask = _donor_mask_from_hex_ids(processed_hex_ids, donor_hex_ids)

    wind_speed = pd.to_numeric(raw_features[WIND_SPEED_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    filtered_donor_mask = donor_mask & (wind_speed >= wind_speed_threshold)
    if not filtered_donor_mask.any():
        raise ValueError(
            f"No donor weather rows with {WIND_SPEED_COLUMN} >= {wind_speed_threshold} "
            f"found among donor_hex_ids={normalized_donor_ids}."
        )

    norm_params = load_weather_normalization_params(norm_params_path)
    z_cols = list(norm_params["z_score"]["cols"])
    if WIND_X_COLUMN not in z_cols or WIND_Y_COLUMN not in z_cols:
        raise ValueError(f"{norm_params_path} does not define z-score parameters for {WIND_X_COLUMN!r}/{WIND_Y_COLUMN!r}.")
    wind_x_mean = float(norm_params["z_score"]["mean"][z_cols.index(WIND_X_COLUMN)])
    wind_x_std = float(norm_params["z_score"]["std"][z_cols.index(WIND_X_COLUMN)]) or 1.0
    wind_y_mean = float(norm_params["z_score"]["mean"][z_cols.index(WIND_Y_COLUMN)])
    wind_y_std = float(norm_params["z_score"]["std"][z_cols.index(WIND_Y_COLUMN)]) or 1.0

    donor_wind_speed = wind_speed[filtered_donor_mask]
    forced_direction = np.full(donor_wind_speed.shape, float(direction_degrees), dtype=np.float64)
    raw_wind_x, raw_wind_y = wind_to_components(ws=donor_wind_speed, wd=forced_direction)

    mean_columns = [
        column for column in processed.columns if column not in NON_AVERAGE_COLUMNS and pd.api.types.is_numeric_dtype(processed[column])
    ]
    donor_subset = processed.loc[filtered_donor_mask, mean_columns].copy()
    donor_subset[WIND_X_COLUMN] = (raw_wind_x - wind_x_mean) / wind_x_std
    donor_subset[WIND_Y_COLUMN] = (raw_wind_y - wind_y_mean) / wind_y_std
    donor_mean = donor_subset.mean(axis=0)
    if not np.isfinite(donor_mean.to_numpy(dtype=np.float64)).all():
        raise ValueError("Donor mean weather vector contains non-finite values.")

    edited = processed.groupby([HEX_ID_COLUMN, WEATHER_ZONE_COLUMN], as_index=False)[mean_columns].mean()
    edited_hex_ids = _validated_processed_hex_ids(edited)
    edited_recipient_mask = edited_hex_ids.isin(normalized_recipient_ids)
    for column in mean_columns:
        edited.loc[edited_recipient_mask, column] = float(donor_mean[column])

    donor_hex_label = ",".join(normalized_donor_ids)
    donor_fwi_mean = float(raw_features.loc[filtered_donor_mask, FWI_COLUMN].mean())
    reports = []
    for recipient_hex_id in normalized_recipient_ids:
        recipient_mask = recipient_masks[recipient_hex_id]
        reports.append(
            WeatherEditReport(
                scenario_name=scenario_name,
                mode=WIND_DIRECTION_ZONE_TRANSPLANT_MODE,
                recipient_hex_id=recipient_hex_id,
                n_recipient_rows=int(recipient_mask.sum()),
                n_recipient_zones=int(processed.loc[recipient_mask, WEATHER_ZONE_COLUMN].nunique()),
                donor_hex_ids=donor_hex_label,
                n_donor_rows=int(filtered_donor_mask.sum()),
                donor_fwi_mean=donor_fwi_mean,
                baseline_fwi_mean=float(raw_features.loc[recipient_mask, FWI_COLUMN].mean()),
                scenario_fwi_mean=donor_fwi_mean,
                wind_speed_threshold=float(wind_speed_threshold),
                direction_degrees=float(direction_degrees),
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
    norm_params_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Dispatch one configured weather edit."""
    if mode not in WEATHER_EDIT_MODES:
        raise ValueError(f"Unknown weather edit mode {mode!r}; expected one of {WEATHER_EDIT_MODES}.")
    donor_hex_ids = params.get("donor_hex_ids")
    if not isinstance(donor_hex_ids, list | tuple) or not donor_hex_ids:
        raise ValueError(f"Weather scenario {scenario_name!r} must define a non-empty donor_hex_ids list.")
    donor_hex_ids = [str(value) for value in donor_hex_ids]

    if mode == WIND_DIRECTION_ZONE_TRANSPLANT_MODE:
        direction_degrees = params.get("direction_degrees")
        if direction_degrees is None:
            raise ValueError(
                f"Weather scenario {scenario_name!r} with mode {WIND_DIRECTION_ZONE_TRANSPLANT_MODE!r} "
                "must define a numeric 'direction_degrees'."
            )
        wind_speed_threshold = params.get("wind_speed_threshold")
        if wind_speed_threshold is None:
            raise ValueError(
                f"Weather scenario {scenario_name!r} with mode {WIND_DIRECTION_ZONE_TRANSPLANT_MODE!r} "
                "must define a numeric 'wind_speed_threshold' (use 0 to include every donor row)."
            )
        if norm_params_path is None:
            raise ValueError(
                f"Weather scenario {scenario_name!r} with mode {WIND_DIRECTION_ZONE_TRANSPLANT_MODE!r} "
                "requires norm_params_path to re-normalize the forced wind_x/wind_y vector."
            )
        return apply_wind_direction_zone_transplant(
            raw_features,
            processed,
            recipient_hex_ids=recipient_hex_ids,
            donor_hex_ids=donor_hex_ids,
            direction_degrees=float(direction_degrees),
            wind_speed_threshold=float(wind_speed_threshold),
            norm_params_path=norm_params_path,
            scenario_name=scenario_name,
        )

    if mode == WINDY_MEAN_ZONE_TRANSPLANT_MODE:
        wind_speed_threshold = params.get("wind_speed_threshold")
        if wind_speed_threshold is None:
            raise ValueError(
                f"Weather scenario {scenario_name!r} with mode {WINDY_MEAN_ZONE_TRANSPLANT_MODE!r} "
                "must define a numeric 'wind_speed_threshold'."
            )
        return apply_windy_mean_zone_transplant(
            raw_features,
            processed,
            recipient_hex_ids=recipient_hex_ids,
            donor_hex_ids=donor_hex_ids,
            wind_speed_threshold=float(wind_speed_threshold),
            scenario_name=scenario_name,
        )

    return apply_external_mean_zone_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
    )
