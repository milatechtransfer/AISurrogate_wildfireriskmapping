"""Weather-table editing helpers for counterfactual analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
WINDY_MEAN_ZONE_DEPENDENT_TRANSPLANT_MODE = "windy_mean_zone_dependent_transplant"
WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE = "wind_direction_zone_dependent_transplant"
WEATHER_EDIT_MODES = (
    EXTERNAL_MEAN_ZONE_TRANSPLANT_MODE,
    WINDY_MEAN_ZONE_TRANSPLANT_MODE,
    WIND_DIRECTION_ZONE_TRANSPLANT_MODE,
    WINDY_MEAN_ZONE_DEPENDENT_TRANSPLANT_MODE,
    WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE,
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
    # `wind_direction_zone_transplant`, and their zone-dependent variants); None
    # otherwise. `wind_speed_percentile` is the configured cutoff (0 = every donor
    # row); `wind_speed_threshold_kmh` is the raw WindSpeed value it resolved to for
    # this donor pool, for reference.
    wind_speed_percentile: float | None = None
    wind_speed_threshold_kmh: float | None = None
    # Set only for `wind_direction_zone_transplant` and its zone-dependent variant;
    # None otherwise.
    direction_degrees: float | None = None
    # Set only for zone-dependent modes (one report row per recipient (hex_id,
    # WeatherZone) pair); None for whole-hex modes.
    weather_zone: int | None = None


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
        raise ValueError(f"Raw/processed hex_id mismatch at row {row}: {raw_hex_ids.iloc[row]!r} != {processed_hex_ids.iloc[row]!r}.")

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


def _percentile_wind_speed_filter(
    wind_speed: np.ndarray,
    base_mask: np.ndarray,
    percentile: float,
    *,
    context: str,
) -> tuple[np.ndarray, float]:
    """Restrict `base_mask` to rows at/above the `percentile`-th WindSpeed value within it.

    `percentile=0` resolves to the pool's minimum and therefore keeps every row in
    `base_mask` (its ordinary/average wind speed); `percentile=90` keeps its windiest
    10%. Returns the filtered mask and the raw WindSpeed cutoff it resolved to.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"wind_speed_percentile must be within [0, 100]; got {percentile} for {context}.")
    if not base_mask.any():
        raise ValueError(f"No weather rows available to compute a wind_speed_percentile for {context}.")
    threshold_value = float(np.percentile(wind_speed[base_mask], percentile))
    filtered_mask = base_mask & (wind_speed >= threshold_value)
    if not filtered_mask.any():
        raise ValueError(
            f"No weather rows with {WIND_SPEED_COLUMN} >= {threshold_value} (the {percentile}th percentile) found for {context}."
        )
    return filtered_mask, threshold_value


def _apply_mean_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    scenario_name: str,
    mode: str,
    wind_speed_percentile: float | None = None,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the exact mean processed-weather vector of donor hexels.

    The returned table contains one row per ``(hex_id, WeatherZone)``. Baseline
    hex-zone means are retained everywhere except recipient hexels, whose zone rows
    receive the donor mean exactly. `donor_hex_ids` may overlap or exactly match
    `recipient_hex_ids` (e.g. a hexel can donate its own filtered rows to itself).
    When `wind_speed_percentile` is set, the donor mean is restricted to donor rows
    whose raw `WindSpeed` is at/above that percentile of the donor pool's own
    WindSpeed distribution (0 = every donor row, 90 = its windiest 10%).
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

    wind_speed_threshold_kmh: float | None = None
    if wind_speed_percentile is not None:
        if WIND_SPEED_COLUMN not in raw_features.columns:
            raise ValueError(f"raw_features is missing {WIND_SPEED_COLUMN!r}, required to apply a wind_speed_percentile.")
        wind_speed = pd.to_numeric(raw_features[WIND_SPEED_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
        donor_mask, wind_speed_threshold_kmh = _percentile_wind_speed_filter(
            wind_speed, donor_mask, wind_speed_percentile, context=f"donor_hex_ids={normalized_donor_ids}"
        )

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
                wind_speed_percentile=wind_speed_percentile,
                wind_speed_threshold_kmh=wind_speed_threshold_kmh,
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
    wind_speed_percentile: float,
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the mean processed-weather vector of donor rows with high wind.

    Identical to `apply_external_mean_zone_transplant`, except the donor mean is
    computed only over donor rows at/above `wind_speed_percentile` of the donor
    pool's own raw WindSpeed distribution (0 = every donor row / its ordinary
    average wind speed, 90 = its windiest 10%). `donor_hex_ids` may equal
    `recipient_hex_ids` to give a hexel the mean of its own windiest days instead of
    an external donor's.
    """
    return _apply_mean_zone_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
        mode=WINDY_MEAN_ZONE_TRANSPLANT_MODE,
        wind_speed_percentile=wind_speed_percentile,
    )


def _load_wind_component_norm_params(norm_params_path: str | Path) -> tuple[float, float, float, float]:
    """Return `(wind_x_mean, wind_x_std, wind_y_mean, wind_y_std)` z-score parameters."""
    norm_params = load_weather_normalization_params(norm_params_path)
    z_cols = list(norm_params["z_score"]["cols"])
    if WIND_X_COLUMN not in z_cols or WIND_Y_COLUMN not in z_cols:
        raise ValueError(f"{norm_params_path} does not define z-score parameters for {WIND_X_COLUMN!r}/{WIND_Y_COLUMN!r}.")
    wind_x_mean = float(norm_params["z_score"]["mean"][z_cols.index(WIND_X_COLUMN)])
    wind_x_std = float(norm_params["z_score"]["std"][z_cols.index(WIND_X_COLUMN)]) or 1.0
    wind_y_mean = float(norm_params["z_score"]["mean"][z_cols.index(WIND_Y_COLUMN)])
    wind_y_std = float(norm_params["z_score"]["std"][z_cols.index(WIND_Y_COLUMN)]) or 1.0
    return wind_x_mean, wind_x_std, wind_y_mean, wind_y_std


def _forced_direction_donor_mean(
    processed: pd.DataFrame,
    mean_columns: list[str],
    donor_mask: np.ndarray,
    *,
    wind_speed: np.ndarray,
    direction_degrees: float,
    wind_component_norm_params: tuple[float, float, float, float],
) -> pd.Series:
    """Donor mean vector with `wind_x`/`wind_y` recomputed for a forced compass direction."""
    wind_x_mean, wind_x_std, wind_y_mean, wind_y_std = wind_component_norm_params
    donor_wind_speed = wind_speed[donor_mask]
    forced_direction = np.full(donor_wind_speed.shape, float(direction_degrees), dtype=np.float64)
    raw_wind_x, raw_wind_y = wind_to_components(ws=donor_wind_speed, wd=forced_direction)

    donor_subset = processed.loc[donor_mask, mean_columns].copy()
    donor_subset[WIND_X_COLUMN] = (raw_wind_x - wind_x_mean) / wind_x_std
    donor_subset[WIND_Y_COLUMN] = (raw_wind_y - wind_y_mean) / wind_y_std
    donor_mean = donor_subset.mean(axis=0)
    if not np.isfinite(donor_mean.to_numpy(dtype=np.float64)).all():
        raise ValueError("Donor mean weather vector contains non-finite values.")
    return donor_mean


def apply_wind_direction_zone_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    direction_degrees: float,
    wind_speed_percentile: float,
    norm_params_path: str | Path,
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Give recipient hexels the donor mean weather vector with wind forced to blow
    from a fixed compass direction.

    Donor rows are first filtered to raw `WindSpeed` at/above `wind_speed_percentile`
    of the donor pool's own WindSpeed distribution (0 = include every donor row and
    use its ordinary/average wind speed instead of only its windiest days). Each
    surviving donor row keeps its own recorded `WindSpeed` magnitude but has its
    `WindDirection` overridden to `direction_degrees` before `wind_x`/`wind_y` are
    recomputed and re-normalized with the same z-score parameters fit at training
    time (loaded from `norm_params_path`, typically `weather_norm_params.json` next
    to the processed weather table). All other weather columns are averaged
    unchanged. Prefer a self-donor (`donor_hex_ids == recipient_hex_ids`) to isolate
    the pure direction effect; an external donor also imports that donor's non-wind
    climate, which conflates two interventions.
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
    filtered_donor_mask, wind_speed_threshold_kmh = _percentile_wind_speed_filter(
        wind_speed, donor_mask, wind_speed_percentile, context=f"donor_hex_ids={normalized_donor_ids}"
    )

    wind_component_norm_params = _load_wind_component_norm_params(norm_params_path)

    mean_columns = [
        column for column in processed.columns if column not in NON_AVERAGE_COLUMNS and pd.api.types.is_numeric_dtype(processed[column])
    ]
    donor_mean = _forced_direction_donor_mean(
        processed,
        mean_columns,
        filtered_donor_mask,
        wind_speed=wind_speed,
        direction_degrees=direction_degrees,
        wind_component_norm_params=wind_component_norm_params,
    )

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
                wind_speed_percentile=float(wind_speed_percentile),
                wind_speed_threshold_kmh=wind_speed_threshold_kmh,
                direction_degrees=float(direction_degrees),
            )
        )
    return edited, reports


def _apply_mean_zone_dependent_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    scenario_name: str,
    mode: str,
    wind_speed_percentile: float,
    direction_degrees: float | None = None,
    norm_params_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Shared zone-scoped implementation for the `*_zone_dependent_transplant` modes.

    Unlike `_apply_mean_zone_transplant`/`apply_wind_direction_zone_transplant`
    (which pool donor rows across the whole donor hex and broadcast one identical
    vector to every recipient WeatherZone), this groups the donor pool by
    WeatherZone: each recipient `(hex_id, WeatherZone)` receives a donor mean
    computed only from donor rows sharing that same WeatherZone id, so zone-to-zone
    weather heterogeneity is preserved instead of collapsed to one hex-wide value.
    `wind_speed_percentile` is resolved separately within each zone's own donor pool
    (0 = every donor row in that zone, 90 = that zone's own windiest 10%), since
    zones can have quite different wind climatologies. Pass `direction_degrees` (and
    `norm_params_path`) to also force wind direction per zone (as
    `wind_direction_zone_dependent_transplant` does); leave both `None` for a
    zone-dependent windy-mean transplant. Emits one `WeatherEditReport` per
    recipient `(hex_id, WeatherZone)` pair.
    """
    if FWI_COLUMN not in raw_features.columns or FWI_COLUMN not in processed.columns:
        raise ValueError(f"Raw and processed weather tables must include {FWI_COLUMN!r}.")
    if WIND_SPEED_COLUMN not in raw_features.columns:
        raise ValueError(f"raw_features is missing {WIND_SPEED_COLUMN!r}, required for {mode!r}.")
    wind_component_norm_params: tuple[float, float, float, float] | None = None
    if direction_degrees is not None:
        if WIND_X_COLUMN not in processed.columns or WIND_Y_COLUMN not in processed.columns:
            raise ValueError(f"Processed weather table must include {WIND_X_COLUMN!r} and {WIND_Y_COLUMN!r}.")
        if norm_params_path is None:
            raise ValueError(f"{mode!r} requires norm_params_path to re-normalize the forced wind_x/wind_y vector.")
        wind_component_norm_params = _load_wind_component_norm_params(norm_params_path)

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
    processed_zones = pd.to_numeric(processed[WEATHER_ZONE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)

    mean_columns = [
        column for column in processed.columns if column not in NON_AVERAGE_COLUMNS and pd.api.types.is_numeric_dtype(processed[column])
    ]

    edited = processed.groupby([HEX_ID_COLUMN, WEATHER_ZONE_COLUMN], as_index=False)[mean_columns].mean()
    edited_hex_ids = _validated_processed_hex_ids(edited)
    edited_zones = pd.to_numeric(edited[WEATHER_ZONE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    edited_recipient_mask = edited_hex_ids.isin(normalized_recipient_ids).to_numpy()

    any_recipient_mask = np.zeros(len(processed), dtype=bool)
    for mask in recipient_masks.values():
        any_recipient_mask |= mask
    recipient_zone_ids = sorted({float(zone) for zone in processed_zones[any_recipient_mask] if np.isfinite(zone)})

    donor_hex_label = ",".join(normalized_donor_ids)
    reports: list[WeatherEditReport] = []
    for zone in recipient_zone_ids:
        zone_donor_mask = donor_mask & (processed_zones == zone)
        if not zone_donor_mask.any():
            raise ValueError(f"No donor weather rows found for WeatherZone={zone!r} among donor_hex_ids={normalized_donor_ids}.")

        filtered_zone_donor_mask, wind_speed_threshold_kmh = _percentile_wind_speed_filter(
            wind_speed,
            zone_donor_mask,
            wind_speed_percentile,
            context=f"donor_hex_ids={normalized_donor_ids}, WeatherZone={zone!r}",
        )

        if direction_degrees is not None:
            assert wind_component_norm_params is not None
            donor_mean = _forced_direction_donor_mean(
                processed,
                mean_columns,
                filtered_zone_donor_mask,
                wind_speed=wind_speed,
                direction_degrees=direction_degrees,
                wind_component_norm_params=wind_component_norm_params,
            )
        else:
            donor_mean = processed.loc[filtered_zone_donor_mask, mean_columns].mean(axis=0)
            if not np.isfinite(donor_mean.to_numpy(dtype=np.float64)).all():
                raise ValueError("Donor mean weather vector contains non-finite values.")

        zone_edit_mask = edited_recipient_mask & (edited_zones == zone)
        for column in mean_columns:
            edited.loc[zone_edit_mask, column] = float(donor_mean[column])

        donor_fwi_mean = float(raw_features.loc[filtered_zone_donor_mask, FWI_COLUMN].mean())
        for recipient_hex_id in normalized_recipient_ids:
            recipient_zone_mask = recipient_masks[recipient_hex_id] & (processed_zones == zone)
            if not recipient_zone_mask.any():
                continue
            reports.append(
                WeatherEditReport(
                    scenario_name=scenario_name,
                    mode=mode,
                    recipient_hex_id=recipient_hex_id,
                    n_recipient_rows=int(recipient_zone_mask.sum()),
                    n_recipient_zones=1,
                    donor_hex_ids=donor_hex_label,
                    n_donor_rows=int(filtered_zone_donor_mask.sum()),
                    donor_fwi_mean=donor_fwi_mean,
                    baseline_fwi_mean=float(raw_features.loc[recipient_zone_mask, FWI_COLUMN].mean()),
                    scenario_fwi_mean=donor_fwi_mean,
                    wind_speed_percentile=float(wind_speed_percentile),
                    wind_speed_threshold_kmh=wind_speed_threshold_kmh,
                    direction_degrees=float(direction_degrees) if direction_degrees is not None else None,
                    weather_zone=int(zone),
                )
            )
    return edited, reports


def apply_windy_mean_zone_dependent_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    wind_speed_percentile: float,
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Zone-scoped `apply_windy_mean_zone_transplant`.

    Each recipient WeatherZone gets the mean of its own (donor-hex, same-zone)
    windiest rows, instead of one hex-wide donor mean broadcast to every zone. See
    `_apply_mean_zone_dependent_transplant` for the shared implementation.
    """
    return _apply_mean_zone_dependent_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
        mode=WINDY_MEAN_ZONE_DEPENDENT_TRANSPLANT_MODE,
        wind_speed_percentile=wind_speed_percentile,
    )


def apply_wind_direction_zone_dependent_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    direction_degrees: float,
    wind_speed_percentile: float,
    norm_params_path: str | Path,
    scenario_name: str,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Zone-scoped `apply_wind_direction_zone_transplant`.

    Each recipient WeatherZone keeps its own (donor-hex, same-zone) wind magnitude
    but forced to `direction_degrees`, instead of one hex-wide forced-direction
    vector broadcast to every zone. See `_apply_mean_zone_dependent_transplant` for
    the shared implementation.
    """
    return _apply_mean_zone_dependent_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
        mode=WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE,
        wind_speed_percentile=wind_speed_percentile,
        direction_degrees=direction_degrees,
        norm_params_path=norm_params_path,
    )


def _required_param(params: dict, key: str, *, scenario_name: str, mode: str, hint: str = "") -> Any:
    value = params.get(key)
    if value is None:
        suffix = f" ({hint})" if hint else ""
        raise ValueError(f"Weather scenario {scenario_name!r} with mode {mode!r} must define a numeric {key!r}{suffix}.")
    return value


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

    if mode in (WIND_DIRECTION_ZONE_TRANSPLANT_MODE, WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE):
        direction_degrees = _required_param(params, "direction_degrees", scenario_name=scenario_name, mode=mode)
        wind_speed_percentile = _required_param(
            params,
            "wind_speed_percentile",
            scenario_name=scenario_name,
            mode=mode,
            hint="use 0 to include every donor row",
        )
        if norm_params_path is None:
            raise ValueError(
                f"Weather scenario {scenario_name!r} with mode {mode!r} requires norm_params_path to re-normalize the forced wind_x/wind_y vector."
            )
        edit_function = (
            apply_wind_direction_zone_dependent_transplant
            if mode == WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE
            else apply_wind_direction_zone_transplant
        )
        return edit_function(
            raw_features,
            processed,
            recipient_hex_ids=recipient_hex_ids,
            donor_hex_ids=donor_hex_ids,
            direction_degrees=float(direction_degrees),
            wind_speed_percentile=float(wind_speed_percentile),
            norm_params_path=norm_params_path,
            scenario_name=scenario_name,
        )

    if mode in (WINDY_MEAN_ZONE_TRANSPLANT_MODE, WINDY_MEAN_ZONE_DEPENDENT_TRANSPLANT_MODE):
        wind_speed_percentile = _required_param(params, "wind_speed_percentile", scenario_name=scenario_name, mode=mode)
        windy_edit_function = (
            apply_windy_mean_zone_dependent_transplant
            if mode == WINDY_MEAN_ZONE_DEPENDENT_TRANSPLANT_MODE
            else apply_windy_mean_zone_transplant
        )
        return windy_edit_function(
            raw_features,
            processed,
            recipient_hex_ids=recipient_hex_ids,
            donor_hex_ids=donor_hex_ids,
            wind_speed_percentile=float(wind_speed_percentile),
            scenario_name=scenario_name,
        )

    return apply_external_mean_zone_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=donor_hex_ids,
        scenario_name=scenario_name,
    )
