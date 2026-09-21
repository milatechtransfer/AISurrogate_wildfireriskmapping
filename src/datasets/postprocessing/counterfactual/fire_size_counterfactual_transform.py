"""Materialize q3 fire-size inputs from spread-event-day counterfactuals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig

FIRE_SIZE_INTERVENTION_CSV_NAME = "fire_size_quantile_intervention.csv"
FIRE_SIZE_GLOBAL_FILL_CSV_NAME = "fire_size_quantile_global_fill.csv"
FIRE_SIZE_NORM_PARAMS_FILENAME = "fire_size_norm_params.json"
SPREAD_DAY_SCALING_MODE = "spread_day_quantile_scaling"
SUPPORTED_FIRE_SIZE_QUANTILES = (0.1, 0.5, 0.9)
NORMALIZATION_EPSILON = 1e-5

_SPREAD_DISTRIBUTION_PATTERN = re.compile(
    r"Spread Event Distribution\s*-\s*(fru\d+)\s*-\s*(s\d+)",
    flags=re.IGNORECASE,
)
_FIRE_ZONE_PATTERN = re.compile(r"fru0*(\d+)", flags=re.IGNORECASE)
_SEASON_PATTERN = re.compile(r"s?0*(\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class FireSizeCounterfactualResult:
    """The q3 lookup tables and audit summary written for one scenario."""

    edited_csv_path: Path
    global_fill_csv_path: Path
    feature_columns: tuple[str, ...]
    summary: pd.DataFrame


@dataclass(frozen=True)
class _SpreadPmf:
    days: np.ndarray
    probabilities: np.ndarray
    original_total_percent: float


def fire_size_intervention_csv_path(prediction_dir: Path) -> Path:
    return prediction_dir / "fire_size_intervention" / FIRE_SIZE_INTERVENTION_CSV_NAME


def fire_size_global_fill_csv_path(prediction_dir: Path) -> Path:
    return prediction_dir / "fire_size_intervention" / FIRE_SIZE_GLOBAL_FILL_CSV_NAME


def _require_columns(frame: pd.DataFrame, columns: list[str], *, source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}.")


def _numeric_series(frame: pd.DataFrame, column: str, *, source: Path | str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & numeric.isna()
    if invalid.any():
        value = frame.loc[invalid, column].iloc[0]
        raise ValueError(f"{source} column {column!r} contains non-numeric value {value!r}.")
    finite = numeric.notna() & ~np.isfinite(numeric)
    if finite.any():
        value = numeric.loc[finite].iloc[0]
        raise ValueError(f"{source} column {column!r} contains non-finite value {value!r}.")
    return numeric


def _fire_zone_id(value: object, *, source: Path | str) -> int:
    match = _FIRE_ZONE_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"{source} contains invalid fire-zone label {value!r}; expected values such as 'fru10'.")
    return int(match.group(1))


def _season_id(value: object, *, source: Path | str) -> str:
    match = _SEASON_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"{source} contains invalid season label {value!r}; expected values such as 's1'.")
    return f"s{int(match.group(1))}"


def _single_file(directory: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {label} in {directory} matching {pattern!r}; found {len(matches)}.")
    return matches[0]


def ignition_season_weights(frame: pd.DataFrame, *, source: Path | str = "ignition distribution") -> dict[int, dict[str, float]]:
    """Aggregate cause-specific ignition likelihoods into P(season | fire zone)."""
    _require_columns(frame, ["FireZone", "Season", "RelativeLikelihood"], source=source)
    likelihood = _numeric_series(frame, "RelativeLikelihood", source=source)
    if likelihood.isna().any():
        raise ValueError(f"{source} column 'RelativeLikelihood' contains missing values.")
    if (likelihood < 0).any():
        raise ValueError(f"{source} column 'RelativeLikelihood' contains negative values.")

    parsed = pd.DataFrame(
        {
            "fire_zone": [_fire_zone_id(value, source=source) for value in frame["FireZone"]],
            "season": [_season_id(value, source=source) for value in frame["Season"]],
            "likelihood": likelihood.astype(float),
        }
    )
    weights: dict[int, dict[str, float]] = {}
    for fire_zone, zone_frame in parsed.groupby("fire_zone", sort=True):
        by_season = zone_frame.groupby("season", sort=True)["likelihood"].sum()
        positive = by_season.loc[by_season > 0]
        total = float(positive.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"{source} has no positive ignition likelihood for fire zone {fire_zone}.")
        weights[int(fire_zone)] = {str(season): float(value / total) for season, value in positive.items()}
    if not weights:
        raise ValueError(f"{source} contains no fire-zone seasonal likelihoods.")
    return weights


def _spread_pmfs(
    frame: pd.DataFrame,
    *,
    source: Path | str,
    total_tolerance_percent: float,
) -> dict[tuple[int, str], _SpreadPmf]:
    _require_columns(frame, ["Name", "Value", "RelativeFrequency"], source=source)
    pmfs: dict[tuple[int, str], _SpreadPmf] = {}
    for name, distribution in frame.groupby("Name", sort=False):
        match = _SPREAD_DISTRIBUTION_PATTERN.fullmatch(str(name).strip())
        if match is None:
            continue
        fire_zone = _fire_zone_id(match.group(1), source=source)
        season = _season_id(match.group(2), source=source)
        days = _numeric_series(distribution, "Value", source=source)
        frequencies = _numeric_series(distribution, "RelativeFrequency", source=source)
        if days.isna().any() or frequencies.isna().any():
            raise ValueError(f"{source} spread distribution {name!r} contains missing values.")
        if (frequencies < 0).any():
            raise ValueError(f"{source} spread distribution {name!r} contains negative frequency.")
        rounded_days = np.rint(days.to_numpy(dtype=np.float64))
        if not np.allclose(days.to_numpy(dtype=np.float64), rounded_days, atol=1e-6):
            raise ValueError(f"{source} spread distribution {name!r} contains non-integer spread-event days.")

        total = float(frequencies.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"{source} spread distribution {name!r} has non-positive total frequency {total}.")
        if abs(total - 100.0) > total_tolerance_percent:
            raise ValueError(
                f"{source} spread distribution {name!r} sums to {total:.6g}%, outside " f"100 +/- {total_tolerance_percent:g}%."
            )

        mass = (
            pd.DataFrame({"day": rounded_days.astype(np.int64), "frequency": frequencies.to_numpy(dtype=np.float64)})
            .groupby("day", sort=True)["frequency"]
            .sum()
        )
        mass = mass.loc[mass > 0]
        if mass.empty:
            raise ValueError(f"{source} spread distribution {name!r} has no positive probability mass.")
        key = (fire_zone, season)
        if key in pmfs:
            raise ValueError(f"{source} contains duplicate spread distribution for fire zone {fire_zone}, season {season}.")
        pmfs[key] = _SpreadPmf(
            days=mass.index.to_numpy(dtype=np.int64),
            probabilities=mass.to_numpy(dtype=np.float64) / float(mass.sum()),
            original_total_percent=total,
        )
    if not pmfs:
        raise ValueError(f"{source} contains no 'Spread Event Distribution - fruNN - sN' rows.")
    return pmfs


def mixture_quantiles(
    seasonal_pmfs: dict[str, tuple[np.ndarray, np.ndarray]],
    seasonal_weights: dict[str, float],
    quantiles: tuple[float, ...],
) -> dict[float, float]:
    """Return inverse-CDF quantiles of an exact weighted mixture of discrete PMFs."""
    if not seasonal_weights:
        raise ValueError("seasonal_weights must not be empty.")
    weights = np.asarray(list(seasonal_weights.values()), dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("seasonal_weights must be finite, non-negative, and have positive total mass.")
    normalized_weights = {season: float(weight / weights.sum()) for season, weight in seasonal_weights.items()}

    mixture_mass: dict[int, float] = {}
    for season, weight in normalized_weights.items():
        if season not in seasonal_pmfs:
            raise ValueError(f"Missing spread-event-day PMF for season {season!r}.")
        days, probabilities = seasonal_pmfs[season]
        days = np.asarray(days, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if days.ndim != 1 or probabilities.ndim != 1 or len(days) != len(probabilities) or len(days) == 0:
            raise ValueError(f"Invalid spread-event-day PMF shape for season {season!r}.")
        if not np.isfinite(days).all() or not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError(f"Spread-event-day PMF for season {season!r} contains invalid values.")
        if not np.allclose(days, np.rint(days), atol=1e-6):
            raise ValueError(f"Spread-event-day PMF for season {season!r} contains non-integer days.")
        probability_total = float(probabilities.sum())
        if probability_total <= 0:
            raise ValueError(f"Spread-event-day PMF for season {season!r} has non-positive mass.")
        for day, probability in zip(np.rint(days).astype(np.int64), probabilities / probability_total, strict=True):
            mixture_mass[int(day)] = mixture_mass.get(int(day), 0.0) + weight * float(probability)

    ordered_days = np.asarray(sorted(mixture_mass), dtype=np.float64)
    probabilities = np.asarray([mixture_mass[int(day)] for day in ordered_days], dtype=np.float64)
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0

    result: dict[float, float] = {}
    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError(f"Mixture quantile must lie strictly between 0 and 1, got {quantile}.")
        index = int(np.searchsorted(cumulative, quantile, side="left"))
        result[quantile] = float(ordered_days[index])
    return result


def _baseline_quantiles(
    frame: pd.DataFrame,
    *,
    zone_id_col: str,
    feature_name: str,
    quantiles: tuple[float, ...],
    source: Path,
) -> tuple[dict[int, dict[float, float]], dict[float, float]]:
    _require_columns(frame, [zone_id_col, feature_name], source=source)
    zone_ids = _numeric_series(frame, zone_id_col, source=source)
    present_zone_ids = zone_ids.dropna().to_numpy(dtype=np.float64)
    if not np.allclose(present_zone_ids, np.rint(present_zone_ids), atol=1e-6):
        raise ValueError(f"{source} column {zone_id_col!r} contains non-integer fire-zone IDs.")
    feature = _numeric_series(frame, feature_name, source=source)
    prepared = pd.DataFrame({"fire_zone": zone_ids.astype("Int64"), "feature": feature})

    by_zone: dict[int, dict[float, float]] = {}
    for fire_zone, zone_frame in prepared.groupby("fire_zone", dropna=True, sort=True):
        values = zone_frame["feature"].dropna()
        if values.empty:
            continue
        by_zone[int(fire_zone)] = {quantile: float(values.quantile(quantile)) for quantile in quantiles}
    if not by_zone:
        raise ValueError(f"{source} contains no usable per-zone values for feature {feature_name!r}.")

    global_values = feature.dropna()
    if global_values.empty:
        raise ValueError(f"{source} contains no usable values for feature {feature_name!r}.")
    global_quantiles = {quantile: float(global_values.quantile(quantile)) for quantile in quantiles}
    return by_zone, global_quantiles


def _normalization_params(path: Path) -> tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Fire-size normalization parameters not found: {path}")
    with path.open() as handle:
        raw = json.load(handle)
    try:
        minimum = float(raw["log_size_min"])
        maximum = float(raw["log_size_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} must define numeric 'log_size_min' and 'log_size_max'.") from error
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError(f"{path} has invalid log-size range [{minimum}, {maximum}].")
    return minimum, maximum


def _feature_to_hectares(value: float, *, minimum: float, maximum: float) -> float:
    hectares = 10 ** (minimum + value * ((maximum - minimum) + NORMALIZATION_EPSILON)) - 1.0
    if not np.isfinite(hectares) or hectares < 0:
        raise ValueError(f"Normalized fire-size feature {value} maps to invalid area {hectares}.")
    return float(hectares)


def _hectares_to_feature(value: float, *, minimum: float, maximum: float) -> float:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Fire size must be finite and non-negative, got {value}.")
    return float((np.log10(value + 1.0) - minimum) / ((maximum - minimum) + NORMALIZATION_EPSILON))


def _quantile_column(feature_name: str, quantile: float) -> str:
    return f"{feature_name}_q{int(round(100 * quantile)):02d}"


def _required_number(params: dict[str, object], name: str, *, minimum_exclusive: float | None = None) -> float:
    if name not in params or isinstance(params[name], bool):
        raise ValueError(f"Fire-size scenario must define numeric {name!r}.")
    raw_value = params.pop(name)
    try:
        value = float(str(raw_value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Fire-size scenario must define numeric {name!r}.") from error
    if not np.isfinite(value):
        raise ValueError(f"Fire-size scenario parameter {name!r} must be finite.")
    if minimum_exclusive is not None and value <= minimum_exclusive:
        raise ValueError(f"Fire-size scenario parameter {name!r} must be greater than {minimum_exclusive}.")
    return value


def materialize_fire_size_scenario(
    *,
    scenario: ScenarioConfig,
    raw_data_dir: Path,
    processed_fire_size_csv: Path,
    recipient_hex_ids: list[str],
    prediction_dir: Path,
    feature_name: str,
    zone_id_col: str,
    quantiles: list[float] | tuple[float, ...],
) -> FireSizeCounterfactualResult:
    """Write exact q3 inputs for a spread-day-derived fire-size intervention."""
    params: dict[str, object] = dict(scenario.fire_size_edit() or {})
    mode = params.pop("mode", None)
    if mode != SPREAD_DAY_SCALING_MODE:
        raise ValueError(f"Fire-size scenario {scenario.name!r} must use mode={SPREAD_DAY_SCALING_MODE!r}; got {mode!r}.")
    delta_q50 = _required_number(params, "spread_day_delta_q50_days")
    delta_q90 = _required_number(params, "spread_day_delta_q90_days")
    exponent = _required_number(params, "size_scaling_exponent", minimum_exclusive=0.0)
    raw_tolerance = params.pop("pmf_total_tolerance_percent", 1.0)
    if isinstance(raw_tolerance, bool):
        raise ValueError("Fire-size scenario parameter 'pmf_total_tolerance_percent' must be numeric.")
    try:
        tolerance = float(str(raw_tolerance))
    except ValueError as error:
        raise ValueError("Fire-size scenario parameter 'pmf_total_tolerance_percent' must be numeric.") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Fire-size scenario parameter 'pmf_total_tolerance_percent' must be finite and non-negative.")
    if params:
        raise ValueError(f"Unknown fire-size scenario parameter(s): {sorted(params)}.")

    configured_quantiles = tuple(float(value) for value in quantiles)
    if configured_quantiles != SUPPORTED_FIRE_SIZE_QUANTILES:
        raise ValueError(
            f"Spread-day fire-size scenarios require q10/q50/q90 inputs {SUPPORTED_FIRE_SIZE_QUANTILES}; " f"got {configured_quantiles}."
        )
    if feature_name != "NORM_LOG_SIZE_HA":
        raise ValueError(
            "Spread-day fire-size scenarios currently require the normalized-log feature " f"'NORM_LOG_SIZE_HA'; got {feature_name!r}."
        )

    normalized_hex_ids = [normalize_hex_id(hex_id) for hex_id in recipient_hex_ids]
    if not normalized_hex_ids:
        raise ValueError("recipient_hex_ids must be non-empty.")
    if len(set(normalized_hex_ids)) != len(normalized_hex_ids):
        raise ValueError(f"recipient_hex_ids contains duplicates after normalization: {normalized_hex_ids}.")

    processed = pd.read_csv(processed_fire_size_csv)
    baseline_by_zone, global_baseline = _baseline_quantiles(
        processed,
        zone_id_col=zone_id_col,
        feature_name=feature_name,
        quantiles=configured_quantiles,
        source=processed_fire_size_csv,
    )
    minimum, maximum = _normalization_params(processed_fire_size_csv.parent / FIRE_SIZE_NORM_PARAMS_FILENAME)
    feature_columns = tuple(_quantile_column(feature_name, quantile) for quantile in configured_quantiles)

    lookup: dict[tuple[int, int], dict[str, float | int]] = {}
    for hex_id in normalized_hex_ids:
        for fire_zone, baseline in baseline_by_zone.items():
            lookup[(int(hex_id), fire_zone)] = {
                "hex_id": int(hex_id),
                zone_id_col: fire_zone,
                **{column: baseline[quantile] for quantile, column in zip(configured_quantiles, feature_columns, strict=True)},
            }

    summary_rows: list[dict[str, object]] = []
    for hex_id in normalized_hex_ids:
        tabular_dir = raw_data_dir / f"hex{hex_id}" / "tabular"
        ignition_path = _single_file(
            tabular_dir,
            f"hex{hex_id}_IgnitionDistribution.csv",
            label="ignition-distribution CSV",
        )
        spread_path = _single_file(
            tabular_dir,
            f"hex{hex_id}_ScenarioDistributions - * - FINAL.csv",
            label="scenario-distributions CSV",
        )
        weights_by_zone = ignition_season_weights(pd.read_csv(ignition_path), source=ignition_path)
        pmfs = _spread_pmfs(
            pd.read_csv(spread_path),
            source=spread_path,
            total_tolerance_percent=tolerance,
        )

        for fire_zone, seasonal_weights in weights_by_zone.items():
            if fire_zone not in baseline_by_zone:
                raise ValueError(
                    f"{processed_fire_size_csv} has no baseline fire-size distribution for " f"hex{hex_id} fire zone {fire_zone}."
                )
            missing_seasons = [season for season in seasonal_weights if (fire_zone, season) not in pmfs]
            if missing_seasons:
                raise ValueError(
                    f"{spread_path} is missing spread distributions for hex{hex_id} fire zone "
                    f"{fire_zone}, season(s) {missing_seasons} with positive ignition likelihood."
                )
            seasonal_pmfs = {
                season: (
                    pmfs[(fire_zone, season)].days,
                    pmfs[(fire_zone, season)].probabilities,
                )
                for season in seasonal_weights
            }
            spread_quantiles = mixture_quantiles(seasonal_pmfs, seasonal_weights, (0.5, 0.9))
            spread_q50 = spread_quantiles[0.5]
            spread_q90 = spread_quantiles[0.9]
            if spread_q50 <= 0 or spread_q90 <= 0:
                raise ValueError(
                    f"Spread-day quantiles must be positive for hex{hex_id} fire zone {fire_zone}; "
                    f"got q50={spread_q50}, q90={spread_q90}."
                )
            if spread_q50 + delta_q50 <= 0 or spread_q90 + delta_q90 <= 0:
                raise ValueError(f"Spread-day deltas make a future quantile non-positive for hex{hex_id} " f"fire zone {fire_zone}.")

            multiplier_q50 = ((spread_q50 + delta_q50) / spread_q50) ** exponent
            multiplier_q90 = ((spread_q90 + delta_q90) / spread_q90) ** exponent
            baseline_features = baseline_by_zone[fire_zone]
            baseline_hectares = {
                quantile: _feature_to_hectares(
                    baseline_features[quantile],
                    minimum=minimum,
                    maximum=maximum,
                )
                for quantile in configured_quantiles
            }
            future_hectares = {
                0.1: baseline_hectares[0.1],
                0.5: baseline_hectares[0.5] * multiplier_q50,
                0.9: baseline_hectares[0.9] * multiplier_q90,
            }
            if not future_hectares[0.1] <= future_hectares[0.5] <= future_hectares[0.9]:
                raise ValueError(
                    f"Fire-size intervention violates q10 <= q50 <= q90 for hex{hex_id} " f"fire zone {fire_zone}: {future_hectares}."
                )
            future_features = {
                quantile: _hectares_to_feature(
                    future_hectares[quantile],
                    minimum=minimum,
                    maximum=maximum,
                )
                for quantile in configured_quantiles
            }
            lookup_row = lookup[(int(hex_id), fire_zone)]
            for quantile, column in zip(configured_quantiles, feature_columns, strict=True):
                lookup_row[column] = future_features[quantile]

            pmf_totals = [pmfs[(fire_zone, season)].original_total_percent for season in seasonal_weights]
            summary_rows.append(
                {
                    "scenario_name": scenario.name,
                    "mode": mode,
                    "hex_id": hex_id,
                    "fire_zone": fire_zone,
                    "season_weights": json.dumps(seasonal_weights, sort_keys=True),
                    "spread_day_q50": spread_q50,
                    "spread_day_q90": spread_q90,
                    "spread_day_delta_q50_days": delta_q50,
                    "spread_day_delta_q90_days": delta_q90,
                    "size_scaling_exponent": exponent,
                    "fire_size_multiplier_q50": multiplier_q50,
                    "fire_size_multiplier_q90": multiplier_q90,
                    "baseline_fire_size_q10_ha": baseline_hectares[0.1],
                    "baseline_fire_size_q50_ha": baseline_hectares[0.5],
                    "baseline_fire_size_q90_ha": baseline_hectares[0.9],
                    "future_fire_size_q10_ha": future_hectares[0.1],
                    "future_fire_size_q50_ha": future_hectares[0.5],
                    "future_fire_size_q90_ha": future_hectares[0.9],
                    "baseline_feature_q10": baseline_features[0.1],
                    "baseline_feature_q50": baseline_features[0.5],
                    "baseline_feature_q90": baseline_features[0.9],
                    "future_feature_q10": future_features[0.1],
                    "future_feature_q50": future_features[0.5],
                    "future_feature_q90": future_features[0.9],
                    "component_frequency_total_min_percent": min(pmf_totals),
                    "component_frequency_total_max_percent": max(pmf_totals),
                    "ignition_distribution_file": ignition_path.name,
                    "scenario_distributions_file": spread_path.name,
                }
            )

    out_path = fire_size_intervention_csv_path(prediction_dir)
    fill_path = fire_size_global_fill_csv_path(prediction_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_frame = pd.DataFrame(lookup.values()).sort_values(["hex_id", zone_id_col]).reset_index(drop=True)
    lookup_frame.to_csv(out_path, index=False)
    fill_frame = pd.DataFrame(
        [
            {
                "hex_id": int(hex_id),
                **{column: global_baseline[quantile] for quantile, column in zip(configured_quantiles, feature_columns, strict=True)},
            }
            for hex_id in normalized_hex_ids
        ]
    )
    fill_frame.to_csv(fill_path, index=False)
    summary = pd.DataFrame(summary_rows).sort_values(["hex_id", "fire_zone"]).reset_index(drop=True)
    return FireSizeCounterfactualResult(
        edited_csv_path=out_path,
        global_fill_csv_path=fill_path,
        feature_columns=feature_columns,
        summary=summary,
    )
