"""Weather re-encoding validation for fixed-model counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.postprocessing.counterfactual import encoded_components_from_from_bearing, flow_bearing_from_from_bearing

RAW_WIND_COLUMNS: tuple[str, ...] = ("WindSpeed", "wind_x", "wind_y")
WIND_DIRECTION_COLUMN = "WindDirection"


@dataclass(frozen=True)
class WindColumnStats:
    """One z-score transform: processed = (raw - mean) / std."""

    column: str
    mean: float
    std: float


@dataclass(frozen=True)
class WindEncodingStats:
    """Z-score transforms for all model-consumed wind columns."""

    columns: tuple[WindColumnStats, ...]

    def by_name(self) -> dict[str, WindColumnStats]:
        return {stat.column: stat for stat in self.columns}


def normalize_raw_weather_ids(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Return raw weather with numeric Season and WeatherZone columns."""

    df = raw_weather.copy()
    for column in ("Season", "WeatherZone"):
        if column not in df.columns:
            raise ValueError(f"Raw weather table is missing required column {column!r}.")
        if df[column].astype(str).str.match(r"^[a-zA-Z]+\d+$").all():
            df[column] = df[column].astype(str).str.extract(r"(\d+)").astype(int)
    return df


def raw_wind_features(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Compute raw-space wind features before z-score normalization.

    BurnP3+/NRCan WindDirection is a from-bearing.  These raw wind_x/wind_y
    features intentionally match the existing model input encoding, which points
    toward the wind source/upwind direction.
    """

    required = {"WindSpeed", "WindDirection"}
    missing = sorted(required - set(raw_weather.columns))
    if missing:
        raise ValueError(f"Raw weather table is missing required columns: {missing}.")

    wind_speed = raw_weather["WindSpeed"].to_numpy(dtype=np.float64)
    wind_direction = raw_weather["WindDirection"].to_numpy(dtype=np.float64)
    wind_x, wind_y = encoded_components_from_from_bearing(wind_speed, wind_direction)
    return pd.DataFrame(
        {
            "WindSpeed": wind_speed,
            "wind_x": wind_x,
            "wind_y": wind_y,
        },
        index=raw_weather.index,
    )


def recover_wind_encoding_stats(
    raw_features: pd.DataFrame,
    processed_weather: pd.DataFrame,
    columns: tuple[str, ...] = RAW_WIND_COLUMNS,
) -> WindEncodingStats:
    """Recover baseline z-score stats from paired raw and processed rows.

    The data-preparation pipeline does not persist the weather StandardScaler.
    Because processed = (raw - mean) / std, we recover ``mean`` and ``std`` by
    fitting raw = std * processed + mean for each consumed wind column.
    """

    if len(raw_features) != len(processed_weather):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_features)} != {len(processed_weather)}.")

    stats: list[WindColumnStats] = []
    for column in columns:
        if column not in raw_features.columns:
            raise ValueError(f"Raw wind features missing column {column!r}.")
        if column not in processed_weather.columns:
            raise ValueError(f"Processed weather missing column {column!r}.")

        raw = raw_features[column].to_numpy(dtype=np.float64)
        processed = processed_weather[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(raw) & np.isfinite(processed)
        if int(finite.sum()) < 2:
            raise ValueError(f"Need at least two finite rows to recover wind stats for {column!r}.")

        x = processed[finite]
        y = raw[finite]
        var_x = float(np.var(x))
        if var_x <= 0.0:
            raise ValueError(f"Processed column {column!r} has zero variance; cannot recover z-score stats.")

        std = float(np.cov(x, y, bias=True)[0, 1] / var_x)
        mean = float(np.mean(y) - std * np.mean(x))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Invalid recovered z-score stats for {column!r}: mean={mean}, std={std}.")
        stats.append(WindColumnStats(column=column, mean=mean, std=std))

    return WindEncodingStats(columns=tuple(stats))


def encode_raw_wind_features(raw_features: pd.DataFrame, stats: WindEncodingStats) -> pd.DataFrame:
    """Apply recovered baseline z-score stats to raw wind features."""

    encoded = pd.DataFrame(index=raw_features.index)
    stats_by_name = stats.by_name()
    for column in RAW_WIND_COLUMNS:
        if column not in stats_by_name:
            raise ValueError(f"Missing wind encoding stats for {column!r}.")
        stat = stats_by_name[column]
        encoded[column] = (raw_features[column].to_numpy(dtype=np.float64) - stat.mean) / stat.std
    return encoded


def validate_wind_roundtrip(
    raw_weather: pd.DataFrame,
    processed_weather: pd.DataFrame,
    stats: WindEncodingStats | None = None,
    columns: tuple[str, ...] = RAW_WIND_COLUMNS,
) -> tuple[WindEncodingStats, pd.DataFrame]:
    """Recover stats if needed and summarize baseline wind roundtrip errors."""

    raw_features = raw_wind_features(raw_weather)
    if stats is None:
        stats = recover_wind_encoding_stats(raw_features, processed_weather, columns=columns)
    encoded = encode_raw_wind_features(raw_features, stats)

    rows = []
    for column in columns:
        expected = processed_weather[column].to_numpy(dtype=np.float64)
        actual = encoded[column].to_numpy(dtype=np.float64)
        diff = actual - expected
        finite = np.isfinite(diff)
        rows.append(
            {
                "column": column,
                "n_rows": int(finite.sum()),
                "max_abs_error": float(np.max(np.abs(diff[finite]))) if finite.any() else float("nan"),
                "mean_abs_error": float(np.mean(np.abs(diff[finite]))) if finite.any() else float("nan"),
                "rmse": float(np.sqrt(np.mean(diff[finite] ** 2))) if finite.any() else float("nan"),
            }
        )
    return stats, pd.DataFrame(rows)


def raw_weather_path(raw_data_dir: Path, hex_id: str) -> Path:
    return raw_data_dir / f"hex{hex_id}" / "tabular" / f"hex{hex_id}_DailyWeather.csv"


def processed_weather_for_hex(
    raw_data_dir: Path,
    processed_weather_path: Path,
    hex_id: str,
) -> pd.DataFrame:
    """Return the processed global weather rows corresponding to one hexel.

    ``weather_table_processed.csv`` is produced by concatenating raw
    ``hex*/tabular/hex*_DailyWeather.csv`` files in sorted path order, then
    applying global preprocessing.  We reproduce that ordering and slice the
    processed table for the requested hexel.
    """

    target_path = raw_weather_path(raw_data_dir, hex_id)
    files = sorted(raw_data_dir.glob("hex*/tabular/hex*_DailyWeather.csv"))
    if target_path not in files:
        raise FileNotFoundError(f"Missing raw weather table for hex{hex_id}: {target_path}")

    offset = 0
    target_count = None
    for path in files:
        count = int(sum(1 for _ in path.open()) - 1)
        if path == target_path:
            target_count = count
            break
        offset += count
    if target_count is None:
        raise FileNotFoundError(f"Missing raw weather table for hex{hex_id}: {target_path}")

    return pd.read_csv(
        processed_weather_path,
        skiprows=range(1, offset + 1),
        nrows=target_count,
    )


def validate_hex_weather_roundtrip(
    raw_data_dir: Path,
    processed_weather_path: Path,
    hex_id: str,
) -> tuple[WindEncodingStats, pd.DataFrame]:
    """Validate baseline wind re-encoding for one hexel."""

    raw_weather = normalize_raw_weather_ids(pd.read_csv(raw_weather_path(raw_data_dir, hex_id)))
    processed_weather = processed_weather_for_hex(raw_data_dir, processed_weather_path, hex_id)
    return validate_wind_roundtrip(raw_weather=raw_weather, processed_weather=processed_weather)


def _scenario_speed(raw_weather: pd.DataFrame, params: dict, default_percentile: float | None = None) -> np.ndarray:
    speed = raw_weather["WindSpeed"].to_numpy(dtype=np.float64)
    if "fixed_speed" in params:
        return np.full(speed.shape, float(params["fixed_speed"]), dtype=np.float64)
    percentile = params.get("speed_percentile", default_percentile)
    if percentile is None:
        return speed.copy()
    return np.full(speed.shape, float(np.percentile(speed, float(percentile))), dtype=np.float64)


def _dominant_from_bearing(wind_speed: np.ndarray, wind_direction: np.ndarray) -> float:
    """Return the speed-weighted dominant meteorological from-bearing."""

    weighted = np.isfinite(wind_speed) & np.isfinite(wind_direction) & (wind_speed > 0.0)
    finite_direction = np.isfinite(wind_direction)
    if not finite_direction.any():
        raise ValueError("Cannot compute dominant wind direction without finite directions.")

    finite = weighted if weighted.any() else finite_direction
    radians = np.deg2rad(wind_direction[finite])
    weights = wind_speed[finite] if weighted.any() else None
    sum_x = float(np.sum(np.sin(radians) * weights)) if weights is not None else float(np.sum(np.sin(radians)))
    sum_y = float(np.sum(np.cos(radians) * weights)) if weights is not None else float(np.sum(np.cos(radians)))
    if np.isclose(sum_x, 0.0) and np.isclose(sum_y, 0.0):
        return float(np.mean(wind_direction[finite]) % 360.0)
    return float(np.rad2deg(np.arctan2(sum_x, sum_y)) % 360.0)


def _set_zone_dominant_directions(raw_weather: pd.DataFrame, *, zone_column: str) -> pd.Series:
    if zone_column not in raw_weather.columns:
        raise ValueError(f"zone_dominant_direction mode requires {zone_column!r} in raw weather.")

    result = pd.Series(index=raw_weather.index, dtype=np.float64)
    for _, group in raw_weather.groupby(zone_column, dropna=False, sort=False):
        result.loc[group.index] = _dominant_from_bearing(
            group["WindSpeed"].to_numpy(dtype=np.float64),
            group["WindDirection"].to_numpy(dtype=np.float64),
        )
    return result


def build_wind_scenario_raw_weather(
    raw_weather: pd.DataFrame,
    params: dict,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create raw-space weather rows for one wind scenario.

    Supported modes:
    - ``roundtrip_placebo``: no raw wind changes.
    - ``fixed_from_bearing``: fixed source direction plus fixed/percentile speed.
    - ``low_wind``: low-percentile or fixed low speed, direction unchanged unless provided.
    - ``speed_only_high``: high-percentile speed, original directions retained.
    - ``direction_only_rotate``: fixed source direction, original speeds retained.
    - ``zone_dominant_direction``: set each zone to its speed-weighted dominant source direction; preserve speeds unless
      speed parameters are provided.
    - ``randomized_direction_control``: permute source directions, original speeds retained.
    """

    mode = str(params.get("mode", "roundtrip_placebo"))
    scenario = raw_weather.copy()
    if "WindSpeed" not in scenario.columns or "WindDirection" not in scenario.columns:
        raise ValueError("Raw weather must include WindSpeed and WindDirection columns.")

    if mode == "roundtrip_placebo":
        return scenario

    if mode == "fixed_from_bearing":
        if "wind_from_bearing_deg" not in params:
            raise ValueError("fixed_from_bearing mode requires 'wind_from_bearing_deg'.")
        scenario["WindSpeed"] = _scenario_speed(scenario, params)
        scenario["WindDirection"] = float(params["wind_from_bearing_deg"]) % 360.0
        return scenario

    if mode == "low_wind":
        scenario["WindSpeed"] = _scenario_speed(scenario, params, default_percentile=5.0)
        if "wind_from_bearing_deg" in params:
            scenario["WindDirection"] = float(params["wind_from_bearing_deg"]) % 360.0
        return scenario

    if mode == "speed_only_high":
        scenario["WindSpeed"] = _scenario_speed(scenario, params, default_percentile=95.0)
        return scenario

    if mode == "direction_only_rotate":
        if "wind_from_bearing_deg" not in params:
            raise ValueError("direction_only_rotate mode requires 'wind_from_bearing_deg'.")
        scenario["WindDirection"] = float(params["wind_from_bearing_deg"]) % 360.0
        return scenario

    if mode == "zone_dominant_direction":
        zone_column = str(params.get("zone_column", "WeatherZone"))
        scenario["WindDirection"] = _set_zone_dominant_directions(scenario, zone_column=zone_column).to_numpy(dtype=np.float64)
        scenario["WindSpeed"] = _scenario_speed(scenario, params)
        return scenario

    if mode == "randomized_direction_control":
        rng = np.random.default_rng(seed)
        scenario["WindDirection"] = rng.permutation(scenario["WindDirection"].to_numpy(dtype=np.float64))
        return scenario

    raise ValueError(f"Unknown wind scenario mode {mode!r}.")


def apply_wind_scenario_to_processed_weather(
    raw_weather: pd.DataFrame,
    processed_weather: pd.DataFrame,
    stats: WindEncodingStats,
    params: dict,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a raw-space wind scenario and return processed rows plus diagnostics."""

    if len(raw_weather) != len(processed_weather):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_weather)} != {len(processed_weather)}.")

    scenario_raw = build_wind_scenario_raw_weather(raw_weather, params, seed=seed)
    scenario_raw_features = raw_wind_features(scenario_raw)
    encoded = encode_raw_wind_features(scenario_raw_features, stats)

    scenario_processed = processed_weather.copy()
    for column in RAW_WIND_COLUMNS:
        scenario_processed[column] = encoded[column].to_numpy(dtype=np.float64)
    scenario_processed[WIND_DIRECTION_COLUMN] = scenario_raw[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64)
    scenario_processed["wd_sin"] = np.sin(np.deg2rad(scenario_processed[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64)))
    scenario_processed["wd_cos"] = np.cos(np.deg2rad(scenario_processed[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64)))

    diagnostics = summarize_wind_scenario(scenario_raw, scenario_processed, stats)
    return scenario_processed, diagnostics


def summarize_wind_scenario(
    scenario_raw_weather: pd.DataFrame,
    scenario_processed_weather: pd.DataFrame,
    stats: WindEncodingStats,
) -> pd.DataFrame:
    """Summarize raw and encoded wind distribution for an OOD/plausibility check."""

    stats_by_name = stats.by_name()
    from_bearing = scenario_raw_weather[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float64)
    flow_bearing = np.asarray(flow_bearing_from_from_bearing(from_bearing), dtype=np.float64)
    rows = [
        {
            "quantity": "wind_from_bearing_deg",
            "raw_min": float(np.min(from_bearing)),
            "raw_mean": float(np.mean(from_bearing)),
            "raw_p95": float(np.percentile(from_bearing, 95)),
            "raw_max": float(np.max(from_bearing)),
            "encoded_min": float("nan"),
            "encoded_mean": float("nan"),
            "encoded_p95": float("nan"),
            "encoded_max": float("nan"),
            "z_abs_max": float("nan"),
        },
        {
            "quantity": "flow_bearing_deg",
            "raw_min": float(np.min(flow_bearing)),
            "raw_mean": float(np.mean(flow_bearing)),
            "raw_p95": float(np.percentile(flow_bearing, 95)),
            "raw_max": float(np.max(flow_bearing)),
            "encoded_min": float("nan"),
            "encoded_mean": float("nan"),
            "encoded_p95": float("nan"),
            "encoded_max": float("nan"),
            "z_abs_max": float("nan"),
        },
    ]
    raw_features = raw_wind_features(scenario_raw_weather)
    for column in RAW_WIND_COLUMNS:
        raw_values = raw_features[column].to_numpy(dtype=np.float64)
        encoded_values = scenario_processed_weather[column].to_numpy(dtype=np.float64)
        stat = stats_by_name[column]
        z_values = (raw_values - stat.mean) / stat.std
        rows.append(
            {
                "quantity": column,
                "raw_min": float(np.min(raw_values)),
                "raw_mean": float(np.mean(raw_values)),
                "raw_p95": float(np.percentile(raw_values, 95)),
                "raw_max": float(np.max(raw_values)),
                "encoded_min": float(np.min(encoded_values)),
                "encoded_mean": float(np.mean(encoded_values)),
                "encoded_p95": float(np.percentile(encoded_values, 95)),
                "encoded_max": float(np.max(encoded_values)),
                "z_abs_max": float(np.max(np.abs(z_values))),
            }
        )
    return pd.DataFrame(rows)
