"""Weather re-encoding validation for fixed-model counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.postprocessing.counterfactual import encoded_components_from_from_bearing

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


def validate_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}.")


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
