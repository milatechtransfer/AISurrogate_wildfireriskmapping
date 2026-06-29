"""Shared IO and plotting helpers for counterfactual map scripts.

Prediction-grid raster IO (reading materialized hexel predictions, aligning to a
reference grid) and small plotting/colour utilities reused across the
counterfactual figure scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def prediction_dirs_from_index(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Read scenario/endpoint prediction directories from the materialization index."""

    index_path = experiment_dir / "scenario_prediction_index.csv"
    index = pd.read_csv(index_path)
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def prediction_raster_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "predicted_hexels" / f"hexel_{int(hex_id):02d}_predicted.tif"


def read_prediction(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def read_prediction_extent(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        bounds = src.bounds
    return bounds.left, bounds.right, bounds.bottom, bounds.top


def prediction_reference_profile(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    *,
    baseline_endpoint: str = "bp",
) -> dict:
    """Reference raster profile defining the prediction grid for a hexel."""

    baseline_dir = prediction_dirs.get(("baseline", baseline_endpoint))
    if baseline_dir is None:
        raise KeyError(f"Missing baseline {baseline_endpoint.upper()} prediction directory; cannot define reference grid.")
    with rasterio.open(prediction_raster_path(baseline_dir, hex_id)) as src:
        return src.profile.copy()


def finite_values(data: np.ma.MaskedArray | np.ndarray) -> np.ndarray:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values[np.isfinite(values)]


def values_and_valid(data: np.ma.MaskedArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values, np.isfinite(values)


def symmetric_percentile_limit(deltas: list[np.ma.MaskedArray], percentile: float = 99.5) -> float:
    """Symmetric color limit from pooled absolute delta values."""

    values = [np.abs(finite_values(delta)) for delta in deltas]
    values = [value for value in values if value.size > 0]
    if not values:
        return 1.0
    limit = float(np.percentile(np.concatenate(values), percentile))
    return max(limit, 1e-9)


def downsample_for_display(data: np.ma.MaskedArray | np.ndarray, factor: int) -> np.ma.MaskedArray:
    """Stride-downsample a raster for plotting only."""

    if factor <= 1:
        return np.ma.asarray(data)
    return np.ma.asarray(data)[::factor, ::factor]


def restrict_to_support(data: np.ma.MaskedArray | np.ndarray, support_mask: np.ndarray) -> np.ma.MaskedArray:
    """Mask an array outside a boolean analysis support mask."""

    arr = np.ma.asarray(data)
    if arr.shape != support_mask.shape:
        raise ValueError(f"Support mask shape {support_mask.shape} does not match data shape {arr.shape}.")
    return np.ma.masked_where(~support_mask | np.ma.getmaskarray(arr), arr)
