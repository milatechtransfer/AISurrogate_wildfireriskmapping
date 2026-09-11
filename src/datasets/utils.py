import math
from typing import overload

import numpy as np
import torch

from data_preparation.spatial.utils import FUEL_GROUP_MAP

SPATIALIZED_TABULAR_SOURCE_NAMES = {"spatialized_weather", "spatialized_fire_size"}
TABULAR_SOURCE_NAMES = {"tabular_weather", "tabular_fire_size"}
AVAILABLE_DATA_SOURCES = ["grid", "tabular_weather", "tabular_fire_size", "spatialized_weather", "spatialized_fire_size"]
MAX_FUEL_GRID = float(max(FUEL_GROUP_MAP.values()))


def finite_difference(values: torch.Tensor, dim: int, spacing: float) -> torch.Tensor:
    """
    Computes first-order finite differences along one spatial dimension.
    """
    grad = torch.zeros_like(values)
    size = values.shape[dim]
    if size < 2:
        return grad

    if dim == 0:
        grad[0, :] = (values[1, :] - values[0, :]) / spacing
        grad[-1, :] = (values[-1, :] - values[-2, :]) / spacing
        if size > 2:
            grad[1:-1, :] = (values[2:, :] - values[:-2, :]) / (2.0 * spacing)
    elif dim == 1:
        grad[:, 0] = (values[:, 1] - values[:, 0]) / spacing
        grad[:, -1] = (values[:, -1] - values[:, -2]) / spacing
        if size > 2:
            grad[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * spacing)
    else:
        raise ValueError(f"Expected dim 0 or 1 for finite differences, got {dim}.")
    return grad


def raster_cell_spacing(transform) -> tuple[float, float]:
    """
    Returns row and column pixel spacing from a raster affine transform, in CRS units.
    """
    col_spacing = math.hypot(float(transform.a), float(transform.d))
    row_spacing = math.hypot(float(transform.b), float(transform.e))
    if row_spacing <= 0.0 or col_spacing <= 0.0:
        raise ValueError(f"Invalid raster transform pixel spacing: row={row_spacing}, col={col_spacing}.")
    return row_spacing, col_spacing


def get_data_source_class(name: str):
    """
    Returns the source class dynamically to avoid circular imports.
    """
    if name in ["grid"]:
        from src.datasets.sources import GridSource

        return GridSource
    elif name in TABULAR_SOURCE_NAMES:
        from src.datasets.sources import TabularSource

        return TabularSource
    elif name in SPATIALIZED_TABULAR_SOURCE_NAMES:
        from src.datasets.sources import SpatializedTabularSource

        return SpatializedTabularSource
    else:
        raise ValueError(f"Unknown data source type: {name}. Available: {AVAILABLE_DATA_SOURCES}")


def get_data_source_param_class(name: str):
    """
    Returns the param class dynamically to avoid circular imports.
    """
    if name in ["grid"]:
        from src.config import GridParams

        return GridParams
    elif name in TABULAR_SOURCE_NAMES:
        from src.config import TabularParams

        return TabularParams
    elif name in SPATIALIZED_TABULAR_SOURCE_NAMES:
        from src.config import SpatializedTabularParams

        return SpatializedTabularParams
    else:
        raise ValueError(f"Unknown data source type: {name}. Available: {AVAILABLE_DATA_SOURCES}")


def get_dataset_dimensions(dataset) -> tuple[int | None, dict[str, int]]:
    """
    Extracts spatial and tabular dimensions from a MultiSourceDataset.
    """
    sources = getattr(dataset, "sources", {})

    spatial_channels = None
    auxiliary_input_dims = {}

    for name, source in sources.items():
        if name == "grid":
            spatial_channels = source.input_dim()
            iros_len = getattr(source, "fuel_curve_len", 0)
            if iros_len > 0:
                auxiliary_input_dims["fuel_curve"] = iros_len
        elif name in SPATIALIZED_TABULAR_SOURCE_NAMES:
            spatial_channels = (spatial_channels or 0) + source.input_dim()
        else:
            auxiliary_input_dims[name] = source.input_dim()

    return spatial_channels, auxiliary_input_dims


def get_fuel_curve_normalization_stats(dataset) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Returns ``(fuel_curve_mean, fuel_curve_std)`` tensors from the GridSource if a
    curve-based fuel encoding is active, otherwise returns ``(None, None)``.

    These stats are computed from training hexels only inside ``GridSource.__init__`` and
    should be passed to the model at construction time so that ``FuelCurveEncoder`` normalizes
    correctly. During evaluation the model loads them from the saved checkpoint instead.
    """
    grid_source = getattr(dataset, "sources", {}).get("grid")
    if grid_source is None or getattr(grid_source, "fuel_curve_len", 0) == 0:
        return None, None
    mean = getattr(grid_source, "fuel_curve_mean", None)
    std = getattr(grid_source, "fuel_curve_std", None)
    if mean is None or std is None:
        return None, None
    return torch.from_numpy(mean), torch.from_numpy(std)


def fill_nan_channel_mean_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Fills NaNs in a (H, W, C) array with the mean of the corresponding channel.
    If a channel is entirely NaN (e.g. a fully-masked patch), falls back to 0.0.
    Modifies the array in-place.
    """
    # 1. Calculate the mean of each channel, ignoring NaNs; entirely-NaN channels → nan
    channel_means = np.nanmean(arr, axis=(0, 1))

    # 2. Fall back to 0.0 for channels that are entirely NaN (no valid pixels at all)
    channel_means = np.where(np.isnan(channel_means), 0.0, channel_means)

    # 3. Find the indices where values are NaN
    nan_mask = np.isnan(arr)

    # 4. Replace NaNs with the channel mean (or 0.0 for fully-masked channels)
    arr[nan_mask] = np.take(channel_means, np.where(nan_mask)[2])

    return arr


def one_hot_encode(arr: np.ndarray, channel_idx: int, num_classes: int) -> np.ndarray:
    """
    Replaces the nth channel with its one-hot encoded version.
    Input: (H, W, C)
    Output: (H, W, C - 1 + num_classes)
    """
    # 1. Split the array
    # left: (H, W, n)
    left_part = arr[:, :, :channel_idx]

    # right: (H, W, C - n - 1)
    right_part = arr[:, :, channel_idx + 1 :]

    # target: (H, W)
    target_channel = arr[:, :, channel_idx]

    nan_mask = np.isnan(target_channel)

    # Replace NaN with 0 (or any safe index) temporarily so .astype(int) doesn't crash
    # We use np.nan_to_num to swap NaN -> 0 safely
    safe_target = np.nan_to_num(target_channel, nan=0).astype(int)

    # 3. One-Hot Encode using the "safe" integers
    encoded_part = np.eye(num_classes, dtype=arr.dtype)[safe_target]

    # 4. Zero out the vectors where the original value was NaN
    # Before this, the NaNs were encoded as Class 0 (because we filled with 0)
    # This step corrects that by setting them to [0, 0, 0...]
    encoded_part[nan_mask] = 0  # Nan is no fuel

    # 3. Concatenate along the channel axis (last axis)
    return np.concatenate([left_part, encoded_part.astype(np.float32), right_part], axis=-1)  # (H,W,C+14)


def log_norm(out_arr: np.ndarray, multiplier: int = 1000) -> np.ndarray:
    """
    Normalize the output target array using log norm.
    """
    return np.log1p(multiplier * out_arr) / np.log1p(multiplier)


def output_target_norm(
    output_arr: np.ndarray,
    target_max: float,
    target_min: float,
    out_norm: str,
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
) -> np.ndarray:
    """
    Normalize the output target map.
    """
    if out_norm == "min_max":
        output_arr = (output_arr - target_min) / (target_max - target_min)
        output_arr = np.clip(output_arr, 0.0, 1.0)
    elif out_norm == "log":
        output_arr = log_norm(output_arr).astype(np.float32)
    elif out_norm == "log_standard":
        if target_log_mean is None or target_log_std is None:
            raise ValueError("target_log_mean and target_log_std are required for out_norm='log_standard'.")
        if target_log_std <= 0.0:
            raise ValueError(f"target_log_std must be positive for out_norm='log_standard', got {target_log_std}.")
        output_arr = (np.log1p(np.clip(output_arr, a_min=0.0, a_max=None)) - target_log_mean) / target_log_std
    elif out_norm in {"none", "total_iters", "season_cause_iters"}:
        pass
    else:
        raise ValueError(f"Unsupported output normalization: {out_norm!r}")
    return output_arr


@overload
def denormalize_output_target(
    data: torch.Tensor,
    target_min: float,
    target_max: float,
    out_norm: str,
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
    multiplier: float = 1000.0,
) -> torch.Tensor: ...


@overload
def denormalize_output_target(
    data: np.ndarray,
    target_min: float,
    target_max: float,
    out_norm: str,
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
    multiplier: float = 1000.0,
) -> np.ndarray: ...


def denormalize_output_target(
    data: np.ndarray | torch.Tensor,
    target_min: float,
    target_max: float,
    out_norm: str,
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
    multiplier: float = 1000.0,
) -> np.ndarray | torch.Tensor:
    """
    Reverse target normalization to recover values in the original target scale.
    """
    if out_norm == "log_standard":
        if target_log_mean is None or target_log_std is None:
            raise ValueError("target_log_mean and target_log_std are required for out_norm='log_standard'.")
        if target_log_std <= 0.0:
            raise ValueError(f"target_log_std must be positive for out_norm='log_standard', got {target_log_std}.")
        log_mean = target_log_mean
        log_std = target_log_std
    else:
        log_mean = 0.0
        log_std = 1.0

    if isinstance(data, torch.Tensor):
        data = data.float()
        if out_norm == "min_max":
            return data * (target_max - target_min) + target_min
        if out_norm == "log":
            return torch.expm1(data * float(np.log1p(multiplier))) / multiplier
        if out_norm == "log_standard":
            return torch.expm1(data * log_std + log_mean).clamp_min(0.0)
        if out_norm in {"none", "total_iters", "season_cause_iters"}:
            return data
    else:
        data = data.astype("float32")
        if out_norm == "min_max":
            return (data * (target_max - target_min) + target_min).astype("float32")
        if out_norm == "log":
            return (np.expm1(data * np.log1p(multiplier)) / multiplier).astype("float32")
        if out_norm == "log_standard":
            return np.clip(np.expm1(data * log_std + log_mean), 0.0, None).astype("float32")
        if out_norm in {"none", "total_iters", "season_cause_iters"}:
            return data

    raise ValueError(f"Unsupported output normalization: {out_norm!r}")


def apply_bp_nodata_zero_range(
    target_name: str,
    max_value: float,
    min_value: float,
    bp_nodata_as_zero: bool,
) -> tuple[float, float]:
    if target_name == "bp" and bp_nodata_as_zero:
        return max_value, 0.0
    return max_value, min_value
