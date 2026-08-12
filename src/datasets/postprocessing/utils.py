import functools
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)
import torch
from rasterio.features import geometry_mask
from rasterio.profiles import Profile

from data_preparation.paths import MaskScope, Paths, normalize_mask_scope
from data_preparation.spatial.utils import (
    denormalize_burn_count,
    get_output_log_stats_cached,
    get_range_output_cached,
    load_spatial_raster,
    read_split_hex_ids,
)
from src.config import Config, GridParams
from src.datasets.postprocessing.stitch_hexel import stitch_windows
from src.datasets.postprocessing.visualize_predictions import (
    as_float_array_with_nan,
    plot_hexbin_distribution,
    plot_histogram_distribution,
    visualize_hexel_iou,
    visualize_target_grids,
)
from src.datasets.targets import TargetSpec, get_target_specs
from src.datasets.utils import apply_bp_nodata_zero_range, denormalize_output_target
from src.logger import CometLogger


@dataclass(frozen=True)
class TargetPostprocessingSettings:
    target: TargetSpec
    target_channel_index: int
    max_target_val: float | None
    min_target_val: float | None
    out_norm: str
    target_log_mean: float | None
    target_log_std: float | None


def get_mask_scope_save_dir(save_dir: str, mask_scope: str) -> str:
    scope = normalize_mask_scope(mask_scope)
    if scope == "actual":
        return save_dir
    return os.path.join(save_dir, f"{scope}_mask_eval")


def effective_robust_plot_percentile(target: TargetSpec, robust_plot_percentile: float | None) -> float | None:
    if robust_plot_percentile is not None:
        return robust_plot_percentile
    if target.name in {"fi", "ros"}:
        return 99.0
    return None


def _accepted_prepared_mask_scopes(mask_scope: MaskScope) -> set[MaskScope]:
    if mask_scope == "buffer_only":
        return {"buffer", "buffer_only"}
    return {mask_scope}


def validate_patch_metadata_mask_scope(metadata: pd.DataFrame, mask_scope: str) -> MaskScope:
    scope = normalize_mask_scope(mask_scope)
    if scope == "actual":
        return scope
    if "mask_scope" not in metadata.columns:
        raise ValueError(
            f"mask_scope={scope!r} requires patch metadata with a matching 'mask_scope' column. "
            "Existing actual-only patch data cannot be safely reinterpreted as buffer data; "
            "prepare or infer on buffer-scope patches first."
        )

    observed = {normalize_mask_scope(str(value)) for value in metadata["mask_scope"].dropna().unique()}
    accepted = _accepted_prepared_mask_scopes(scope)
    if not observed or not observed.issubset(accepted):
        raise ValueError(f"mask_scope={scope!r} requires patch metadata mask_scope in {sorted(accepted)}, got {sorted(observed)}.")
    return scope


def _actual_area_mask(mask_path: Path, profile: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    crs = profile.get("crs")
    transform = profile.get("transform")
    if crs is None or transform is None:
        raise ValueError("buffer_only masking requires a geospatial profile with 'crs' and 'transform'.")

    actual_gdf = gpd.read_file(mask_path)
    if actual_gdf.empty:
        raise ValueError(f"Actual mask contains no geometries: {mask_path}")

    return geometry_mask(actual_gdf.to_crs(crs).geometry, out_shape=shape, transform=transform, invert=True)


def apply_mask_scope_to_grids(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    profile: dict[str, Any],
    mask_path: Path,
    mask_scope: str,
    hex_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    scope = normalize_mask_scope(mask_scope)
    if scope != "buffer_only":
        return gt_grid, pred_grid

    gt_arr = as_float_array_with_nan(gt_grid)
    pred_arr = as_float_array_with_nan(pred_grid)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            f"mask_scope='buffer_only' requires target and prediction grids with the same shape, "
            f"got target={gt_arr.shape}, prediction={pred_arr.shape} for hex {str(hex_id).zfill(2)}."
        )

    actual_mask = _actual_area_mask(mask_path=mask_path, profile=profile, shape=gt_arr.shape)
    gt_arr = np.where(actual_mask, np.nan, gt_arr)
    pred_arr = np.where(actual_mask, np.nan, pred_arr)
    if not np.any(np.isfinite(gt_arr) & np.isfinite(pred_arr)):
        raise ValueError(
            f"mask_scope='buffer_only' left no finite overlapping target/prediction pixels for hex {str(hex_id).zfill(2)}. "
            "This usually means the predictions came from actual-scope patches; prepare or infer on buffer-scope patches first."
        )
    return gt_arr, pred_arr


def fill_bp_target_nodata_as_zero(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    target: TargetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if target.name != "bp":
        return gt_grid, pred_grid

    gt_arr = as_float_array_with_nan(gt_grid)
    pred_arr = as_float_array_with_nan(pred_grid)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            f"BP nodata-to-zero fill requires target and prediction grids with the same shape, "
            f"got target={gt_arr.shape}, prediction={pred_arr.shape}."
        )

    fill_mask = np.isfinite(pred_arr) & ~np.isfinite(gt_arr)
    return np.where(fill_mask, 0.0, gt_arr), pred_arr


def mask_grids_by_support(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support_mask = np.asarray(support_mask, dtype=bool)
    gt_arr = as_float_array_with_nan(gt_grid)
    pred_arr = as_float_array_with_nan(pred_grid)
    if gt_arr.shape != pred_arr.shape or gt_arr.shape != support_mask.shape:
        raise ValueError(
            f"Support mask shape must match target and prediction grids, got target={gt_arr.shape}, "
            f"prediction={pred_arr.shape}, support={support_mask.shape}."
        )
    return np.where(support_mask, gt_arr, np.nan), np.where(support_mask, pred_arr, np.nan)


def load_target_grid_for_mask_scope(
    paths: Paths,
    target: TargetSpec,
    pred_grid: np.ndarray,
    profile: dict[str, Any],
    mask_scope: str,
    hex_id: str,
    bp_nodata_as_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    scope = normalize_mask_scope(mask_scope)
    target_path = getattr(paths, target.path_method)()
    loaded_target_grid, _ = load_spatial_raster(
        path=target_path,
        mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=scope),
        reference_profile=profile,
    )
    target_grid, pred_grid = apply_mask_scope_to_grids(
        gt_grid=as_float_array_with_nan(loaded_target_grid),
        pred_grid=pred_grid,
        profile=profile,
        mask_path=paths.mask_grid_actual(hex_id=hex_id),
        mask_scope=scope,
        hex_id=hex_id,
    )
    if bp_nodata_as_zero:
        return fill_bp_target_nodata_as_zero(gt_grid=target_grid, pred_grid=pred_grid, target=target)
    return target_grid, pred_grid


def save_predicted_hexels(
    predicted_hexel: np.ndarray,
    hexel_profile: Profile,
    hex_id: str,
    save_dir: str,
    target_name: str | None = None,
):
    """
    Save the predicted (reconstructed) hexel
    Args:
        predicted_hexel (np.ndarray) : 2d array of shape (height, width)
        hexel_profile (rasterio.profile): Profile for the hexel, required by rasterio for saving geospatial data
        hex_id (str): The id of the hex to be saved
        save_dir (str): directory to save the hexel
    """
    suffix = f"_{target_name}" if target_name else ""
    out_path = os.path.join(save_dir, "predicted_hexels", f"hexel_{hex_id}{suffix}_predicted.tif")
    os.makedirs(os.path.join(save_dir, "predicted_hexels"), exist_ok=True)
    nodata = hexel_profile.get("nodata", -9999)
    write_array = np.where(np.isfinite(predicted_hexel), predicted_hexel, nodata).astype(hexel_profile["dtype"])
    with rasterio.open(out_path, "w", **hexel_profile) as dst:
        dst.write(write_array, 1)


def get_stitched_windows(
    base_dir: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    start_idx: int,
    gt_shape: tuple,
    target_channel_index: int = 0,
    prediction_mask_channel_indices: list[int] | None = None,
    stitch_mode: str = "mean",
) -> np.ndarray:
    """
    Accumulate and stitch all the windows together to build the hexel
    """
    all_data_points, all_locations, all_masks = [], [], []
    for i, data in enumerate(np.array(df)):
        path = data[0]
        array = np.load(os.path.join(base_dir, path))
        if target_channel_index >= array.shape[2]:
            raise ValueError(f"target_channel_index={target_channel_index} is out of bounds for patch with shape {array.shape}.")
        if prediction_mask_channel_indices is None:
            mask = np.isfinite(array[:, :, target_channel_index])
        else:
            if not prediction_mask_channel_indices:
                raise ValueError("prediction_mask_channel_indices cannot be empty.")
            invalid_indices = [index for index in prediction_mask_channel_indices if index < 0 or index >= array.shape[2]]
            if invalid_indices:
                raise ValueError(f"prediction mask channel indices {invalid_indices} are out of bounds for patch with shape {array.shape}.")
            mask = np.logical_and.reduce([np.isfinite(array[:, :, index]) for index in prediction_mask_channel_indices])
        patch_h, patch_w = array.shape[:2]
        all_data_points.append(predictions[start_idx + i].reshape((patch_h, patch_w)))
        all_locations.append((data[5], data[6]))
        all_masks.append(mask.reshape((patch_h, patch_w)))
    reconstructed_hexel = stitch_windows(
        all_data_points,
        all_locations,
        all_masks,
        gt_shape,
        mode=stitch_mode,
    )
    return reconstructed_hexel  # gt_shape


def get_predicted_hexel(
    base_dir: str,
    raw_data_dir: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    min_target_val: float | None,
    max_target_val: float | None,
    hex_id: str,
    modelling_approach: str = "1",
    out_norm: str = "min_max",
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
    stitch_mode: str = "mean",
    target_channel_index: int = 0,
    prediction_mask_channel_indices: list[int] | None = None,
    mask_scope: str = "actual",
) -> tuple[np.ndarray, Profile]:
    """
    Returns the reconstructed hexel
    """
    start_idx = 0

    all_paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    scope = normalize_mask_scope(mask_scope)

    gt_elevation_grid, gt_elevation_grid_profile = load_spatial_raster(
        path=all_paths.elevation_grid(hex_id=hex_id),
        mask_path=all_paths.mask_grid(hex_id=hex_id, mask_scope=scope),
    )

    if out_norm in {"min_max", "log"}:
        predictions = np.clip(predictions, 0, 1)

    requires_target_range = out_norm == "min_max" or modelling_approach != "1"
    if requires_target_range and (min_target_val is None or max_target_val is None):
        raise ValueError(
            f"min_target_val and max_target_val are required when out_norm={out_norm!r} or modelling_approach={modelling_approach!r}."
        )

    if modelling_approach == "1":
        reconstructed_hexel = get_stitched_windows(
            base_dir=base_dir,
            df=test_df,
            predictions=predictions,
            start_idx=start_idx,
            gt_shape=tuple(gt_elevation_grid.shape),
            target_channel_index=target_channel_index,
            stitch_mode=stitch_mode,
            prediction_mask_channel_indices=prediction_mask_channel_indices,
        )
        reconstructed_hexel_denorm = denormalize_model_target(
            data=reconstructed_hexel,
            min_val=0.0 if min_target_val is None else min_target_val,
            max_val=0.0 if max_target_val is None else max_target_val,
            out_norm=out_norm,
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
        )
        gt_elevation_grid_profile.update(dtype="float32", compress="lzw", nodata=-9999)  # type: ignore
    else:
        unique_season_cause = list(set(zip(test_df["season"], test_df["cause"], strict=False)))
        season_cause_hexels = []
        for season, cause in unique_season_cause:
            filtered_season_cause_df = test_df[(test_df["season"] == season) & (test_df["cause"] == cause)]
            reconstructed_season_cause_hexel = get_stitched_windows(
                base_dir=base_dir,
                df=filtered_season_cause_df,
                predictions=predictions,
                start_idx=start_idx,
                gt_shape=tuple(gt_elevation_grid.data.shape),
                target_channel_index=target_channel_index,
                stitch_mode=stitch_mode,
                prediction_mask_channel_indices=prediction_mask_channel_indices,
            )
            reconstructed_season_cause_hexel_denorm = denormalize_burn_count(
                data=reconstructed_season_cause_hexel,
                min_val=0.0 if min_target_val is None else min_target_val,
                max_val=0.0 if max_target_val is None else max_target_val,
            )
            season_cause_hexels.append(reconstructed_season_cause_hexel_denorm)
            start_idx += len(filtered_season_cause_df)
        # merge the counts
        reconstructed_hexel_denorm = np.sum(np.stack(season_cause_hexels), axis=0)
        reconstructed_hexel_denorm = np.rint(reconstructed_hexel_denorm).astype("int32")
        # clip values to the true range, in case of outliers
        reconstructed_hexel_denorm = np.clip(
            reconstructed_hexel_denorm,
            0.0 if min_target_val is None else min_target_val,
            0.0 if max_target_val is None else max_target_val,
        )
        gt_elevation_grid_profile.update(dtype="int32", compress="lzw", nodata=-9999)  # type: ignore

    return reconstructed_hexel_denorm, gt_elevation_grid_profile


def get_config_target_specs(config: Config) -> list[TargetSpec]:
    for source in config.data.input_sources:
        if source.name == "grid" and isinstance(source.params, GridParams):
            return get_target_specs([target.name for target in source.params.resolved_targets()])
    return get_target_specs("bp")


def get_config_target_spec(config: Config) -> TargetSpec:
    targets = get_config_target_specs(config)
    if len(targets) != 1:
        target_names = [target.name for target in targets]
        raise ValueError(f"Expected a single grid target, got {target_names}.")
    return targets[0]


def get_config_grid_params(config: Config) -> GridParams | None:
    for source in config.data.input_sources:
        if source.name == "grid" and isinstance(source.params, GridParams):
            return source.params
    return None


def denormalize_model_target(
    data: np.ndarray,
    min_val: float,
    max_val: float,
    out_norm: str,
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
) -> np.ndarray:
    if out_norm == "log_standard":
        if target_log_mean is None or target_log_std is None:
            raise ValueError("target_log_mean and target_log_std are required for out_norm='log_standard'.")
        if target_log_std <= 0.0:
            raise ValueError(f"target_log_std must be positive for out_norm='log_standard', got {target_log_std}.")
        return np.clip(np.expm1(data.astype("float32") * target_log_std + target_log_mean), 0.0, None).astype("float32")

    return denormalize_output_target(data=data, target_min=min_val, target_max=max_val, out_norm=out_norm)


def get_target_channel_index(data_dir: str, modelling_approach: str, target: TargetSpec) -> int:
    feature_map_path = os.path.join(data_dir, f"feature_channel_map_{modelling_approach}.json")
    with open(feature_map_path) as f:
        channel_feature_map = json.load(f)

    channel_indices = channel_feature_map.get(target.channel_key)
    if not channel_indices:
        raise ValueError(
            f"Missing target channel {target.channel_key!r} in {feature_map_path}. Available keys: {list(channel_feature_map.keys())}"
        )
    return int(channel_indices[0])


def get_prediction_mask_channel_indices(
    data_dir: str,
    modelling_approach: str,
    grid_params: GridParams | dict[str, Any] | None,
    prediction_support_policy: str = "input",
) -> list[int] | None:
    if prediction_support_policy == "target":
        return None
    if prediction_support_policy != "input":
        raise ValueError("prediction_support_policy must be one of ('input', 'target').")
    if grid_params is None:
        return None

    feature_names = grid_params.feature_names_list if isinstance(grid_params, GridParams) else grid_params.get("feature_names_list", [])
    if not feature_names:
        return None

    feature_map_path = os.path.join(data_dir, f"feature_channel_map_{modelling_approach}.json")
    with open(feature_map_path) as f:
        channel_feature_map = json.load(f)

    indices: list[int] = []
    for feature_name in feature_names:
        channel_indices = channel_feature_map.get(feature_name)
        if not channel_indices:
            raise ValueError(
                f"Missing input channel {feature_name!r} in {feature_map_path}. Available keys: {list(channel_feature_map.keys())}"
            )
        indices.extend(int(index) for index in channel_indices)
    return sorted(set(indices))


def get_target_out_norm(grid_params: GridParams | None, target: TargetSpec, fallback_out_norm: str) -> str:
    if grid_params is None:
        return fallback_out_norm
    return grid_params.target_config(target.name).out_norm


def get_target_log_stats(grid_params: GridParams | None, target: TargetSpec) -> tuple[float | None, float | None]:
    if grid_params is None:
        return None, None
    target_config = grid_params.target_config(target.name)
    return target_config.log_mean, target_config.log_std


def get_target_postprocessing_settings(config: Config, out_norm: str) -> list[TargetPostprocessingSettings]:
    """Build per-target channel, range, and normalization settings for postprocessing."""
    data_dir = config.data.root_dir
    raw_data_dir = config.data.raw_data_dir
    train_hex_ids: set[int] | None = None
    if data_dir and config.data.train_split:
        train_hex_ids = read_split_hex_ids(os.path.join(data_dir, config.data.train_split))

    grid_params = get_config_grid_params(config)
    settings = []
    for target in get_config_target_specs(config):
        target_channel_index = get_target_channel_index(data_dir=data_dir, modelling_approach=config.modelling_approach, target=target)
        target_out_norm = get_target_out_norm(grid_params=grid_params, target=target, fallback_out_norm=out_norm)

        max_target_val: float | None = None
        min_target_val: float | None = None
        if target_out_norm == "min_max":
            max_target_val, min_target_val = get_range_output_cached(
                root_dir=data_dir or raw_data_dir,
                output_type=target.output_type,
                allowed_hex_ids=train_hex_ids,
                raw_data_dir=raw_data_dir,
            )
            max_target_val, min_target_val = apply_bp_nodata_zero_range(
                target_name=target.name,
                max_value=max_target_val,
                min_value=min_target_val,
                bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero,
            )
        target_log_mean, target_log_std = get_target_log_stats(grid_params=grid_params, target=target)
        if target_out_norm == "log_standard" and (target_log_mean is None or target_log_std is None):
            target_log_mean, target_log_std = get_output_log_stats_cached(
                str(data_dir), target.output_type, allowed_hex_ids=train_hex_ids, raw_data_dir=raw_data_dir
            )
        settings.append(
            TargetPostprocessingSettings(
                target=target,
                target_channel_index=target_channel_index,
                max_target_val=max_target_val,
                min_target_val=min_target_val,
                out_norm=target_out_norm,
                target_log_mean=target_log_mean,
                target_log_std=target_log_std,
            )
        )
    return settings


def select_prediction_target_channel(
    predictions: np.ndarray,
    target_name: str,
    target_index: int = 0,
) -> np.ndarray:
    if predictions.ndim == 3:
        if target_index != 0:
            raise ValueError(f"Cannot select target index {target_index} from channel-free predictions with shape {predictions.shape}.")
        return predictions
    if predictions.ndim == 4 and 0 <= target_index < predictions.shape[1]:
        return predictions[:, target_index]
    raise ValueError(f"Cannot select target={target_name!r} at index {target_index} from predictions with shape {predictions.shape}.")


def calculate_hexel_metrics_pytorch(
    gt_grid: np.ndarray, pred_grid: np.ndarray, device: str | torch.device, metric_functions: dict[str, Callable]
) -> dict[str, float]:
    """
    Utils to convert 2D numpy hexels into torch tensors to run the global per-hexel eval. metrics.
    """
    gt_grid = as_float_array_with_nan(gt_grid)
    pred_grid = as_float_array_with_nan(pred_grid)

    valid_mask_np = np.isfinite(gt_grid) & np.isfinite(pred_grid)
    valid_mask_np = valid_mask_np & (gt_grid >= 0.0)

    gt_clean = np.nan_to_num(gt_grid, nan=0.0)
    pred_clean = np.nan_to_num(pred_grid, nan=0.0)

    compute_device = "cpu" if torch.device(device).type == "mps" else device

    t_targets = torch.from_numpy(gt_clean).to(device=compute_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_preds = torch.from_numpy(pred_clean).to(device=compute_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_mask = torch.from_numpy(valid_mask_np).to(device=compute_device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    results = {}
    # compute metrics requested in config.
    with torch.no_grad():
        for name, metric_fn in metric_functions.items():
            val = metric_fn(t_preds, t_targets, t_mask)
            results[name] = val.item()

    return results


def get_hexel_binary_maps(pred_grid: np.ndarray, gt_grid: np.ndarray, percentile: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Utils to get the Top K percentile thresholds (binary maps) for full 2D numpy hexel grids.
    """
    gt_grid = as_float_array_with_nan(gt_grid)
    pred_grid = as_float_array_with_nan(pred_grid)

    valid_mask = np.isfinite(gt_grid) & np.isfinite(pred_grid)

    p_valid = pred_grid[valid_mask]
    t_valid = gt_grid[valid_mask]

    pred_bin = np.zeros_like(pred_grid, dtype=bool)
    gt_bin = np.zeros_like(gt_grid, dtype=bool)

    n_valid = len(p_valid)
    if n_valid > 0:
        # get count (number of elements) for the specific top K %
        top_fraction = 1.0 - percentile
        k = int(np.ceil(top_fraction * n_valid))
        if k >= n_valid:
            pred_bin[valid_mask] = True
            gt_bin[valid_mask] = True
        elif k > 0:
            # select exactly k highest values within the valid area
            pred_valid_bin = np.zeros_like(p_valid, dtype=bool)
            gt_valid_bin = np.zeros_like(t_valid, dtype=bool)
            pred_topk_idx = np.argpartition(p_valid, -k)[-k:]
            gt_topk_idx = np.argpartition(t_valid, -k)[-k:]
            pred_valid_bin[pred_topk_idx] = True
            gt_valid_bin[gt_topk_idx] = True
            pred_bin[valid_mask] = pred_valid_bin
            gt_bin[valid_mask] = gt_valid_bin

    return pred_bin, gt_bin


def evaluate_and_visualize_hexels(
    test_predictions: np.ndarray,
    config: Config,
    out_norm: str,
    device: str | torch.device,
    experiment_logger: CometLogger | None = None,
    metric_functions: dict[str, Callable] | None = None,
    stitch_mode: str = "mean",
    save_artifacts: bool = True,
    save_plots: bool = True,
    robust_plot_percentile: float | None = None,
    mask_scope: str = "actual",
    split_csv: str | None = None,
    save_dir_suffix: str | None = None,
    test_metadata: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    A util function to re-construct predicted hexels out of test predictions, and visualize side-by-side with the Groundtruth.
    Also computes and aggregates stitched hexel-level metrics.

    ``split_csv`` selects which patch metadata split to stitch (defaults to
    ``config.data.test_split``); pass ``config.data.val_split`` to compute
    hexel-level metrics on the validation split instead. ``save_dir_suffix``,
    if given, isolates that split's artifacts under a dedicated subdirectory
    (e.g. "val") so they don't collide with the default test-split outputs.
    """
    prediction_support_label = "input support" if config.evaluation.prediction_support_policy == "input" else "target support"
    scope = normalize_mask_scope(mask_scope)
    show_prediction_support_outline = config.evaluation.prediction_support_policy == "input" and scope == "actual"
    base_save_dir = os.path.join(config.save_dir, save_dir_suffix) if save_dir_suffix else config.save_dir
    artifacts_save_dir = get_mask_scope_save_dir(base_save_dir, scope)

    from src.datasets.postprocessing.hexel_reconstruction import reconstruct_denormalized_hexels

    all_hexel_metrics: list[tuple[str, str | None, str | None, dict[str, float]]] = []
    current_hex_id: str | None = None
    multi_target = len(get_config_target_specs(config)) > 1

    for stitched_hexel in reconstruct_denormalized_hexels(
        test_predictions=test_predictions,
        config=config,
        out_norm=out_norm,
        stitch_mode=stitch_mode,
        mask_scope=scope,
        split_csv=split_csv,
        test_metadata=test_metadata,
    ):
        if stitched_hexel.hex_id != current_hex_id:
            if current_hex_id is not None and save_artifacts:
                artifact_label = "subplots" if save_plots else "predicted rasters"
                print(f"=======Saved {artifact_label} for hex{current_hex_id}==============")
            current_hex_id = stitched_hexel.hex_id
            print(f"======Working with hex{current_hex_id}========")

        target = stitched_hexel.target
        target_name_for_artifacts = target.name if multi_target else None
        grid_gt = stitched_hexel.gt_grid
        reconstructed_hexel_denorm = stitched_hexel.pred_grid
        actual_support_mask = stitched_hexel.actual_support_mask
        buffer_support_mask = stitched_hexel.buffer_support_mask

        if save_artifacts:
            save_predicted_hexels(
                reconstructed_hexel_denorm,
                stitched_hexel.profile,
                stitched_hexel.hex_id,
                artifacts_save_dir,
                target_name=target_name_for_artifacts,
            )
            if save_plots:
                visualize_target_grids(
                    gt_grid=grid_gt,
                    pred_grid=reconstructed_hexel_denorm,
                    hex_id=stitched_hexel.hex_id,
                    save_dir=artifacts_save_dir,
                    experiment_logger=experiment_logger,
                    target_label=target.label,
                    target_name=target_name_for_artifacts,
                    actual_support_mask=actual_support_mask,
                    buffer_support_mask=buffer_support_mask,
                    prediction_support_label=prediction_support_label,
                    show_prediction_support_outline=show_prediction_support_outline,
                )
                target_robust_plot_percentile = effective_robust_plot_percentile(
                    target=target,
                    robust_plot_percentile=robust_plot_percentile,
                )
                if target_robust_plot_percentile is not None:
                    visualize_target_grids(
                        gt_grid=grid_gt,
                        pred_grid=reconstructed_hexel_denorm,
                        hex_id=stitched_hexel.hex_id,
                        save_dir=artifacts_save_dir,
                        experiment_logger=experiment_logger,
                        target_label=target.label,
                        target_name=target_name_for_artifacts,
                        value_percentile=target_robust_plot_percentile,
                        diff_percentile=target_robust_plot_percentile,
                        filename_suffix=f"_p{target_robust_plot_percentile:g}",
                        actual_support_mask=actual_support_mask,
                        buffer_support_mask=buffer_support_mask,
                        prediction_support_label=prediction_support_label,
                        show_prediction_support_outline=show_prediction_support_outline,
                    )
                plot_hexbin_distribution(
                    gt_grid=grid_gt,
                    pred_grid=reconstructed_hexel_denorm,
                    hex_id=stitched_hexel.hex_id,
                    save_dir=artifacts_save_dir,
                    experiment_logger=experiment_logger,
                    target_label=target.label,
                    probability_scale=target.probability_scale,
                    target_name=target_name_for_artifacts,
                )
                plot_histogram_distribution(
                    gt_grid=grid_gt,
                    pred_grid=reconstructed_hexel_denorm,
                    hex_id=stitched_hexel.hex_id,
                    save_dir=artifacts_save_dir,
                    experiment_logger=experiment_logger,
                    target_label=target.label,
                    probability_scale=target.probability_scale,
                    target_name=target_name_for_artifacts,
                )

        # compute per-hexel metrics
        if metric_functions is not None:
            hex_metrics = calculate_hexel_metrics_pytorch(
                gt_grid=grid_gt, pred_grid=reconstructed_hexel_denorm, device=device, metric_functions=metric_functions
            )
            metric_scope: str | None = None if scope == "actual" else scope
            metric_target_name = target.name if multi_target else None
            all_hexel_metrics.append((stitched_hexel.hex_id, metric_target_name, metric_scope, hex_metrics))

            if scope == "buffer" and actual_support_mask is not None:
                actual_gt, actual_pred = mask_grids_by_support(
                    gt_grid=grid_gt,
                    pred_grid=reconstructed_hexel_denorm,
                    support_mask=actual_support_mask,
                )
                buffer_only_gt, buffer_only_pred = mask_grids_by_support(
                    gt_grid=grid_gt,
                    pred_grid=reconstructed_hexel_denorm,
                    support_mask=np.isfinite(reconstructed_hexel_denorm) & ~actual_support_mask,
                )
                actual_metrics = calculate_hexel_metrics_pytorch(
                    gt_grid=actual_gt,
                    pred_grid=actual_pred,
                    device=device,
                    metric_functions=metric_functions,
                )
                buffer_only_metrics = calculate_hexel_metrics_pytorch(
                    gt_grid=buffer_only_gt,
                    pred_grid=buffer_only_pred,
                    device=device,
                    metric_functions=metric_functions,
                )
                all_hexel_metrics.append((stitched_hexel.hex_id, metric_target_name, "actual", actual_metrics))
                all_hexel_metrics.append((stitched_hexel.hex_id, metric_target_name, "buffer_only", buffer_only_metrics))

            percentiles_to_plot = [
                fn.keywords["percentile"]
                for _, fn in metric_functions.items()
                if isinstance(fn, functools.partial) and "percentile" in fn.keywords
            ]

            if save_artifacts and save_plots:
                for p in percentiles_to_plot:
                    pred_bin, gt_bin = get_hexel_binary_maps(reconstructed_hexel_denorm, grid_gt, percentile=p)
                    visualize_hexel_iou(
                        grid_gt,
                        reconstructed_hexel_denorm,
                        gt_bin,
                        pred_bin,
                        stitched_hexel.hex_id,
                        artifacts_save_dir,
                        p,
                        target_label=target.label,
                        target_name=target_name_for_artifacts,
                        actual_support_mask=actual_support_mask,
                        buffer_support_mask=buffer_support_mask,
                    )

    if current_hex_id is not None and save_artifacts:
        artifact_label = "subplots" if save_plots else "predicted rasters"
        print(f"=======Saved {artifact_label} for hex{current_hex_id}==============")

    # aggregate final scores
    hexel_metrics = {}

    if metric_functions is not None and len(all_hexel_metrics) > 0:
        # get per-hexel metrics
        metric_values_by_key: dict[str, list[float]] = {}
        for hex_id_str, target_name, metric_scope, hex_metric in all_hexel_metrics:
            for key, val in hex_metric.items():
                metric_key = f"{target_name}_{key}" if target_name is not None else key
                if metric_scope is not None:
                    metric_key = f"{metric_scope}_{metric_key}"
                metric_value = float(val) if not np.isnan(val) else float("nan")
                # we create keys such as "hex12/mse" or "hex12/bp_mse" for clarity
                hexel_metrics[f"hex{hex_id_str}/{metric_key}"] = metric_value
                metric_values_by_key.setdefault(metric_key, []).append(metric_value)

        # get the aggregated averages over all hexels
        for key, values in metric_values_by_key.items():
            finite_values = [value for value in values if not np.isnan(value)]
            mean_val = float(np.mean(finite_values)) if finite_values else float("nan")
            # we create keys such as "all/mse" or "all/bp_mse"
            hexel_metrics[f"all/{key}"] = mean_val

    return hexel_metrics


def print_and_log_eval_metrics(
    test_metrics: str | dict[str, float] | None,
    hexel_metrics: dict[str, float] | None,
    experiment_logger: CometLogger | None = None,
    split_label: str = "Test",
    metric_prefix: str = "test_hexel",
) -> None:
    """
    Prints terminal metrics and Comet logging for both patch-level and hexel-level metrics.

    ``split_label`` customizes the printed section headers (e.g. "Val") and
    ``metric_prefix`` customizes the Comet metric key prefix (e.g. "val_hexel")
    so metrics from different splits don't collide when logged side by side.
    """
    # patch-level metrics
    if isinstance(test_metrics, dict):
        print(f"\n[{split_label} patch-level metrics]")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

        if experiment_logger:
            experiment_logger.log_metrics({f"{split_label.lower()}_patch_{k}": v for k, v in test_metrics.items()})

    # hexel-level metrics
    if hexel_metrics:
        print(f"\n[{split_label} per-hexel and aggregated metrics]")
        current_group = None

        for k, v in hexel_metrics.items():
            group, metric_name = k.split("/")

            if current_group is not None and current_group != group:
                print("")
            current_group = group
            print(f"  [{group}] {metric_name}: {v:.6f}")

        if experiment_logger:
            experiment_logger.log_metrics({f"{metric_prefix}/{k}": v for k, v in hexel_metrics.items()})
