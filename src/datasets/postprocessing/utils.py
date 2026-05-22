import functools
import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.profiles import Profile

from data_preparation.paths import Paths
from data_preparation.spatial.utils import (
    denormalize_burn_count,
    get_output_log_stats,
    get_range_output,
    load_spatial_raster,
)
from src.config import Config, GridParams
from src.datasets.postprocessing.stitch_hexel import stitch_windows
from src.datasets.postprocessing.visualize_predictions import (
    plot_hexbin_distribution,
    plot_histogram_distribution,
    visualize_hexel_iou,
    visualize_target_grids,
)
from src.datasets.targets import TargetSpec, get_target_spec
from src.datasets.utils import denormalize_output_target
from src.logger import CometLogger


def save_predicted_hexels(predicted_hexel: np.ndarray, hexel_profile: Profile, hex_id: str, save_dir: str):
    """
    Save the predicted (reconstructed) hexel
    Args:
        predicted_hexel (np.ndarray) : 2d array of shape (height, width)
        hexel_profile (rasterio.profile): Profile for the hexel, required by rasterio for saving geospatial data
        hex_id (str): The id of the hex to be saved
        save_dir (str): directory to save the hexel
    """
    out_path = os.path.join(save_dir, "predicted_hexels", f"hexel_{hex_id}_predicted.tif")
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
    stitch_mode: str = "mean",
    win_h: int = 128,
    win_w: int = 128,
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
        target_array = array[:, :, target_channel_index]
        mask = ~np.isnan(target_array)
        all_data_points.append(predictions[start_idx + i].reshape((win_h, win_w)))
        all_locations.append((data[5], data[6]))
        all_masks.append(mask.reshape((win_h, win_w)))
    reconstructed_hexel = stitch_windows(all_data_points, all_locations, all_masks, gt_shape, mode=stitch_mode)
    return reconstructed_hexel  # gt_shape


def get_predicted_hexel(
    base_dir: str,
    raw_data_dir: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    min_target_val: float,
    max_target_val: float,
    hex_id: str,
    modelling_approach: str = "1",
    out_norm: str = "min_max",
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
    stitch_mode: str = "mean",
    target_channel_index: int = 0,
    win_h: int = 128,
    win_w: int = 128,
) -> tuple[np.ndarray, Profile]:
    """
    Returns the reconstructed hexel
    """
    start_idx = 0

    all_paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)

    gt_elevation_grid, gt_elevation_grid_profile = load_spatial_raster(
        path=all_paths.elevation_grid(hex_id=hex_id), actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id)
    )

    if out_norm in {"min_max", "log"}:
        predictions = np.clip(predictions, 0, 1)

    if str(modelling_approach) == "1":
        reconstructed_hexel = get_stitched_windows(
            base_dir=base_dir,
            df=test_df,
            predictions=predictions,
            start_idx=start_idx,
            gt_shape=tuple(gt_elevation_grid.shape),
            target_channel_index=target_channel_index,
            stitch_mode=stitch_mode,
            win_h=win_h,
            win_w=win_w,
        )
        reconstructed_hexel_denorm = denormalize_output_target(
            data=reconstructed_hexel,
            target_min=min_target_val,
            target_max=max_target_val,
            out_norm=out_norm,
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
        )
        gt_elevation_grid_profile.update(dtype="float32", compress="lzw", nodata=-9999)  # type: ignore
    else:
        grouped_test_df = test_df.reset_index(drop=True)
        if "season" not in grouped_test_df.columns:
            raise ValueError("modelling_approach=2 evaluation requires a 'season' column in the metadata dataframe.")

        season_hexels = []
        unique_seasons = list(grouped_test_df["season"].drop_duplicates())
        for season in unique_seasons:
            filtered_season_df = grouped_test_df[grouped_test_df["season"] == season]
            filtered_predictions = predictions[filtered_season_df.index.to_numpy()]
            reconstructed_season_hexel = get_stitched_windows(
                base_dir=base_dir,
                df=filtered_season_df,
                predictions=filtered_predictions,
                start_idx=0,
                gt_shape=tuple(gt_elevation_grid.data.shape),
                target_channel_index=target_channel_index,
                stitch_mode=stitch_mode,
                win_h=win_h,
                win_w=win_w,
            )
            reconstructed_season_hexel_denorm = denormalize_burn_count(
                data=reconstructed_season_hexel, min_val=min_target_val, max_val=max_target_val
            )
            season_hexels.append(reconstructed_season_hexel_denorm)
        # merge the counts
        reconstructed_hexel_denorm = np.sum(np.stack(season_hexels), axis=0)
        reconstructed_hexel_denorm = np.rint(reconstructed_hexel_denorm).astype("int32")
        # clip values to the true range, in case of outliers
        reconstructed_hexel_denorm = np.clip(reconstructed_hexel_denorm, min_target_val, max_target_val)
        gt_elevation_grid_profile.update(dtype="int32", compress="lzw", nodata=-9999)  # type: ignore

    return reconstructed_hexel_denorm, gt_elevation_grid_profile


def get_config_target_spec(config: Config) -> TargetSpec:
    for source in config.data.input_sources:
        if source.name == "grid" and isinstance(source.params, GridParams):
            return get_target_spec(source.params.target_name)
    return get_target_spec("bp")


def get_config_grid_params(config: Config) -> GridParams | None:
    for source in config.data.input_sources:
        if source.name == "grid" and isinstance(source.params, GridParams):
            return source.params
    return None


def as_float_array_with_nan(data: np.ndarray) -> np.ndarray:
    data_ma = np.ma.masked_invalid(np.ma.asarray(data).astype("float32"))
    return np.asarray(data_ma.filled(np.nan), dtype=np.float32)


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


def get_modelling_approach_two_bp_ground_truth(
    raw_data_dir: str,
    hex_id: str,
    test_df: pd.DataFrame,
    reference_profile: Profile,
) -> np.ma.MaskedArray:
    all_paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    season_values = []
    for season in test_df["season"]:
        if pd.isna(season):
            continue
        if isinstance(season, str) and season.strip().lower() == "all":
            continue
        season_values.append(int(float(season)))

    unique_seasons = list(dict.fromkeys(season_values))
    if not unique_seasons:
        raise ValueError("modelling_approach=2 bp evaluation requires at least one concrete season in the metadata dataframe.")

    season_grids = []
    for season in unique_seasons:
        season_grid, _ = load_spatial_raster(
            path=all_paths.output_burn_prob(season=season),
            actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
            reference_profile=reference_profile,
        )
        season_grids.append(season_grid)

    return np.ma.sum(np.ma.stack(season_grids, axis=0), axis=0)


def calculate_hexel_metrics_pytorch(
    gt_grid: np.ndarray, pred_grid: np.ndarray, device: torch.device, metric_functions: dict[str, Callable]
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

    t_targets = torch.from_numpy(gt_clean).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_preds = torch.from_numpy(pred_clean).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_mask = torch.from_numpy(valid_mask_np).to(device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

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
    device: torch.device,
    experiment_logger: CometLogger | None = None,
    metric_functions: dict[str, Callable] | None = None,
) -> dict[str, float]:
    """
    A util function to re-construct predicted hexels out of test predictions, and visualize side-by-side with the Groundtruth.
    Also computes and aggregates stitched hexel-level metrics.
    """
    data_dir = config.data.root_dir
    raw_data_dir = config.data.raw_data_dir
    modelling_approach = config.modelling_approach
    valid_mask_threshold = config.data.valid_mask_threshold
    target = get_config_target_spec(config)
    grid_params = get_config_grid_params(config)
    target_channel_index = get_target_channel_index(data_dir=data_dir, modelling_approach=modelling_approach, target=target)

    max_target_val, min_target_val = get_range_output(root_dir=raw_data_dir, output_type=target.output_type)
    target_log_mean = grid_params.target_log_mean if grid_params is not None else None
    target_log_std = grid_params.target_log_std if grid_params is not None else None
    if out_norm == "log_standard" and (target_log_mean is None or target_log_std is None):
        target_log_mean, target_log_std = get_output_log_stats(root_dir=raw_data_dir, output_type=target.output_type)

    if isinstance(test_predictions, str):
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")

    try:
        test_df = pd.read_csv(os.path.join(data_dir, config.data.test_split))
    except (FileNotFoundError, AttributeError):
        raise ValueError("Test df file does not exist.")  # noqa: B904

    # separate hexels by their IDs
    test_df = test_df[test_df["valid_ratio"] > valid_mask_threshold].reset_index(drop=True)  # type: ignore
    all_hex_ids = list(test_df["hex_id"].unique())

    # init. hexel metrics
    all_hexel_metrics = []

    # loop over test hexels
    for hex_id in all_hex_ids:
        print(f"======Working with hex{hex_id}========")
        one_hexel_df = test_df[test_df["hex_id"] == hex_id]
        hexel_indices = test_df[test_df["hex_id"] == hex_id].index.tolist()
        if len(str(hex_id)) != 2:
            hex_id = "0" + str(hex_id)

        hex_test_predictions = test_predictions[hexel_indices]
        reconstructed_hexel_denorm, gt_elevation_grid_profile = get_predicted_hexel(
            base_dir=data_dir,
            raw_data_dir=raw_data_dir,
            test_df=one_hexel_df,
            predictions=hex_test_predictions,
            min_target_val=min_target_val,
            max_target_val=max_target_val,
            hex_id=hex_id,
            modelling_approach=modelling_approach,
            out_norm=out_norm,
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
            stitch_mode="mean",
            target_channel_index=target_channel_index,
            win_h=config.data_prep.win_h,
            win_w=config.data_prep.win_w,
        )
        save_predicted_hexels(reconstructed_hexel_denorm, gt_elevation_grid_profile, hex_id, config.save_dir)
        # Save the hex as plt plot
        if str(modelling_approach) == "2" and target.name == "bp":
            grid_gt = get_modelling_approach_two_bp_ground_truth(
                raw_data_dir=raw_data_dir,
                hex_id=hex_id,
                test_df=one_hexel_df,
                reference_profile=gt_elevation_grid_profile,
            )
        else:
            all_paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
            target_path = getattr(all_paths, target.path_method)()
            grid_gt, _ = load_spatial_raster(
                path=target_path,
                actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
                reference_profile=gt_elevation_grid_profile,
            )

        visualize_target_grids(
            gt_grid=grid_gt,
            pred_grid=reconstructed_hexel_denorm,
            hex_id=hex_id,
            save_dir=config.save_dir,
            experiment_logger=experiment_logger,
            target_label=target.label,
        )

        # plot and save hexbin figures (for calibration)
        plot_hexbin_distribution(
            gt_grid=grid_gt,
            pred_grid=reconstructed_hexel_denorm,
            hex_id=hex_id,
            save_dir=config.save_dir,
            experiment_logger=experiment_logger,
            target_label=target.label,
            probability_scale=target.probability_scale,
        )

        # plot and save hist. figures
        plot_histogram_distribution(
            gt_grid=grid_gt,
            pred_grid=reconstructed_hexel_denorm,
            hex_id=hex_id,
            save_dir=config.save_dir,
            experiment_logger=experiment_logger,
            target_label=target.label,
            probability_scale=target.probability_scale,
        )

        # compute per-hexel metrics
        if metric_functions is not None:
            hex_metrics = calculate_hexel_metrics_pytorch(
                gt_grid=grid_gt, pred_grid=reconstructed_hexel_denorm, device=device, metric_functions=metric_functions
            )
            all_hexel_metrics.append(hex_metrics)

            # get top k perc. values dynamically
            percentiles_to_plot = [
                fn.keywords["percentile"]
                for _, fn in metric_functions.items()
                if isinstance(fn, functools.partial) and "percentile" in fn.keywords
            ]

            # generate the TopK IoU plots
            for p in percentiles_to_plot:
                pred_bin, gt_bin = get_hexel_binary_maps(reconstructed_hexel_denorm, grid_gt, percentile=p)
                visualize_hexel_iou(
                    grid_gt,
                    reconstructed_hexel_denorm,
                    gt_bin,
                    pred_bin,
                    hex_id,
                    config.save_dir,
                    p,
                    target_label=target.label,
                )

        print(f"=======Saved subplots for hex{hex_id}==============")

    # aggregate final scores
    hexel_metrics = {}

    if metric_functions is not None and len(all_hexel_metrics) > 0:
        # get per-hexel metrics
        for hex_id, hex_metric in zip(all_hex_ids, all_hexel_metrics, strict=False):
            hex_id_str = str(hex_id).zfill(2)
            for key, val in hex_metric.items():
                # we create keys such as "hex12/mse" for clarity
                hexel_metrics[f"hex{hex_id_str}/{key}"] = float(val) if not np.isnan(val) else float("nan")

        # get the aggregated averages over all hexels
        for key in metric_functions:
            mean_val = np.nanmean([hm[key] for hm in all_hexel_metrics if key in hm and not np.isnan(hm[key])])
            # we create keys such as "all/mse"
            hexel_metrics[f"all/{key}"] = float(mean_val)

    return hexel_metrics


def print_and_log_eval_metrics(
    test_metrics: str | dict[str, float] | None, hexel_metrics: dict[str, float] | None, experiment_logger: CometLogger | None = None
) -> None:
    """
    Prints terminal metrics and Comet logging for both patch-level and hexel-level metrics.
    """
    # patch-level metrics
    if isinstance(test_metrics, dict):
        print("\n[Test patch-level metrics]")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

        if experiment_logger:
            experiment_logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # hexel-level metrics
    if hexel_metrics:
        print("\n[Test per-hexel and aggregated metrics]")
        current_group = None

        for k, v in hexel_metrics.items():
            group, metric_name = k.split("/")

            if current_group is not None and current_group != group:
                print("")
            current_group = group
            print(f"  [{group}] {metric_name}: {v:.6f}")

        if experiment_logger:
            experiment_logger.log_metrics({f"hexel/{k}": v for k, v in hexel_metrics.items()})
