"""
End-to-end inference pipeline for a single hexel.
Orchestrates data preparation, dataset building, and prediction.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation.hexel_loader import load_spatial_features_per_hexel
from data_preparation.paths import MASK_SCOPE_CHOICES, Paths, normalize_mask_scope, prepared_mask_scope
from data_preparation.process_hexels_into_grids import get_split_hexel_window
from data_preparation.process_tabular_data import build_weather_table, process_fire_size_distribution_table
from data_preparation.spatial.utils import get_output_log_stats_cached, get_range_output_cached, read_split_hex_ids
from data_preparation.utils import find_hex_ids
from inference.predictor import BurnRiskPredictor
from src.datasets.dataset import MultiSourceDataset
from src.datasets.postprocessing.hazard import compute_raw_hazard
from src.datasets.postprocessing.utils import (
    get_mask_scope_save_dir,
    get_predicted_hexel,
    get_prediction_mask_channel_indices,
    get_target_channel_index,
    load_target_grid_for_mask_scope,
    save_predicted_hexels,
    validate_patch_metadata_mask_scope,
)
from src.datasets.postprocessing.visualize_predictions import visualize_target_grids
from src.datasets.targets import TargetSpec, get_target_spec, get_target_specs
from src.datasets.utils import apply_bp_nodata_zero_range, get_data_source_class, get_data_source_param_class, get_dataset_dimensions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference/run_hexel_inference.log", mode="w"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetNormalization:
    out_norm: str
    min_value: float | None = None
    max_value: float | None = None
    log_mean: float | None = None
    log_std: float | None = None


def prepare_hexel_data(
    data_dir: Path,
    hex_id: str,
    win_h: int = 128,
    win_w: int = 128,
    overlap_ratio: float = 0.2,
    modelling_approach: int = 1,
    output_type: str = "prob",
    weather_sampling: str = "weather_zone_id",
    mask_scope: str = "actual",
    weather_norm_params_path: Path | None = None,
    fire_size_norm_params_path: Path | None = None,
) -> Path:
    """
    Prepare data patches for a single hexel.

    Handles all the CSV building, raw data loading, and patch splitting.

    Args:
        data_dir: Directory containing hexel data.
        hex_id: Hexel ID to process (e.g., "02").
        win_h: Patch height in pixels.
        win_w: Patch width in pixels.
        overlap_ratio: Overlap between patches (0.0 to 1.0).
        modelling_approach: 1 for joint season-cause, 2 for separate.
        output_type: "count" or "prob" for fire output type.
        weather_sampling: Weather sampling strategy.
        weather_norm_params_path: Path to a JSON file with weather normalization parameters.
            If the file exists, parameters are loaded and applied (inference mode) instead of
            being refit, preventing leakage from the hexel(s) being predicted. Defaults to
            ``weather_norm_params.json`` inside the processed data directory.
        fire_size_norm_params_path: Path to a JSON file with fire size normalization parameters.
            Same semantics as ``weather_norm_params_path``. Defaults to
            ``fire_size_norm_params.json`` inside the processed data directory.

    Returns:
        Path to the output directory containing patches and metadata CSV.
    """
    scope = prepared_mask_scope(mask_scope)
    suffix = "" if scope == "actual" else f"_{scope}"
    processed_data_dir = data_dir / f"data_samples_approach_{modelling_approach}{suffix}"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    (processed_data_dir / "numpy_files").mkdir(parents=True, exist_ok=True)

    if weather_norm_params_path is None:
        weather_norm_params_path = processed_data_dir / "weather_norm_params.json"
    if fire_size_norm_params_path is None:
        fire_size_norm_params_path = processed_data_dir / "fire_size_norm_params.json"

    weather_table_path = processed_data_dir / "weather_table_processed.csv"
    logger.info("Building weather table...")
    build_weather_table(root_dir=data_dir, save_path=weather_table_path, norm_params_path=weather_norm_params_path)

    fire_size_input = data_dir / "df_fire_fru.csv"
    fire_size_output = processed_data_dir / "df_fire_fru_processed.csv"
    if fire_size_input.exists():
        logger.info("Processing fire size distribution table...")
        process_fire_size_distribution_table(
            input_path=fire_size_input, output_path=fire_size_output, norm_params_path=fire_size_norm_params_path
        )

    # Load features for the hexel
    feature_channel_map_path = processed_data_dir / f"feature_channel_map_{modelling_approach}.json"

    available_hex_ids = find_hex_ids(str(data_dir))
    if hex_id not in available_hex_ids:
        logger.error(f"Hexel ID {hex_id} not found in {data_dir}. Available hexel IDs: {available_hex_ids}")
        raise ValueError(f"Hexel ID {hex_id} not found in {data_dir}. Check logs for details.")

    logger.info(f"Loading features for hexel {hex_id}...")
    stacked_feats, mask, season_cause_mapping = load_spatial_features_per_hexel(
        root_dir=str(data_dir),
        hex_id=hex_id,
        feature_channel_map_path=str(feature_channel_map_path),
        modelling_approach=modelling_approach,
        mask_scope=scope,
    )

    if stacked_feats is None or mask is None:
        logger.error(f"Failed to load features or mask for hexel {hex_id}. Aborting data preparation.")
        raise ValueError(f"Failed to load features or mask for hexel {hex_id}. Check logs for details.")

    logger.info(f"Splitting hexel {hex_id} into {win_h}x{win_w} patches...")
    get_split_hexel_window(
        season_cause_stacked_feats=stacked_feats,
        season_cause_mask=mask,
        season_cause_mapping=season_cause_mapping,
        out_dir=str(processed_data_dir),
        root_dir=str(data_dir),
        hex_id=hex_id,
        win_h=win_h,
        win_w=win_w,
        overlap_ratio=overlap_ratio,
        mask_scope=scope,
    )

    logger.info(f"Data preparation complete. Output saved to {processed_data_dir}")
    return processed_data_dir


def create_dataset(processed_data_dir: Path, hex_id: str, config_dict: dict) -> MultiSourceDataset:
    """
    Build the PyTorch Dataset based on the saved checkpoint config.

    Args:
        processed_data_dir: Directory containing processed data.
        hex_id: Hexel ID.
        config_dict: Checkpoint config dict.

    Returns:
        MultiSourceDataset ready for inference.
    """
    csv_name = f"meta_hex_{hex_id}.csv"
    filename_col = config_dict["filename_col"]
    valid_mask_threshold = config_dict["valid_mask_threshold"]
    sources = {}
    for source in config_dict["input_sources"]:
        source_name = source["name"]
        source_class = get_data_source_class(source_name)
        source_param_class = get_data_source_param_class(source_name)
        source_kwargs: dict[str, Any] = {
            "root_dir": processed_data_dir,
            "params": source_param_class(**source["params"]),
        }
        if source_name == "grid":
            source_kwargs.update(
                root_dir=config_dict.get("root_dir", processed_data_dir),
                raw_data_dir=config_dict.get("raw_data_dir"),
                train_split_csv_name=config_dict.get("train_split"),
            )
        sources[source_name] = source_class(**source_kwargs)

    return MultiSourceDataset(
        csv_name=csv_name,
        root_dir=str(processed_data_dir),
        sources=sources,
        filename_col=filename_col,
        valid_mask_threshold=valid_mask_threshold,
    )


def get_target_specs_from_data_config(data_config: dict) -> list[TargetSpec]:
    """Read ordered target metadata from checkpoint data config, defaulting to BP."""
    for source in data_config.get("input_sources", []):
        if source.get("name") == "grid":
            params = source.get("params", {})
            if params.get("targets"):
                return get_target_specs([target["name"] for target in params["targets"]])
            return get_target_specs(params.get("target_name", "bp"))
    return get_target_specs("bp")


def get_target_spec_from_data_config(data_config: dict) -> TargetSpec:
    targets = get_target_specs_from_data_config(data_config)
    if len(targets) != 1:
        raise ValueError(f"Expected one inference target, got {[target.name for target in targets]}.")
    return targets[0]


def get_grid_params_from_data_config(data_config: dict) -> dict[str, Any]:
    for source in data_config.get("input_sources", []):
        if source.get("name") == "grid":
            return source.get("params", {})
    return {}


def get_target_params_from_grid_config(grid_params: dict[str, Any], target: TargetSpec) -> dict[str, Any]:
    target_configs = grid_params.get("targets")
    if target_configs:
        for target_config in target_configs:
            if get_target_spec(target_config["name"]).name == target.name:
                return target_config
        raise KeyError(f"Missing target config for {target.name!r}.")
    return {
        "name": target.name,
        "out_norm": grid_params.get("out_norm", "min_max"),
        "log_mean": grid_params.get("target_log_mean"),
        "log_std": grid_params.get("target_log_std"),
    }


def resolve_target_normalization(
    data_config: dict[str, Any],
    target: TargetSpec,
    target_params: dict[str, Any],
    *,
    bp_nodata_as_zero: bool,
) -> TargetNormalization:
    root_dir = str(data_config["root_dir"])
    raw_data_dir = str(data_config.get("raw_data_dir") or root_dir)
    train_hex_ids = None
    train_split = data_config.get("train_split")
    if train_split:
        train_split_path = Path(root_dir) / str(train_split)
        if train_split_path.is_file():
            train_hex_ids = read_split_hex_ids(str(train_split_path))

    out_norm = str(target_params["out_norm"])
    min_value = None
    max_value = None
    log_mean = target_params.get("log_mean")
    log_std = target_params.get("log_std")
    if out_norm == "min_max":
        max_value, min_value = get_range_output_cached(
            root_dir=root_dir,
            output_type=target.output_type,
            allowed_hex_ids=train_hex_ids,
            raw_data_dir=raw_data_dir,
        )
        max_value, min_value = apply_bp_nodata_zero_range(
            target_name=target.name,
            max_value=max_value,
            min_value=min_value,
            bp_nodata_as_zero=bp_nodata_as_zero,
        )
    elif out_norm == "log_standard" and (log_mean is None or log_std is None):
        log_mean, log_std = get_output_log_stats_cached(
            root_dir=root_dir,
            output_type=target.output_type,
            allowed_hex_ids=train_hex_ids,
            raw_data_dir=raw_data_dir,
        )
    return TargetNormalization(
        out_norm=out_norm,
        min_value=min_value,
        max_value=max_value,
        log_mean=log_mean,
        log_std=log_std,
    )


def run_single_hexel_pipeline(
    checkpoint_path: Path,
    data_dir: Path,
    hex_id: str,
    batch_size: int = 32,
    num_workers: int = 4,
    prepare_data: bool = False,
    save_dir: Path = Path("outputs"),
    mask_scope: str = "actual",
    weather_norm_params_path: Path | None = None,
    fire_size_norm_params_path: Path | None = None,
) -> tuple[np.ndarray | dict[str, np.ndarray], Any]:
    """
    Orchestrate the end-to-end (data preparation + inference + post-processing) for one specific hexel.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        data_dir: Directory containing hexel data.
        hex_id: Hexel ID to process.
        batch_size: Batch size for inference.
        num_workers: Number of dataloader workers.
        prepare_data: If True, run data preparation step.
        win_h: Patch height in pixels.
        win_w: Patch width in pixels.
        overlap_ratio: Overlap ratio between patches.
        modelling_approach: 1 for joint season-cause, 2 for separate.
        output_type: "count" or "prob" for fire output.
        weather_sampling: Weather sampling strategy.
        save_dir: Directory to save predictions and visualizations.
        weather_norm_params_path: Path to a JSON file with weather normalization parameters,
            forwarded to `prepare_hexel_data` when `prepare_data` is True.
        fire_size_norm_params_path: Path to a JSON file with fire size normalization parameters,
            forwarded to `prepare_hexel_data` when `prepare_data` is True.

    Returns:
        Reconstructed target hexel grid (denormalized), and the ground truth elevation grid profile.
    """
    # Step 1: Load checkpoint
    logger.info("Step 1: Loading checkpoint and config...")
    scope = normalize_mask_scope(mask_scope)
    data_scope = prepared_mask_scope(scope)
    artifact_save_dir = Path(get_mask_scope_save_dir(str(save_dir), scope))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_config = checkpoint["config"]["data"]  # We use this to build dataset class
    data_prep_config = checkpoint["config"]["data_prep"]  # We use this to prepare data

    # Step 2: Prepare the Data (if requested)
    if prepare_data:
        logger.info("Step 2: Preparing Hexel Data...")
        processed_data_dir = prepare_hexel_data(
            data_dir=data_dir,
            hex_id=hex_id,
            win_h=data_prep_config["win_h"],
            win_w=data_prep_config["win_w"],
            overlap_ratio=data_prep_config["overlap_ratio"],
            modelling_approach=data_prep_config["modelling_approach"],
            mask_scope=data_scope,
            weather_norm_params_path=weather_norm_params_path,
            fire_size_norm_params_path=fire_size_norm_params_path,
        )
    else:
        suffix = "" if data_scope == "actual" else f"_{data_scope}"
        processed_data_dir = data_dir / f"data_samples_approach_{data_prep_config['modelling_approach']}{suffix}"
        logger.info(f"Step 2: Using existing data at {processed_data_dir}")

    # Step 3: Build Dataset
    logger.info("Step 3: Building Dataset...")
    dataset = create_dataset(processed_data_dir, hex_id, data_config)
    validate_patch_metadata_mask_scope(dataset.metadata, scope)
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(dataset)
    if spatial_channels is None:
        raise ValueError("Could not determine spatial channels from dataset")
    logger.info(f"Dataset: {len(dataset)} samples | Spatial channels: {spatial_channels} | Auxiliary dims: {auxiliary_input_dims}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Step 4: Instantiate the Predictor (pass pre-loaded checkpoint)
    logger.info("Step 4: Initializing Model Predictor...")
    predictor = BurnRiskPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path, spatial_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims
    )

    # Step 5: Run Inference Loop
    logger.info("Step 5: Running Inference...")
    pred_start_time = time.time()
    predictions_list = []
    for batch in tqdm(dataloader, desc="Predicting Batches"):
        spatial_inputs = batch["grid"][0]
        batch_preds = predictor(spatial_inputs, auxiliary_inputs=batch)  # Predictor handles device placement internally
        predictions_list.append(batch_preds)
    pred_time = time.time() - pred_start_time
    logger.info(f"TIME - Inference completed in {pred_time:.2f} seconds for {len(predictions_list)} batches.")
    predictions = torch.cat(predictions_list, dim=0).numpy()

    # Step 6: Save patch predictions
    save_pred_path = artifact_save_dir / "predicted_patches" / f"hexel_{hex_id}.npy"
    save_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_pred_path, predictions)
    logger.info(f"Step 6: Saved predictions patches to {save_pred_path}")

    # Step 7: Post-process predictions back to denormalized hexel
    logger.info("Step 7: Post-processing prediction patches into denormalized hexel...")
    targets = get_target_specs_from_data_config(data_config)
    grid_params = get_grid_params_from_data_config(data_config)
    prediction_mask_channel_indices = get_prediction_mask_channel_indices(
        data_dir=str(processed_data_dir),
        modelling_approach=str(data_prep_config["modelling_approach"]),
        grid_params=grid_params,
        prediction_support_policy=checkpoint["config"].get("evaluation", {}).get("prediction_support_policy", "input"),
    )
    all_paths = Paths(hex_id=hex_id, root_dir=data_dir)
    reconstructed_targets: dict[str, np.ndarray] = {}
    ground_truth_targets: dict[str, np.ndarray] = {}
    gt_elevation_grid_profile = None
    multi_target = len(targets) > 1
    bp_nodata_as_zero = checkpoint["config"].get("evaluation", {}).get("bp_nodata_as_zero", True)
    if predictions.shape[1] != len(targets):
        raise ValueError(f"Prediction channels {predictions.shape[1]} do not match targets {[target.name for target in targets]}.")

    for target_index, target in enumerate(targets):
        target_params = get_target_params_from_grid_config(grid_params, target)
        normalization = resolve_target_normalization(
            data_config,
            target,
            target_params,
            bp_nodata_as_zero=bp_nodata_as_zero,
        )

        target_channel_index = get_target_channel_index(
            data_dir=str(processed_data_dir),
            modelling_approach=str(data_prep_config["modelling_approach"]),
            target=target,
        )
        reconstructed_target, target_profile = get_predicted_hexel(
            base_dir=str(processed_data_dir),
            raw_data_dir=str(data_dir),
            test_df=dataset.metadata,
            predictions=predictions[:, target_index],
            min_target_val=normalization.min_value,
            max_target_val=normalization.max_value,
            hex_id=hex_id,
            out_norm=normalization.out_norm,
            target_log_mean=normalization.log_mean,
            target_log_std=normalization.log_std,
            target_channel_index=target_channel_index,
            prediction_mask_channel_indices=prediction_mask_channel_indices,
            mask_scope=scope,
        )
        gt_grid, reconstructed_target = load_target_grid_for_mask_scope(
            paths=all_paths,
            target=target,
            pred_grid=reconstructed_target,
            profile=target_profile,
            mask_scope=scope,
            hex_id=hex_id,
            bp_nodata_as_zero=bp_nodata_as_zero,
        )
        artifact_target_name = target.name if multi_target else None
        save_predicted_hexels(
            predicted_hexel=reconstructed_target,
            hexel_profile=target_profile,
            hex_id=hex_id,
            save_dir=str(artifact_save_dir),
            target_name=artifact_target_name,
        )
        visualize_target_grids(
            gt_grid=gt_grid,
            pred_grid=reconstructed_target,
            hex_id=hex_id,
            save_dir=str(artifact_save_dir),
            target_label=target.label,
            target_name=artifact_target_name,
        )
        reconstructed_targets[target.name] = reconstructed_target
        ground_truth_targets[target.name] = gt_grid
        gt_elevation_grid_profile = target_profile

    if gt_elevation_grid_profile is None:
        raise RuntimeError("No target outputs were reconstructed.")
    if {"bp", "fi"} <= set(reconstructed_targets):
        fi_cap = checkpoint["config"].get("evaluation", {}).get("hazard_fi_cap", 10000.0)
        reconstructed_targets["hazard"] = compute_raw_hazard(
            reconstructed_targets["bp"],
            reconstructed_targets["fi"],
            fi_cap=fi_cap,
        )
        ground_truth_targets["hazard"] = compute_raw_hazard(
            ground_truth_targets["bp"],
            ground_truth_targets["fi"],
            fi_cap=fi_cap,
        )
        save_predicted_hexels(
            predicted_hexel=reconstructed_targets["hazard"],
            hexel_profile=gt_elevation_grid_profile,
            hex_id=hex_id,
            save_dir=str(artifact_save_dir),
            target_name="hazard",
        )
        visualize_target_grids(
            gt_grid=ground_truth_targets["hazard"],
            pred_grid=reconstructed_targets["hazard"],
            hex_id=hex_id,
            save_dir=str(artifact_save_dir),
            target_label="Raw Hazard",
            target_name="hazard",
        )

    logger.info(f"Step 8: Saved reconstructed hexel and visualization for hexel {hex_id} in {artifact_save_dir}")

    if not multi_target:
        return reconstructed_targets[targets[0].name], gt_elevation_grid_profile
    return reconstructed_targets, gt_elevation_grid_profile


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end inference on a single hexel.")
    parser.add_argument("--config", type=str, default="inference/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing hexel data (overrides config).")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (overrides config).")
    parser.add_argument("--hex_id", type=str, default=None, help="Hexel ID (overrides config).")
    parser.add_argument("--prepare_data", type=str, default=None, help="Whether to prepare data (overrides config).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers (overrides config).")
    parser.add_argument("--post_process", type=str, default=None, help="Whether to post-process predictions (overrides config).")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predictions and visualizations (overrides config).")
    parser.add_argument("--mask_scope", choices=MASK_SCOPE_CHOICES, default=None, help="Evaluation/inference mask scope.")
    parser.add_argument(
        "--weather_norm_params_path",
        type=str,
        default=None,
        help="Path to JSON file with weather normalization parameters to reuse at inference (overrides config).",
    )
    parser.add_argument(
        "--fire_size_norm_params_path",
        type=str,
        default=None,
        help="Path to JSON file with fire size normalization parameters to reuse at inference (overrides config).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI args override config (use 'is not None' to allow falsy values like 0)
    data_dir = args.data_dir if args.data_dir is not None else config["data_dir"]
    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else config["checkpoint_path"]
    save_dir = args.save_dir if args.save_dir is not None else config["save_dir"]
    hex_id = args.hex_id if args.hex_id is not None else config["hex_id"]
    prepare_data = (args.prepare_data == "True") if args.prepare_data else config["prepare_data"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["num_workers"]
    mask_scope = args.mask_scope if args.mask_scope is not None else config.get("mask_scope", "actual")
    weather_norm_params_path = (
        args.weather_norm_params_path if args.weather_norm_params_path is not None else config.get("weather_norm_params_path")
    )
    fire_size_norm_params_path = (
        args.fire_size_norm_params_path if args.fire_size_norm_params_path is not None else config.get("fire_size_norm_params_path")
    )
    weather_norm_params_path = Path(weather_norm_params_path) if weather_norm_params_path else None
    fire_size_norm_params_path = Path(fire_size_norm_params_path) if fire_size_norm_params_path else None

    # Resolve "all" into the list of available hex IDs
    if hex_id == "all":
        hex_ids_to_run = sorted(find_hex_ids(str(Path(data_dir))))
        logger.info(f"Running inference for all hexels: {hex_ids_to_run}")
    else:
        hex_ids_to_run = [hex_id]

    start_time = time.time()

    for hid in hex_ids_to_run:
        logger.info(f"\n\n========== Hexel {hid} ==========\n")
        run_single_hexel_pipeline(
            checkpoint_path=Path(checkpoint_path),
            data_dir=Path(data_dir),
            hex_id=hid,
            batch_size=batch_size,
            num_workers=num_workers,
            prepare_data=prepare_data,
            save_dir=Path(save_dir),
            mask_scope=mask_scope,
            weather_norm_params_path=weather_norm_params_path,
            fire_size_norm_params_path=fire_size_norm_params_path,
        )

    elapsed_time = time.time() - start_time
    logger.info(
        f"TIME - Pipeline for {len(hex_ids_to_run)} hexels processed. Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
