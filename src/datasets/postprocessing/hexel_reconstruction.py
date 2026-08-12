"""Reusable reconstruction of patch predictions into denormalized hexel rasters.

The producer yields one target/hexel at a time so callers can stream large
buffer extents instead of materializing every reconstructed raster at once.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rasterio.profiles import Profile

from data_preparation.paths import Paths, normalize_mask_scope
from src.config import Config
from src.datasets.postprocessing import utils as post_utils
from src.datasets.targets import TargetSpec


@dataclass(frozen=True)
class StitchedHexel:
    """Denormalized prediction and target grids for one target on one hexel."""

    hex_id: str
    target: TargetSpec
    gt_grid: np.ndarray
    pred_grid: np.ndarray
    profile: Profile
    actual_support_mask: np.ndarray | None = None
    buffer_support_mask: np.ndarray | None = None


def load_filtered_test_metadata(config: Config, mask_scope: str, split_csv: str | None = None) -> pd.DataFrame:
    """Load and filter split patch metadata used to stitch hexels.

    ``split_csv`` defaults to ``config.data.test_split`` but can be set to
    another split (e.g. ``config.data.val_split``) to reconstruct hexels for
    a different split.
    """
    csv_name = split_csv if split_csv is not None else config.data.test_split
    try:
        test_df = pd.read_csv(os.path.join(config.data.root_dir, csv_name))
    except (FileNotFoundError, AttributeError):
        raise ValueError(f"Split metadata file does not exist: {csv_name}")  # noqa: B904

    test_df = test_df[test_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)  # type: ignore
    post_utils.validate_patch_metadata_mask_scope(test_df, mask_scope)
    return test_df


def reconstruct_denormalized_hexels(
    *,
    test_predictions: np.ndarray,
    config: Config,
    out_norm: str,
    stitch_mode: str = "mean",
    mask_scope: str = "actual",
    split_csv: str | None = None,
    test_metadata: pd.DataFrame | None = None,
) -> Iterator[StitchedHexel]:
    """Yield stitched, denormalized hexel grids using the same settings as training/evaluation.

    ``split_csv`` selects which patch metadata split to stitch (defaults to
    ``config.data.test_split``); pass ``config.data.val_split`` to stitch the
    validation split instead.
    """
    if isinstance(test_predictions, str):
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")

    scope = normalize_mask_scope(mask_scope)
    settings_list = post_utils.get_target_postprocessing_settings(config=config, out_norm=out_norm)
    if test_metadata is None:
        test_df = load_filtered_test_metadata(config=config, mask_scope=scope, split_csv=split_csv)
    else:
        test_df = test_metadata.reset_index(drop=True).copy()
        post_utils.validate_patch_metadata_mask_scope(test_df, scope)
    prediction_mask_channel_indices = post_utils.get_prediction_mask_channel_indices(
        data_dir=config.data.root_dir,
        modelling_approach=config.modelling_approach,
        grid_params=post_utils.get_config_grid_params(config),
        prediction_support_policy=config.evaluation.prediction_support_policy,
    )

    for raw_hex_id in test_df["hex_id"].unique():
        hex_id = str(raw_hex_id).zfill(2)
        one_hexel_df = test_df[test_df["hex_id"] == raw_hex_id]
        hexel_indices = test_df[test_df["hex_id"] == raw_hex_id].index.tolist()
        hex_test_predictions = test_predictions[hexel_indices]

        for target_index, settings in enumerate(settings_list):
            print(
                f"[Postprocess] Reconstructing {settings.target.name.upper()} hex {hex_id} from {len(one_hexel_df)} {scope} patches...",
                flush=True,
            )
            target_predictions = post_utils.select_prediction_target_channel(
                predictions=hex_test_predictions,
                target_name=settings.target.name,
                target_index=target_index,
            )
            pred_grid, profile = post_utils.get_predicted_hexel(
                base_dir=config.data.root_dir,
                raw_data_dir=config.data.raw_data_dir,
                test_df=one_hexel_df,
                predictions=target_predictions,
                min_target_val=settings.min_target_val,
                max_target_val=settings.max_target_val,
                hex_id=hex_id,
                modelling_approach=config.modelling_approach,
                out_norm=settings.out_norm,
                target_log_mean=settings.target_log_mean,
                target_log_std=settings.target_log_std,
                stitch_mode=stitch_mode,
                target_channel_index=settings.target_channel_index,
                prediction_mask_channel_indices=prediction_mask_channel_indices,
                mask_scope=scope,
            )
            paths = Paths(hex_id=hex_id, root_dir=config.data.raw_data_dir)
            gt_grid, pred_grid = post_utils.load_target_grid_for_mask_scope(
                paths=paths,
                target=settings.target,
                pred_grid=pred_grid,
                profile=profile,
                mask_scope=scope,
                hex_id=hex_id,
                bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero,
            )
            actual_support_mask = None
            buffer_support_mask = None
            if scope != "actual" and profile.get("crs") is not None and profile.get("transform") is not None:
                buffer_support_mask = post_utils._actual_area_mask(
                    mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=scope),
                    profile=profile,
                    shape=pred_grid.shape,
                )
                actual_support_mask = post_utils._actual_area_mask(
                    mask_path=paths.mask_grid_actual(hex_id=hex_id),
                    profile=profile,
                    shape=pred_grid.shape,
                )
            yield StitchedHexel(
                hex_id=hex_id,
                target=settings.target,
                gt_grid=gt_grid,
                pred_grid=pred_grid,
                profile=profile,
                actual_support_mask=actual_support_mask,
                buffer_support_mask=buffer_support_mask,
            )
