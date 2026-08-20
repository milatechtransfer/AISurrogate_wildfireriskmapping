from __future__ import annotations

import json
import os

import numpy as np

from data_preparation.paths import Paths, normalize_mask_scope
from data_preparation.spatial import (
    NODATA,
    load_fuel_grid,
    load_ignition_grid,
    load_ignition_grid_probability_mass,
    load_ignition_grid_weighted,
    load_spatial_raster,
)
from data_preparation.utils import feature_names, feature_names_weighted_ignition

IGNITION_WEIGHTING_CHOICES = ("max", "distribution", "probability_mass")
FUEL_GRID_CHOICES = ("raw", "group")


def get_num_channels_array(arr: np.ndarray) -> int:
    """Returns the number of channels a particular feature will take"""
    if len(arr.shape) == 2:
        return 1
    return arr.shape[-1]


def generate_feature_channel_map(feature_list: list[np.ndarray], feature_channel_map_path: str, names: list[str] | None = None):
    """Maps the feature names to the corresponding channels in our input stack"""
    names = names or feature_names
    feature_channel_map = dict()
    channel = 0
    for i, feature in enumerate(feature_list):
        feature_channels = get_num_channels_array(feature)
        feature_channel_map[names[i]] = list(range(channel, channel + feature_channels))
        channel += feature_channels
    os.makedirs(os.path.dirname(feature_channel_map_path), exist_ok=True)
    with open(feature_channel_map_path, "w") as f:
        json.dump(feature_channel_map, f, indent=4)


def load_spatial_features_per_hexel(
    root_dir: str,
    hex_id: str,
    feature_channel_map_path: str,
    modelling_approach: int = 1,
    mask_scope: str = "actual",
    ignition_weighting: str = "distribution",
    fuel_representation: str = "raw",
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, tuple[int, int]] | None]:
    """
    Load all data (features and output) per hexel
    root_dir: Root directory containing all hexels.
    hex_id: Hexel id to load.
    modelling_approach: 1 for joint season-cause modelling, 2 for separate season-cause modelling.
    ignition_weighting: "distribution" for legacy zone-area-weighted blending,
        "probability_mass" for exact BurnP3+ per-ignition spatial mass
        (both produce Human + Lightning channels), or "max" for one max channel.
    Returns:
        all_features: np.ndarray of shape (N, H, W, num_features)
        all_masks: np.ndarray of shape (N, H, W)
        season_cause_mapping: dict mapping index to (season, cause)
    """
    if ignition_weighting not in IGNITION_WEIGHTING_CHOICES:
        raise ValueError(f"ignition_weighting must be one of {IGNITION_WEIGHTING_CHOICES}, got {ignition_weighting!r}.")

    if fuel_representation not in FUEL_GRID_CHOICES:
        raise ValueError(f"fuel_representation must be one of {FUEL_GRID_CHOICES}, got {fuel_representation!r}.")

    use_two_channel_ignition = ignition_weighting in {"distribution", "probability_mass"}

    def stack_sample(
        fuel_grid: np.ma.MaskedArray,
        elevation_grid: np.ma.MaskedArray,
        ignition_grid: np.ma.MaskedArray,
        firezones_grid: np.ma.MaskedArray,
        bp_out_grid: np.ma.MaskedArray,
        fi_out_grid: np.ma.MaskedArray,
        ros_out_grid: np.ma.MaskedArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack all features and compute mask.

        ignition_grid may be (H, W) for the max-aggregation path or
        (H, W, 2) for the distribution-weighted path.
        """
        if use_two_channel_ignition:
            # ignition_grid is (H, W, 2) — split into two separate (H, W, 1) channels
            # so that generate_feature_channel_map maps each to its own name
            ign_features_list = [ignition_grid[:, :, 0:1], ignition_grid[:, :, 1:2]]
        else:
            ign_features_list = [ignition_grid[:, :, np.newaxis]]  # (H, W, 1)

        features_list = [
            fuel_grid[:, :, np.newaxis],
            elevation_grid[:, :, np.newaxis],
            *ign_features_list,
            firezones_grid[:, :, np.newaxis],
            bp_out_grid[:, :, np.newaxis],
            fi_out_grid[:, :, np.newaxis],
            ros_out_grid[:, :, np.newaxis],
        ]

        if not os.path.exists(feature_channel_map_path):
            names = feature_names_weighted_ignition if use_two_channel_ignition else feature_names
            generate_feature_channel_map(features_list, feature_channel_map_path, names=names)

        fuel_mask = np.ma.getmaskarray(fuel_grid)
        elevation_mask = np.ma.getmaskarray(elevation_grid)
        firezones_mask = np.ma.getmaskarray(firezones_grid)
        # For 2-channel ignition, a pixel is masked if ANY channel is masked
        raw_ign_mask = np.ma.getmaskarray(ignition_grid)
        ignition_mask = np.any(raw_ign_mask, axis=-1) if raw_ign_mask.ndim == 3 else raw_ign_mask
        input_mask = fuel_mask | elevation_mask | ignition_mask | firezones_mask

        stacked_ma = np.ma.concatenate(features_list, axis=-1)
        stacked = stacked_ma.filled(NODATA).astype(np.float32)
        stacked[input_mask, :] = NODATA
        return stacked, input_mask

    # identify all seasons and causes first
    scope = normalize_mask_scope(mask_scope)
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    scope_mask_path = all_paths.mask_grid(hex_id=hex_id, mask_scope=scope)

    elevation_grid, reference_profile = load_spatial_raster(path=all_paths.elevation_grid(hex_id=hex_id), mask_path=scope_mask_path)
    # load all common grids on the elevation reference grid
    fuel_grid = load_fuel_grid(
        root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile, fuel_representation=fuel_representation, mask_scope=scope
    )

    firezones_grid, _ = load_spatial_raster(
        path=all_paths.firezones_grid(hex_id=hex_id),
        mask_path=scope_mask_path,
        reference_profile=reference_profile,
    )

    if modelling_approach == 1:
        # input
        if ignition_weighting == "distribution":
            ignition_grid = load_ignition_grid_weighted(
                root_dir=root_dir,
                hex_id=hex_id,
                firezones_grid=firezones_grid,
                reference_profile=reference_profile,
                mask_scope=scope,
            )
        elif ignition_weighting == "probability_mass":
            ignition_grid = load_ignition_grid_probability_mass(
                root_dir=root_dir,
                hex_id=hex_id,
                firezones_grid=firezones_grid,
                reference_profile=reference_profile,
                mask_scope=scope,
            )
        else:
            ignition_grid = load_ignition_grid(root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile, mask_scope=scope)

        bp_out_grid, _ = load_spatial_raster(
            all_paths.output_burn_prob(),
            mask_path=scope_mask_path,
            reference_profile=reference_profile,
        )
        fi_out_grid, _ = load_spatial_raster(
            all_paths.output_fire_intensity(),
            mask_path=scope_mask_path,
            reference_profile=reference_profile,
        )
        ros_out_grid, _ = load_spatial_raster(
            all_paths.output_ros(),
            mask_path=scope_mask_path,
            reference_profile=reference_profile,
        )

        stacked_features, mask = stack_sample(
            fuel_grid, elevation_grid, ignition_grid, firezones_grid, bp_out_grid, fi_out_grid, ros_out_grid
        )
        return np.expand_dims(stacked_features, axis=0), np.expand_dims(mask, axis=0), None

    # modelling approach 2
    raise ValueError("Data Season mapping not supported yet!")
