from __future__ import annotations

import json
import os
from itertools import product

import numpy as np

from data_preparation.paths import Paths
from data_preparation.spatial import NODATA, load_fuel_grid, load_ignition_grid, load_spatial_raster
from data_preparation.utils import feature_names, read_ids_from_csv


def get_num_channels_array(arr: np.ndarray) -> int:
    """Returns the number of channels a particular feature will take"""
    if len(arr.shape) == 2:
        return 1
    return arr.shape[-1]


def generate_feature_channel_map(feature_list: list[np.ndarray], feature_channel_map_path: str):
    """Maps the feature names to the corresponding channels in our input stack"""
    feature_channel_map = dict()
    channel = 0
    for i, feature in enumerate(feature_list):
        feature_channels = get_num_channels_array(feature)
        feature_channel_map[feature_names[i]] = list(range(channel, channel + feature_channels))
        channel += feature_channels
    os.makedirs(os.path.dirname(feature_channel_map_path), exist_ok=True)
    with open(feature_channel_map_path, "w") as f:
        json.dump(feature_channel_map, f, indent=4)


def load_spatial_features_per_hexel(
    root_dir: str,
    hex_id: str,
    feature_channel_map_path: str,
    modelling_approach: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, tuple[int, int | str]] | None]:
    """
    Load all data (features and output) per hexel
    root_dir: Root directory containing all hexels.
    hex_id: Hexel id to load.
    modelling_approach: 1 for joint season-cause modelling, 2 for separate season modelling.
    Returns:
        all_features: np.ndarray of shape (N, H, W, num_features)
        all_masks: np.ndarray of shape (N, H, W)
        season_cause_mapping: dict mapping index to seasons
    """

    def stack_sample(
        fuel_grid: np.ma.MaskedArray,
        elevation_grid: np.ma.MaskedArray,
        ignition_grid: np.ma.MaskedArray,
        firezones_grid: np.ma.MaskedArray,
        bp_out_grid: np.ma.MaskedArray,
        fi_out_grid: np.ma.MaskedArray,
        ros_out_grid: np.ma.MaskedArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack all features and compute mask."""
        features_list = [
            fuel_grid[:, :, np.newaxis],
            elevation_grid[:, :, np.newaxis],
            ignition_grid[:, :, np.newaxis],
            firezones_grid[:, :, np.newaxis],
            bp_out_grid[:, :, np.newaxis],
            fi_out_grid[:, :, np.newaxis],
            ros_out_grid[:, :, np.newaxis],
        ]

        if not os.path.exists(feature_channel_map_path):
            generate_feature_channel_map(features_list, feature_channel_map_path)

        stacked_ma = np.ma.concatenate(features_list, axis=-1)
        mask = np.logical_or.reduce(
            [
                np.ma.getmaskarray(fuel_grid),
                np.ma.getmaskarray(elevation_grid),
                np.ma.getmaskarray(ignition_grid),
                np.ma.getmaskarray(firezones_grid),
                np.ma.getmaskarray(bp_out_grid),
                np.ma.getmaskarray(fi_out_grid),
                np.ma.getmaskarray(ros_out_grid),
            ]
        )

        fuel_mask = np.ma.getmaskarray(fuel_grid)
        elevation_mask = np.ma.getmaskarray(elevation_grid)
        ignition_mask = np.ma.getmaskarray(ignition_grid)
        firezones_mask = np.ma.getmaskarray(firezones_grid)
        bp_out_mask = np.ma.getmaskarray(bp_out_grid)
        fi_out_mask = np.ma.getmaskarray(fi_out_grid)
        ros_out_mask = np.ma.getmaskarray(ros_out_grid)

        assert np.array_equal(mask, fuel_mask | elevation_mask | ignition_mask | firezones_mask | bp_out_mask | fi_out_mask | ros_out_mask)

        stacked = stacked_ma.filled(NODATA).astype(np.float32)
        stacked[mask] = NODATA
        return stacked, mask

    # identify all seasons and causes first
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)

    elevation_grid, reference_profile = load_spatial_raster(
        path=all_paths.elevation_grid(hex_id=hex_id), actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id)
    )
    # load all common grids on the elevation reference grid
    fuel_grid = load_fuel_grid(root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile)

    firezones_grid, _ = load_spatial_raster(
        path=all_paths.firezones_grid(hex_id=hex_id),
        actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
        reference_profile=reference_profile,
    )

    if modelling_approach == 1:
        # input
        ignition_grid = load_ignition_grid(root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile)

        bp_out_grid, _ = load_spatial_raster(
            all_paths.output_burn_prob(),
            actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
            reference_profile=reference_profile,
        )
        fi_out_grid, _ = load_spatial_raster(
            all_paths.output_fire_intensity(),
            actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
            reference_profile=reference_profile,
        )
        ros_out_grid, _ = load_spatial_raster(
            all_paths.output_ros(),
            actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
            reference_profile=reference_profile,
        )

        stacked_features, mask = stack_sample(
            fuel_grid, elevation_grid, ignition_grid, firezones_grid, bp_out_grid, fi_out_grid, ros_out_grid
        )
        return np.expand_dims(stacked_features, axis=0), np.expand_dims(mask, axis=0), None

    # modelling approach 2 - seasonal mapping
    elif modelling_approach == 2:
        seasons = read_ids_from_csv(all_paths.season_table(hex_id=hex_id))
        samples_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        season_mapping: dict[int, tuple[int, int | str]] = {}

        for i, season in enumerate(seasons):
            ignition_grid = load_ignition_grid(root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile, season=season)

            bp_out_grid, _ = load_spatial_raster(
                all_paths.output_burn_prob(season=season),
                actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
                reference_profile=reference_profile,
            )
            fi_out_grid, _ = load_spatial_raster(
                all_paths.output_fire_intensity(),
                actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
                reference_profile=reference_profile,
            )
            ros_out_grid, _ = load_spatial_raster(
                all_paths.output_ros(),
                actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id),
                reference_profile=reference_profile,
            )

            stacked_features, mask = stack_sample(
                fuel_grid, elevation_grid, ignition_grid, firezones_grid, bp_out_grid, fi_out_grid, ros_out_grid
            )

            samples_list.append(stacked_features)
            masks_list.append(mask)
            season_mapping[i] = (int(season), "all")

        samples = np.stack(samples_list, axis=0)  # shape: (N, H, W, C)
        masks = np.stack(masks_list, axis=0)  # shape: (N, H, W)

        return samples, masks, season_mapping
    else:
        raise ValueError(f"Unsupported modelling_approach: {modelling_approach}")


# if __name__ == "__main__":
#     s, masks, mapping = load_spatial_features_per_hexel(
#         root_dir="../burnp3plus", hex_id="05", feature_channel_map_path=".", modelling_approach=2
#     )
#     print(s.shape)
#     print(masks.shape)
#     print(mapping)
