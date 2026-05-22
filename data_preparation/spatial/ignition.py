import os
from typing import Any

import numpy as np

from data_preparation.paths import Paths
from data_preparation.spatial.utils import fire_cause_mapping, load_spatial_raster


def load_ignition_grid(
    root_dir: str,
    hex_id: str,
    cause: int | None = None,
    season: int | None = None,
    reference_profile: dict[str, Any] | None = None,
) -> np.ma.MaskedArray:
    """Load ignition grid for a specific season/cause, or aggregate using max."""

    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ignition_grids_folder_path = all_paths.ignition_prob_dir()
    actual_mask_path = all_paths.mask_grid_actual(hex_id=hex_id)

    def _load_one_raster(file_name: str) -> np.ma.MaskedArray:
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            actual_mask_path=actual_mask_path,
            reference_profile=reference_profile,
        )
        return ignition_raster

    def _max_over_files(file_names: list[str]) -> np.ma.MaskedArray:
        if not file_names:
            raise FileNotFoundError(f"No ignition grid files found in {ignition_grids_folder_path}")

        ignition_grids = [_load_one_raster(file_name) for file_name in file_names]
        ignition_grids = np.ma.stack(ignition_grids, axis=0)  # shape: (N, H, W)

        return np.ma.max(ignition_grids, axis=0)

    # Case 1: specific cause and season
    if season is not None and cause is not None:
        file_name = f"hex{hex_id}_ignGrid_{fire_cause_mapping[cause]}_s{season}.tif"
        return _load_one_raster(file_name)

    # Case 2: season only, cause is None
    # Load all causes for the given season, then take max grid.
    if season is not None and cause is None:
        file_names = [f"hex{hex_id}_ignGrid_{cause_name}_s{season}.tif" for cause_name in fire_cause_mapping.values()]

        existing_file_names = [file_name for file_name in file_names if (ignition_grids_folder_path / file_name).exists()]

        return _max_over_files(existing_file_names)

    # Case 3: no specific season/cause
    # Load all ignition grids across all causes/seasons, then take max grid.
    file_names = [file_name for file_name in os.listdir(ignition_grids_folder_path) if file_name.endswith(".tif")]

    return _max_over_files(file_names)
