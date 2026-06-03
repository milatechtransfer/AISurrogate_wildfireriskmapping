import os
from pathlib import Path
from typing import Any

import numpy as np

from data_preparation.paths import Paths
from data_preparation.spatial.utils import fire_cause_mapping, load_spatial_raster


def load_ignition_grid(
    root_dir: str,
    hex_id: str,
    cause: int = None,
    season: int = None,
    reference_profile: dict[str, Any] | None = None,
    mask_path: Path = None,
) -> np.ma.MaskedArray:
    """Load ignition grids for a specific season/cause or all seasons/causes"""
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ignition_grids_folder_path = all_paths.ignition_prob_dir()

    if not mask_path:
        mask_path = all_paths.mask_grid_actual(hex_id=hex_id)

    if season and cause and hex_id:
        file_name = f"hex{hex_id}_ignGrid_{fire_cause_mapping[cause]}_s{season}.tif"
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            mask_path=mask_path,
            reference_profile=reference_profile,
        )

        return ignition_raster

    # else, loop over all ignition grids (across causes/seasons) and output one grid
    ignition_raster_files = [f for f in os.listdir(ignition_grids_folder_path) if f.endswith(".tif")]
    out_ignition_grids = []
    for file_name in ignition_raster_files:
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            mask_path=mask_path,
            reference_profile=reference_profile,
        )
        out_ignition_grids.append(ignition_raster)

    # size (H, W)
    out_ignition_grids = np.ma.stack(out_ignition_grids, axis=0)  # type: ignore
    max_ignition_grid = np.ma.max(out_ignition_grids, axis=0)
    return max_ignition_grid
