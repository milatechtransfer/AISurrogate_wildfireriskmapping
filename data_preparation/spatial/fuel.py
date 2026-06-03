from pathlib import Path
from typing import Any

import numpy as np

from data_preparation.paths import Paths
from data_preparation.spatial.utils import FUEL_GROUP_MAP, load_spatial_raster


def load_fuel_grid(
    root_dir: str, hex_id: str, reference_profile: dict[str, Any] | None = None, mask_path: Path = None
) -> np.ma.MaskedArray:
    """
    Load an FBP fuel raster and group fuel types if selected
    group_fuels: boolean flag to choose if we can group fuels into 5 distinct groups
    rerank_fuels: boolean flag to choose if we can rerank fuel IDs for ordinal encoding. Fuels are ranked from lowest to highest based on their Rate of Spread.
    """
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    if not mask_path:
        mask_path = all_paths.mask_grid_actual(hex_id=hex_id)

    fuel_grid, _ = load_spatial_raster(
        all_paths.fuel_grid(hex_id=hex_id),
        mask_path=mask_path,
        reference_profile=reference_profile,
    )

    data = fuel_grid.data
    mask = np.ma.getmaskarray(fuel_grid)

    grouped = np.full(data.shape, -1, dtype=np.int16)

    for fuel_id, group_id in FUEL_GROUP_MAP.items():
        grouped[data == fuel_id] = group_id

    return np.ma.masked_array(grouped, mask=mask)
