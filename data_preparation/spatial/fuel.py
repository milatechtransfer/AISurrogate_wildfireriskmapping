from typing import Any

import numpy as np

from data_preparation.paths import Paths

# from data_preparation.spatial.ignition import load_ignition_grid, load_ignition_grid_weighted
# from data_preparation.visualizations import visualize_elevation_grid, visualize_fuel_grid
from data_preparation.spatial.utils import FUEL_GROUP_MAP, load_spatial_raster


def load_fuel_grid(
    root_dir: str,
    hex_id: str,
    fuel_representation: str = "raw",
    reference_profile: dict[str, Any] | None = None,
    mask_scope: str | None = None,
    scenario_name: str | None = None,
) -> np.ma.MaskedArray:
    """
    Load an FBP fuel raster and group fuel types if selected
    group_fuels: boolean flag to choose if we can group fuels into 5 distinct groups
    rerank_fuels: boolean flag to choose if we can rerank fuel IDs for ordinal encoding. Fuels are ranked from lowest to highest based on their Rate of Spread.
    """
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    mask_path = all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope) if mask_scope is not None else None
    fuel_grid, _ = load_spatial_raster(
        all_paths.fuel_grid(hex_id=hex_id, scenario_name=scenario_name),
        mask_path=mask_path,
        reference_profile=reference_profile,
    )

    data = fuel_grid.data
    mask = np.ma.getmaskarray(fuel_grid)

    if fuel_representation == "group":
        grouped = np.full(data.shape, -1, dtype=np.int16)

        for fuel_id, group_id in FUEL_GROUP_MAP.items():
            grouped[data == fuel_id] = group_id

        return np.ma.masked_array(grouped, mask=mask)

    return np.ma.masked_array(data, mask=mask)


# if __name__ == "__main__":
#     root_dir = "../NWT_data/fortsimpson_data_Jun2026"
#     hex_id="100"
#     scenario_name="FireSpotting"
#     all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
#     elevation_grid, reference_profile = load_spatial_raster(path=all_paths.elevation_grid(hex_id=hex_id))
#     # visualize_elevation_grid(elevation_grid)
#     arr = load_fuel_grid(root_dir, hex_id=hex_id, scenario_name="FireSpotting", reference_profile=reference_profile)
#     # visualize_fuel_grid(arr)
#     # bp_out_grid, _ = load_spatial_raster(
#     #     all_paths.output_ros(scenario_name=scenario_name),
#     #     reference_profile=reference_profile,
#     # )
#     # visualize_elevation_grid(bp_out_grid)
#     print(arr.shape)
#     firezones_grid, _ = load_spatial_raster(
#         path=all_paths.firezones_grid(hex_id=hex_id),
#         reference_profile=reference_profile,
#     )
#     ignition_grid = load_ignition_grid_weighted(
#         root_dir=root_dir,
#         hex_id=hex_id,
#         firezones_grid=firezones_grid,
#         reference_profile=reference_profile,
#     )
#     print(ignition_grid.shape)
#     visualize_elevation_grid(ignition_grid[:,:,0:1])
#     visualize_elevation_grid(ignition_grid[:, :, 1:2])
