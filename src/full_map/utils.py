import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.colors import LogNorm, Normalize
from rasterio.warp import Resampling, reproject


def find_hex_files(folder: Path, pattern: str) -> dict[int, Path]:
    """Scans a folder for files matching the pattern."""
    files = list(folder.glob(pattern))
    file_map = {}

    for f in files:
        # option that matches the predictions format
        match = re.search(r"hexel_(\d+)", f.name)

        # option for the original data folder (targets)
        if not match and len(f.parents) >= 2:
            match = re.search(r"hex_?(\d+)", f.parent.parent.name)

        # looks for any other potential matches
        if not match:
            match = re.search(r"hex(?:el)?_?(\d+)", str(f))

        if match:
            file_map[int(match.group(1))] = f
        else:
            print(f"Warning: Could not extract hex ID from path {f}.")

    return file_map


def group_predicted_hexel_files_by_target(folder: Path, pattern: str) -> dict[str, dict[int, Path]]:
    """Scans a folder for predicted hexel rasters and groups them by target name.

    Expects the ``save_predicted_hexels`` naming convention:
    ``hexel_{hex_id}[_{target_name}]_predicted.tif``. Single-target models (no ``target_name``
    suffix) are grouped under the key ``"default"``; multi-target models (e.g. bp/fi/ros) are
    grouped by their respective target name, so each target can be mosaicked/compared
    independently.
    """
    files = list(folder.glob(pattern))
    grouped: dict[str, dict[int, Path]] = {}

    for f in files:
        match = re.search(r"hexel_(?P<hex_id>\d+)(?:_(?P<target>[A-Za-z]+))?_predicted", f.name)
        if not match:
            print(f"Warning: Could not extract hex ID/target from path {f}.")
            continue
        hex_id = int(match.group("hex_id"))
        target_name = match.group("target") or "default"
        grouped.setdefault(target_name, {})[hex_id] = f

    return grouped


def calculate_global_stats(file_map: dict[int, Path]) -> tuple[float, float]:
    """
    Returns (global_min_positive, global_max).
    """
    print(f"Scanning {len(file_map)} files for global statistics...")
    global_max = -np.inf
    global_min_pos = np.inf

    for f in file_map.values():
        try:
            with rasterio.open(f) as src:
                data = src.read(1)

                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)

                if data.count() == 0:
                    continue

                cmax = data.max()
                if cmax > global_max:
                    global_max = cmax

                # For Log scale, find smallest positive non-zero
                valid_pos = data[data > 0]
                if valid_pos.size > 0:
                    cmin_pos = valid_pos.min()
                    if cmin_pos < global_min_pos:
                        global_min_pos = cmin_pos
        except Exception as e:
            print(f"Warning skipping stats for {f}: {e}")

    # fallbacks if inf. values
    if global_max == -np.inf:
        global_max = 1.0
    if global_min_pos == np.inf:
        global_min_pos = 1e-6

    return global_min_pos, global_max


def get_scale_settings(scale: str, pos_min: float, global_max: float) -> Normalize:
    """Configures the normalization based on the scale type."""
    if scale == "log":
        # For log scale, we use positive min. and global max.
        print(f"Using LOG scale: {pos_min:.2e} to {global_max:.2e}")
        norm = LogNorm(vmin=pos_min, vmax=global_max)
    elif scale == "linear":
        # For linear scale, we start at 0 prob/count.
        linear_min = 0
        print(f"Using LINEAR scale: {linear_min} to {global_max:.2e}")
        norm = Normalize(vmin=linear_min, vmax=global_max)  # type: ignore
    else:
        raise ValueError(f"Unknown scale type: '{scale}'. Please use 'log' or 'linear'.")

    return norm


def load_hexel_shapefile(shapefile_path: str, hexel_id_column: str = "hex_id") -> gpd.GeoDataFrame:
    """Loads the national hexel-polygon shapefile used to place each predicted hexel raster
    at its real geographic location. ``hexel_id_column`` is normalized to a plain string dtype
    so it can be matched against hex IDs parsed from predicted-hexel filenames (also strings).
    """
    if not Path(shapefile_path).exists():
        raise FileNotFoundError(f"National hexel shapefile not found: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if hexel_id_column not in gdf.columns:
        raise KeyError(f"Column {hexel_id_column!r} not found in shapefile {shapefile_path} (columns: {list(gdf.columns)}).")
    gdf = gdf.copy()
    gdf[hexel_id_column] = gdf[hexel_id_column].astype(str)
    return gdf


def mosaic_predicted_hexels(
    file_map: dict[int, Path],
    shapefile_gdf: gpd.GeoDataFrame,
    hexel_id_column: str,
    reference_raster_path: str,
) -> tuple[np.ndarray, dict]:
    """Mosaics per-hexel predicted rasters onto the real national grid, for one target.

    Each predicted hexel `.tif` lives in its own local raster grid/transform (from patch
    stitching), not aligned to the national reference raster. For every hexel in ``file_map``
    (already grouped by target, e.g. via ``group_predicted_hexel_files_by_target``):
      1. look up its polygon by hex_id in ``shapefile_gdf`` (used only to sanity-check the hexel
         is a real national hexel; placement itself comes from each raster's own CRS/transform),
      2. reproject that hexel's array onto the reference raster's grid (CRS/transform/shape),
      3. paste the reprojected pixels into the output canvas wherever they are valid (non-nodata),
         so overlapping/adjacent hexels don't overwrite each other's valid pixels.

    Returns ``(mosaic, profile)`` where ``mosaic`` is a 2D array shaped like the reference
    raster and ``profile`` is that reference raster's rasterio profile (with ``count=1``).
    """
    if not file_map:
        raise ValueError("file_map is empty -- nothing to mosaic.")

    known_hex_ids = set(shapefile_gdf[hexel_id_column].astype(int))
    missing_from_shapefile = sorted(set(file_map) - known_hex_ids)
    if missing_from_shapefile:
        print(f"Warning: {len(missing_from_shapefile)} predicted hexel(s) not found in the national shapefile: {missing_from_shapefile}")

    with rasterio.open(reference_raster_path) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_nodata = ref_src.nodata if ref_src.nodata is not None else -9999.0
        mosaic = np.full((ref_src.height, ref_src.width), ref_nodata, dtype="float32")

        for hex_id, hex_path in file_map.items():
            with rasterio.open(hex_path) as hex_src:
                hex_nodata = hex_src.nodata if hex_src.nodata is not None else -9999.0
                reprojected = np.full((ref_src.height, ref_src.width), ref_nodata, dtype="float32")
                reproject(
                    source=rasterio.band(hex_src, 1),
                    destination=reprojected,
                    src_transform=hex_src.transform,
                    src_crs=hex_src.crs,
                    src_nodata=hex_nodata,
                    dst_transform=ref_src.transform,
                    dst_crs=ref_src.crs,
                    dst_nodata=ref_nodata,
                    resampling=Resampling.nearest,
                )
            valid = reprojected != ref_nodata
            mosaic[valid] = reprojected[valid]
            print(f"Pasted hex_id={hex_id} ({hex_path.name}) onto national mosaic ({valid.sum()} valid px).")

    ref_profile.update(count=1, dtype="float32", nodata=ref_nodata)
    return mosaic, ref_profile
