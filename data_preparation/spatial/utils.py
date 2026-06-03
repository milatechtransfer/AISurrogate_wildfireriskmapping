import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

from data_preparation.paths import Paths
from data_preparation.utils import find_hex_ids

# value for nodata in the rasters
NODATA = np.nan

# mapping cause to cause index
fire_cause_mapping = {1: "H", 2: "N"}

# TODO: revisit fuel grouping
FUEL_GROUP_MAP = {
    # Non-fuel
    **{k: 0 for k in [100, 101, 102, 105, 106, 110]},
    # Conifer
    1: 1,  # Spruce-Lichen Woodland
    2: 2,  # Boreal Spruce
    3: 3,  # Mature Jack or Lodgepole Pine
    4: 4,  # Immature Jack or Lodgepole Pine
    5: 5,  # Red and White Pine
    6: 6,  # Conifer Plantation
    7: 7,  # Ponderosa Pine - Douglas-Fir
    # Aspen
    **{k: 8 for k in [11, 12, 13]},
    # Slash
    **{k: 9 for k in [21, 22, 23]},
    # Grass
    **{k: 10 for k in [31, 32]},
    # Boreal Mixedwood
    **{k: 11 for k in [40, 50, 60]},
    **{k: 12 for k in range(405, 500, 5)},
    **{k: 13 for k in range(505, 600, 5)},
    **{k: 14 for k in range(605, 700, 5)},
    # Dead Balsam Fir Mixedwood
    **{k: 15 for k in [70, 80, 90]},
    **{k: 16 for k in range(705, 800, 5)},
    **{k: 17 for k in range(805, 900, 5)},
    **{k: 18 for k in range(905, 1000, 5)},
}


def load_csv(path: str) -> pd.DataFrame:
    """Load csv file from given path"""
    if os.path.exists(path):  # noqa: F821
        df = pd.read_csv(path)
        print("File loaded successfully.")
        return df
    else:
        raise FileNotFoundError(f"File not found: {path}")


def load_raster(path: str) -> np.ma.MaskedArray:
    """Load raster from given path"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        raster = src.read(1, masked=True)  # mask out the nodata
        return raster


def load_spatial_raster(
    path: Path,
    reproject_flag: bool = True,
    mask_path: Path | None = None,
    reference_profile: dict[str, Any] | None = None,
) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
    """Load one raster band, optionally reproject/clip/crop it, and return updated profile."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        raster = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        profile = src.profile.copy()

    if reproject_flag:
        raster, transform, profile = reproject_raster(
            raster=raster,
            src_transform=transform,
            src_crs=crs,
            src_nodata=nodata,
            profile=profile,
            dst_crs=reference_profile["crs"] if reference_profile is not None else "ESRI:102002",
            dst_transform=reference_profile["transform"] if reference_profile is not None else None,
            dst_width=reference_profile["width"] if reference_profile is not None else None,
            dst_height=reference_profile["height"] if reference_profile is not None else None,
        )
        crs = profile["crs"]

    if mask_path:
        raster, transform, profile = clip_array_to_mask(
            raster=raster,
            transform=transform,
            profile=profile,
            mask_path=mask_path,
            crs=crs,
            nodata=nodata,
        )

    return raster, profile


def clip_array_to_mask(
    raster: np.ma.MaskedArray,
    transform: Affine,
    profile: dict[str, Any],
    crs: Any,
    mask_path: Path,
    crop: bool = True,
    filled: bool = False,
    nodata: float | int | None = None,
) -> tuple[np.ma.MaskedArray, Affine, dict[str, Any]]:
    """Clip a loaded 2D raster array to a polygon mask in the raster CRS."""
    if raster.ndim != 2:
        raise ValueError("Input raster must be 2D.")

    mask_gdf = gpd.read_file(mask_path).to_crs(crs)

    if nodata is None:
        nodata = profile.get("nodata")

    data_to_write = raster.filled(nodata) if np.ma.isMaskedArray(raster) and nodata is not None else np.asarray(raster)

    meta = profile.copy()
    meta.update(
        {
            "driver": meta.get("driver", "GTiff"),
            "height": raster.shape[0],
            "width": raster.shape[1],
            "count": 1,
            "dtype": data_to_write.dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
        }
    )

    with MemoryFile() as memfile, memfile.open(**meta) as ds:
        ds.write(data_to_write, 1)

        out_image, out_transform = mask(
            ds,
            mask_gdf.geometry,
            crop=crop,
            filled=filled,
        )

        out_profile = ds.profile.copy()
        out_profile.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": crs,
                "nodata": nodata,
                "count": 1,
            }
        )

    clipped = out_image[0]

    if filled:
        clipped = np.ma.masked_equal(clipped, nodata) if nodata is not None else np.ma.masked_array(clipped)
    else:
        if not np.ma.isMaskedArray(clipped):
            clipped = np.ma.masked_array(clipped)

    return clipped, out_transform, out_profile


def crop_masked_raster(
    raster: np.ma.MaskedArray,
    transform: Affine,
    profile: dict[str, Any],
) -> tuple[np.ma.MaskedArray, Affine, dict[str, Any]]:
    """Crop fully masked borders and update transform and profile."""
    if not np.ma.isMaskedArray(raster):
        raise ValueError("Input raster must be a masked array.")

    valid = ~np.ma.getmaskarray(raster)

    if not np.any(valid):
        raise ValueError("Raster contains no valid pixels.")

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]

    row_min, row_max = rows[0], rows[-1]
    col_min, col_max = cols[0], cols[-1]

    cropped = raster[row_min : row_max + 1, col_min : col_max + 1]
    new_transform = transform * Affine.translation(col_min, row_min)

    out_profile = profile.copy()
    out_profile.update(
        {
            "height": cropped.shape[0],
            "width": cropped.shape[1],
            "transform": new_transform,
            "count": 1,
        }
    )

    return cropped, new_transform, out_profile


def reproject_raster(
    raster: np.ma.MaskedArray,
    src_transform: Affine,
    src_crs: Any,
    src_nodata: float | int | None,
    profile: dict[str, Any],
    dst_crs: str = "ESRI:102002",
    resampling: Resampling = Resampling.nearest,
    dst_transform: Affine | None = None,
    dst_width: int | None = None,
    dst_height: int | None = None,
) -> tuple[np.ma.MaskedArray, Affine, dict[str, Any]]:
    """Reproject an already loaded raster while preserving mask/nodata and updating profile."""
    height, width = raster.shape

    src_filled = raster.filled(src_nodata) if src_nodata is not None else raster.filled()

    has_reference_grid = dst_transform is not None and dst_width is not None and dst_height is not None

    if not has_reference_grid:
        if dst_transform is not None or dst_width is not None or dst_height is not None:
            raise ValueError("dst_transform, dst_width, and dst_height must be provided together.")

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            width,
            height,
            *rasterio.transform.array_bounds(height, width, src_transform),
        )

    if dst_transform is None or dst_width is None or dst_height is None:
        raise ValueError("dst_transform, dst_width, and dst_height must be resolved before reprojection.")

    if src_nodata is not None:
        dst = np.full((dst_height, dst_width), src_nodata, dtype=src_filled.dtype)
    else:
        dst = np.empty((dst_height, dst_width), dtype=src_filled.dtype)

    reproject(
        source=src_filled,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=src_nodata,
        resampling=resampling,
    )

    if src_nodata is not None:
        if isinstance(src_nodata, float) and np.isnan(src_nodata):
            # masked_equal uses == which returns False for NaN; use isnan instead.
            dst_masked = np.ma.masked_where(np.isnan(dst), dst)
        else:
            dst_masked = np.ma.masked_equal(dst, src_nodata)
    else:
        dst_masked = np.ma.masked_array(dst)

    out_profile = profile.copy()
    out_profile.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "count": 1,
            "nodata": src_nodata,
            "dtype": dst.dtype,
        }
    )

    return dst_masked, dst_transform, out_profile


def get_range_elevation(root_dir: str) -> tuple[float, float]:
    """Get the global range of elevation for normalization"""
    all_hex_ids = find_hex_ids(root_dir)
    min_value, max_value = np.inf, -np.inf
    for hex_id in all_hex_ids:
        paths = Paths(hex_id=hex_id, root_dir=root_dir)
        path_elev_grid = paths.elevation_grid(hex_id=hex_id)
        elevation_grid = load_raster(str(path_elev_grid))
        masked_data = np.ma.masked_invalid(np.ma.masked_equal(elevation_grid, -9999))
        values = masked_data.compressed()
        if values.size == 0:
            continue
        max_value = max(max_value, float(np.max(values)))
        min_value = min(min_value, float(np.min(values)))

    if not np.isfinite(max_value) or not np.isfinite(min_value) or max_value <= min_value:
        raise ValueError(
            f"Invalid elevation normalization range from root_dir={root_dir!r}: min={min_value}, max={max_value}. "
            "Check that root_dir points to the raw hexel dataset, not only the prepared patch directory."
        )
    return float(max_value), float(min_value)


def get_range_output(root_dir: str, output_type: str) -> tuple[float, float]:
    """
    Get global max and min for one output type across all valid hexelss.
    Usage:
    get_output_range(root_dir, "fire_intensity")
    get_output_range(root_dir, "fire_ros")
    get_output_range(root_dir, "fire_burn_probability")
    """
    path_methods = {
        "fire_intensity": "output_fire_intensity",
        "fire_ros": "output_ros",
        "fire_burn_probability": "output_burn_prob",
    }

    if output_type not in path_methods:
        raise ValueError(f"Unsupported output_type: {output_type}")

    all_hex_ids = find_hex_ids(root_dir)
    min_value, max_value = np.inf, -np.inf

    for hex_id in all_hex_ids:
        paths = Paths(hex_id=hex_id, root_dir=root_dir)
        output_path = getattr(paths, path_methods[output_type])()
        output_grid = load_raster(str(output_path))
        output_values = np.ma.masked_invalid(output_grid).compressed()
        if output_values.size == 0:
            continue

        max_value = max(max_value, float(np.max(output_values)))
        min_value = min(min_value, float(np.min(output_values)))

    if not np.isfinite(max_value) or not np.isfinite(min_value) or max_value <= min_value:
        raise ValueError(
            f"Invalid {output_type} normalization range from root_dir={root_dir!r}: "
            f"min={min_value}, max={max_value}. Check that root_dir points to the raw hexel dataset, "
            "not only the prepared patch directory."
        )
    return max_value, min_value


def get_output_log_stats(root_dir: str, output_type: str) -> tuple[float, float]:
    """
    Get global mean/std of log1p target values for one output type across all valid hexels.
    """
    path_methods = {
        "fire_intensity": "output_fire_intensity",
        "fire_ros": "output_ros",
        "fire_burn_probability": "output_burn_prob",
    }

    if output_type not in path_methods:
        raise ValueError(f"Unsupported output_type: {output_type}")

    count = 0
    total = 0.0
    total_sq = 0.0

    for hex_id in find_hex_ids(root_dir):
        paths = Paths(hex_id=hex_id, root_dir=root_dir)
        output_path = getattr(paths, path_methods[output_type])()
        output_grid = load_raster(str(output_path))
        output_values = np.ma.masked_invalid(output_grid).compressed().astype(np.float64, copy=False)
        if output_values.size == 0:
            continue
        log_values = np.log1p(np.clip(output_values, a_min=0.0, a_max=None))
        count += int(log_values.size)
        total += float(log_values.sum())
        total_sq += float(np.square(log_values).sum())

    if count == 0:
        raise ValueError(f"No valid target pixels found for output_type={output_type!r} in root_dir={root_dir!r}.")

    mean = total / count
    variance = max((total_sq / count) - mean**2, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError(f"Invalid log-standard normalization std for output_type={output_type!r} in root_dir={root_dir!r}: {std}.")
    return float(mean), std


def denormalize_burn_count(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Reverse the count normalization to recover true counts."""
    data = data.astype("float32")
    return data * (max_val - min_val) + min_val
