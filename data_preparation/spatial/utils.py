import logging
import os
from collections.abc import Collection
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

logger = logging.getLogger(__name__)

# Canonical filename for the precomputed normalization stats JSON artifact.
# Written by compute_dataset_normalization_stats; read by all cached range/stat functions.
NORM_STATS_JSON = "dataset_norm_stats.json"

# value for nodata in the rasters
NODATA = np.nan

# mapping cause to cause index
fire_cause_mapping = {1: "H", 2: "N"}

# cause letter to the label used in IgnitionDistribution.csv
fire_cause_label_mapping = {"H": "Human", "N": "Lightning"}

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


def assert_raster_grids_match(paths: Collection[Path]) -> None:
    """Require rasters to share one exact CRS, affine grid, and shape."""
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("At least one raster path is required.")

    with rasterio.open(resolved[0]) as src:
        reference = (src.crs, src.transform, src.width, src.height)

    mismatches = []
    for path in resolved[1:]:
        with rasterio.open(path) as src:
            candidate = (src.crs, src.transform, src.width, src.height)
        if candidate != reference:
            mismatches.append(path)

    if mismatches:
        raise ValueError(
            f"Native-grid preparation requires exact raster alignment with {resolved[0]}; " f"mismatched rasters: {mismatches}"
        )


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

    dst_masked = np.ma.masked_equal(dst, src_nodata) if src_nodata is not None else np.ma.masked_array(dst)

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


def read_split_hex_ids(split_csv_path: str | Path, hex_id_col: str = "hex_id") -> set[int]:
    """Return the set of integer hex ids referenced by a split index CSV."""
    path = Path(split_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Split index CSV not found: {path}")
    df = pd.read_csv(path)
    if hex_id_col not in df.columns:
        raise KeyError(f"Column {hex_id_col!r} not found in split CSV {path}. Available columns: {list(df.columns)}")
    hex_ids = {int(value) for value in df[hex_id_col].dropna()}
    if not hex_ids:
        raise ValueError(f"Split CSV {path} contains no hex ids in column {hex_id_col!r}.")
    return hex_ids


def _restrict_hex_ids(all_hex_ids: list, allowed_hex_ids: Collection[int] | None) -> list:
    """Filter raw hex-id directory names to those whose integer id is in allowed_hex_ids."""
    if allowed_hex_ids is None:
        return all_hex_ids
    allowed = {int(value) for value in allowed_hex_ids}
    return [hex_id for hex_id in all_hex_ids if int(hex_id) in allowed]


def get_range_elevation(root_dir: str, allowed_hex_ids: Collection[int] | None = None) -> tuple[float, float]:
    """Get the global range of elevation for normalization.

    When allowed_hex_ids is provided, only those hexes contribute to the range so the
    statistic is derived from the training split rather than the full (leak-prone) dataset.
    """
    all_hex_ids = _restrict_hex_ids(find_hex_ids(root_dir), allowed_hex_ids)
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


def get_range_elevation_cached(
    root_dir: str,
    allowed_hex_ids: Collection[int] | None = None,
    raw_data_dir: str | None = None,
) -> tuple[float, float]:
    """Return (max, min) for elevation, reading from ``dataset_norm_stats.json`` if available.

    Falls back to scanning raw rasters via ``get_range_elevation``.  ``raw_data_dir`` is the
    raster tree location used for the fallback scan; defaults to ``root_dir`` if not provided.
    """
    import json as _json

    cache_path = os.path.join(root_dir, NORM_STATS_JSON)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = _json.load(f)
        entry = cached.get("elevation", {})
        min_val = entry.get("min")
        max_val = entry.get("max")
        if min_val is not None and max_val is not None:
            logger.debug("Elevation range loaded from %s: min=%.4f, max=%.4f", cache_path, min_val, max_val)
            return float(max_val), float(min_val)
    logger.warning("Elevation range not found in cache — scanning raw rasters (allowed_hex_ids=%s).", allowed_hex_ids)
    return get_range_elevation(raw_data_dir if raw_data_dir is not None else root_dir, allowed_hex_ids)


def get_range_output(root_dir: str, output_type: str, allowed_hex_ids: Collection[int] | None = None) -> tuple[float, float]:
    """
    Get global max and min for one output type across all valid hexels.
    Usage:
    get_output_range(root_dir, "fire_intensity")
    get_output_range(root_dir, "fire_ros")
    get_output_range(root_dir, "fire_burn_probability")

    When allowed_hex_ids is provided, only those hexes contribute so the range is derived
    from the training split rather than the full (leak-prone) dataset.
    """
    path_methods = {
        "fire_intensity": "output_fire_intensity",
        "fire_ros": "output_ros",
        "fire_burn_probability": "output_burn_prob",
    }

    if output_type not in path_methods:
        raise ValueError(f"Unsupported output_type: {output_type}")

    all_hex_ids = _restrict_hex_ids(find_hex_ids(root_dir), allowed_hex_ids)
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


def get_output_log_stats_cached(
    root_dir: str,
    output_type: str,
    allowed_hex_ids: Collection[int] | None = None,
    raw_data_dir: str | None = None,
) -> tuple[float, float]:
    """
    Return log1p mean/std for a target, reading from a cached JSON file if available.
    Falls back to scanning raw rasters via get_output_log_stats.

    The cached JSON (``dataset_norm_stats.json``) is the canonical, train-only artifact
    produced by ``compute_dataset_normalization_stats`` / ``write_dataset_norm_stats``. The fallback scan honours allowed_hex_ids
    (and an explicit raw_data_dir holding the per-hex rasters) so it stays train-only too.
    """
    import json as _json

    cache_path = os.path.join(root_dir, NORM_STATS_JSON)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = _json.load(f)
        entry = cached.get(output_type, {})
        mean = entry.get("log_mean")
        std = entry.get("log_std")
        if mean is not None and std is not None:
            logger.debug("Log stats for %r loaded from %s: mean=%.4f, std=%.4f", output_type, cache_path, mean, std)
            return float(mean), float(std)
    logger.warning("Log stats for %r not found in cache — scanning raw rasters (allowed_hex_ids=%s).", output_type, allowed_hex_ids)
    return get_output_log_stats(raw_data_dir or root_dir, output_type, allowed_hex_ids)


def get_range_output_cached(
    root_dir: str,
    output_type: str,
    allowed_hex_ids: Collection[int] | None = None,
    raw_data_dir: str | None = None,
) -> tuple[float, float]:
    """Return (max, min) for a target, reading from ``dataset_norm_stats.json`` if available.

    Falls back to scanning raw rasters via ``get_range_output``.  ``raw_data_dir`` is the
    raster tree location used for the fallback scan; defaults to ``root_dir`` if not provided.
    The cached JSON is produced by ``compute_dataset_norm_stats`` and stores ``min``/``max``
    for ``fire_burn_probability``.
    """
    import json as _json

    cache_path = os.path.join(root_dir, NORM_STATS_JSON)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = _json.load(f)
        entry = cached.get(output_type, {})
        min_val = entry.get("min")
        max_val = entry.get("max")
        if min_val is not None and max_val is not None:
            logger.debug("Range for %r loaded from %s: min=%.4f, max=%.4f", output_type, cache_path, min_val, max_val)
            return float(max_val), float(min_val)
    logger.warning("Range for %r not found in cache — scanning raw rasters (allowed_hex_ids=%s).", output_type, allowed_hex_ids)
    return get_range_output(raw_data_dir if raw_data_dir is not None else root_dir, output_type, allowed_hex_ids)


def get_output_log_stats(
    root_dir: str,
    output_type: str,
    allowed_hex_ids: Collection[int] | None = None,
) -> tuple[float, float]:
    """
    Get global mean/std of log1p target values for one output type across all valid hexels.

    When allowed_hex_ids is provided, only those hexes contribute so the statistic is
    derived from the training split rather than the full (leak-prone) dataset.
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

    for hex_id in _restrict_hex_ids(find_hex_ids(root_dir), allowed_hex_ids):
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


def write_dataset_norm_stats(
    *,
    raw_data_dir: str | Path,
    root_dir: str | Path | None = None,
    output_path: str | Path,
    types: Collection[str],
    allowed_hex_ids: Collection[int],
) -> dict[str, dict[str, float]]:
    """Compute train-only normalization stats for each requested type and persist to JSON.

    Supported types and their computed stats:

    - ``elevation``: min/max from DEM rasters (for min-max norm).
    - ``fire_burn_probability``: min/max from BP rasters (for min-max norm).
    - ``fire_intensity``, ``fire_ros``: log1p mean/std (for log-standard norm).
    - ``fuel_curve_iROS``, ``fuel_curve_HFI``: log1p mean/std of fuel curve vectors.
      Requires ``root_dir`` (prepared patch dataset with fuel curve CSV and ignition tables).

    All stats are derived solely from ``allowed_hex_ids`` (the training split) so held-out
    hexes never leak into the normalization constants used by training, evaluation, and
    inference.

    Output JSON structure::

        {
          "elevation":             {"min": ..., "max": ...},
          "fire_burn_probability": {"min": ..., "max": ...},
          "fire_intensity":        {"log_mean": ..., "log_std": ...},
          "fire_ros":              {"log_mean": ..., "log_std": ...},
          "fuel_curve_iROS":       {"log_mean": ..., "log_std": ...},
          "fuel_curve_HFI":        {"log_mean": ..., "log_std": ...}
        }
    """
    import json as _json

    _LOG_STAT_TYPES = {"fire_intensity", "fire_ros"}
    _MIN_MAX_OUTPUT_TYPES = {"fire_burn_probability"}
    _FUEL_CURVE_PREFIX = "fuel_curve_"

    stats: dict[str, dict[str, float]] = {}

    for type_name in types:
        entry: dict[str, float] = {}
        try:
            if type_name == "elevation":
                max_val, min_val = get_range_elevation(str(raw_data_dir), allowed_hex_ids)
                entry = {"min": min_val, "max": max_val}

            elif type_name in _MIN_MAX_OUTPUT_TYPES:
                max_val, min_val = get_range_output(str(raw_data_dir), type_name, allowed_hex_ids)
                entry = {"min": min_val, "max": max_val}

            elif type_name in _LOG_STAT_TYPES:
                mean, std = get_output_log_stats(str(raw_data_dir), type_name, allowed_hex_ids)
                entry = {"log_mean": mean, "log_std": std}

            elif type_name.startswith(_FUEL_CURVE_PREFIX):
                feature_name = type_name[len(_FUEL_CURVE_PREFIX) :]
                if root_dir is None:
                    logger.warning("Skipping %r: --root_dir is required for fuel curve stats.", type_name)
                    continue
                from src.datasets.fuel_utils import compute_fuel_curve_norm_stats

                mean, std = compute_fuel_curve_norm_stats(
                    root_dir=root_dir,
                    raw_data_dir=raw_data_dir,
                    feature_name=feature_name,
                    allowed_hex_ids=set(allowed_hex_ids) if allowed_hex_ids is not None else None,
                )
                entry = {"log_mean": mean, "log_std": std}

            else:
                logger.warning("Unknown type %r — skipping (not in elevation, min-max, log-stat, or fuel-curve sets).", type_name)
                continue

        except ValueError as exc:
            logger.warning("Could not compute stats for %r: %s", type_name, exc)
            continue

        stats[type_name] = entry
        logger.info("Stats for %r: %s", type_name, entry)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        _json.dump(stats, handle, indent=2)
    return stats


def denormalize_burn_count(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Reverse the count normalization to recover true counts."""
    data = data.astype("float32")
    return data * (max_val - min_val) + min_val
