"""Shared IO helpers for spatialized-weather counterfactual response maps.

Common loaders and aggregations used by both the daily FWI-regime FI maps and
the per-zone peak-wind ROS maps: resolving the raw data directory from a
generated baseline config, loading ground-truth/burnable-support rasters onto
the prediction grid, and pooling per-pixel responses into coarse blocks for
hotspot location.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.fuel_barrier_geometry import parse_fuel_barrier_info

FUEL_NODATA: int = -32768
FIREZONES_RELATIVE_PATH = "spatial/hex{hex_int:02d}_firezones.tif"


def load_zone_labels(
    raw_data_dir: Path,
    hex_id: str,
    reference_profile: dict,
    *,
    support: np.ndarray | None = None,
) -> np.ma.MaskedArray:
    """Firezone label raster aligned to the prediction grid.

    The firezone raster is valid across the full hex tile, which extends beyond
    the sharp hexagon of pixels that the model actually predicts.  Pass the
    displayed map's boolean ``support`` mask to clip the labels to it so that the
    boundary overlay never traces borders outside the predicted hexagon.
    """
    firezones_path = raw_data_dir / f"hex{int(hex_id):02d}" / FIREZONES_RELATIVE_PATH.format(hex_int=int(hex_id))
    zones, _ = load_spatial_raster(path=firezones_path, reference_profile=reference_profile)
    if support is not None:
        zones = np.ma.masked_where(~support, zones)
    return zones


def extent_km(extent_m: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Recentre a metre extent on its lower-left origin and convert to kilometres."""
    left, right, bottom, top = extent_m
    return (0.0, (right - left) / 1000.0, 0.0, (top - bottom) / 1000.0)


def raw_data_dir_from_config(experiment_dir: Path, *, endpoint: str) -> Path:
    config_path = experiment_dir / "generated_configs" / f"baseline_{endpoint}.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return Path(config["data"]["raw_data_dir"])


def load_ground_truth(
    raw_data_dir: Path,
    hex_id: str,
    reference_profile: dict,
    *,
    gt_relative_path: str,
) -> np.ma.MaskedArray:
    gt_path = raw_data_dir / f"hex{int(hex_id):02d}" / gt_relative_path
    gt, _ = load_spatial_raster(path=gt_path, reference_profile=reference_profile)
    return gt


def load_burnable_support(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ndarray:
    """Boolean mask of burnable pixels (non-fuel/water excluded) on the prediction grid."""
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_ma, _ = load_spatial_raster(path=paths.fuel_grid(hex_id), reference_profile=reference_profile)
    fuel_values = np.ma.asarray(fuel_ma).filled(FUEL_NODATA).astype(np.int32)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    return ~np.isin(fuel_values, fuel_info.nonfuel_ids)


def block_response(delta: np.ma.MaskedArray, block: int) -> np.ndarray:
    """Mean absolute response pooled into ``block``x``block`` cells over valid pixels."""
    abs_delta = np.abs(np.ma.filled(delta, 0.0))
    valid = (~np.ma.getmaskarray(delta)).astype(np.float64)
    rows = (abs_delta.shape[0] // block) * block
    cols = (abs_delta.shape[1] // block) * block
    summed = abs_delta[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    counts = valid[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    return np.where(counts > 0, summed / np.maximum(counts, 1.0), 0.0)
