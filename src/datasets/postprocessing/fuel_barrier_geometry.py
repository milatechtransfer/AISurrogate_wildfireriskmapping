"""Shared fuel-barrier geometry utilities for counterfactual figures.

Pixel-grid helpers for reasoning about non-fuel spread barriers: parsing
per-hexel barrier IDs, distance-to-barrier fields, distance-bin labels,
directional sectors relative to a dominant wind, and per-zone wind consistency.

Distance convention: all distances are pixel-centre-to-pixel-centre (EDT), so
the immediately adjacent burnable pixel is ~100 m from a barrier on a 100 m
grid.

Wind direction convention: BurnP3+/NRCan WindDirection is a meteorological
from-bearing.  Physical downwind flow is therefore WindDirection + 180 degrees,
or equivalently (-wind_x, -wind_y) relative to the raw component encoding used
by the model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from data_preparation.paths import Paths
from src.datasets.postprocessing.utils import bp_nonfuel_restricted_ids

FUEL_NODATA: int = -32768
ZONE_NODATA: int = -128

WIND_DIRECTION_CONVENTION_UNKNOWN: str = (
    "WindDirection is treated as a from-bearing; physical downwind flow is " "WindDirection + 180 degrees."
)

SECTOR_LABELS: tuple[str, ...] = (
    "downwind_of_barrier",
    "crosswind_right",
    "upwind_of_barrier",
    "crosswind_left",
)
# Half-angle for each 90-degree sector.
_SECTOR_HW_RAD: float = np.pi / 4

DEFAULT_DIST_BIN_EDGES_M: tuple[float, ...] = (
    100.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
)


@dataclass
class FuelBarrierInfo:
    """Per-hexel fuel barrier and ignition-restriction IDs."""

    hex_id: str
    # IDs whose FuelTypes.csv Description == "Non-fuel" — true spread barriers.
    nonfuel_ids: list[int]
    # IDs that are ignition-restricted but NOT non-fuel (can still burn).
    restricted_ids: list[int]
    restricted_names: list[str]


def parse_fuel_barrier_info(paths: Paths, hex_id: str) -> FuelBarrierInfo:
    """Return non-fuel and ignition-restricted IDs separately for one hexel."""
    _, nonfuel_ids, restricted_names = bp_nonfuel_restricted_ids(paths, hex_id)

    fuel_table = pd.read_csv(paths.fuel_table(hex_id))
    name_to_id = {str(n): int(i) for n, i in zip(fuel_table["Name"], fuel_table["ID"], strict=False)}
    nonfuel_set = set(nonfuel_ids)
    restricted_ids = sorted({name_to_id[name] for name in restricted_names if name in name_to_id and name_to_id[name] not in nonfuel_set})
    return FuelBarrierInfo(
        hex_id=hex_id,
        nonfuel_ids=sorted(nonfuel_ids),
        restricted_ids=restricted_ids,
        restricted_names=restricted_names,
    )


def compute_distance_fields(
    nonfuel_mask: np.ndarray,
    pixel_h_m: float,
    pixel_w_m: float,
    *,
    return_nearest_indices: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Compute pixel-centre-to-pixel-centre distance to nearest non-fuel barrier.

    Parameters
    ----------
    nonfuel_mask:
        Boolean array, True where non-fuel barrier pixels are located.
    pixel_h_m, pixel_w_m:
        Pixel dimensions in metres (row-spacing first).
    return_nearest_indices:
        If True, also return row/col indices of the nearest barrier pixel for
        each pixel (used for directional sector classification).

    Returns
    -------
    dist_m:
        float32 distance in metres (centre-to-centre).  Barrier pixels = 0.
    nearest_row, nearest_col:
        int32 arrays of nearest-barrier pixel position, or None if not requested.
    """
    if not nonfuel_mask.any():
        raise ValueError("No non-fuel barrier pixels found; cannot compute distance fields.")

    not_barrier = ~nonfuel_mask
    if return_nearest_indices:
        dist_m, (nearest_row, nearest_col) = distance_transform_edt(not_barrier, sampling=(pixel_h_m, pixel_w_m), return_indices=True)
        return (
            dist_m.astype(np.float32),
            nearest_row.astype(np.int32),
            nearest_col.astype(np.int32),
        )
    dist_m = distance_transform_edt(not_barrier, sampling=(pixel_h_m, pixel_w_m))
    return dist_m.astype(np.float32), None, None


def dist_bin_label(bin_idx: int, bin_edges_m: tuple[float, ...]) -> str:
    edges = [0.0, *bin_edges_m, np.inf]
    lo, hi = edges[bin_idx], edges[bin_idx + 1]
    return f">{lo:.0f}m" if np.isinf(hi) else f"{lo:.0f}-{hi:.0f}m"


def compute_zone_wind_consistency(weather_path: Path) -> pd.DataFrame:
    """Compute speed-weighted wind consistency per weather zone.

    For each zone:
      source_x = Σ(WindSpeed * sin(WindDirection_rad))
      source_y = Σ(WindSpeed * cos(WindDirection_rad))
      consistency = ||[sum_x, sum_y]|| / Σ(WindSpeed)   ∈ [0, 1]
      dominant_from_direction_deg = atan2(source_x, source_y) mod 360
      dominant_direction_deg = (dominant_from_direction_deg + 180) mod 360

    consistency = 1: all strong winds point the same direction.
    consistency = 0: winds cancel out after speed-weighting.

    Note: dominant_direction_deg is the physical flow/downwind bearing.
    """
    df = pd.read_csv(weather_path)
    wd_rad = np.deg2rad(df["WindDirection"].to_numpy(dtype=np.float64))
    ws = df["WindSpeed"].to_numpy(dtype=np.float64)
    df = df.copy()
    df["_wx"] = ws * np.sin(wd_rad)
    df["_wy"] = ws * np.cos(wd_rad)

    rows = []
    for zone, grp in df.groupby("WeatherZone"):
        sum_x = float(grp["_wx"].sum())
        sum_y = float(grp["_wy"].sum())
        sum_speed = float(grp["WindSpeed"].sum())
        n = int(len(grp))

        if sum_speed == 0.0:
            consistency = float("nan")
            dominant_from_dir = float("nan")
            dominant_flow_dir = float("nan")
        else:
            consistency = float(np.sqrt(sum_x**2 + sum_y**2) / sum_speed)
            dominant_from_dir = float(np.degrees(np.arctan2(sum_x, sum_y)) % 360)
            dominant_flow_dir = float((dominant_from_dir + 180.0) % 360.0)

        rows.append(
            {
                "zone": str(zone),
                "n_rows": n,
                "sum_speed": sum_speed,
                "mean_speed": float(grp["WindSpeed"].mean()),
                "consistency": consistency,
                "dominant_from_direction_deg": dominant_from_dir,
                "dominant_direction_deg": dominant_flow_dir,
                "wind_convention_warning": WIND_DIRECTION_CONVENTION_UNKNOWN,
            }
        )
    return pd.DataFrame(rows)


def _normalize_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def assign_directional_sectors(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    nearest_row: np.ndarray,
    nearest_col: np.ndarray,
    dominant_dir_deg: float,
    pixel_h_m: float,
    pixel_w_m: float,
) -> np.ndarray:
    """Assign pixels to directional sectors relative to dominant wind.

    ``dominant_dir_deg`` is the physical flow/downwind bearing.  The bearing is
    measured from the nearest barrier pixel to the analysis pixel
    (barrier → pixel direction).  Sectors are:
      0  downwind_of_barrier  bearing ≈ dominant_dir
      1  crosswind_right      bearing ≈ dominant_dir + 90°
      2  upwind_of_barrier    bearing ≈ dominant_dir ± 180°
      3  crosswind_left       bearing ≈ dominant_dir − 90°

    North-up raster convention: rows increase downward, so north = −row direction.
    Returns int8 with sector index (0–3) or −1 for on-boundary pixels (rare).
    """
    delta_row = row_idx - nearest_row  # positive = pixel south of barrier
    delta_col = col_idx - nearest_col  # positive = pixel east of barrier
    # Geographic bearing: 0 = North (−row), π/2 = East (+col).
    bearing = np.arctan2(delta_col * pixel_w_m, -delta_row * pixel_h_m)

    dominant_rad = np.deg2rad(dominant_dir_deg)
    diff = _normalize_angle_rad(bearing - dominant_rad)

    hw = _SECTOR_HW_RAD  # 45°
    sectors = np.full(row_idx.shape, -1, dtype=np.int8)
    sectors[np.abs(diff) <= hw] = 0  # downwind_of_barrier
    sectors[(diff > hw) & (diff <= 3 * hw)] = 1  # crosswind_right
    sectors[np.abs(diff) > 3 * hw] = 2  # upwind_of_barrier
    sectors[(diff < -hw) & (diff >= -3 * hw)] = 3  # crosswind_left
    return sectors
