"""Diagnose BP barrier halo effect and wind-consistency shadow.

For each hexel, this script:
  1. Identifies non-fuel (spread-barrier) pixels from FuelTypes.csv.
  2. Computes per-pixel distance to the nearest barrier using EDT.
  3. Summarises mean BP as a function of barrier distance (distance profiles).
  4. Estimates a matched near/far halo contrast within (zone × fuel-group) strata.
  5. Computes speed-weighted wind consistency per weather zone.
  6. Classifies near-barrier pixels into directional sectors and reports sector
     contrasts for zones with sufficient wind consistency.
  7. Reports ignition-restricted fuel BP suppression separately.

Distance convention: all distances are pixel-centre-to-pixel-centre (EDT),
so the immediately adjacent burnable pixel is ~100 m from a barrier on a 100 m
grid.  Bins are labelled accordingly.

Wind direction convention: BurnP3+/NRCan WindDirection is treated as a
meteorological from-bearing.  Physical downwind flow is therefore
WindDirection + 180 degrees, or equivalently (-wind_x, -wind_y) relative to the
raw component encoding used by the model features.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy.ndimage import distance_transform_edt

from data_preparation.paths import Paths
from data_preparation.spatial.utils import FUEL_GROUP_MAP
from src.datasets.postprocessing.bp_restricted_zero import bp_nonfuel_restricted_ids

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FUEL_NODATA: int = -32768
ZONE_NODATA: int = -128
BP_NODATA: float = -9999.0

# BurnP3+/NRCan WindDirection is a meteorological from-bearing.  Keep the old
# constant name for compatibility with older tests/imports.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HaloConfig:
    raw_data_dir: Path
    hex_ids: list[str]
    save_dir: Path
    # Bin edges for barrier distance histograms.
    # Distances are pixel-centre-to-pixel-centre from EDT.
    distance_bin_edges_m: tuple[float, ...]
    # Analysis bands: "near" = dist < near_band_m; "far" = dist > far_band_m.
    near_band_m: float
    far_band_m: float
    # Strata with fewer than this many near or far pixels are skipped.
    min_pixels_per_stratum: int
    # Exclude analysis pixels within this distance of boundary-artifact nodata
    # (burnable pixels that are nodata because the simulation did not cover them).
    boundary_exclusion_m: float
    # Minimum speed-weighted consistency to run directional analysis.
    wind_consistency_threshold: float
    # Minimum near-barrier pixels per sector for directional output.
    min_pixels_per_sector: int
    # Minimum mean_bp_far to compute halo_rel (below this it is set to NaN).
    min_bp_for_halo_rel: float
    seed: int


def load_halo_config(path: Path) -> HaloConfig:
    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping; got {type(raw).__name__}.")
    for key in ("raw_data_dir", "save_dir"):
        if key not in raw:
            raise ValueError(f"Missing required config key: {key!r}.")

    hex_ids_raw = raw.get("hex_ids", "all")
    if hex_ids_raw == "all":
        from data_preparation.utils import find_hex_ids

        hex_ids = [str(h).zfill(2) for h in find_hex_ids(raw["raw_data_dir"])]
    else:
        hex_ids = [str(h).zfill(2) for h in hex_ids_raw]

    bin_edges = raw.get("distance_bin_edges_m", list(DEFAULT_DIST_BIN_EDGES_M))

    return HaloConfig(
        raw_data_dir=Path(raw["raw_data_dir"]),
        hex_ids=hex_ids,
        save_dir=Path(raw["save_dir"]),
        distance_bin_edges_m=tuple(float(e) for e in bin_edges),
        near_band_m=float(raw.get("near_band_m", 500.0)),
        far_band_m=float(raw.get("far_band_m", 2000.0)),
        min_pixels_per_stratum=int(raw.get("min_pixels_per_stratum", 30)),
        boundary_exclusion_m=float(raw.get("boundary_exclusion_m", 500.0)),
        wind_consistency_threshold=float(raw.get("wind_consistency_threshold", 0.3)),
        min_pixels_per_sector=int(raw.get("min_pixels_per_sector", 50)),
        min_bp_for_halo_rel=float(raw.get("min_bp_for_halo_rel", 1e-4)),
        seed=int(raw.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FuelBarrierInfo:
    """Per-hexel fuel barrier and ignition-restriction IDs."""

    hex_id: str
    # IDs whose FuelTypes.csv Description == "Non-fuel" — true spread barriers.
    nonfuel_ids: list[int]
    # IDs that are ignition-restricted but NOT non-fuel (can still burn).
    restricted_ids: list[int]
    restricted_names: list[str]


@dataclass
class HexelLayers:
    """All raster layers for one hexel on the same 100 m grid."""

    hex_id: str
    # float32; NaN where BP is nodata.
    bp: np.ndarray
    # int32; FUEL_NODATA where nodata.
    fuel: np.ndarray
    # int32; ZONE_NODATA where nodata.
    firezones: np.ndarray
    # float32; NaN where nodata.
    elevation: np.ndarray
    pixel_h_m: float
    pixel_w_m: float


# ---------------------------------------------------------------------------
# Fuel group lookup (vectorised, nodata-safe)
# ---------------------------------------------------------------------------

_MAX_FUEL_ID: int = max(FUEL_GROUP_MAP) + 1
_FUEL_GROUP_LOOKUP: np.ndarray = np.full(_MAX_FUEL_ID, -1, dtype=np.int8)
for _fid, _fg in FUEL_GROUP_MAP.items():
    _FUEL_GROUP_LOOKUP[_fid] = _fg


def fuel_to_groups(fuel: np.ndarray) -> np.ndarray:
    """Map raw fuel-ID raster to fuel-group raster.  Nodata and unknown → -1."""
    safe = np.where((fuel >= 0) & (fuel < _MAX_FUEL_ID), fuel, 0)
    groups = _FUEL_GROUP_LOOKUP[safe]
    groups = np.where((fuel >= 0) & (fuel < _MAX_FUEL_ID), groups, np.int8(-1))
    return groups


# ---------------------------------------------------------------------------
# Fuel barrier info
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------


def load_hexel_layers(paths: Paths, hex_id: str) -> HexelLayers:
    """Load all raster layers directly (co-registered; no reprojection needed)."""

    def _read_f32_nan(path: Path, nodata: float | None = None) -> tuple[np.ndarray, float, float]:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nd = float(src.nodata) if src.nodata is not None else nodata
            ph = abs(float(src.transform.e))
            pw = abs(float(src.transform.a))
        arr = np.where(data == nd, np.nan, data).astype(np.float32) if nd is not None else data
        return arr, ph, pw

    bp_path = paths.output_burn_prob()
    with rasterio.open(bp_path) as src:
        bp_raw = src.read(1).astype(np.float32)
        bp_nd = float(src.nodata) if src.nodata is not None else BP_NODATA
        pixel_h_m = abs(float(src.transform.e))
        pixel_w_m = abs(float(src.transform.a))
    bp = np.where(bp_raw == bp_nd, np.nan, bp_raw).astype(np.float32)

    with rasterio.open(paths.fuel_grid(hex_id)) as src:
        fuel = src.read(1).astype(np.int32)
        fuel_nd = int(src.nodata) if src.nodata is not None else FUEL_NODATA
    fuel = np.where(fuel == fuel_nd, FUEL_NODATA, fuel).astype(np.int32)

    with rasterio.open(paths.firezones_grid(hex_id)) as src:
        zones_raw = src.read(1)
        zone_nd = int(src.nodata) if src.nodata is not None else ZONE_NODATA
    firezones = np.where(zones_raw == zone_nd, ZONE_NODATA, zones_raw).astype(np.int32)

    with rasterio.open(paths.elevation_grid(hex_id)) as src:
        dem_raw = src.read(1).astype(np.float32)
        dem_nd = float(src.nodata) if src.nodata is not None else None
    elevation = np.where(dem_raw == dem_nd, np.nan, dem_raw).astype(np.float32) if dem_nd is not None else dem_raw

    return HexelLayers(
        hex_id=hex_id,
        bp=bp,
        fuel=fuel,
        firezones=firezones,
        elevation=elevation,
        pixel_h_m=pixel_h_m,
        pixel_w_m=pixel_w_m,
    )


# ---------------------------------------------------------------------------
# Distance fields
# ---------------------------------------------------------------------------


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


def compute_boundary_exclusion_mask(
    bp: np.ndarray,
    nonfuel_mask: np.ndarray,
    pixel_h_m: float,
    pixel_w_m: float,
    boundary_exclusion_m: float,
) -> np.ndarray:
    """Return a boolean mask of pixels near simulation-boundary nodata.

    The simulation extent does not cover the full raster; burnable pixels
    outside the extent are BP nodata.  These are structurally different from
    non-fuel nodata (barrier) and must not inflate the "far from barrier" group.

    Only pixels that are (a) BP nodata AND (b) not a known non-fuel barrier are
    considered boundary-artifact nodata.  Analysis pixels within
    ``boundary_exclusion_m`` of these are flagged for exclusion.
    """
    if boundary_exclusion_m <= 0:
        return np.zeros(bp.shape, dtype=bool)

    # Boundary-artifact nodata: BP is nodata but fuel is burnable (not non-fuel).
    boundary_nodata = np.isnan(bp) & ~nonfuel_mask
    if not boundary_nodata.any():
        return np.zeros(bp.shape, dtype=bool)

    dist_to_boundary = distance_transform_edt(~boundary_nodata, sampling=(pixel_h_m, pixel_w_m))
    return dist_to_boundary < boundary_exclusion_m


# ---------------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------------


def assign_distance_bins(
    dist_m: np.ndarray,
    bin_edges_m: tuple[float, ...],
) -> np.ndarray:
    """Assign each pixel an integer bin index.

    Bins are [0, e0), [e0, e1), ..., [e_{n-1}, ∞).
    Returns int16; value is the bin index (0-based).
    Barrier pixels (dist=0) land in bin 0 but are excluded by the analysis mask.
    """
    edges = np.array([0.0, *bin_edges_m, np.inf], dtype=np.float64)
    bins = np.searchsorted(edges[1:], dist_m, side="right").astype(np.int16)
    return bins


def dist_bin_label(bin_idx: int, bin_edges_m: tuple[float, ...]) -> str:
    edges = [0.0, *bin_edges_m, np.inf]
    lo, hi = edges[bin_idx], edges[bin_idx + 1]
    return f">{lo:.0f}m" if np.isinf(hi) else f"{lo:.0f}-{hi:.0f}m"


def dist_bin_labels(bin_edges_m: tuple[float, ...]) -> list[str]:
    return [dist_bin_label(i, bin_edges_m) for i in range(len(bin_edges_m) + 1)]


# ---------------------------------------------------------------------------
# Wind consistency
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Distance profiles
# ---------------------------------------------------------------------------


def compute_distance_profiles(
    layers: HexelLayers,
    fuel_info: FuelBarrierInfo,
    dist_m: np.ndarray,
    boundary_excl_mask: np.ndarray,
    bin_edges_m: tuple[float, ...],
) -> pd.DataFrame:
    """Descriptive BP statistics for each (zone_id, fuel_group, dist_bin).

    Only includes analysis pixels: burnable, unrestricted, BP-valid, not near
    boundary-artifact nodata.
    """
    bp = layers.bp
    fuel = layers.fuel
    zones = layers.firezones

    nonfuel_set = set(fuel_info.nonfuel_ids)
    restricted_set = set(fuel_info.restricted_ids)

    analysis = (
        (fuel != FUEL_NODATA)
        & ~np.isin(fuel, list(nonfuel_set))
        & ~np.isin(fuel, list(restricted_set))
        & ~np.isnan(bp)
        & (zones != ZONE_NODATA)
        & ~boundary_excl_mask
    )

    if not analysis.any():
        return pd.DataFrame()

    fg_arr = fuel_to_groups(fuel)
    bins_arr = assign_distance_bins(dist_m, bin_edges_m)

    idx = np.where(analysis)
    df_pix = pd.DataFrame(
        {
            "bp": bp[idx].astype(np.float64),
            "zone_id": zones[idx],
            "fuel_group": fg_arr[idx].astype(np.int16),
            "dist_bin_idx": bins_arr[idx],
        }
    )

    agg = (
        df_pix.groupby(["zone_id", "fuel_group", "dist_bin_idx"])["bp"]
        .agg(
            n_pixels="count",
            bp_mean="mean",
            bp_median="median",
            bp_p25=lambda x: float(np.percentile(x, 25)),
            bp_p75=lambda x: float(np.percentile(x, 75)),
        )
        .reset_index()
    )
    agg["hex_id"] = layers.hex_id
    agg["dist_bin"] = agg["dist_bin_idx"].apply(lambda i: dist_bin_label(int(i), bin_edges_m))
    return agg


# ---------------------------------------------------------------------------
# Matched halo contrasts
# ---------------------------------------------------------------------------


def _analytical_halo_ci(
    near: np.ndarray,
    far: np.ndarray,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Return halo SE and 95% CI assuming independent pixels (lower bound due to autocorr)."""
    se_near = float(np.std(near, ddof=1) / np.sqrt(len(near))) if len(near) > 1 else 0.0
    se_far = float(np.std(far, ddof=1) / np.sqrt(len(far))) if len(far) > 1 else 0.0
    se_halo = float(np.sqrt(se_near**2 + se_far**2))
    halo_abs = float(far.mean() - near.mean())
    return halo_abs, halo_abs - z * se_halo, halo_abs + z * se_halo


def compute_matched_contrasts(
    layers: HexelLayers,
    fuel_info: FuelBarrierInfo,
    dist_m: np.ndarray,
    boundary_excl_mask: np.ndarray,
    near_band_m: float,
    far_band_m: float,
    min_pixels_per_stratum: int,
    min_bp_for_halo_rel: float,
) -> pd.DataFrame:
    """Near/far halo contrast within (zone_id × fuel_group) strata.

    halo_abs = mean_bp_far − mean_bp_near.  Positive means near-barrier
    suppression (lower BP close to the barrier).

    CIs assume pixel-level independence; they are anticonservative under spatial
    autocorrelation and should be treated as indicative rather than inferential.
    """
    bp = layers.bp
    fuel = layers.fuel
    zones = layers.firezones

    nonfuel_set = set(fuel_info.nonfuel_ids)
    restricted_set = set(fuel_info.restricted_ids)

    analysis = (
        (fuel != FUEL_NODATA)
        & ~np.isin(fuel, list(nonfuel_set))
        & ~np.isin(fuel, list(restricted_set))
        & ~np.isnan(bp)
        & (zones != ZONE_NODATA)
        & ~boundary_excl_mask
    )

    near_mask = analysis & (dist_m < near_band_m)
    far_mask = analysis & (dist_m > far_band_m)

    fg_arr = fuel_to_groups(fuel)

    rows = []
    for zone_id in np.unique(zones[zones != ZONE_NODATA]):
        in_zone = zones == zone_id
        fgs = np.unique(fg_arr[analysis & in_zone])
        for fg in fgs:
            in_fg = fg_arr == fg
            near_bp = bp[near_mask & in_zone & in_fg]
            far_bp = bp[far_mask & in_zone & in_fg]

            if len(near_bp) < min_pixels_per_stratum or len(far_bp) < min_pixels_per_stratum:
                continue

            halo_abs, ci_lo, ci_hi = _analytical_halo_ci(near_bp, far_bp)
            mean_far = float(far_bp.mean())
            halo_rel = (halo_abs / mean_far) if mean_far >= min_bp_for_halo_rel else float("nan")

            rows.append(
                {
                    "hex_id": layers.hex_id,
                    "zone_id": int(zone_id),
                    "fuel_group": int(fg),
                    "n_near": len(near_bp),
                    "n_far": len(far_bp),
                    "mean_bp_near": float(near_bp.mean()),
                    "mean_bp_far": mean_far,
                    "std_bp_near": float(near_bp.std(ddof=1)),
                    "std_bp_far": float(far_bp.std(ddof=1)),
                    "halo_abs": halo_abs,
                    "halo_rel": halo_rel,
                    "halo_ci_lo_95_naive": ci_lo,
                    "halo_ci_hi_95_naive": ci_hi,
                    "ci_note": "independent-pixel SE; anticonservative under spatial autocorrelation",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Directional sectors
# ---------------------------------------------------------------------------


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


def compute_directional_shadow(
    layers: HexelLayers,
    fuel_info: FuelBarrierInfo,
    dist_m: np.ndarray,
    nearest_row: np.ndarray,
    nearest_col: np.ndarray,
    boundary_excl_mask: np.ndarray,
    wind_df: pd.DataFrame,
    near_band_m: float,
    wind_consistency_threshold: float,
    min_pixels_per_sector: int,
) -> pd.DataFrame:
    """Sector-wise BP contrasts near barriers for high-consistency wind zones."""
    bp = layers.bp
    fuel = layers.fuel
    zones = layers.firezones

    nonfuel_set = set(fuel_info.nonfuel_ids)
    restricted_set = set(fuel_info.restricted_ids)

    # Near-barrier analysis pixels only (> 0 excludes the barrier pixels themselves).
    analysis = (
        (fuel != FUEL_NODATA)
        & ~np.isin(fuel, list(nonfuel_set))
        & ~np.isin(fuel, list(restricted_set))
        & ~np.isnan(bp)
        & (zones != ZONE_NODATA)
        & ~boundary_excl_mask
        & (dist_m > 0)
        & (dist_m < near_band_m)
    )

    h, w = bp.shape
    row_idx, col_idx = np.indices((h, w), dtype=np.int32)

    rows = []
    for _, wrow in wind_df.iterrows():
        consistency = float(wrow["consistency"])
        if not np.isfinite(consistency) or consistency < wind_consistency_threshold:
            continue

        zone_name = str(wrow["zone"])
        dominant_dir = float(wrow["dominant_direction_deg"])

        # Map zone name (e.g., "fru21") → integer zone ID.
        digits = "".join(c for c in zone_name if c.isdigit())
        if not digits:
            log.warning("hex%s: cannot parse zone_id from zone name %r; skipping.", layers.hex_id, zone_name)
            continue
        zone_id = int(digits)

        in_zone = zones == zone_id
        zone_analysis = analysis & in_zone
        if not zone_analysis.any():
            continue

        sectors = assign_directional_sectors(
            row_idx,
            col_idx,
            nearest_row,
            nearest_col,
            dominant_dir,
            layers.pixel_h_m,
            layers.pixel_w_m,
        )

        for s_idx, s_label in enumerate(SECTOR_LABELS):
            in_sector = (sectors == s_idx) & zone_analysis
            n = int(in_sector.sum())
            if n < min_pixels_per_sector:
                continue
            bp_vals = bp[in_sector]
            rows.append(
                {
                    "hex_id": layers.hex_id,
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "sector": s_label,
                    "n_pixels": n,
                    "bp_mean": float(bp_vals.mean()),
                    "bp_median": float(np.median(bp_vals)),
                    "bp_std": float(bp_vals.std(ddof=1)),
                    "consistency": consistency,
                    "dominant_direction_deg": dominant_dir,
                    "wind_convention_warning": WIND_DIRECTION_CONVENTION_UNKNOWN,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Restricted fuel diagnostics
# ---------------------------------------------------------------------------


def compute_restricted_fuel_diagnostics(
    layers: HexelLayers,
    fuel_info: FuelBarrierInfo,
) -> pd.DataFrame:
    """BP suppression on ignition-restricted pixels vs matched unrestricted pixels.

    Restricted pixels are burnable but cannot be ignition sources.  They are
    not spread barriers, so BP on these pixels reflects fire reaching them from
    adjacent ignitable areas.
    """
    if not fuel_info.restricted_ids:
        return pd.DataFrame()

    bp = layers.bp
    fuel = layers.fuel
    zones = layers.firezones

    fuel_valid = fuel != FUEL_NODATA
    zone_valid = zones != ZONE_NODATA
    nonfuel_set = set(fuel_info.nonfuel_ids)
    restricted_set = set(fuel_info.restricted_ids)

    fg_arr = fuel_to_groups(fuel)

    unrestricted_mask = fuel_valid & ~np.isin(fuel, list(nonfuel_set)) & ~np.isin(fuel, list(restricted_set)) & zone_valid

    rows = []
    for zone_id in np.unique(zones[zone_valid]):
        in_zone = zones == zone_id
        for restr_id in fuel_info.restricted_ids:
            in_fuel = fuel == restr_id
            restr_mask = fuel_valid & in_fuel & in_zone & zone_valid
            n_total = int(restr_mask.sum())
            if n_total == 0:
                continue

            bp_valid_restr = restr_mask & ~np.isnan(bp)
            bp_nodata_restr = restr_mask & np.isnan(bp)
            bp_vals_restr = bp[bp_valid_restr]

            fg = int(FUEL_GROUP_MAP.get(restr_id, -1))
            ref_mask = unrestricted_mask & in_zone & ~np.isnan(bp) & (fg_arr == fg)
            bp_ref = bp[ref_mask]

            suppression = float(bp_ref.mean() - bp_vals_restr.mean()) if len(bp_vals_restr) > 0 and len(bp_ref) > 0 else float("nan")
            rows.append(
                {
                    "hex_id": layers.hex_id,
                    "zone_id": int(zone_id),
                    "fuel_id": restr_id,
                    "fuel_group": fg,
                    "n_total": n_total,
                    "n_bp_nodata": int(bp_nodata_restr.sum()),
                    "n_bp_valid": len(bp_vals_restr),
                    "mean_bp_restricted": float(bp_vals_restr.mean()) if len(bp_vals_restr) > 0 else float("nan"),
                    "n_reference": len(bp_ref),
                    "mean_bp_reference": float(bp_ref.mean()) if len(bp_ref) > 0 else float("nan"),
                    "suppression_abs": suppression,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Barrier support summary
# ---------------------------------------------------------------------------


def compute_barrier_support(
    layers: HexelLayers,
    fuel_info: FuelBarrierInfo,
    dist_m: np.ndarray,
    boundary_excl_mask: np.ndarray,
    near_band_m: float,
    far_band_m: float,
) -> pd.DataFrame:
    """Pixel-count summary for this hexel."""
    bp = layers.bp
    fuel = layers.fuel

    fuel_valid = fuel != FUEL_NODATA
    nonfuel_set = set(fuel_info.nonfuel_ids)
    restricted_set = set(fuel_info.restricted_ids)

    nonfuel_mask = fuel_valid & np.isin(fuel, list(nonfuel_set))
    burnable_mask = fuel_valid & ~np.isin(fuel, list(nonfuel_set))
    bp_valid = ~np.isnan(bp)
    analysis_mask = burnable_mask & bp_valid & ~np.isin(fuel, list(restricted_set)) & ~boundary_excl_mask
    near_analysis = analysis_mask & (dist_m < near_band_m)
    far_analysis = analysis_mask & (dist_m > far_band_m)

    return pd.DataFrame(
        [
            {
                "hex_id": layers.hex_id,
                "total_pixels": int(fuel.size),
                "nonfuel_barrier_pixels": int(nonfuel_mask.sum()),
                "burnable_pixels": int(burnable_mask.sum()),
                "burnable_bp_nodata": int((burnable_mask & ~bp_valid).sum()),
                "boundary_excluded_pixels": int(boundary_excl_mask.sum()),
                "analysis_pixels": int(analysis_mask.sum()),
                "near_band_pixels": int(near_analysis.sum()),
                "far_band_pixels": int(far_analysis.sum()),
                "analysis_pct_of_total": round(100.0 * int(analysis_mask.sum()) / int(fuel.size), 2),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_distance_profiles(profiles_df: pd.DataFrame, save_dir: Path) -> None:
    if profiles_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for (hx, zid), grp in profiles_df.groupby(["hex_id", "zone_id"]):
        grp = grp.sort_values("dist_bin_idx")
        ax.plot(grp["dist_bin_idx"], grp["bp_mean"], alpha=0.35, linewidth=0.9)
    ax.set_xticks(range(len(DEFAULT_DIST_BIN_EDGES_M) + 1))
    ax.set_xticklabels(dist_bin_labels(DEFAULT_DIST_BIN_EDGES_M), rotation=30, ha="right")
    ax.set_xlabel("Distance to non-fuel barrier (centre-to-centre)")
    ax.set_ylabel("Mean BP")
    ax.set_title("BP vs distance to non-fuel barrier (all hexels × zones)")
    fig.tight_layout()
    fig.savefig(save_dir / "halo_distance_profiles.png", dpi=130)
    plt.close(fig)


def _plot_halo_contrasts(contrasts_df: pd.DataFrame, save_dir: Path) -> None:
    if contrasts_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = sorted(contrasts_df["fuel_group"].unique())
    data = [contrasts_df.loc[contrasts_df["fuel_group"] == g, "halo_abs"].dropna().values for g in groups]
    ax.boxplot(data, tick_labels=[str(g) for g in groups], notch=False)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Fuel group")
    ax.set_ylabel("Halo strength  (mean BP far − near)")
    ax.set_title("Matched halo contrast by fuel group")
    fig.tight_layout()
    fig.savefig(save_dir / "halo_matched_contrasts.png", dpi=130)
    plt.close(fig)


def _plot_wind_consistency(wind_df: pd.DataFrame, save_dir: Path) -> None:
    if wind_df.empty:
        return
    fig = plt.figure(figsize=(13, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(wind_df["consistency"].dropna(), bins=30, edgecolor="white")
    ax1.set_xlabel("Wind consistency")
    ax1.set_ylabel("Weather-zone count")
    ax1.set_title("Speed-weighted wind consistency distribution")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    valid = wind_df.dropna(subset=["consistency", "dominant_direction_deg"])
    theta = np.deg2rad(valid["dominant_direction_deg"].values)
    r = valid["consistency"].values
    ax2.scatter(theta, r, alpha=0.6, s=18)
    ax2.set_title("Dominant physical flow direction vs consistency")

    fig.tight_layout()
    fig.savefig(save_dir / "wind_consistency.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-hexel orchestration
# ---------------------------------------------------------------------------


def run_hexel(hex_id: str, config: HaloConfig) -> dict[str, pd.DataFrame]:
    paths = Paths(hex_id=hex_id, root_dir=config.raw_data_dir)

    log.info("hex%s: loading layers", hex_id)
    layers = load_hexel_layers(paths, hex_id)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)

    nonfuel_mask = np.isin(layers.fuel, fuel_info.nonfuel_ids) & (layers.fuel != FUEL_NODATA)
    if not nonfuel_mask.any():
        log.warning("hex%s: no non-fuel barrier pixels found; skipping barrier diagnostics.", hex_id)
        return {k: pd.DataFrame() for k in ("profiles", "contrasts", "wind", "shadow", "restricted", "support")}

    log.info("hex%s: computing distance fields", hex_id)
    dist_m, nearest_row, nearest_col = compute_distance_fields(
        nonfuel_mask, layers.pixel_h_m, layers.pixel_w_m, return_nearest_indices=True
    )

    log.info("hex%s: computing boundary exclusion mask", hex_id)
    boundary_excl = compute_boundary_exclusion_mask(
        layers.bp, nonfuel_mask, layers.pixel_h_m, layers.pixel_w_m, config.boundary_exclusion_m
    )

    log.info("hex%s: computing wind consistency", hex_id)
    weather_path = paths.tabular_dir / f"hex{hex_id}_DailyWeather.csv"
    wind_df = compute_zone_wind_consistency(weather_path)
    wind_df.insert(0, "hex_id", hex_id)

    log.info("hex%s: computing distance profiles", hex_id)
    profiles_df = compute_distance_profiles(layers, fuel_info, dist_m, boundary_excl, config.distance_bin_edges_m)

    log.info("hex%s: computing matched contrasts", hex_id)
    contrasts_df = compute_matched_contrasts(
        layers,
        fuel_info,
        dist_m,
        boundary_excl,
        config.near_band_m,
        config.far_band_m,
        config.min_pixels_per_stratum,
        config.min_bp_for_halo_rel,
    )

    log.info("hex%s: computing directional shadow", hex_id)
    assert nearest_row is not None and nearest_col is not None
    shadow_df = compute_directional_shadow(
        layers,
        fuel_info,
        dist_m,
        nearest_row,
        nearest_col,
        boundary_excl,
        wind_df,
        config.near_band_m,
        config.wind_consistency_threshold,
        config.min_pixels_per_sector,
    )

    log.info("hex%s: computing restricted-fuel diagnostics", hex_id)
    restricted_df = compute_restricted_fuel_diagnostics(layers, fuel_info)

    log.info("hex%s: computing barrier support summary", hex_id)
    support_df = compute_barrier_support(layers, fuel_info, dist_m, boundary_excl, config.near_band_m, config.far_band_m)

    return {
        "profiles": profiles_df,
        "contrasts": contrasts_df,
        "wind": wind_df,
        "shadow": shadow_df,
        "restricted": restricted_df,
        "support": support_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: HaloConfig) -> None:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    buckets: dict[str, list[pd.DataFrame]] = {k: [] for k in ("profiles", "contrasts", "wind", "shadow", "restricted", "support")}

    for hex_id in config.hex_ids:
        try:
            results = run_hexel(hex_id, config)
        except Exception:
            log.exception("hex%s: unexpected error; skipping.", hex_id)
            continue
        for key, df in results.items():
            if not df.empty:
                buckets[key].append(df)

    def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    profiles_df = _cat(buckets["profiles"])
    contrasts_df = _cat(buckets["contrasts"])
    wind_df = _cat(buckets["wind"])
    shadow_df = _cat(buckets["shadow"])
    restricted_df = _cat(buckets["restricted"])
    support_df = _cat(buckets["support"])

    profiles_df.to_csv(config.save_dir / "halo_distance_profiles.csv", index=False)
    contrasts_df.to_csv(config.save_dir / "halo_matched_contrasts.csv", index=False)
    wind_df.to_csv(config.save_dir / "wind_consistency_by_zone.csv", index=False)
    shadow_df.to_csv(config.save_dir / "directional_shadow_contrasts.csv", index=False)
    restricted_df.to_csv(config.save_dir / "restricted_fuel_diagnostics.csv", index=False)
    support_df.to_csv(config.save_dir / "barrier_support_diagnostics.csv", index=False)

    _plot_distance_profiles(profiles_df, plot_dir)
    _plot_halo_contrasts(contrasts_df, plot_dir)
    _plot_wind_consistency(wind_df, plot_dir)

    log.info(
        "Done. Analysed %d hexels. Results saved to %s",
        len(buckets["support"]),
        config.save_dir,
    )


def _parse_args() -> HaloConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    args = parser.parse_args()
    return load_halo_config(args.config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main(_parse_args())
