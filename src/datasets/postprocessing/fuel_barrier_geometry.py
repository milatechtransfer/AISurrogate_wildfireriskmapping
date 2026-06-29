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
from data_preparation.spatial.utils import FUEL_GROUP_MAP, load_spatial_raster
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


SECTOR_ALL = "all_pixels"
SECTOR_CROSSWIND = "crosswind"
PROFILE_SECTORS: tuple[str, ...] = (
    SECTOR_ALL,
    "downwind_of_barrier",
    SECTOR_CROSSWIND,
    "upwind_of_barrier",
)
LOCALIZATION_THRESHOLDS_M: tuple[float, ...] = (250.0, 500.0, 1000.0)

DISPLAY_SCENARIO = {
    "remove_barriers_adjacent_modal": "Remove non-fuel barriers",
}


@dataclass(frozen=True)
class BarrierProfileLayers:
    hex_id: str
    fuel: np.ndarray
    firezones: np.ndarray
    pixel_h_m: float
    pixel_w_m: float


def load_barrier_layers_on_prediction_grid(
    *,
    raw_data_dir: Path,
    reference_profile: dict,
    hex_id: str,
) -> BarrierProfileLayers:
    """Load raw fuel/firezone layers aligned to a prediction reference grid."""

    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_ma, fuel_profile = load_spatial_raster(
        paths.fuel_grid(hex_id),
        reference_profile=reference_profile,
    )
    firezones_ma, _ = load_spatial_raster(
        paths.firezones_grid(hex_id),
        reference_profile=reference_profile,
    )
    fuel = np.ma.asarray(fuel_ma).filled(FUEL_NODATA).astype(np.int32)
    firezones = np.ma.asarray(firezones_ma).filled(ZONE_NODATA).astype(np.int32)
    transform = fuel_profile["transform"]
    return BarrierProfileLayers(
        hex_id=hex_id,
        fuel=fuel,
        firezones=firezones,
        pixel_h_m=abs(float(transform.e)),
        pixel_w_m=abs(float(transform.a)),
    )


def load_grouped_fuel_on_prediction_grid(
    *,
    raw_data_dir: Path,
    reference_profile: dict,
    hex_id: str,
) -> np.ndarray:
    """Load raw fuel on the prediction grid and map raw fuel IDs to model fuel groups."""

    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_ma, _ = load_spatial_raster(paths.fuel_grid(hex_id), reference_profile=reference_profile)
    raw_fuel = np.ma.asarray(fuel_ma).astype(np.float32).filled(np.nan)
    grouped = np.full(raw_fuel.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(raw_fuel)
    raw_int = np.full(raw_fuel.shape, -9999, dtype=np.int32)
    raw_int[finite] = raw_fuel[finite].astype(np.int32)
    for raw_id, group_id in FUEL_GROUP_MAP.items():
        grouped[raw_int == int(raw_id)] = float(group_id)
    return grouped


def _finite_values(data: np.ndarray | np.ma.MaskedArray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.ma.asarray(data)
    values = np.asarray(arr.filled(np.nan), dtype=np.float64)
    return values, np.isfinite(values)


def _safe_share(numerator: float, denominator: float) -> float:
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(numerator / denominator)


def profile_from_arrays(
    *,
    baseline: np.ndarray | np.ma.MaskedArray,
    scenario: np.ndarray | np.ma.MaskedArray,
    analysis_mask: np.ndarray,
    dist_m: np.ndarray,
    sector_labels: np.ndarray,
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
    scenario_name: str,
    endpoint: str,
    hex_id: str,
    sector_source: str,
) -> pd.DataFrame:
    """Compute binned scenario-baseline deltas versus barrier distance."""

    baseline_values, baseline_valid = _finite_values(baseline)
    scenario_values, scenario_valid = _finite_values(scenario)
    valid = analysis_mask & baseline_valid & scenario_valid & np.isfinite(dist_m)
    if not valid.any():
        return pd.DataFrame()

    bin_edges = [0.0, *bin_edges_m, np.inf]
    rows: list[dict] = []
    for sector in PROFILE_SECTORS:
        sector_mask = valid if sector == SECTOR_ALL else valid & (sector_labels == sector)
        if not sector_mask.any():
            continue
        for bin_idx in range(len(bin_edges) - 1):
            lo = bin_edges[bin_idx]
            hi = bin_edges[bin_idx + 1]
            if np.isinf(hi):
                in_bin = sector_mask & (dist_m >= lo)
            else:
                in_bin = sector_mask & (dist_m >= lo) & (dist_m < hi)
            if not in_bin.any():
                continue
            baseline_bin = baseline_values[in_bin]
            scenario_bin = scenario_values[in_bin]
            delta = scenario_bin - baseline_bin
            rows.append(
                {
                    "scenario": scenario_name,
                    "endpoint": endpoint,
                    "hex_id": hex_id,
                    "sector": sector,
                    "sector_source": sector_source,
                    "dist_bin_idx": bin_idx,
                    "dist_bin": dist_bin_label(bin_idx, bin_edges_m),
                    "dist_min_m": lo,
                    "dist_max_m": hi,
                    "n_pixels": int(delta.size),
                    "baseline_sum": float(np.sum(baseline_bin)),
                    "scenario_sum": float(np.sum(scenario_bin)),
                    "delta_sum": float(np.sum(delta)),
                    "baseline_mean": float(np.mean(baseline_bin)),
                    "scenario_mean": float(np.mean(scenario_bin)),
                    "delta_mean": float(np.mean(delta)),
                    "delta_median": float(np.median(delta)),
                    "delta_p25": float(np.percentile(delta, 25)),
                    "delta_p75": float(np.percentile(delta, 75)),
                    "delta_abs_sum": float(np.sum(np.abs(delta))),
                    "delta_abs_mean": float(np.mean(np.abs(delta))),
                    "delta_positive_sum": float(np.sum(delta[delta > 0.0])),
                    "delta_negative_sum": float(np.sum(delta[delta < 0.0])),
                    "frac_delta_positive": float(np.mean(delta > 0.0)),
                    "frac_delta_negative": float(np.mean(delta < 0.0)),
                }
            )
    return pd.DataFrame(rows)


def hazard_decomposition_from_arrays(
    *,
    baseline_bp: np.ndarray | np.ma.MaskedArray,
    baseline_fi: np.ndarray | np.ma.MaskedArray,
    scenario_bp: np.ndarray | np.ma.MaskedArray,
    scenario_fi: np.ndarray | np.ma.MaskedArray,
    analysis_mask: np.ndarray,
    dist_m: np.ndarray,
    sector_labels: np.ndarray,
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
    scenario_name: str,
    hex_id: str,
    sector_source: str,
) -> pd.DataFrame:
    """Decompose delta(BP x FI) exactly into BP, FI, and interaction terms."""

    bp0, bp0_valid = _finite_values(baseline_bp)
    fi0, fi0_valid = _finite_values(baseline_fi)
    bp1, bp1_valid = _finite_values(scenario_bp)
    fi1, fi1_valid = _finite_values(scenario_fi)
    valid = analysis_mask & bp0_valid & fi0_valid & bp1_valid & fi1_valid & np.isfinite(dist_m)
    if not valid.any():
        return pd.DataFrame()

    delta_bp = bp1 - bp0
    delta_fi = fi1 - fi0
    baseline_hazard = bp0 * fi0
    scenario_hazard = bp1 * fi1
    bp_component = fi0 * delta_bp
    fi_component = bp0 * delta_fi
    interaction_component = delta_bp * delta_fi
    delta_hazard = bp_component + fi_component + interaction_component

    bin_edges = [0.0, *bin_edges_m, np.inf]
    rows: list[dict] = []
    for sector in PROFILE_SECTORS:
        sector_mask = valid if sector == SECTOR_ALL else valid & (sector_labels == sector)
        if not sector_mask.any():
            continue
        for bin_idx in range(len(bin_edges) - 1):
            lo = bin_edges[bin_idx]
            hi = bin_edges[bin_idx + 1]
            if np.isinf(hi):
                in_bin = sector_mask & (dist_m >= lo)
            else:
                in_bin = sector_mask & (dist_m >= lo) & (dist_m < hi)
            if not in_bin.any():
                continue

            bp_term = bp_component[in_bin]
            fi_term = fi_component[in_bin]
            interaction = interaction_component[in_bin]
            hazard_delta = delta_hazard[in_bin]
            baseline_hazard_bin = baseline_hazard[in_bin]
            scenario_hazard_bin = scenario_hazard[in_bin]
            rows.append(
                {
                    "scenario": scenario_name,
                    "hex_id": hex_id,
                    "sector": sector,
                    "sector_source": sector_source,
                    "dist_bin_idx": bin_idx,
                    "dist_bin": dist_bin_label(bin_idx, bin_edges_m),
                    "dist_min_m": lo,
                    "dist_max_m": hi,
                    "n_pixels": int(hazard_delta.size),
                    "baseline_hazard_sum": float(np.sum(baseline_hazard_bin)),
                    "scenario_hazard_sum": float(np.sum(scenario_hazard_bin)),
                    "delta_hazard_sum": float(np.sum(hazard_delta)),
                    "bp_component_sum": float(np.sum(bp_term)),
                    "fi_component_sum": float(np.sum(fi_term)),
                    "interaction_component_sum": float(np.sum(interaction)),
                    "delta_hazard_mean": float(np.mean(hazard_delta)),
                    "bp_component_mean": float(np.mean(bp_term)),
                    "fi_component_mean": float(np.mean(fi_term)),
                    "interaction_component_mean": float(np.mean(interaction)),
                    "delta_bp_mean": float(np.mean(delta_bp[in_bin])),
                    "delta_fi_mean": float(np.mean(delta_fi[in_bin])),
                    "baseline_bp_mean": float(np.mean(bp0[in_bin])),
                    "baseline_fi_mean": float(np.mean(fi0[in_bin])),
                    "scenario_bp_mean": float(np.mean(bp1[in_bin])),
                    "scenario_fi_mean": float(np.mean(fi1[in_bin])),
                }
            )
    return pd.DataFrame(rows)


def cumulative_hazard_delta_share(
    profile_df: pd.DataFrame,
    *,
    scenario: str = "remove_barriers_adjacent_modal",
    endpoint: str = "hazard_bp_x_fi",
    thresholds_m: tuple[float, ...] = LOCALIZATION_THRESHOLDS_M,
) -> pd.DataFrame:
    """Summarize cumulative delta-hazard shares within distance thresholds."""

    sub = profile_df[profile_df["scenario"].eq(scenario) & profile_df["endpoint"].eq(endpoint) & profile_df["sector"].eq(SECTOR_ALL)].copy()
    if sub.empty:
        return pd.DataFrame()

    if "delta_sum" not in sub.columns:
        sub["delta_sum"] = sub["delta_mean"] * sub["n_pixels"]
    if "baseline_sum" not in sub.columns:
        sub["baseline_sum"] = sub["baseline_mean"] * sub["n_pixels"]
    if "scenario_sum" not in sub.columns:
        sub["scenario_sum"] = sub["scenario_mean"] * sub["n_pixels"]
    has_signed_sums = {"delta_positive_sum", "delta_negative_sum", "delta_abs_sum"}.issubset(sub.columns)

    sub = sub.sort_values("dist_bin_idx")
    total_delta = float(sub["delta_sum"].sum())
    total_baseline = float(sub["baseline_sum"].sum())
    total_scenario = float(sub["scenario_sum"].sum())
    total_positive = float(sub["delta_positive_sum"].sum()) if has_signed_sums else float("nan")
    total_abs = float(sub["delta_abs_sum"].sum()) if has_signed_sums else float("nan")

    rows: list[dict] = []
    for threshold_m in thresholds_m:
        within = sub[sub["dist_max_m"].astype(float) <= threshold_m]
        delta_within = float(within["delta_sum"].sum())
        positive_within = float(within["delta_positive_sum"].sum()) if has_signed_sums else float("nan")
        abs_within = float(within["delta_abs_sum"].sum()) if has_signed_sums else float("nan")
        rows.append(
            {
                "scenario": scenario,
                "endpoint": endpoint,
                "threshold_m": float(threshold_m),
                "threshold_label": f"<= {threshold_m / 1000:g} km" if threshold_m >= 1000 else f"<= {threshold_m:g} m",
                "n_pixels_within": int(within["n_pixels"].sum()),
                "n_pixels_total": int(sub["n_pixels"].sum()),
                "delta_hazard_sum_within": delta_within,
                "delta_hazard_sum_total": total_delta,
                "percent_of_total_delta_hazard": 100.0 * _safe_share(delta_within, total_delta),
                "positive_delta_hazard_sum_within": positive_within,
                "positive_delta_hazard_sum_total": total_positive,
                "percent_of_positive_delta_hazard": 100.0 * _safe_share(positive_within, total_positive),
                "absolute_delta_hazard_sum_within": abs_within,
                "absolute_delta_hazard_sum_total": total_abs,
                "percent_of_absolute_delta_hazard": 100.0 * _safe_share(abs_within, total_abs),
                "baseline_hazard_sum_total": total_baseline,
                "scenario_hazard_sum_total": total_scenario,
            }
        )
    return pd.DataFrame(rows)


def relative_hazard_decomposition_summary(
    decomposition_df: pd.DataFrame,
    *,
    scenario: str = "remove_barriers_adjacent_modal",
) -> pd.DataFrame:
    """Summarize relative delta-hazard contributions as summed terms over summed baseline hazard."""

    sub = decomposition_df[decomposition_df["scenario"].eq(scenario) & decomposition_df["sector"].eq(SECTOR_ALL)].copy()
    if sub.empty:
        return pd.DataFrame()

    required = {
        "baseline_hazard_sum",
        "scenario_hazard_sum",
        "delta_hazard_sum",
        "bp_component_sum",
        "fi_component_sum",
        "interaction_component_sum",
    }
    missing = sorted(required - set(sub.columns))
    if missing:
        raise ValueError(f"Missing required hazard decomposition columns: {missing}")

    sub = sub.sort_values("dist_bin_idx")
    rows: list[dict] = []
    for row in sub.itertuples(index=False):
        baseline_hazard_sum = float(row.baseline_hazard_sum)
        bp_percent = 100.0 * _safe_share(float(row.bp_component_sum), baseline_hazard_sum)
        fi_percent = 100.0 * _safe_share(float(row.fi_component_sum), baseline_hazard_sum)
        interaction_percent = 100.0 * _safe_share(float(row.interaction_component_sum), baseline_hazard_sum)
        total_percent = 100.0 * _safe_share(float(row.delta_hazard_sum), baseline_hazard_sum)
        rows.append(
            {
                "scenario": scenario,
                "hex_id": row.hex_id,
                "sector": row.sector,
                "sector_source": row.sector_source,
                "dist_bin_idx": int(row.dist_bin_idx),
                "dist_bin": row.dist_bin,
                "dist_min_m": float(row.dist_min_m),
                "dist_max_m": float(row.dist_max_m),
                "n_pixels": int(row.n_pixels),
                "baseline_hazard_sum": baseline_hazard_sum,
                "scenario_hazard_sum": float(row.scenario_hazard_sum),
                "delta_hazard_sum": float(row.delta_hazard_sum),
                "bp_component_sum": float(row.bp_component_sum),
                "fi_component_sum": float(row.fi_component_sum),
                "interaction_component_sum": float(row.interaction_component_sum),
                "relative_delta_hazard_percent": total_percent,
                "relative_bp_component_percent": bp_percent,
                "relative_fi_component_percent": fi_percent,
                "relative_interaction_component_percent": interaction_percent,
            }
        )
    return pd.DataFrame(rows)
