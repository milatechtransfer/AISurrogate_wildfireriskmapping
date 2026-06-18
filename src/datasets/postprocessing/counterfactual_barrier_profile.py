"""Barrier-relative profiles for fixed-model counterfactual predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_weather import raw_weather_path
from src.datasets.postprocessing.diagnose_bp_barrier_halo import (
    DEFAULT_DIST_BIN_EDGES_M,
    FUEL_NODATA,
    SECTOR_LABELS,
    ZONE_NODATA,
    assign_directional_sectors,
    compute_distance_fields,
    compute_zone_wind_consistency,
    dist_bin_label,
    parse_fuel_barrier_info,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SECTOR_ALL = "all_pixels"
SECTOR_CROSSWIND = "crosswind"
PROFILE_SECTORS: tuple[str, ...] = (
    SECTOR_ALL,
    "downwind_of_barrier",
    SECTOR_CROSSWIND,
    "upwind_of_barrier",
)

ENDPOINT_ORDER: tuple[str, ...] = ("bp", "ros", "fi", "hazard_bp_x_fi")
SCENARIO_ORDER: tuple[str, ...] = (
    "wind_roundtrip_placebo",
    "wind_from_west_p95",
    "wind_zone_consistent_direction",
    "wind_speed50_zone_consistent_direction",
    "wind_speed100_zone_consistent_direction",
    "wind_speed50_original_direction",
    "remove_barriers_adjacent_modal",
)
LOCALIZATION_THRESHOLDS_M: tuple[float, ...] = (250.0, 500.0, 1000.0)

SECTOR_COLOURS = {
    SECTOR_ALL: "#222222",
    "downwind_of_barrier": "#d73027",
    SECTOR_CROSSWIND: "#4575b4",
    "upwind_of_barrier": "#1a9850",
}
SCENARIO_COLOURS = {
    "wind_roundtrip_placebo": "#888888",
    "wind_from_west_p95": "#4575b4",
    "wind_zone_consistent_direction": "#4daf4a",
    "wind_speed50_zone_consistent_direction": "#e7298a",
    "wind_speed100_zone_consistent_direction": "#a65628",
    "wind_speed50_original_direction": "#984ea3",
    "remove_barriers_adjacent_modal": "#d73027",
}

DISPLAY_ENDPOINT = {
    "bp": "BP",
    "ros": "ROS",
    "fi": "FI",
    "hazard_bp_x_fi": "Hazard = BP x FI",
}
DISPLAY_SCENARIO = {
    "wind_roundtrip_placebo": "Wind roundtrip placebo",
    "wind_from_west_p95": "Wind from west, p95 speed",
    "wind_zone_consistent_direction": "Wind zone-consistent directions",
    "wind_speed50_zone_consistent_direction": "Wind speed 50, zone-consistent directions",
    "wind_speed100_zone_consistent_direction": "Wind speed 100, zone-consistent directions",
    "wind_speed50_original_direction": "Wind speed 50, original directions",
    "remove_barriers_adjacent_modal": "Remove non-fuel barriers",
}
DISPLAY_SECTOR = {
    SECTOR_ALL: "All pixels",
    "downwind_of_barrier": "Downwind of barrier",
    SECTOR_CROSSWIND: "Crosswind",
    "upwind_of_barrier": "Upwind of barrier",
}


@dataclass(frozen=True)
class BarrierProfileLayers:
    hex_id: str
    fuel: np.ndarray
    firezones: np.ndarray
    pixel_h_m: float
    pixel_w_m: float


def _prediction_dirs(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    index_path = experiment_dir / "scenario_prediction_index.csv"
    index = pd.read_csv(index_path)
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def _prediction_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "predicted_hexels" / f"hexel_{int(hex_id):02d}_predicted.tif"


def _read_prediction(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def _prediction_reference_profile(prediction_dirs: dict[tuple[str, str], Path], hex_id: str) -> dict:
    baseline_bp_dir = prediction_dirs.get(("baseline", "bp"))
    if baseline_bp_dir is None:
        raise KeyError("Missing baseline BP prediction directory; cannot define profile grid.")
    path = _prediction_path(baseline_bp_dir, hex_id)
    with rasterio.open(path) as src:
        return src.profile.copy()


def _load_barrier_layers_on_prediction_grid(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
) -> BarrierProfileLayers:
    """Load raw fuel/firezone layers aligned to the stitched prediction grid."""

    reference_profile = _prediction_reference_profile(prediction_dirs, hex_id)
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


def _finite_values(data: np.ndarray | np.ma.MaskedArray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.ma.asarray(data)
    values = np.asarray(arr.filled(np.nan), dtype=np.float64)
    return values, np.isfinite(values)


def _endpoint_array(
    prediction_dirs: dict[tuple[str, str], Path],
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> np.ma.MaskedArray:
    if endpoint != "hazard_bp_x_fi":
        prediction_dir = prediction_dirs.get((scenario, endpoint))
        if prediction_dir is None:
            raise KeyError(f"Missing prediction directory for scenario={scenario!r}, endpoint={endpoint!r}.")
        return _read_prediction(_prediction_path(prediction_dir, hex_id))

    bp_dir = prediction_dirs.get((scenario, "bp"))
    fi_dir = prediction_dirs.get((scenario, "fi"))
    if bp_dir is None or fi_dir is None:
        raise KeyError(f"Missing BP/FI prediction directories for hazard scenario={scenario!r}.")
    return _read_prediction(_prediction_path(bp_dir, hex_id)) * _read_prediction(_prediction_path(fi_dir, hex_id))


def _fixed_flow_direction_for_scenario(experiment_dir: Path, scenario: str) -> float | None:
    """Return a scenario-level fixed physical flow bearing if diagnostics prove one."""

    diagnostics_path = experiment_dir / "scenario_ood_diagnostics.csv"
    if not diagnostics_path.exists():
        return None
    diagnostics = pd.read_csv(diagnostics_path)
    rows = diagnostics[diagnostics["scenario"].astype(str).eq(scenario) & diagnostics["quantity"].astype(str).eq("flow_bearing_deg")]
    if rows.empty:
        return None

    raw_min = rows["raw_min"].to_numpy(dtype=np.float64)
    raw_max = rows["raw_max"].to_numpy(dtype=np.float64)
    raw_mean = rows["raw_mean"].to_numpy(dtype=np.float64)
    if np.all(np.isfinite(raw_min)) and np.all(np.isfinite(raw_max)) and np.allclose(raw_min, raw_max, atol=1e-9):
        return float(np.mean(raw_mean) % 360.0)
    return None


def _zone_id_from_weather_zone(value: object) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _sector_grid(
    *,
    experiment_dir: Path,
    scenario: str,
    raw_data_dir: Path,
    hex_id: str,
    firezones: np.ndarray,
    nearest_row: np.ndarray,
    nearest_col: np.ndarray,
    pixel_h_m: float,
    pixel_w_m: float,
) -> tuple[np.ndarray, str]:
    """Assign each pixel to a barrier-relative wind sector.

    Wind scenarios with a fixed flow direction use the scenario intervention
    direction. Other scenarios use baseline raw-weather zone-specific dominant
    flow directions.
    """

    h, w = firezones.shape
    row_idx, col_idx = np.indices((h, w), dtype=np.int32)
    fixed_flow_direction = _fixed_flow_direction_for_scenario(experiment_dir, scenario)
    if fixed_flow_direction is not None:
        sectors = assign_directional_sectors(
            row_idx,
            col_idx,
            nearest_row,
            nearest_col,
            fixed_flow_direction,
            pixel_h_m,
            pixel_w_m,
        )
        return sectors, "scenario_fixed_flow_direction"

    sectors = np.full(firezones.shape, -1, dtype=np.int8)
    wind_df = compute_zone_wind_consistency(raw_weather_path(raw_data_dir, hex_id))
    for row in wind_df.itertuples(index=False):
        zone_id = _zone_id_from_weather_zone(getattr(row, "zone"))
        if zone_id is None:
            continue
        dominant_dir = float(getattr(row, "dominant_direction_deg"))
        if not np.isfinite(dominant_dir):
            continue
        in_zone = firezones == zone_id
        if not in_zone.any():
            continue
        zone_sectors = assign_directional_sectors(
            row_idx,
            col_idx,
            nearest_row,
            nearest_col,
            dominant_dir,
            pixel_h_m,
            pixel_w_m,
        )
        sectors[in_zone] = zone_sectors[in_zone]
    return sectors, "baseline_zone_dominant_flow_direction"


def _collapsed_sector_values(sector_grid: np.ndarray) -> np.ndarray:
    labels = np.full(sector_grid.shape, "", dtype=object)
    labels[sector_grid == 0] = "downwind_of_barrier"
    labels[(sector_grid == 1) | (sector_grid == 3)] = SECTOR_CROSSWIND
    labels[sector_grid == 2] = "upwind_of_barrier"
    return labels


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
    """Decompose Δ(BP×FI) exactly into BP, FI, and interaction terms."""

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


@dataclass(frozen=True)
class BarrierProfileContext:
    prediction_dirs: dict[tuple[str, str], Path]
    layers: BarrierProfileLayers
    dist_m: np.ndarray
    nearest_row: np.ndarray
    nearest_col: np.ndarray
    analysis_mask: np.ndarray


def _barrier_profile_context(
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str,
) -> BarrierProfileContext:
    prediction_dirs = _prediction_dirs(experiment_dir)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    layers = _load_barrier_layers_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    nonfuel_mask = np.isin(layers.fuel, fuel_info.nonfuel_ids)
    dist_m, nearest_row, nearest_col = compute_distance_fields(
        nonfuel_mask,
        layers.pixel_h_m,
        layers.pixel_w_m,
        return_nearest_indices=True,
    )
    if nearest_row is None or nearest_col is None:
        raise RuntimeError("Nearest-barrier indices were not returned.")

    analysis_mask = (
        (layers.fuel != FUEL_NODATA)
        & ~np.isin(layers.fuel, fuel_info.nonfuel_ids)
        & ~np.isin(layers.fuel, fuel_info.restricted_ids)
        & (layers.firezones != ZONE_NODATA)
        & (dist_m > 0.0)
    )
    return BarrierProfileContext(
        prediction_dirs=prediction_dirs,
        layers=layers,
        dist_m=dist_m,
        nearest_row=nearest_row,
        nearest_col=nearest_col,
        analysis_mask=analysis_mask,
    )


def compute_barrier_relative_profiles(
    experiment_dir: Path,
    raw_data_dir: Path,
    *,
    hex_id: str = "16",
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
) -> pd.DataFrame:
    """Compute counterfactual delta profiles by distance and wind sector."""

    context = _barrier_profile_context(experiment_dir, raw_data_dir, hex_id)
    prediction_dirs = context.prediction_dirs
    layers = context.layers

    scenarios = [scenario for scenario in SCENARIO_ORDER if any(key[0] == scenario for key in prediction_dirs)]
    extra_scenarios = sorted({scenario for scenario, _ in prediction_dirs if scenario not in {"baseline", *SCENARIO_ORDER}})
    scenarios.extend(extra_scenarios)

    endpoints = [endpoint for endpoint in ENDPOINT_ORDER if endpoint == "hazard_bp_x_fi" or ("baseline", endpoint) in prediction_dirs]
    rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        sector_grid, sector_source = _sector_grid(
            experiment_dir=experiment_dir,
            scenario=scenario,
            raw_data_dir=raw_data_dir,
            hex_id=hex_id,
            firezones=layers.firezones,
            nearest_row=context.nearest_row,
            nearest_col=context.nearest_col,
            pixel_h_m=layers.pixel_h_m,
            pixel_w_m=layers.pixel_w_m,
        )
        sector_labels = _collapsed_sector_values(sector_grid)
        for endpoint in endpoints:
            try:
                baseline = _endpoint_array(prediction_dirs, "baseline", endpoint, hex_id)
                scenario_arr = _endpoint_array(prediction_dirs, scenario, endpoint, hex_id)
            except KeyError:
                continue
            rows.append(
                profile_from_arrays(
                    baseline=baseline,
                    scenario=scenario_arr,
                    analysis_mask=context.analysis_mask,
                    dist_m=context.dist_m,
                    sector_labels=sector_labels,
                    bin_edges_m=bin_edges_m,
                    scenario_name=scenario,
                    endpoint=endpoint,
                    hex_id=hex_id,
                    sector_source=sector_source,
                )
            )

    nonempty = [frame for frame in rows if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def compute_hazard_decomposition_profiles(
    experiment_dir: Path,
    raw_data_dir: Path,
    *,
    hex_id: str = "16",
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
) -> pd.DataFrame:
    """Compute exact BP/FI decomposition of hazard deltas by distance and sector."""

    context = _barrier_profile_context(experiment_dir, raw_data_dir, hex_id)
    prediction_dirs = context.prediction_dirs
    scenarios = [scenario for scenario in SCENARIO_ORDER if (scenario, "bp") in prediction_dirs and (scenario, "fi") in prediction_dirs]
    baseline_bp = _endpoint_array(prediction_dirs, "baseline", "bp", hex_id)
    baseline_fi = _endpoint_array(prediction_dirs, "baseline", "fi", hex_id)

    rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        sector_grid, sector_source = _sector_grid(
            experiment_dir=experiment_dir,
            scenario=scenario,
            raw_data_dir=raw_data_dir,
            hex_id=hex_id,
            firezones=context.layers.firezones,
            nearest_row=context.nearest_row,
            nearest_col=context.nearest_col,
            pixel_h_m=context.layers.pixel_h_m,
            pixel_w_m=context.layers.pixel_w_m,
        )
        rows.append(
            hazard_decomposition_from_arrays(
                baseline_bp=baseline_bp,
                baseline_fi=baseline_fi,
                scenario_bp=_endpoint_array(prediction_dirs, scenario, "bp", hex_id),
                scenario_fi=_endpoint_array(prediction_dirs, scenario, "fi", hex_id),
                analysis_mask=context.analysis_mask,
                dist_m=context.dist_m,
                sector_labels=_collapsed_sector_values(sector_grid),
                bin_edges_m=bin_edges_m,
                scenario_name=scenario,
                hex_id=hex_id,
                sector_source=sector_source,
            )
        )

    nonempty = [frame for frame in rows if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _ordered_labels(profile_df: pd.DataFrame) -> list[str]:
    labels = profile_df[["dist_bin_idx", "dist_bin"]].drop_duplicates().sort_values("dist_bin_idx")["dist_bin"].tolist()
    return [str(label) for label in labels]


def _bin_x_lookup(profile_df: pd.DataFrame) -> tuple[dict[int, int], list[str]]:
    ordered = profile_df[["dist_bin_idx", "dist_bin"]].drop_duplicates().sort_values("dist_bin_idx")
    labels = ordered["dist_bin"].astype(str).tolist()
    x_by_bin = {int(row.dist_bin_idx): idx for idx, row in enumerate(ordered.itertuples(index=False))}
    return x_by_bin, labels


def plot_hazard_profile_by_sector(profile_df: pd.DataFrame, out_dir: Path) -> Path | None:
    hazard = profile_df[profile_df["endpoint"].eq("hazard_bp_x_fi")].copy()
    hazard = hazard[hazard["scenario"].isin(SCENARIO_ORDER)]
    if hazard.empty:
        return None

    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(hazard["scenario"])]
    x_by_bin, labels = _bin_x_lookup(hazard)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 4.5), sharey=True)
    if len(scenarios) == 1:
        axes = np.array([axes])

    for ax, scenario in zip(axes, scenarios, strict=False):
        sub = hazard[hazard["scenario"].eq(scenario)]
        for sector in PROFILE_SECTORS:
            line = sub[sub["sector"].eq(sector)].sort_values("dist_bin_idx")
            if line.empty:
                continue
            x = line["dist_bin_idx"].map(x_by_bin).to_numpy(dtype=float)
            ax.plot(
                x,
                line["delta_mean"],
                marker="o",
                linewidth=2.0 if sector == SECTOR_ALL else 1.5,
                color=SECTOR_COLOURS[sector],
                label=DISPLAY_SECTOR[sector],
            )
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.9)
        ax.set_title(DISPLAY_SCENARIO.get(scenario, scenario))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Distance to original non-fuel barrier")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Mean Δhazard  (scenario − baseline)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Barrier-relative counterfactual hazard profile", y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "barrier_relative_hazard_profile_by_sector.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_endpoint_profile_all_pixels(profile_df: pd.DataFrame, out_dir: Path) -> Path | None:
    sub = profile_df[profile_df["sector"].eq(SECTOR_ALL)].copy()
    sub = sub[sub["scenario"].isin(SCENARIO_ORDER)]
    if sub.empty:
        return None

    endpoints = [endpoint for endpoint in ENDPOINT_ORDER if endpoint in set(sub["endpoint"])]
    x_by_bin, labels = _bin_x_lookup(sub)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes_flat = axes.ravel()
    for ax, endpoint in zip(axes_flat, endpoints, strict=False):
        endpoint_df = sub[sub["endpoint"].eq(endpoint)]
        for scenario in SCENARIO_ORDER:
            line = endpoint_df[endpoint_df["scenario"].eq(scenario)].sort_values("dist_bin_idx")
            if line.empty:
                continue
            x = line["dist_bin_idx"].map(x_by_bin).to_numpy(dtype=float)
            ax.plot(
                x,
                line["delta_mean"],
                marker="o",
                linewidth=1.7,
                color=SCENARIO_COLOURS.get(scenario, "0.4"),
                label=DISPLAY_SCENARIO.get(scenario, scenario),
            )
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.9)
        ax.set_title(DISPLAY_ENDPOINT.get(endpoint, endpoint))
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
    for ax in axes_flat[len(endpoints) :]:
        ax.axis("off")
    axes_flat[0].legend(frameon=False, loc="best")
    fig.supxlabel("Distance to original non-fuel barrier")
    fig.supylabel("Mean Δ endpoint  (scenario − baseline)")
    fig.suptitle("Barrier-relative endpoint profiles, all sectors combined", y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "barrier_relative_endpoint_profile_all_pixels.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _profile_x_positions(profile_df: pd.DataFrame) -> tuple[list[float], list[str]]:
    ordered = profile_df[["dist_bin_idx", "dist_bin", "dist_min_m", "dist_max_m"]].drop_duplicates().sort_values("dist_bin_idx")
    x_positions: list[float] = []
    labels: list[str] = []
    for row in ordered.itertuples(index=False):
        lo = float(row.dist_min_m)
        hi = float(row.dist_max_m)
        if np.isinf(hi):
            hi = lo * 1.5
        x_positions.append((lo + hi) / 2.0)
        labels.append(str(row.dist_bin))
    return x_positions, labels


def plot_remove_barriers_fi_hazard_response_profile(profile_df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Plot the focused communication profile: ΔFI/Δhazard versus barrier distance."""

    scenario = "remove_barriers_adjacent_modal"
    sub = profile_df[
        profile_df["scenario"].eq(scenario) & profile_df["sector"].eq(SECTOR_ALL) & profile_df["endpoint"].isin(["fi", "hazard_bp_x_fi"])
    ].copy()
    if sub.empty:
        return None

    x_positions, labels = _profile_x_positions(sub)
    x_by_bin = {
        int(row.dist_bin_idx): x
        for x, row in zip(
            x_positions,
            sub[["dist_bin_idx", "dist_bin", "dist_min_m", "dist_max_m"]]
            .drop_duplicates()
            .sort_values("dist_bin_idx")
            .itertuples(index=False),
            strict=False,
        )
    }
    panels = [
        ("fi", "Mean ΔFI", "ΔFI = counterfactual − baseline", "#d95f02"),
        ("hazard_bp_x_fi", "Mean Δhazard", "Δhazard = counterfactual − baseline", "#7b3294"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for ax, (endpoint, title, ylabel, colour) in zip(axes, panels, strict=False):
        line = sub[sub["endpoint"].eq(endpoint)].sort_values("dist_bin_idx")
        if line.empty:
            ax.axis("off")
            continue
        x = np.array([x_by_bin[int(idx)] for idx in line["dist_bin_idx"]], dtype=float)
        mean = line["delta_mean"].to_numpy(dtype=float)
        p25 = line["delta_p25"].to_numpy(dtype=float)
        p75 = line["delta_p75"].to_numpy(dtype=float)
        ax.fill_between(x, p25, p75, color=colour, alpha=0.18, linewidth=0.0, label="IQR")
        ax.plot(x, mean, color=colour, marker="o", linewidth=2.2, label="Mean")
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for xi, yi, n_pixels in zip(x, mean, line["n_pixels"], strict=False):
            ax.annotate(
                f"n={int(n_pixels)/1_000_000:.1f}M",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="0.25",
            )
        ax.legend(frameon=False, loc="best")

    axes[-1].set_xscale("log")
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(labels, rotation=25, ha="right")
    axes[-1].set_xlabel("Distance to original non-fuel barrier")
    fig.suptitle("Barrier-relative response to local non-fuel replacement", y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remove_barriers_barrier_relative_fi_hazard_profile.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_wind_speed50_endpoint_response_profile(profile_df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Plot BP, FI, and ROS distance profiles for the extreme speed-only wind intervention."""

    scenario = "wind_speed50_original_direction"
    sub = profile_df[
        profile_df["scenario"].eq(scenario) & profile_df["sector"].eq(SECTOR_ALL) & profile_df["endpoint"].isin(["bp", "fi", "ros"])
    ].copy()
    if sub.empty:
        return None

    x_positions, labels = _profile_x_positions(sub)
    x_by_bin = {
        int(row.dist_bin_idx): x
        for x, row in zip(
            x_positions,
            sub[["dist_bin_idx", "dist_bin", "dist_min_m", "dist_max_m"]]
            .drop_duplicates()
            .sort_values("dist_bin_idx")
            .itertuples(index=False),
            strict=False,
        )
    }
    panels = [
        ("bp", "Mean ΔBP", "ΔBP = WindSpeed 50 − baseline", "#7570b3"),
        ("fi", "Mean ΔFI", "ΔFI = WindSpeed 50 − baseline", "#1b9e77"),
        ("ros", "Mean ΔROS", "ΔROS = WindSpeed 50 − baseline", "#d95f02"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.0), sharex=True)
    for ax, (endpoint, title, ylabel, colour) in zip(axes, panels, strict=False):
        line = sub[sub["endpoint"].eq(endpoint)].sort_values("dist_bin_idx")
        if line.empty:
            ax.axis("off")
            continue
        x = np.array([x_by_bin[int(idx)] for idx in line["dist_bin_idx"]], dtype=float)
        mean = line["delta_mean"].to_numpy(dtype=float)
        p25 = line["delta_p25"].to_numpy(dtype=float)
        p75 = line["delta_p75"].to_numpy(dtype=float)
        ax.fill_between(x, p25, p75, color=colour, alpha=0.18, linewidth=0.0, label="IQR")
        ax.plot(x, mean, color=colour, marker="o", linewidth=2.2, label="Mean")
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="best")

    axes[-1].set_xscale("log")
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(labels, rotation=25, ha="right")
    axes[-1].set_xlabel("Distance to original non-fuel barrier")
    fig.suptitle("Extreme wind-speed response by endpoint", y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wind_speed50_bp_fi_ros_barrier_relative_profile.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _safe_share(numerator: float, denominator: float) -> float:
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(numerator / denominator)


def cumulative_hazard_delta_share(
    profile_df: pd.DataFrame,
    *,
    scenario: str = "remove_barriers_adjacent_modal",
    endpoint: str = "hazard_bp_x_fi",
    thresholds_m: tuple[float, ...] = LOCALIZATION_THRESHOLDS_M,
) -> pd.DataFrame:
    """Summarize cumulative Δhazard shares within distance thresholds."""

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


def write_cumulative_hazard_delta_share(
    profile_df: pd.DataFrame,
    experiment_dir: Path,
    *,
    scenario: str = "remove_barriers_adjacent_modal",
    endpoint: str = "hazard_bp_x_fi",
) -> Path:
    summary = cumulative_hazard_delta_share(profile_df, scenario=scenario, endpoint=endpoint)
    out_path = experiment_dir / "counterfactual_remove_barriers_hazard_delta_localization_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def plot_cumulative_hazard_delta_share(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    labels = summary_df["threshold_label"].astype(str).tolist()
    values = summary_df["percent_of_total_delta_hazard"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(labels, values, color="#7b3294", width=0.62)
    for bar, value in zip(bars, values, strict=False):
        if not np.isfinite(value):
            continue
        ax.annotate(
            f"{value:.1f}%",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5 if value >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    finite_values = values[np.isfinite(values)]
    ymax = max(100.0, float(np.max(finite_values)) + 8.0) if finite_values.size else 100.0
    ymin = min(0.0, float(np.min(finite_values)) - 8.0) if finite_values.size else 0.0
    ax.set_ylim(ymin, ymax)
    ax.axhline(0.0, color="0.45", linewidth=0.9)
    ax.set_ylabel("% of total Δhazard")
    ax.set_xlabel("Distance to original non-fuel barrier")
    ax.set_title("Cumulative share of hazard change near barriers")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remove_barriers_hazard_delta_localization.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def relative_hazard_decomposition_summary(
    decomposition_df: pd.DataFrame,
    *,
    scenario: str = "remove_barriers_adjacent_modal",
) -> pd.DataFrame:
    """Summarize relative Δhazard contributions as summed terms divided by summed baseline hazard."""

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


def write_relative_hazard_decomposition_summary(
    decomposition_df: pd.DataFrame,
    experiment_dir: Path,
    *,
    scenario: str = "remove_barriers_adjacent_modal",
) -> Path:
    summary = relative_hazard_decomposition_summary(decomposition_df, scenario=scenario)
    out_path = experiment_dir / "counterfactual_remove_barriers_relative_hazard_decomposition_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def plot_relative_hazard_decomposition(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    summary = summary_df.sort_values("dist_bin_idx")
    labels = summary["dist_bin"].astype(str).tolist()
    x = np.arange(len(summary))
    terms = [
        ("relative_bp_component_percent", "BP term", "#7b3294"),
        ("relative_fi_component_percent", "FI term", "#008837"),
        ("relative_interaction_component_percent", "Interaction", "#fdb863"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bottoms_positive = np.zeros(len(summary), dtype=float)
    bottoms_negative = np.zeros(len(summary), dtype=float)
    all_values: list[np.ndarray] = []
    for column, label, colour in terms:
        values = summary[column].to_numpy(dtype=float)
        all_values.append(values)
        bottoms = np.where(values >= 0.0, bottoms_positive, bottoms_negative)
        ax.bar(x, values, bottom=bottoms, color=colour, label=label, width=0.72)
        bottoms_positive += np.where(values >= 0.0, values, 0.0)
        bottoms_negative += np.where(values < 0.0, values, 0.0)

    total = summary["relative_delta_hazard_percent"].to_numpy(dtype=float)
    ax.plot(x, total, color="black", marker="o", linewidth=1.9, label="Total ΔH / H0")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Distance to original non-fuel barrier")
    ax.set_ylabel("% of baseline hazard in distance bin")
    ax.set_title("Relative hazard-change drivers after local non-fuel replacement")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    finite_values = np.concatenate([*all_values, total])
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        y_min = min(0.0, float(np.min(finite_values)) - 5.0)
        y_max = max(0.0, float(np.max(finite_values)) + 5.0)
        ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "remove_barriers_relative_hazard_decomposition.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hazard_decomposition(decomposition_df: pd.DataFrame, out_dir: Path) -> Path | None:
    sub = decomposition_df[decomposition_df["scenario"].eq("wind_from_west_p95") & decomposition_df["sector"].eq(SECTOR_ALL)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("dist_bin_idx")
    labels = sub["dist_bin"].astype(str).tolist()
    x = np.arange(len(labels))
    terms = [
        ("bp_component_mean", "ΔBP × baseline FI", "#7b3294"),
        ("fi_component_mean", "baseline BP × ΔFI", "#008837"),
        ("interaction_component_mean", "ΔBP × ΔFI", "#fdb863"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottoms_positive = np.zeros(len(sub), dtype=float)
    bottoms_negative = np.zeros(len(sub), dtype=float)
    for column, label, colour in terms:
        values = sub[column].to_numpy(dtype=float)
        bottoms = np.where(values >= 0.0, bottoms_positive, bottoms_negative)
        ax.bar(x, values, bottom=bottoms, color=colour, label=label, width=0.72)
        bottoms_positive += np.where(values >= 0.0, values, 0.0)
        bottoms_negative += np.where(values < 0.0, values, 0.0)

    ax.plot(x, sub["delta_hazard_mean"], color="black", marker="o", linewidth=1.8, label="Total Δhazard")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Distance to original non-fuel barrier")
    ax.set_ylabel("Mean contribution to Δhazard")
    ax.set_title("Wind-from-west p95 hazard decrease decomposition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wind_from_west_hazard_decomposition_all_pixels.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_barrier_relative_profiles(
    experiment_dir: Path,
    raw_data_dir: Path,
    *,
    hex_id: str = "16",
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
) -> tuple[Path, Path, Path, Path, list[Path]]:
    profiles = compute_barrier_relative_profiles(
        experiment_dir,
        raw_data_dir,
        hex_id=hex_id,
        bin_edges_m=bin_edges_m,
    )
    decomposition = compute_hazard_decomposition_profiles(
        experiment_dir,
        raw_data_dir,
        hex_id=hex_id,
        bin_edges_m=bin_edges_m,
    )
    out_path = experiment_dir / "counterfactual_barrier_relative_profiles.csv"
    decomposition_path = experiment_dir / "counterfactual_hazard_decomposition_profiles.csv"
    profiles.to_csv(out_path, index=False)
    decomposition.to_csv(decomposition_path, index=False)

    localization_path = write_cumulative_hazard_delta_share(profiles, experiment_dir)
    localization = pd.read_csv(localization_path)
    relative_decomposition_path = write_relative_hazard_decomposition_summary(decomposition, experiment_dir)
    relative_decomposition = pd.read_csv(relative_decomposition_path)

    plot_dir = experiment_dir / "plots"
    plot_paths = [
        path
        for path in (
            plot_remove_barriers_fi_hazard_response_profile(profiles, plot_dir),
            plot_cumulative_hazard_delta_share(localization, plot_dir),
            plot_relative_hazard_decomposition(relative_decomposition, plot_dir),
            plot_hazard_profile_by_sector(profiles, plot_dir),
            plot_endpoint_profile_all_pixels(profiles, plot_dir),
            plot_hazard_decomposition(decomposition, plot_dir),
            plot_wind_speed50_endpoint_response_profile(profiles, plot_dir),
        )
        if path is not None
    ]
    return out_path, decomposition_path, localization_path, relative_decomposition_path, plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot barrier-relative counterfactual profiles.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--hex_id", default="16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, decomposition_path, localization_path, relative_decomposition_path, plot_paths = write_barrier_relative_profiles(
        args.experiment_dir,
        args.raw_data_dir,
        hex_id=str(args.hex_id).zfill(2),
    )
    print(f"Wrote barrier-relative profiles: {csv_path}")
    print(f"Wrote hazard decomposition profiles: {decomposition_path}")
    print(f"Wrote hazard localization summary: {localization_path}")
    print(f"Wrote relative hazard decomposition summary: {relative_decomposition_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
