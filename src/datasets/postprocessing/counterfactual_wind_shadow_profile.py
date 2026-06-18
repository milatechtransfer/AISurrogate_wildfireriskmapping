"""Zone-wise wind-relative endpoint profiles around non-fuel barriers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_hazard_map import (
    endpoint_prediction,
    ground_truth_endpoint_on_prediction_grid,
    mask_scope_on_prediction_grid,
    prediction_dirs_from_index,
    prediction_reference_profile,
)
from src.datasets.postprocessing.diagnose_bp_barrier_halo import (
    DEFAULT_DIST_BIN_EDGES_M,
    ZONE_NODATA,
    assign_directional_sectors,
    compute_distance_fields,
    compute_zone_wind_consistency,
    dist_bin_label,
    parse_fuel_barrier_info,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SECTOR_DOWNWIND = "downwind_of_barrier"
SECTOR_CROSSWIND = "crosswind"
SECTOR_UPWIND = "upwind_of_barrier"
PROFILE_SECTORS: tuple[str, ...] = (SECTOR_DOWNWIND, SECTOR_CROSSWIND, SECTOR_UPWIND)
ENDPOINT_ORDER: tuple[str, ...] = ("bp", "fi", "ros")
ENDPOINT_LABELS = {
    "bp": "BP",
    "fi": "FI",
    "ros": "ROS",
}
MODEL_SCENARIOS: tuple[str, ...] = (
    "wind_zone_consistent_direction",
    "wind_speed50_original_direction",
    "wind_speed50_zone_consistent_direction",
    "wind_speed100_zone_consistent_direction",
)
MODEL_SCENARIO_SOURCE = {
    "wind_zone_consistent_direction": "model_consistent_direction",
    "wind_speed50_original_direction": "model_speed50_original_direction",
    "wind_speed50_zone_consistent_direction": "model_speed50_consistent_direction",
    "wind_speed100_zone_consistent_direction": "model_speed100_consistent_direction",
}
SOURCE_ORDER: tuple[str, ...] = (
    "gt",
    "model_baseline",
    "model_consistent_direction",
    "model_speed50_original_direction",
    "model_speed50_consistent_direction",
    "model_speed100_consistent_direction",
)
SOURCE_LABELS = {
    "gt": "GT",
    "model_baseline": "Model baseline",
    "model_consistent_direction": "Model consistent direction",
    "model_speed50_original_direction": "Model speed50 original directions",
    "model_speed50_consistent_direction": "Model speed50 + consistent direction",
    "model_speed100_consistent_direction": "Model speed100 + consistent direction",
}
INTERVENTION_PLOT_ORDER: tuple[tuple[str, str, str], ...] = (
    ("model_speed50_original_direction", "speed50_original_direction", "Speed50 with original directions"),
    ("model_consistent_direction", "consistent_direction", "Consistent direction"),
    ("model_speed50_consistent_direction", "speed50_consistent_direction", "Speed50 + consistent direction"),
    ("model_speed100_consistent_direction", "speed100_consistent_direction", "Speed100 + consistent direction"),
)
SECTOR_COLOURS = {
    SECTOR_DOWNWIND: "#d73027",
    SECTOR_CROSSWIND: "#4575b4",
    SECTOR_UPWIND: "#1a9850",
}
SECTOR_LABELS = {
    SECTOR_DOWNWIND: "Downwind",
    SECTOR_CROSSWIND: "Crosswind",
    SECTOR_UPWIND: "Upwind",
}


def collapsed_sector_labels(sector_indices: np.ndarray) -> np.ndarray:
    labels = np.full(sector_indices.shape, "", dtype=object)
    labels[sector_indices == 0] = SECTOR_DOWNWIND
    labels[(sector_indices == 1) | (sector_indices == 3)] = SECTOR_CROSSWIND
    labels[sector_indices == 2] = SECTOR_UPWIND
    return labels


def endpoint_sector_distance_profile(
    *,
    values: np.ndarray,
    analysis_mask: np.ndarray,
    dist_m: np.ndarray,
    sector_labels: np.ndarray,
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
    source: str,
    endpoint: str,
    hex_id: str,
    zone: str,
    zone_id: int,
) -> pd.DataFrame:
    """Summarize an endpoint by distance-to-barrier bin and wind-relative sector."""

    if not (values.shape == analysis_mask.shape == dist_m.shape == sector_labels.shape):
        raise ValueError("values, analysis_mask, dist_m, and sector_labels must have matching shapes.")

    valid = analysis_mask & np.isfinite(values) & np.isfinite(dist_m) & (dist_m > 0.0)
    if not valid.any():
        return pd.DataFrame()

    edges = [0.0, *bin_edges_m, np.inf]
    rows: list[dict] = []
    for sector in PROFILE_SECTORS:
        sector_mask = valid & (sector_labels == sector)
        if not sector_mask.any():
            continue
        for bin_idx in range(len(edges) - 1):
            lo = edges[bin_idx]
            hi = edges[bin_idx + 1]
            in_bin = sector_mask & (dist_m >= lo) if np.isinf(hi) else sector_mask & (dist_m >= lo) & (dist_m < hi)
            if not in_bin.any():
                continue
            bin_values = values[in_bin].astype(np.float64)
            rows.append(
                {
                    "hex_id": hex_id,
                    "zone": zone,
                    "zone_id": int(zone_id),
                    "endpoint": endpoint,
                    "source": source,
                    "sector": sector,
                    "dist_bin_idx": bin_idx,
                    "dist_bin": dist_bin_label(bin_idx, bin_edges_m),
                    "dist_min_m": lo,
                    "dist_max_m": hi,
                    "n_pixels": int(bin_values.size),
                    "value_mean": float(np.mean(bin_values)),
                    "value_median": float(np.median(bin_values)),
                    "value_p25": float(np.percentile(bin_values, 25)),
                    "value_p75": float(np.percentile(bin_values, 75)),
                    "value_p95": float(np.percentile(bin_values, 95)),
                }
            )
    return pd.DataFrame(rows)


def bp_sector_distance_profile(
    *,
    bp: np.ndarray,
    analysis_mask: np.ndarray,
    dist_m: np.ndarray,
    sector_labels: np.ndarray,
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
    source: str,
    hex_id: str,
    zone: str,
    zone_id: int,
) -> pd.DataFrame:
    """Summarize BP by distance-to-barrier bin and wind-relative sector."""

    profile = endpoint_sector_distance_profile(
        values=bp,
        analysis_mask=analysis_mask,
        dist_m=dist_m,
        sector_labels=sector_labels,
        bin_edges_m=bin_edges_m,
        source=source,
        endpoint="bp",
        hex_id=hex_id,
        zone=zone,
        zone_id=zone_id,
    )
    if profile.empty:
        return profile
    return profile.rename(
        columns={
            "value_mean": "bp_mean",
            "value_median": "bp_median",
            "value_p25": "bp_p25",
            "value_p75": "bp_p75",
            "value_p95": "bp_p95",
        }
    )


def largest_component_summary(mask: np.ndarray) -> dict[str, int]:
    if not mask.any():
        return {
            "largest_component_pixels": 0,
            "largest_component_bbox_row_min": -1,
            "largest_component_bbox_row_max": -1,
            "largest_component_bbox_col_min": -1,
            "largest_component_bbox_col_max": -1,
        }

    labels, _ = label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.int8))
    slices = find_objects(labels)
    best: tuple[int, int, int, int, int] | None = None
    for component_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        row_slice, col_slice = component_slice
        component_pixels = int((labels[row_slice, col_slice] == component_id).sum())
        candidate = (
            component_pixels,
            int(row_slice.start),
            int(row_slice.stop),
            int(col_slice.start),
            int(col_slice.stop),
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    return {
        "largest_component_pixels": best[0],
        "largest_component_bbox_row_min": best[1],
        "largest_component_bbox_row_max": best[2],
        "largest_component_bbox_col_min": best[3],
        "largest_component_bbox_col_max": best[4],
    }


def weighted_mean_from_profile(profile_df: pd.DataFrame, value_column: str = "value_mean") -> float:
    if profile_df.empty:
        return float("nan")
    weights = profile_df["n_pixels"].to_numpy(dtype=np.float64)
    values = profile_df[value_column].to_numpy(dtype=np.float64)
    if weights.sum() <= 0.0:
        return float("nan")
    return float(np.average(values, weights=weights))


def summarize_near_band(
    profile_df: pd.DataFrame,
    *,
    near_min_m: float = 100.0,
    near_max_m: float = 500.0,
) -> pd.DataFrame:
    """Return weighted 100-500 m sector means and downwind/upwind ratios."""

    value_column = "value_mean" if "value_mean" in profile_df.columns else "bp_mean"
    rows: list[dict] = []
    band = profile_df[(profile_df["dist_min_m"].astype(float) >= near_min_m) & (profile_df["dist_max_m"].astype(float) <= near_max_m)]
    group_columns = ["hex_id", "zone", "zone_id", "source"]
    if "endpoint" in band.columns:
        group_columns.insert(3, "endpoint")
    for group_key, group in band.groupby(group_columns, sort=True):
        group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,), strict=False))
        means = {}
        counts = {}
        for sector in PROFILE_SECTORS:
            sector_df = group[group["sector"].eq(sector)]
            means[sector] = weighted_mean_from_profile(sector_df, value_column=value_column)
            counts[sector] = int(sector_df["n_pixels"].sum()) if not sector_df.empty else 0
        downwind = means[SECTOR_DOWNWIND]
        upwind = means[SECTOR_UPWIND]
        row = {
            "hex_id": group_values["hex_id"],
            "zone": group_values["zone"],
            "zone_id": int(group_values["zone_id"]),
            "source": group_values["source"],
            "distance_band_m": f"{near_min_m:g}-{near_max_m:g}",
            "downwind_n": counts[SECTOR_DOWNWIND],
            "crosswind_n": counts[SECTOR_CROSSWIND],
            "upwind_n": counts[SECTOR_UPWIND],
            "downwind_value_mean": downwind,
            "crosswind_value_mean": means[SECTOR_CROSSWIND],
            "upwind_value_mean": upwind,
            "downwind_minus_upwind": downwind - upwind,
            "downwind_over_upwind": downwind / upwind if np.isfinite(upwind) and upwind != 0.0 else float("nan"),
        }
        if value_column == "bp_mean":
            row["downwind_bp_mean"] = downwind
            row["crosswind_bp_mean"] = means[SECTOR_CROSSWIND]
            row["upwind_bp_mean"] = upwind
        if "endpoint" in group_values:
            row["endpoint"] = group_values["endpoint"]
        rows.append(row)
    return pd.DataFrame(rows)


def _zone_id_from_name(zone_name: str) -> int | None:
    digits = "".join(ch for ch in str(zone_name) if ch.isdigit())
    return int(digits) if digits else None


def _firezone_lookup(paths: Paths, hex_id: str) -> dict[str, int]:
    table = pd.read_csv(paths.firezones_table(hex_id))
    if not {"Name", "ID"}.issubset(table.columns):
        raise ValueError(f"{paths.firezones_table(hex_id)} must contain Name and ID columns.")
    return {str(row.Name): int(row.ID) for row in table.itertuples(index=False)}


def compute_zone_wind_shadow_profiles(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str = "16",
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
    min_zone_pixels: int = 10_000,
    min_nonfuel_pixels: int = 10_000,
    min_largest_component_pixels: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute zone-wise wind-relative endpoint profiles around all non-fuel barriers in each usable zone."""

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    actual_mask = mask_scope_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
        mask_scope="actual",
    )

    firezones_ma, _ = load_spatial_raster(paths.firezones_grid(hex_id), reference_profile=reference_profile)
    fuel_ma, fuel_profile = load_spatial_raster(paths.fuel_grid(hex_id), reference_profile=reference_profile)
    firezones = np.asarray(np.ma.asarray(firezones_ma).astype(np.int32).filled(ZONE_NODATA), dtype=np.int32)
    fuel = np.asarray(np.ma.asarray(fuel_ma).astype(np.int32).filled(-32768), dtype=np.int32)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    nonfuel = np.isin(fuel, fuel_info.nonfuel_ids) & actual_mask
    restricted = np.isin(fuel, fuel_info.restricted_ids) & actual_mask

    endpoint_arrays: dict[tuple[str, str], np.ndarray] = {}
    for endpoint in ENDPOINT_ORDER:
        endpoint_arrays[(endpoint, "gt")] = np.asarray(
            np.ma.asarray(
                ground_truth_endpoint_on_prediction_grid(
                    raw_data_dir=raw_data_dir,
                    prediction_dirs=prediction_dirs,
                    hex_id=hex_id,
                    endpoint=endpoint,
                )
            ).filled(np.nan),
            dtype=np.float64,
        )
        endpoint_arrays[(endpoint, "model_baseline")] = np.asarray(
            np.ma.asarray(endpoint_prediction(prediction_dirs, "baseline", endpoint, hex_id)).filled(np.nan),
            dtype=np.float64,
        )
        for scenario in MODEL_SCENARIOS:
            try:
                scenario_values = endpoint_prediction(prediction_dirs, scenario, endpoint, hex_id)
            except (KeyError, FileNotFoundError):
                continue
            endpoint_arrays[(endpoint, MODEL_SCENARIO_SOURCE[scenario])] = np.asarray(
                np.ma.asarray(scenario_values).filled(np.nan),
                dtype=np.float64,
            )

    transform = fuel_profile["transform"]
    pixel_h_m = abs(float(transform.e))
    pixel_w_m = abs(float(transform.a))
    h, w = fuel.shape
    row_idx, col_idx = np.indices((h, w), dtype=np.int32)

    wind_df = compute_zone_wind_consistency(paths.weather_table(hex_id))
    zone_lookup = _firezone_lookup(paths, hex_id)

    profile_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for wind_row in wind_df.itertuples(index=False):
        zone_name = str(wind_row.zone)
        zone_id = zone_lookup.get(zone_name, _zone_id_from_name(zone_name))
        if zone_id is None:
            continue
        zone_mask = (firezones == zone_id) & actual_mask
        zone_pixels = int(zone_mask.sum())
        zone_nonfuel = zone_mask & nonfuel
        nonfuel_pixels = int(zone_nonfuel.sum())
        largest = largest_component_summary(zone_nonfuel)
        usable = (
            zone_pixels >= min_zone_pixels
            and nonfuel_pixels >= min_nonfuel_pixels
            and largest["largest_component_pixels"] >= min_largest_component_pixels
        )
        summary_base = {
            "hex_id": hex_id,
            "zone": zone_name,
            "zone_id": int(zone_id),
            "usable": bool(usable),
            "zone_pixels": zone_pixels,
            "nonfuel_pixels": nonfuel_pixels,
            "nonfuel_density": nonfuel_pixels / zone_pixels if zone_pixels else float("nan"),
            "wind_consistency": float(wind_row.consistency),
            "dominant_from_bearing_deg": float(wind_row.dominant_from_direction_deg),
            "dominant_flow_bearing_deg": float(wind_row.dominant_direction_deg),
            **largest,
        }
        if not usable:
            summary_rows.append(summary_base)
            continue

        dist_m, nearest_row, nearest_col = compute_distance_fields(
            zone_nonfuel,
            pixel_h_m,
            pixel_w_m,
            return_nearest_indices=True,
        )
        assert nearest_row is not None and nearest_col is not None
        sectors = assign_directional_sectors(
            row_idx,
            col_idx,
            nearest_row,
            nearest_col,
            float(wind_row.dominant_direction_deg),
            pixel_h_m,
            pixel_w_m,
        )
        sector_labels = collapsed_sector_labels(sectors)
        analysis_base = zone_mask & ~nonfuel & ~restricted & (fuel != -32768) & np.isfinite(dist_m) & (dist_m > 0.0)
        for (endpoint, source), values in endpoint_arrays.items():
            profile = endpoint_sector_distance_profile(
                values=values,
                analysis_mask=analysis_base,
                dist_m=dist_m,
                sector_labels=sector_labels,
                bin_edges_m=bin_edges_m,
                source=source,
                endpoint=endpoint,
                hex_id=hex_id,
                zone=zone_name,
                zone_id=int(zone_id),
            )
            if profile.empty:
                continue
            for key, value in summary_base.items():
                if key not in profile.columns:
                    profile[key] = value
            profile_frames.append(profile)
        summary_rows.append(summary_base)

    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    near = summarize_near_band(profiles) if not profiles.empty else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not near.empty:
        summary = summary.merge(
            near,
            on=["hex_id", "zone", "zone_id"],
            how="left",
            suffixes=("", "_near"),
        )
    return profiles, summary


def _profile_for_legacy_bp(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty:
        return profile_df
    bp = profile_df[profile_df["endpoint"].eq("bp")].copy() if "endpoint" in profile_df.columns else profile_df.copy()
    if bp.empty:
        return bp
    bp["source"] = bp["source"].replace({"gt": "gt_bp", "model_baseline": "model_baseline_bp"})
    return bp.rename(
        columns={
            "value_mean": "bp_mean",
            "value_median": "bp_median",
            "value_p25": "bp_p25",
            "value_p75": "bp_p75",
            "value_p95": "bp_p95",
        }
    ).drop(columns=["endpoint"], errors="ignore")


def plot_zone_wind_shadow_profiles(
    profile_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
    *,
    hex_id: str,
    endpoint: str = "bp",
    sources: tuple[str, ...] | None = None,
    filename_suffix: str | None = None,
    title_suffix: str | None = None,
) -> Path | None:
    if profile_df.empty:
        return None
    endpoint_profile = profile_df[profile_df["endpoint"].eq(endpoint)].copy() if "endpoint" in profile_df.columns else profile_df.copy()
    if endpoint_profile.empty:
        return None

    usable_zones = summary_df[summary_df["usable"].eq(True)]["zone"].drop_duplicates().tolist()
    if not usable_zones:
        return None
    selected_sources = sources or SOURCE_ORDER
    source_order = [source for source in selected_sources if source in set(endpoint_profile["source"])]
    if not source_order:
        return None
    ordered_bins = endpoint_profile[["dist_bin_idx", "dist_bin"]].drop_duplicates().sort_values("dist_bin_idx")
    labels = ordered_bins["dist_bin"].astype(str).tolist()
    x_by_bin = {int(row.dist_bin_idx): idx for idx, row in enumerate(ordered_bins.itertuples(index=False))}

    fig, axes = plt.subplots(
        len(usable_zones),
        len(source_order),
        figsize=(5.6 * len(source_order), 3.5 * len(usable_zones)),
        squeeze=False,
        sharex=True,
    )
    for row_idx, zone in enumerate(usable_zones):
        zone_summary = summary_df[summary_df["zone"].eq(zone)].iloc[0]
        for col_idx, source in enumerate(source_order):
            ax = axes[row_idx, col_idx]
            sub = endpoint_profile[endpoint_profile["zone"].eq(zone) & endpoint_profile["source"].eq(source)]
            for sector in PROFILE_SECTORS:
                line = sub[sub["sector"].eq(sector)].sort_values("dist_bin_idx")
                if line.empty:
                    continue
                x = line["dist_bin_idx"].map(x_by_bin).to_numpy(dtype=float)
                ax.plot(
                    x,
                    line["value_mean"],
                    marker="o",
                    linewidth=2,
                    color=SECTOR_COLOURS[sector],
                    label=SECTOR_LABELS[sector],
                )
                ax.fill_between(
                    x,
                    line["value_p25"].to_numpy(dtype=float),
                    line["value_p75"].to_numpy(dtype=float),
                    color=SECTOR_COLOURS[sector],
                    alpha=0.12,
                    linewidth=0,
                )
            if row_idx == 0:
                ax.set_title(SOURCE_LABELS.get(source, source))
            if col_idx == 0:
                ax.set_ylabel(
                    f"{zone}\n{ENDPOINT_LABELS.get(endpoint, endpoint.upper())}\n"
                    f"cons={zone_summary.wind_consistency:.2f}, flow={zone_summary.dominant_flow_bearing_deg:.0f}°"
                )
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right")

    for ax in axes[-1, :]:
        ax.set_xlabel("Distance to non-fuel barrier in zone")
    axes[0, -1].legend(frameon=False, loc="best")
    endpoint_label = ENDPOINT_LABELS.get(endpoint, endpoint.upper())
    suffix_text = f": {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Hex{int(hex_id):02d} zone-wise wind-relative {endpoint_label} profiles{suffix_text}", y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = out_dir / f"hex{int(hex_id):02d}_zone_wind_shadow_{endpoint}{suffix}_profiles.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_split_zone_wind_shadow_profiles(
    profile_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
    *,
    hex_id: str,
) -> list[Path]:
    """Write one 3-column GT/baseline/intervention profile per endpoint and intervention."""

    plot_paths: list[Path] = []
    available_sources = set(profile_df["source"]) if "source" in profile_df.columns else set()
    for intervention_source, filename_suffix, title_suffix in INTERVENTION_PLOT_ORDER:
        if intervention_source not in available_sources:
            continue
        sources = ("gt", "model_baseline", intervention_source)
        for endpoint in ENDPOINT_ORDER:
            plot_path = plot_zone_wind_shadow_profiles(
                profile_df,
                summary_df,
                out_dir,
                hex_id=hex_id,
                endpoint=endpoint,
                sources=sources,
                filename_suffix=filename_suffix,
                title_suffix=title_suffix,
            )
            if plot_path is not None:
                plot_paths.append(plot_path)
    return plot_paths


def summarize_consistency_intervention(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare near-band shadow ratios before/after the direction-consistency intervention."""

    if summary_df.empty or "source" not in summary_df.columns:
        return pd.DataFrame()
    intervention_sources = [source for source in MODEL_SCENARIO_SOURCE.values() if source in set(summary_df["source"])]
    if not intervention_sources or "model_baseline" not in set(summary_df["source"]):
        return pd.DataFrame()
    sub = summary_df[summary_df["usable"].eq(True) & summary_df["source"].isin(["model_baseline", *intervention_sources])].copy()
    if sub.empty:
        return pd.DataFrame()

    baseline = sub[sub["source"].eq("model_baseline")].set_index(["endpoint", "zone"])
    rows: list[dict] = []
    for source in intervention_sources:
        intervention = sub[sub["source"].eq(source)].set_index(["endpoint", "zone"])
        shared_index = baseline.index.intersection(intervention.index)
        for endpoint, zone in shared_index:
            base_row = baseline.loc[(endpoint, zone)]
            int_row = intervention.loc[(endpoint, zone)]
            base_ratio = float(base_row["downwind_over_upwind"])
            int_ratio = float(int_row["downwind_over_upwind"])
            base_shadow = 1.0 - base_ratio
            int_shadow = 1.0 - int_ratio
            rows.append(
                {
                    "endpoint": endpoint,
                    "zone": zone,
                    "intervention_source": source,
                    "intervention_label": SOURCE_LABELS.get(source, source),
                    "downwind_over_upwind_model_baseline": base_ratio,
                    "downwind_over_upwind_intervention": int_ratio,
                    "ratio_delta_intervention_minus_baseline": int_ratio - base_ratio,
                    "ratio_pct_change": 100.0 * (int_ratio - base_ratio) / base_ratio if base_ratio != 0.0 else float("nan"),
                    "shadow_strength_baseline": base_shadow,
                    "shadow_strength_intervention": int_shadow,
                    "shadow_strength_delta": int_shadow - base_shadow,
                    "downwind_minus_upwind_model_baseline": float(base_row["downwind_minus_upwind"]),
                    "downwind_minus_upwind_intervention": float(int_row["downwind_minus_upwind"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["intervention_source", "endpoint", "zone"]) if rows else pd.DataFrame()


def plot_consistency_intervention_summary(summary_df: pd.DataFrame, out_dir: Path, *, hex_id: str) -> Path | None:
    if summary_df.empty:
        return None

    endpoints = [endpoint for endpoint in ENDPOINT_ORDER if endpoint in set(summary_df["endpoint"])]
    interventions = summary_df["intervention_source"].drop_duplicates().astype(str).tolist()
    if not endpoints or not interventions:
        return None

    fig, axes = plt.subplots(1, len(interventions), figsize=(7.2 * len(interventions), 4.5), squeeze=False, sharey=True)
    for ax, intervention in zip(axes.ravel(), interventions, strict=False):
        sub_intervention = summary_df[summary_df["intervention_source"].eq(intervention)]
        zones = sub_intervention["zone"].drop_duplicates().astype(str).tolist()
        x = np.arange(len(zones), dtype=float)
        width = min(0.24, 0.78 / max(len(endpoints), 1))
        for idx, endpoint in enumerate(endpoints):
            sub = sub_intervention[sub_intervention["endpoint"].eq(endpoint)].set_index("zone").reindex(zones)
            offset = (idx - (len(endpoints) - 1) / 2.0) * width
            values = 100.0 * sub["shadow_strength_delta"].to_numpy(dtype=float)
            ax.bar(
                x + offset,
                values,
                width=width,
                label=ENDPOINT_LABELS.get(endpoint, endpoint.upper()),
            )

        ax.axhline(0.0, color="0.45", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(zones)
        ax.set_title(SOURCE_LABELS.get(intervention, intervention))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Change in near-band shadow strength\npercentage points of 1 - downwind/upwind")
    axes[0, -1].legend(frameon=False, ncols=len(endpoints))
    fig.suptitle("Effect of wind-consistency interventions on barrier shadow", y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hex{int(hex_id):02d}_zone_wind_shadow_consistency_intervention_summary.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_zone_wind_shadow_outputs(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str = "16",
) -> tuple[Path, Path, Path, Path, list[Path]]:
    profiles, summary = compute_zone_wind_shadow_profiles(
        experiment_dir=experiment_dir,
        raw_data_dir=raw_data_dir,
        hex_id=hex_id,
    )
    endpoint_profile_path = experiment_dir / "counterfactual_zone_wind_shadow_endpoint_profile.csv"
    endpoint_summary_path = experiment_dir / "counterfactual_zone_wind_shadow_endpoint_summary.csv"
    bp_profile_path = experiment_dir / "counterfactual_zone_wind_shadow_bp_profile.csv"
    summary_path = experiment_dir / "counterfactual_zone_wind_shadow_summary.csv"
    profiles.to_csv(endpoint_profile_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(endpoint_summary_path, index=False)
    _profile_for_legacy_bp(profiles).to_csv(bp_profile_path, index=False)
    consistency_summary = summarize_consistency_intervention(summary)
    consistency_summary_path = experiment_dir / "counterfactual_zone_wind_shadow_consistency_intervention_summary.csv"
    consistency_summary.to_csv(consistency_summary_path, index=False)
    plot_paths = plot_split_zone_wind_shadow_profiles(profiles, summary, experiment_dir / "plots", hex_id=hex_id)
    consistency_plot_path = plot_consistency_intervention_summary(
        consistency_summary,
        experiment_dir / "plots",
        hex_id=hex_id,
    )
    if consistency_plot_path is not None:
        plot_paths.append(consistency_plot_path)
    return endpoint_profile_path, endpoint_summary_path, bp_profile_path, summary_path, plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute zone-wise wind-relative endpoint profiles.")
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
    endpoint_profile_path, endpoint_summary_path, bp_profile_path, summary_path, plot_paths = write_zone_wind_shadow_outputs(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        hex_id=str(args.hex_id).zfill(2),
    )
    print(f"Wrote zone wind-shadow endpoint profile: {endpoint_profile_path}")
    print(f"Wrote zone wind-shadow endpoint summary: {endpoint_summary_path}")
    print(f"Wrote legacy BP wind-shadow profile: {bp_profile_path}")
    print(f"Wrote zone wind-shadow summary: {summary_path}")
    for plot_path in plot_paths:
        print(f"Wrote zone wind-shadow plot: {plot_path}")


if __name__ == "__main__":
    main()
