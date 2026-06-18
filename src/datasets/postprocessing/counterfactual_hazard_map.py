"""Hex-level hazard delta maps for fixed-model counterfactual predictions."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm
from rasterio.features import geometry_mask

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_barrier_profile import (
    DISPLAY_SCENARIO,
    _load_barrier_layers_on_prediction_grid,
)
from src.datasets.postprocessing.diagnose_bp_barrier_halo import parse_fuel_barrier_info

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_MAP_SCENARIOS: tuple[str, ...] = (
    "wind_from_west_p95",
    "wind_speed50_original_direction",
    "remove_barriers_adjacent_modal",
)
REFERENCE_SUPPORT_POLICY = "native_actual_gt_model_prediction"


@dataclass(frozen=True)
class HazardDeltaMapSummary:
    scenario: str
    hex_id: str
    support_policy: str
    n_pixels: int
    baseline_hazard_mean: float
    scenario_hazard_mean: float
    delta_mean: float
    delta_median: float
    delta_p01: float
    delta_p05: float
    delta_p95: float
    delta_p99: float
    delta_min: float
    delta_max: float
    delta_abs_plot_limit: float
    frac_delta_positive: float
    frac_delta_negative: float


@dataclass(frozen=True)
class HazardReferenceMapSummary:
    panel: str
    scenario: str
    hex_id: str
    support_policy: str
    n_pixels: int
    hazard_mean: float
    hazard_median: float
    hazard_p95: float
    hazard_p99: float
    hazard_min: float
    hazard_max: float
    display_vmax: float
    display_percentile: float


@dataclass(frozen=True)
class EndpointReferenceMapSummary:
    endpoint: str
    panel: str
    scenario: str
    hex_id: str
    support_policy: str
    n_pixels: int
    value_mean: float
    value_median: float
    value_p95: float
    value_p99: float
    value_min: float
    value_max: float
    display_vmax: float
    display_percentile: float


@dataclass(frozen=True)
class EndpointResponseMapSummary:
    endpoint: str
    panel: str
    scenario: str
    hex_id: str
    support_policy: str
    n_pixels: int
    value_mean: float
    value_median: float
    value_p05: float
    value_p95: float
    value_min: float
    value_max: float
    display_abs_vmax: float
    display_delta_abs_vmax: float
    display_percentile: float


ENDPOINT_LABELS = {
    "bp": "BP",
    "fi": "FI",
    "ros": "ROS",
}


def prediction_dirs_from_index(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Read scenario/endpoint prediction directories from the materialization index."""

    index_path = experiment_dir / "scenario_prediction_index.csv"
    index = pd.read_csv(index_path)
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def prediction_raster_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "predicted_hexels" / f"hexel_{int(hex_id):02d}_predicted.tif"


def read_prediction(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def read_prediction_extent(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        bounds = src.bounds
    return bounds.left, bounds.right, bounds.bottom, bounds.top


def hazard_prediction(
    prediction_dirs: dict[tuple[str, str], Path],
    scenario: str,
    hex_id: str,
) -> np.ma.MaskedArray:
    """Return stitched predicted hazard = predicted BP x predicted FI."""

    bp_dir = prediction_dirs.get((scenario, "bp"))
    fi_dir = prediction_dirs.get((scenario, "fi"))
    if bp_dir is None or fi_dir is None:
        raise KeyError(f"Missing BP/FI predictions for scenario={scenario!r}.")
    return read_prediction(prediction_raster_path(bp_dir, hex_id)) * read_prediction(prediction_raster_path(fi_dir, hex_id))


def endpoint_prediction(
    prediction_dirs: dict[tuple[str, str], Path],
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> np.ma.MaskedArray:
    prediction_dir = prediction_dirs.get((scenario, endpoint))
    if prediction_dir is None:
        raise KeyError(f"Missing {endpoint.upper()} prediction for scenario={scenario!r}.")
    return read_prediction(prediction_raster_path(prediction_dir, hex_id))


def prediction_reference_profile(prediction_dirs: dict[tuple[str, str], Path], hex_id: str) -> dict:
    baseline_bp_dir = prediction_dirs.get(("baseline", "bp"))
    if baseline_bp_dir is None:
        raise KeyError("Missing baseline BP prediction directory; cannot define reference grid.")
    with rasterio.open(prediction_raster_path(baseline_bp_dir, hex_id)) as src:
        return src.profile.copy()


def ground_truth_hazard_on_prediction_grid(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
) -> np.ma.MaskedArray:
    """Return raw GT hazard = raw BP x raw FI aligned to the prediction grid."""

    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    bp_ma, _ = load_spatial_raster(paths.output_burn_prob(), reference_profile=reference_profile)
    fi_ma, _ = load_spatial_raster(paths.output_fire_intensity(), reference_profile=reference_profile)
    support = original_valid_hazard_support(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    bp = np.asarray(np.ma.asarray(bp_ma).filled(np.nan), dtype=np.float64)
    fi = np.asarray(np.ma.asarray(fi_ma).filled(np.nan), dtype=np.float64)
    hazard = bp * fi
    return np.ma.masked_invalid(np.where(support, hazard, np.nan))


def ground_truth_endpoint_on_prediction_grid(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    endpoint: str,
) -> np.ma.MaskedArray:
    """Return a raw GT endpoint aligned to the prediction grid and clipped to the actual hex."""

    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    if endpoint == "bp":
        source_path = paths.output_burn_prob()
    elif endpoint == "fi":
        source_path = paths.output_fire_intensity()
    elif endpoint == "ros":
        source_path = paths.output_ros()
    else:
        raise ValueError("endpoint must be one of {'bp', 'fi', 'ros'}.")

    endpoint_ma, _ = load_spatial_raster(source_path, reference_profile=reference_profile)
    values = np.asarray(np.ma.asarray(endpoint_ma).filled(np.nan), dtype=np.float64)
    support = np.isfinite(values) & mask_scope_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
        mask_scope="actual",
    )
    return np.ma.masked_invalid(np.where(support, values, np.nan))


def hazard_delta(
    prediction_dirs: dict[tuple[str, str], Path],
    scenario: str,
    hex_id: str,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]:
    """Return baseline hazard, scenario hazard, and paired scenario-baseline delta."""

    baseline = hazard_prediction(prediction_dirs, "baseline", hex_id)
    scenario_hazard = hazard_prediction(prediction_dirs, scenario, hex_id)
    return baseline, scenario_hazard, scenario_hazard - baseline


def finite_values(data: np.ma.MaskedArray | np.ndarray) -> np.ndarray:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values[np.isfinite(values)]


def pooled_positive_percentile(arrays: list[np.ma.MaskedArray | np.ndarray], percentile: float) -> float:
    values = [finite_values(arr) for arr in arrays]
    values = [value[value >= 0.0] for value in values if value.size > 0]
    values = [value for value in values if value.size > 0]
    if not values:
        return 1.0
    vmax = float(np.percentile(np.concatenate(values), percentile))
    return max(vmax, 1e-9)


def values_and_valid(data: np.ma.MaskedArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values, np.isfinite(values)


def symmetric_percentile_limit(deltas: list[np.ma.MaskedArray], percentile: float = 99.5) -> float:
    """Symmetric color limit from pooled absolute delta values."""

    values = [np.abs(finite_values(delta)) for delta in deltas]
    values = [value for value in values if value.size > 0]
    if not values:
        return 1.0
    limit = float(np.percentile(np.concatenate(values), percentile))
    return max(limit, 1e-9)


def summarize_hazard_delta(
    *,
    scenario: str,
    hex_id: str,
    baseline_hazard: np.ma.MaskedArray,
    scenario_hazard: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    delta_abs_plot_limit: float,
    support_policy: str = "prediction",
) -> HazardDeltaMapSummary:
    baseline_values, baseline_valid = values_and_valid(baseline_hazard)
    scenario_values, scenario_valid = values_and_valid(scenario_hazard)
    delta_values_all, delta_valid = values_and_valid(delta)
    valid = baseline_valid & scenario_valid & delta_valid
    if not valid.any():
        raise ValueError(f"No finite hazard deltas for scenario={scenario!r}, hex={hex_id!r}.")
    baseline = baseline_values[valid]
    scenario_values = scenario_values[valid]
    delta_values = delta_values_all[valid]
    return HazardDeltaMapSummary(
        scenario=scenario,
        hex_id=hex_id,
        support_policy=support_policy,
        n_pixels=int(delta_values.size),
        baseline_hazard_mean=float(np.mean(baseline)),
        scenario_hazard_mean=float(np.mean(scenario_values)),
        delta_mean=float(np.mean(delta_values)),
        delta_median=float(np.median(delta_values)),
        delta_p01=float(np.percentile(delta_values, 1)),
        delta_p05=float(np.percentile(delta_values, 5)),
        delta_p95=float(np.percentile(delta_values, 95)),
        delta_p99=float(np.percentile(delta_values, 99)),
        delta_min=float(np.min(delta_values)),
        delta_max=float(np.max(delta_values)),
        delta_abs_plot_limit=float(delta_abs_plot_limit),
        frac_delta_positive=float(np.mean(delta_values > 0.0)),
        frac_delta_negative=float(np.mean(delta_values < 0.0)),
    )


def downsample_for_display(data: np.ma.MaskedArray | np.ndarray, factor: int) -> np.ma.MaskedArray:
    """Stride-downsample a raster for plotting only."""

    if factor <= 1:
        return np.ma.asarray(data)
    return np.ma.asarray(data)[::factor, ::factor]


def restrict_to_support(data: np.ma.MaskedArray | np.ndarray, support_mask: np.ndarray) -> np.ma.MaskedArray:
    """Mask an array outside a boolean analysis support mask."""

    arr = np.ma.asarray(data)
    if arr.shape != support_mask.shape:
        raise ValueError(f"Support mask shape {support_mask.shape} does not match data shape {arr.shape}.")
    return np.ma.masked_where(~support_mask | np.ma.getmaskarray(arr), arr)


def mask_scope_on_prediction_grid(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    mask_scope: str = "actual",
) -> np.ndarray:
    """Return the actual/buffer polygon mask on the stitched prediction grid."""

    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    mask_gdf = gpd.read_file(paths.mask_grid(hex_id, mask_scope)).to_crs(reference_profile["crs"])
    return geometry_mask(
        mask_gdf.geometry,
        out_shape=(int(reference_profile["height"]), int(reference_profile["width"])),
        transform=reference_profile["transform"],
        invert=True,
    )


def original_valid_hazard_support(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    mask_scope: str | None = "actual",
) -> np.ndarray:
    """Original finite BP/FI burnable support aligned to the prediction grid."""

    baseline_bp_dir = prediction_dirs.get(("baseline", "bp"))
    if baseline_bp_dir is None:
        raise KeyError("Missing baseline BP prediction directory; cannot define reference grid.")
    with rasterio.open(prediction_raster_path(baseline_bp_dir, hex_id)) as src:
        reference_profile = src.profile.copy()

    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    bp_ma, _ = load_spatial_raster(paths.output_burn_prob(), reference_profile=reference_profile)
    fi_ma, _ = load_spatial_raster(paths.output_fire_intensity(), reference_profile=reference_profile)
    fuel_ma, _ = load_spatial_raster(paths.fuel_grid(hex_id), reference_profile=reference_profile)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)

    bp_values = np.ma.asarray(bp_ma).filled(np.nan)
    fi_values = np.ma.asarray(fi_ma).filled(np.nan)
    fuel_values = np.ma.asarray(fuel_ma).filled(-32768).astype(np.int32)
    support = np.isfinite(bp_values) & np.isfinite(fi_values) & ~np.isin(fuel_values, fuel_info.nonfuel_ids)
    if mask_scope is not None:
        support &= mask_scope_on_prediction_grid(
            raw_data_dir=raw_data_dir,
            prediction_dirs=prediction_dirs,
            hex_id=hex_id,
            mask_scope=mask_scope,
        )
    return support


def summarize_hazard_reference_map(
    *,
    panel: str,
    scenario: str,
    hex_id: str,
    support_policy: str,
    hazard: np.ma.MaskedArray,
    display_vmax: float,
    display_percentile: float,
) -> HazardReferenceMapSummary:
    values = finite_values(hazard)
    if values.size == 0:
        raise ValueError(f"No finite hazard values for panel={panel!r}, scenario={scenario!r}, hex={hex_id!r}.")
    return HazardReferenceMapSummary(
        panel=panel,
        scenario=scenario,
        hex_id=hex_id,
        support_policy=support_policy,
        n_pixels=int(values.size),
        hazard_mean=float(np.mean(values)),
        hazard_median=float(np.median(values)),
        hazard_p95=float(np.percentile(values, 95)),
        hazard_p99=float(np.percentile(values, 99)),
        hazard_min=float(np.min(values)),
        hazard_max=float(np.max(values)),
        display_vmax=float(display_vmax),
        display_percentile=float(display_percentile),
    )


def plot_hazard_reference_maps(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str = "16",
    scenario: str = "remove_barriers_adjacent_modal",
    percentile: float = 99.5,
    downsample: int = 2,
) -> tuple[Path, Path]:
    """Write whole-hex GT/model/intervention hazard reference panels."""

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    gt_hazard = ground_truth_hazard_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    baseline_hazard = hazard_prediction(prediction_dirs, "baseline", hex_id)
    scenario_hazard = hazard_prediction(prediction_dirs, scenario, hex_id)

    panels = [
        ("Ground truth baseline", "ground_truth", gt_hazard),
        ("Model baseline", "baseline", baseline_hazard),
        (f"Model intervention\n{DISPLAY_SCENARIO.get(scenario, scenario)}", scenario, scenario_hazard),
    ]
    display_vmax = pooled_positive_percentile([hazard for _, _, hazard in panels], percentile=percentile)
    norm = Normalize(vmin=0.0, vmax=display_vmax)
    cmap = copy.copy(plt.get_cmap("magma"))
    cmap.set_bad(color="#eeeeee", alpha=1.0)

    baseline_bp_dir = prediction_dirs[("baseline", "bp")]
    extent = read_prediction_extent(prediction_raster_path(baseline_bp_dir, hex_id))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 7.0), squeeze=False)
    image = None
    for ax, (title, _, hazard) in zip(axes.ravel(), panels, strict=False):
        ax.set_facecolor("#eeeeee")
        image = ax.imshow(
            downsample_for_display(hazard, downsample),
            cmap=cmap,
            norm=norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(f"Hazard = BP x FI (shared scale, clipped at pooled p{percentile:g})")

    fig.suptitle(f"Hex{int(hex_id):02d} hazard reference maps", y=0.96)
    out_dir = experiment_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"hex{int(hex_id):02d}_gt_model_intervention_hazard.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summaries = [
        asdict(
            summarize_hazard_reference_map(
                panel=title.replace("\n", " "),
                scenario=panel_scenario,
                hex_id=hex_id,
                support_policy=REFERENCE_SUPPORT_POLICY,
                hazard=hazard,
                display_vmax=display_vmax,
                display_percentile=percentile,
            )
        )
        for title, panel_scenario, hazard in panels
    ]
    summary_path = experiment_dir / "counterfactual_hazard_reference_map_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return plot_path, summary_path


def summarize_endpoint_reference_map(
    *,
    endpoint: str,
    panel: str,
    scenario: str,
    hex_id: str,
    support_policy: str,
    values_map: np.ma.MaskedArray,
    display_vmax: float,
    display_percentile: float,
) -> EndpointReferenceMapSummary:
    values = finite_values(values_map)
    if values.size == 0:
        raise ValueError(f"No finite {endpoint} values for panel={panel!r}, scenario={scenario!r}, hex={hex_id!r}.")
    return EndpointReferenceMapSummary(
        endpoint=endpoint,
        panel=panel,
        scenario=scenario,
        hex_id=hex_id,
        support_policy=support_policy,
        n_pixels=int(values.size),
        value_mean=float(np.mean(values)),
        value_median=float(np.median(values)),
        value_p95=float(np.percentile(values, 95)),
        value_p99=float(np.percentile(values, 99)),
        value_min=float(np.min(values)),
        value_max=float(np.max(values)),
        display_vmax=float(display_vmax),
        display_percentile=float(display_percentile),
    )


def summarize_endpoint_response_map(
    *,
    endpoint: str,
    panel: str,
    scenario: str,
    hex_id: str,
    support_policy: str,
    values_map: np.ma.MaskedArray,
    display_abs_vmax: float,
    display_delta_abs_vmax: float,
    display_percentile: float,
) -> EndpointResponseMapSummary:
    values = finite_values(values_map)
    if values.size == 0:
        raise ValueError(f"No finite {endpoint} values for panel={panel!r}, scenario={scenario!r}, hex={hex_id!r}.")
    return EndpointResponseMapSummary(
        endpoint=endpoint,
        panel=panel,
        scenario=scenario,
        hex_id=hex_id,
        support_policy=support_policy,
        n_pixels=int(values.size),
        value_mean=float(np.mean(values)),
        value_median=float(np.median(values)),
        value_p05=float(np.percentile(values, 5)),
        value_p95=float(np.percentile(values, 95)),
        value_min=float(np.min(values)),
        value_max=float(np.max(values)),
        display_abs_vmax=float(display_abs_vmax),
        display_delta_abs_vmax=float(display_delta_abs_vmax),
        display_percentile=float(display_percentile),
    )


def plot_endpoint_reference_maps(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    endpoint: str,
    hex_id: str = "16",
    scenario: str = "remove_barriers_adjacent_modal",
    percentile: float = 99.5,
    downsample: int = 2,
) -> tuple[Path, Path]:
    """Write whole-hex GT/model/intervention reference panels for one endpoint."""

    if endpoint not in set(ENDPOINT_LABELS):
        raise ValueError("endpoint must be one of {'bp', 'fi', 'ros'}.")

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    gt_endpoint = ground_truth_endpoint_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
        endpoint=endpoint,
    )
    baseline_endpoint = endpoint_prediction(prediction_dirs, "baseline", endpoint, hex_id)
    scenario_endpoint = endpoint_prediction(prediction_dirs, scenario, endpoint, hex_id)

    endpoint_label = ENDPOINT_LABELS.get(endpoint, endpoint.upper())
    panels = [
        (f"Ground truth baseline {endpoint_label}", "ground_truth", gt_endpoint),
        (f"Model baseline {endpoint_label}", "baseline", baseline_endpoint),
        (f"Model intervention {endpoint_label}\n{DISPLAY_SCENARIO.get(scenario, scenario)}", scenario, scenario_endpoint),
    ]
    display_vmax = pooled_positive_percentile([values_map for _, _, values_map in panels], percentile=percentile)
    norm = Normalize(vmin=0.0, vmax=display_vmax)
    cmap = copy.copy(plt.get_cmap("magma"))
    cmap.set_bad(color="#eeeeee", alpha=1.0)

    baseline_bp_dir = prediction_dirs[("baseline", "bp")]
    extent = read_prediction_extent(prediction_raster_path(baseline_bp_dir, hex_id))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 7.0), squeeze=False)
    image = None
    for ax, (title, _, values_map) in zip(axes.ravel(), panels, strict=False):
        ax.set_facecolor("#eeeeee")
        image = ax.imshow(
            downsample_for_display(values_map, downsample),
            cmap=cmap,
            norm=norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(f"{endpoint_label} (shared scale, clipped at pooled p{percentile:g})")

    fig.suptitle(f"Hex{int(hex_id):02d} {endpoint_label} reference maps", y=0.96)
    out_dir = experiment_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"hex{int(hex_id):02d}_gt_model_intervention_{endpoint}.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summaries = [
        asdict(
            summarize_endpoint_reference_map(
                endpoint=endpoint,
                panel=title.replace("\n", " "),
                scenario=panel_scenario,
                hex_id=hex_id,
                support_policy=REFERENCE_SUPPORT_POLICY,
                values_map=values_map,
                display_vmax=display_vmax,
                display_percentile=percentile,
            )
        )
        for title, panel_scenario, values_map in panels
    ]
    summary_path = experiment_dir / f"counterfactual_{endpoint}_reference_map_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return plot_path, summary_path


def plot_endpoint_response_maps(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    scenario: str,
    endpoints: tuple[str, ...] = ("fi", "ros"),
    hex_id: str = "16",
    percentile: float = 99.5,
    downsample: int = 2,
    support_policy: str = "prediction",
) -> tuple[Path, Path]:
    """Write baseline/scenario/delta endpoint panels for selected endpoints."""

    if support_policy not in {"prediction", "raw_valid_output"}:
        raise ValueError("support_policy must be one of {'prediction', 'raw_valid_output'}.")
    unknown = sorted(set(endpoints) - set(ENDPOINT_LABELS))
    if unknown:
        raise ValueError(f"Unknown endpoints: {unknown}.")

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    endpoint_data: dict[str, tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]] = {}
    for endpoint in endpoints:
        baseline = endpoint_prediction(prediction_dirs, "baseline", endpoint, hex_id)
        scenario_endpoint = endpoint_prediction(prediction_dirs, scenario, endpoint, hex_id)
        delta = scenario_endpoint - baseline
        if support_policy == "raw_valid_output":
            support = np.isfinite(
                np.asarray(
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
            )
            baseline = restrict_to_support(baseline, support)
            scenario_endpoint = restrict_to_support(scenario_endpoint, support)
            delta = restrict_to_support(delta, support)
        endpoint_data[endpoint] = baseline, scenario_endpoint, delta

    baseline_bp_dir = prediction_dirs[("baseline", "bp")]
    extent = read_prediction_extent(prediction_raster_path(baseline_bp_dir, hex_id))
    fig, axes = plt.subplots(len(endpoints), 3, figsize=(15.5, 5.2 * len(endpoints)), squeeze=False)
    abs_cmap = copy.copy(plt.get_cmap("magma"))
    abs_cmap.set_bad(color="#eeeeee", alpha=1.0)
    delta_cmap = copy.copy(plt.get_cmap("RdBu_r"))
    delta_cmap.set_bad(color="#eeeeee", alpha=1.0)

    summaries: list[dict] = []
    for row_idx, endpoint in enumerate(endpoints):
        baseline, scenario_endpoint, delta = endpoint_data[endpoint]
        endpoint_label = ENDPOINT_LABELS.get(endpoint, endpoint.upper())
        abs_vmax = pooled_positive_percentile([baseline, scenario_endpoint], percentile=percentile)
        delta_limit = symmetric_percentile_limit([delta], percentile=percentile)
        abs_norm = Normalize(vmin=0.0, vmax=abs_vmax)
        delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
        panels = [
            ("Model baseline", "baseline", baseline, abs_cmap, abs_norm),
            (DISPLAY_SCENARIO.get(scenario, scenario), scenario, scenario_endpoint, abs_cmap, abs_norm),
            ("Δ scenario − baseline", "delta", delta, delta_cmap, delta_norm),
        ]
        abs_image = None
        delta_image = None
        for col_idx, (title, panel_name, values_map, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#eeeeee")
            image = ax.imshow(
                downsample_for_display(values_map, downsample),
                cmap=cmap,
                norm=norm,
                extent=extent,
                origin="upper",
                interpolation="nearest",
            )
            if col_idx < 2:
                abs_image = image
            else:
                delta_image = image
            ax.set_title(f"{endpoint_label}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            summaries.append(
                asdict(
                    summarize_endpoint_response_map(
                        endpoint=endpoint,
                        panel=panel_name,
                        scenario=scenario if panel_name != "baseline" else "baseline",
                        hex_id=hex_id,
                        support_policy=support_policy,
                        values_map=values_map,
                        display_abs_vmax=abs_vmax,
                        display_delta_abs_vmax=delta_limit,
                        display_percentile=percentile,
                    )
                )
            )
        if abs_image is not None:
            cbar = fig.colorbar(abs_image, ax=axes[row_idx, :2].tolist(), fraction=0.025, pad=0.02)
            cbar.set_label(f"{endpoint_label} (shared baseline/scenario scale, p{percentile:g})")
        if delta_image is not None:
            cbar = fig.colorbar(delta_image, ax=axes[row_idx, 2], fraction=0.046, pad=0.03)
            cbar.set_label(f"Δ{endpoint_label} (symmetric p{percentile:g})")

    fig.suptitle(f"Hex{int(hex_id):02d} endpoint response maps: {DISPLAY_SCENARIO.get(scenario, scenario)}", y=0.99)
    out_dir = experiment_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_slug = "_".join(endpoints)
    plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_{endpoint_slug}_endpoint_response_maps_{support_policy}.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = experiment_dir / f"counterfactual_{scenario}_{endpoint_slug}_endpoint_response_map_summary_{support_policy}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return plot_path, summary_path


def original_barrier_mask(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
) -> np.ndarray:
    """Return original raw non-fuel barriers aligned to the prediction grid."""

    layers = _load_barrier_layers_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    return np.isin(layers.fuel, fuel_info.nonfuel_ids)


def plot_hazard_delta_maps(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str = "16",
    scenarios: tuple[str, ...] = DEFAULT_MAP_SCENARIOS,
    percentile: float = 99.5,
    downsample: int = 2,
    overlay_barriers: bool = True,
    support_policy: str = "raw_valid_output",
) -> tuple[Path, Path]:
    """Write a full-hex map of counterfactual hazard deltas."""

    if support_policy not in {"prediction", "raw_valid_output"}:
        raise ValueError("support_policy must be one of {'prediction', 'raw_valid_output'}.")

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    scenario_data: dict[str, tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]] = {}
    for scenario in scenarios:
        baseline, scenario_hazard, delta = hazard_delta(prediction_dirs, scenario, hex_id)
        scenario_data[scenario] = baseline, scenario_hazard, delta

    support_mask = None
    if support_policy == "raw_valid_output":
        support_mask = original_valid_hazard_support(
            raw_data_dir=raw_data_dir,
            prediction_dirs=prediction_dirs,
            hex_id=hex_id,
        )
        scenario_data = {
            scenario: (
                restrict_to_support(baseline, support_mask),
                restrict_to_support(scenario_hazard, support_mask),
                restrict_to_support(delta, support_mask),
            )
            for scenario, (baseline, scenario_hazard, delta) in scenario_data.items()
        }

    deltas = [triple[2] for triple in scenario_data.values()]
    plot_limit = symmetric_percentile_limit(deltas, percentile=percentile)
    norm = TwoSlopeNorm(vmin=-plot_limit, vcenter=0.0, vmax=plot_limit)
    cmap = copy.copy(plt.get_cmap("RdBu_r"))
    cmap.set_bad(color="white", alpha=0.0)

    baseline_bp_dir = prediction_dirs[("baseline", "bp")]
    extent = read_prediction_extent(prediction_raster_path(baseline_bp_dir, hex_id))
    barrier_mask = None
    if overlay_barriers:
        barrier_mask = original_barrier_mask(
            raw_data_dir=raw_data_dir,
            prediction_dirs=prediction_dirs,
            hex_id=hex_id,
        )

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.0 * len(scenarios), 7.0), squeeze=False)
    image = None
    for ax, scenario in zip(axes.ravel(), scenarios, strict=False):
        _, _, delta = scenario_data[scenario]
        image = ax.imshow(
            downsample_for_display(delta, downsample),
            cmap=cmap,
            norm=norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        if barrier_mask is not None:
            ax.contour(
                downsample_for_display(barrier_mask.astype(np.float32), downsample),
                levels=[0.5],
                colors="black",
                linewidths=0.25,
                alpha=0.35,
                extent=extent,
                origin="upper",
            )
        ax.set_title(DISPLAY_SCENARIO.get(scenario, scenario))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(f"Δ hazard = scenario − baseline (clipped at pooled p{percentile:g})")

    fig.suptitle(f"Hex{int(hex_id):02d} counterfactual hazard delta maps", y=0.96)
    out_dir = experiment_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"hex{int(hex_id):02d}_hazard_delta_maps_{support_policy}.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summaries = [
        asdict(
            summarize_hazard_delta(
                scenario=scenario,
                hex_id=hex_id,
                baseline_hazard=baseline,
                scenario_hazard=scenario_hazard,
                delta=delta,
                delta_abs_plot_limit=plot_limit,
                support_policy=support_policy,
            )
        )
        for scenario, (baseline, scenario_hazard, delta) in scenario_data.items()
    ]
    summary_path = experiment_dir / f"counterfactual_hazard_delta_map_summary_{support_policy}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return plot_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full-hex counterfactual hazard delta maps.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario to plot; may be repeated.")
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument(
        "--reference_maps",
        action="store_true",
        help="Also write GT baseline, model baseline, and selected model-intervention hazard panels.",
    )
    parser.add_argument(
        "--support_policy",
        choices=("raw_valid_output", "prediction"),
        default="raw_valid_output",
        help=(
            "raw_valid_output masks to original finite raw BP/FI burnable support; "
            "prediction uses every finite stitched model prediction pixel."
        ),
    )
    parser.add_argument("--no_barrier_overlay", action="store_true")
    parser.add_argument(
        "--endpoint_response_maps",
        action="store_true",
        help="Also write baseline/scenario/delta maps for selected endpoints.",
    )
    parser.add_argument(
        "--endpoint_response_endpoint",
        action="append",
        dest="endpoint_response_endpoints",
        choices=tuple(ENDPOINT_LABELS),
        help="Endpoint for --endpoint_response_maps; may be repeated. Defaults to BP, FI, and ROS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = tuple(args.scenarios) if args.scenarios else DEFAULT_MAP_SCENARIOS
    plot_path, summary_path = plot_hazard_delta_maps(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        hex_id=str(args.hex_id).zfill(2),
        scenarios=scenarios,
        percentile=args.percentile,
        downsample=max(1, int(args.downsample)),
        overlay_barriers=not args.no_barrier_overlay,
        support_policy=args.support_policy,
    )
    print(f"Wrote hazard delta map: {plot_path}")
    print(f"Wrote hazard delta map summary: {summary_path}")
    if args.reference_maps:
        reference_plot_path, reference_summary_path = plot_hazard_reference_maps(
            experiment_dir=args.experiment_dir,
            raw_data_dir=args.raw_data_dir,
            hex_id=str(args.hex_id).zfill(2),
            scenario=scenarios[-1],
            percentile=args.percentile,
            downsample=max(1, int(args.downsample)),
        )
        print(f"Wrote hazard reference map: {reference_plot_path}")
        print(f"Wrote hazard reference map summary: {reference_summary_path}")
        for endpoint in ("bp", "fi", "ros"):
            endpoint_plot_path, endpoint_summary_path = plot_endpoint_reference_maps(
                experiment_dir=args.experiment_dir,
                raw_data_dir=args.raw_data_dir,
                endpoint=endpoint,
                hex_id=str(args.hex_id).zfill(2),
                scenario=scenarios[-1],
                percentile=args.percentile,
                downsample=max(1, int(args.downsample)),
            )
            print(f"Wrote {endpoint.upper()} reference map: {endpoint_plot_path}")
            print(f"Wrote {endpoint.upper()} reference map summary: {endpoint_summary_path}")
    if args.endpoint_response_maps:
        response_endpoints = tuple(args.endpoint_response_endpoints) if args.endpoint_response_endpoints else ("bp", "fi", "ros")
        response_plot_path, response_summary_path = plot_endpoint_response_maps(
            experiment_dir=args.experiment_dir,
            raw_data_dir=args.raw_data_dir,
            scenario=scenarios[-1],
            endpoints=response_endpoints,
            hex_id=str(args.hex_id).zfill(2),
            percentile=args.percentile,
            downsample=max(1, int(args.downsample)),
            support_policy=args.support_policy,
        )
        print(f"Wrote endpoint response maps: {response_plot_path}")
        print(f"Wrote endpoint response map summary: {response_summary_path}")


if __name__ == "__main__":
    main()
