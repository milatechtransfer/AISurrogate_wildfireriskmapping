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
from matplotlib.colors import TwoSlopeNorm
from rasterio.features import geometry_mask

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_viz import (
    downsample_for_display,
    prediction_dirs_from_index,
    prediction_raster_path,
    prediction_reference_profile,
    read_prediction,
    read_prediction_extent,
    restrict_to_support,
    symmetric_percentile_limit,
    values_and_valid,
)
from src.datasets.postprocessing.fuel_barrier_geometry import (
    DISPLAY_SCENARIO,
    load_barrier_layers_on_prediction_grid,
    parse_fuel_barrier_info,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_MAP_SCENARIOS: tuple[str, ...] = ("remove_barriers_adjacent_modal",)
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


ENDPOINT_LABELS = {
    "bp": "BP",
    "fi": "FI",
    "ros": "ROS",
}


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


def hazard_delta(
    prediction_dirs: dict[tuple[str, str], Path],
    scenario: str,
    hex_id: str,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]:
    """Return baseline hazard, scenario hazard, and paired scenario-baseline delta."""

    baseline = hazard_prediction(prediction_dirs, "baseline", hex_id)
    scenario_hazard = hazard_prediction(prediction_dirs, scenario, hex_id)
    return baseline, scenario_hazard, scenario_hazard - baseline


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


def original_barrier_mask(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
) -> np.ndarray:
    """Return original raw non-fuel barriers aligned to the prediction grid."""

    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    layers = load_barrier_layers_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        reference_profile=reference_profile,
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
        "--support_policy",
        choices=("raw_valid_output", "prediction"),
        default="raw_valid_output",
        help=(
            "raw_valid_output masks to original finite raw BP/FI burnable support; "
            "prediction uses every finite stitched model prediction pixel."
        ),
    )
    parser.add_argument("--no_barrier_overlay", action="store_true")
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


if __name__ == "__main__":
    main()
