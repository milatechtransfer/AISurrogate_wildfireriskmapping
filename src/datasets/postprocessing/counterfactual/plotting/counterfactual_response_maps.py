"""Generic response-map and patch-zoom figures for any counterfactual scenario.

Ground-truth/baseline/scenario/\u0394 maps and high-response patch zoom-ins for one
(scenario \u00d7 endpoint) pair from ``scenario_prediction_index.csv``. Parameterized by
endpoint (bp/fi/ros) so any future counterfactual scenario family (fuel, weather,
wind, ...) reuses this instead of writing a new per-endpoint figure script.

For fuel-editing scenarios, baseline and scenario support come from the exact fuel
rasters persisted during evaluation. Non-burnable pixels contribute zero, so
barrier removal and insertion are represented symmetrically.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    load_counterfactual_config,
    resolve_counterfactual_paths,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map import (
    burnable_fuel_support,
    load_evaluated_fuel_pair,
    load_static_burnable_support,
    load_zone_labels_on_prediction_grid,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    DEFAULT_ZONE_OVERLAY_ALPHA,
    DEFAULT_ZONE_OVERLAY_COLOR,
    DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    add_zone_overlay_args,
    build_endpoint_response,
    delta_norm,
    downsample_for_display,
    finite_values,
    load_baseline_scenario_pair,
    overlay_zone_boundaries,
    plot_delta_concentration,
    plot_delta_histogram,
    prediction_dirs_from_index,
    prediction_footprint,
    prediction_raster_path,
    read_prediction_extent,
    restrict_to_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 17,
    }
)

PANEL_TITLE_WIDTH = 30
DELTA_COLOR = "#b2182b"


@dataclass(frozen=True)
class EndpointSpec:
    """Display metadata for one prediction endpoint."""

    label: str
    units: str
    cmap: str
    ground_truth_path: Callable[[Paths], Path]


ENDPOINT_SPECS: dict[str, EndpointSpec] = {
    "bp": EndpointSpec("BP", "Burn probability", "viridis", lambda paths: paths.output_burn_prob()),
    "fi": EndpointSpec("FI", "Fire intensity (kW/m)", "viridis", lambda paths: paths.output_fire_intensity()),
    "ros": EndpointSpec("ROS", "ROS (m/min)", "viridis", lambda paths: paths.output_ros()),
}


def _panel_title(title: str) -> str:
    return textwrap.fill(title, width=PANEL_TITLE_WIDTH)


def _response_norms(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_values: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
) -> tuple[Normalize, TwoSlopeNorm]:
    """Shared sequential value scale and zero-centered \u0394 scale for a response-map panel row."""
    pooled = np.concatenate([finite_values(ground_truth), finite_values(baseline), finite_values(scenario_values)])
    value_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
    return value_norm, delta_norm([delta], percentile=99.0)


def _extent_km(extent_m: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Recentre a metre extent on its lower-left origin and convert to kilometres."""
    left, right, bottom, top = extent_m
    return (0.0, (right - left) / 1000.0, 0.0, (top - bottom) / 1000.0)


def load_ground_truth(raw_data_dir: Path, hex_id: str, reference_profile: dict, *, endpoint: str) -> np.ma.MaskedArray:
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    gt, _ = load_spatial_raster(path=ENDPOINT_SPECS[endpoint].ground_truth_path(paths), reference_profile=reference_profile)
    return gt


def block_response(delta: np.ma.MaskedArray, block: int) -> np.ndarray:
    """Mean absolute response pooled into `block`x`block` cells over valid pixels."""
    abs_delta = np.abs(np.ma.filled(delta, 0.0))
    valid = (~np.ma.getmaskarray(delta)).astype(np.float64)
    rows = (abs_delta.shape[0] // block) * block
    cols = (abs_delta.shape[1] // block) * block
    summed = abs_delta[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    counts = valid[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    return np.where(counts > 0, summed / np.maximum(counts, 1.0), 0.0)


def hotspot_centers(delta: np.ma.MaskedArray, block: int, *, count: int, window: int) -> list[tuple[int, int]]:
    """Centres of the `count` highest-response windows, suppressed to be distinct."""
    scores = block_response(delta, block)
    suppress = max(window // block, 1)
    centers: list[tuple[int, int]] = []
    for _ in range(count):
        if not np.any(scores > 0):
            break
        flat_index = int(np.argmax(scores))
        row_block, col_block = (int(idx) for idx in np.unravel_index(flat_index, scores.shape))
        centers.append((row_block * block + block // 2, col_block * block + block // 2))
        scores[
            max(row_block - suppress, 0) : row_block + suppress + 1,
            max(col_block - suppress, 0) : col_block + suppress + 1,
        ] = 0.0
    return centers


def load_endpoint_response(
    experiment_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    raw_data_dir: Path,
    *,
    scenario: str,
    endpoint: str,
    nonfuel_ids: list[int] | None = None,
    static_nonfuel_ids: list[int] | None = None,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray, tuple[float, float, float, float], dict]:
    """Load ground-truth and a support-aware baseline/scenario response.

    `nonfuel_ids` restricts to burnable land using the persisted baseline/scenario fuel
    raster pair (for scenarios that edit fuel). `static_nonfuel_ids` instead restricts
    using a single raw fuel raster shared by baseline and scenario (for scenarios, e.g.
    weather counterfactuals, that leave fuel unchanged). At most one should be set.
    """
    if endpoint not in ENDPOINT_SPECS:
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {sorted(ENDPOINT_SPECS)}.")
    if nonfuel_ids is not None and static_nonfuel_ids is not None:
        raise ValueError("nonfuel_ids and static_nonfuel_ids are mutually exclusive.")

    baseline, scenario_values = load_baseline_scenario_pair(prediction_dirs, hex_id, endpoint=endpoint, scenario=scenario)
    baseline_path = prediction_raster_path(prediction_dirs[("baseline", endpoint)], hex_id, target_name=endpoint)
    extent = read_prediction_extent(baseline_path)
    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()

    if nonfuel_ids is not None:
        baseline_fuel, scenario_fuel = load_evaluated_fuel_pair(
            experiment_dir=experiment_dir,
            scenario=scenario,
            endpoint=endpoint,
            hex_id=hex_id,
        )
        response = build_endpoint_response(
            baseline,
            scenario_values,
            baseline_support=burnable_fuel_support(baseline_fuel, nonfuel_ids),
            scenario_support=burnable_fuel_support(scenario_fuel, nonfuel_ids),
        )
    elif static_nonfuel_ids is not None:
        support = load_static_burnable_support(raw_data_dir, hex_id, reference_profile, static_nonfuel_ids)
        response = build_endpoint_response(baseline, scenario_values, baseline_support=support, scenario_support=support)
    else:
        response = build_endpoint_response(baseline, scenario_values)

    ground_truth = load_ground_truth(raw_data_dir, hex_id, reference_profile, endpoint=endpoint)
    if ground_truth.shape != baseline.shape:
        raise ValueError(f"Ground-truth shape {ground_truth.shape} does not match prediction grid {baseline.shape}.")

    ground_truth = restrict_to_support(ground_truth, response.baseline_support)
    return ground_truth, response.baseline, response.scenario, response.delta, extent, reference_profile


def _render_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    data: np.ma.MaskedArray,
    title: str,
    cmap: str,
    norm: Normalize,
    cbar_label: str,
    *,
    extent: tuple[float, float, float, float] | None = None,
    downsample: int = 1,
    zone_window: np.ma.MaskedArray | None = None,
    zone_overlay_color: str,
    zone_overlay_linewidth: float,
    zone_overlay_alpha: float,
) -> None:
    """Render one imshow + colourbar + zone-overlay panel shared by all response-map figures."""
    image = ax.imshow(
        downsample_for_display(data, downsample),
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    overlay_zone_boundaries(
        ax,
        zone_window,
        extent=extent,
        color=zone_overlay_color,
        linewidth=zone_overlay_linewidth,
        alpha=zone_overlay_alpha,
    )
    ax.set_title(_panel_title(title), pad=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=14)


def plot_response_maps(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_values: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    extent_m: tuple[float, float, float, float],
    *,
    endpoint: str,
    out_path: Path,
    scenario_label: str,
    suptitle: str,
    downsample: int,
    zone_labels: np.ma.MaskedArray | None = None,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Ground-truth / baseline / scenario / \u0394 maps across the whole hex."""
    spec = ENDPOINT_SPECS[endpoint]
    value_norm, response_delta_norm = _response_norms(ground_truth, baseline, scenario_values, delta)
    extent = _extent_km(extent_m)

    panels = [
        (ground_truth, f"Ground truth {spec.label} (BurnP3+)", spec.cmap, value_norm, spec.units),
        (baseline, f"Baseline {spec.label} (model)", spec.cmap, value_norm, spec.units),
        (scenario_values, f"Scenario {spec.label} \u2014 {scenario_label}", spec.cmap, value_norm, spec.units),
        (delta, f"\u0394{spec.label} (scenario \u2212 baseline)", "RdBu_r", response_delta_norm, f"\u0394 {spec.units}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.6))
    for ax, (data, title, cmap, norm, cbar_label) in zip(axes, panels, strict=True):
        _render_panel(
            fig,
            ax,
            data,
            title,
            cmap,
            norm,
            cbar_label,
            extent=extent,
            downsample=downsample,
            zone_window=zone_labels,
            zone_overlay_color=zone_overlay_color,
            zone_overlay_linewidth=zone_overlay_linewidth,
            zone_overlay_alpha=zone_overlay_alpha,
        )
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_patch_zoom(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_values: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    *,
    endpoint: str,
    out_path: Path,
    scenario_label: str,
    suptitle: str,
    window: int,
    hotspot_block: int,
    patch_count: int,
    zone_labels: np.ma.MaskedArray | None = None,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Ground-truth/baseline/scenario/\u0394 zoom-ins on the highest-response windows."""
    spec = ENDPOINT_SPECS[endpoint]
    value_norm, response_delta_norm = _response_norms(ground_truth, baseline, scenario_values, delta)
    half = window // 2
    centers = hotspot_centers(delta, hotspot_block, count=patch_count, window=window)
    if not centers:
        print(f"No positive response found for {out_path.name}; skipping patch zoom plot.")
        return

    fig, axes = plt.subplots(len(centers), 4, figsize=(20.0, 5.0 * len(centers)), squeeze=False)
    for row, (center_row, center_col) in enumerate(centers):
        r0 = max(center_row - half, 0)
        c0 = max(center_col - half, 0)
        r1 = min(r0 + window, baseline.shape[0])
        c1 = min(c0 + window, baseline.shape[1])
        zone_window = zone_labels[r0:r1, c0:c1] if zone_labels is not None else None
        panels = [
            (ground_truth[r0:r1, c0:c1], f"Ground truth {spec.label} (BurnP3+)", spec.cmap, value_norm, spec.units),
            (baseline[r0:r1, c0:c1], f"Baseline {spec.label}", spec.cmap, value_norm, spec.units),
            (scenario_values[r0:r1, c0:c1], f"Scenario {spec.label} \u2014 {scenario_label}", spec.cmap, value_norm, spec.units),
            (delta[r0:r1, c0:c1], f"\u0394{spec.label}", "RdBu_r", response_delta_norm, f"\u0394 {spec.units}"),
        ]
        for col, (data, title, cmap, norm, cbar_label) in enumerate(panels):
            _render_panel(
                fig,
                axes[row, col],
                data,
                title,
                cmap,
                norm,
                cbar_label,
                zone_window=zone_window,
                zone_overlay_color=zone_overlay_color,
                zone_overlay_linewidth=zone_overlay_linewidth,
                zone_overlay_alpha=zone_overlay_alpha,
            )
        axes[row, 0].set_ylabel(f"Window #{row + 1}", labelpad=12)
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, choices=sorted(ENDPOINT_SPECS), help="Prediction endpoint to render.")
    parser.add_argument("--scenario", required=True, help="Scenario name from scenario_prediction_index.csv.")
    parser.add_argument("--label", default=None, help="Human-readable scenario label for titles (defaults to the name).")
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_fuel.yaml"))
    parser.add_argument("--experiment_dir", type=Path, default=None, help="Overrides save_dir from --config.")
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Overrides raw_data_dir from --config.")
    parser.add_argument("--patch_window", type=int, default=400, help="Side length (pixels) of patch zoom windows.")
    parser.add_argument("--hotspot_block", type=int, default=64, help="Block size for locating high-response windows.")
    parser.add_argument("--patch_count", type=int, default=3, help="Number of distinct high-response windows to render.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/<scenario>_<endpoint>.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.hex_id = normalize_hex_id(args.hex_id)
    label = args.label if args.label is not None else args.scenario
    config = load_counterfactual_config(args.config)
    experiment_dir, raw_data_dir = resolve_counterfactual_paths(
        config,
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
    )
    out_dir = args.out_dir if args.out_dir is not None else experiment_dir / "figures" / f"{args.scenario}_{args.endpoint}"
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    scenario_cfg = config.scenario(args.scenario)

    nonfuel_ids: list[int] | None = None
    static_nonfuel_ids: list[int] | None = None
    fuel_edit = scenario_cfg.fuel_edit()
    if fuel_edit is not None:
        nonfuel_ids = [int(value) for value in fuel_edit["nonfuel_ids"]]
    elif config.nonfuel_ids:
        static_nonfuel_ids = config.nonfuel_ids

    ground_truth, baseline, scenario_values, delta, extent, reference_profile = load_endpoint_response(
        experiment_dir,
        prediction_dirs,
        args.hex_id,
        raw_data_dir,
        scenario=args.scenario,
        endpoint=args.endpoint,
        nonfuel_ids=nonfuel_ids,
        static_nonfuel_ids=static_nonfuel_ids,
    )
    zone_labels = None
    if args.zone_overlay:
        zone_labels = load_zone_labels_on_prediction_grid(
            raw_data_dir=raw_data_dir,
            reference_profile=reference_profile,
            hex_id=args.hex_id,
            support=prediction_footprint(prediction_dirs, args.hex_id, endpoint=args.endpoint),
        )

    spec = ENDPOINT_SPECS[args.endpoint]
    plot_response_maps(
        ground_truth,
        baseline,
        scenario_values,
        delta,
        extent,
        endpoint=args.endpoint,
        out_path=out_dir / f"{args.scenario}_{args.endpoint}_response_maps.png",
        scenario_label=label,
        suptitle=f"{spec.label} response \u2014 {label} (hex {args.hex_id})",
        downsample=args.downsample,
        zone_labels=zone_labels,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
    )
    plot_patch_zoom(
        ground_truth,
        baseline,
        scenario_values,
        delta,
        endpoint=args.endpoint,
        out_path=out_dir / f"{args.scenario}_{args.endpoint}_patch_zoom.png",
        scenario_label=label,
        suptitle=f"High-response {args.patch_window}\u00d7{args.patch_window}-pixel windows \u2014 {label} (hex {args.hex_id})",
        window=args.patch_window,
        hotspot_block=args.hotspot_block,
        patch_count=args.patch_count,
        zone_labels=zone_labels,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
    )
    plot_delta_histogram(
        delta,
        out_path=out_dir / f"{args.scenario}_{args.endpoint}_delta_histogram.png",
        xlabel=f"\u0394{spec.label} per pixel (scenario \u2212 baseline, {spec.units})",
        title=f"Per-pixel {spec.label} change \u2014 {label} (hex {args.hex_id})",
        color=DELTA_COLOR,
    )
    plot_delta_concentration(
        delta,
        out_path=out_dir / f"{args.scenario}_{args.endpoint}_delta_concentration.png",
        title=f"{spec.label} change concentration \u2014 {label} (hex {args.hex_id})",
        color=DELTA_COLOR,
    )

    values = finite_values(delta)
    print(f"{args.scenario}/{args.endpoint}: n={values.size} mean_d={float(np.mean(values)):+.3f} median={float(np.median(values)):+.3f}")
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
