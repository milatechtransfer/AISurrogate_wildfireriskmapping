"""Generic BP response figures for a single counterfactual scenario."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm

from src.datasets.postprocessing.counterfactual import load_counterfactual_config
from src.datasets.postprocessing.counterfactual_viz import (
    DEFAULT_ZONE_OVERLAY_ALPHA,
    DEFAULT_ZONE_OVERLAY_COLOR,
    DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    add_zone_overlay_args,
    downsample_for_display,
    finite_values,
    overlay_zone_boundaries,
    plot_delta_histogram,
    prediction_dirs_from_index,
    prediction_footprint,
    prediction_raster_path,
    read_prediction,
    read_prediction_extent,
    restrict_to_support,
    symmetric_percentile_limit,
)
from src.datasets.postprocessing.counterfactual_weather_maps import (
    block_response,
    extent_km,
    load_burnable_support,
    load_ground_truth,
    load_zone_labels,
    raw_data_dir_from_config,
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

GT_RELATIVE_PATH = "results/burnP3Plus_OutputBurnProbability/burnProbability-sn2.tif"
ENDPOINT = "bp"
DELTA_COLOR = "#b2182b"
PANEL_TITLE_WIDTH = 30


def _panel_title(title: str) -> str:
    return textwrap.fill(title, width=PANEL_TITLE_WIDTH)


def load_bp_response(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    raw_data_dir: Path,
    *,
    scenario: str,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray, tuple[float, float, float, float], dict]:
    baseline_dir = prediction_dirs.get(("baseline", ENDPOINT))
    scenario_dir = prediction_dirs.get((scenario, ENDPOINT))
    if baseline_dir is None:
        raise KeyError("Missing baseline BP prediction directory.")
    if scenario_dir is None:
        raise KeyError(f"Missing BP prediction directory for scenario={scenario!r}.")

    baseline_path = prediction_raster_path(baseline_dir, hex_id)
    baseline = read_prediction(baseline_path)
    scenario_bp = read_prediction(prediction_raster_path(scenario_dir, hex_id))
    if scenario_bp.shape != baseline.shape:
        raise ValueError(f"Scenario BP shape {scenario_bp.shape} does not match baseline grid {baseline.shape}.")

    extent = read_prediction_extent(baseline_path)
    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()

    finite = ~np.ma.getmaskarray(baseline) & ~np.ma.getmaskarray(scenario_bp)
    support = load_burnable_support(raw_data_dir, hex_id, reference_profile) & finite
    ground_truth = load_ground_truth(raw_data_dir, hex_id, reference_profile, gt_relative_path=GT_RELATIVE_PATH)
    if ground_truth.shape != baseline.shape:
        raise ValueError(f"Ground-truth shape {ground_truth.shape} does not match prediction grid {baseline.shape}.")

    ground_truth = restrict_to_support(ground_truth, support)
    baseline = restrict_to_support(baseline, support)
    scenario_bp = restrict_to_support(scenario_bp, support)
    delta = restrict_to_support(scenario_bp - baseline, support)
    return ground_truth, baseline, scenario_bp, delta, extent, reference_profile


def hotspot_centers(delta: np.ma.MaskedArray, block: int, *, count: int, window: int) -> list[tuple[int, int]]:
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


def plot_bp_response_maps(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_bp: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    extent_m: tuple[float, float, float, float],
    *,
    out_path: Path,
    scenario_label: str,
    suptitle: str,
    downsample: int,
    zone_labels: np.ma.MaskedArray | None = None,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    pooled = np.concatenate([finite_values(ground_truth), finite_values(baseline), finite_values(scenario_bp)])
    bp_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
    delta_limit = symmetric_percentile_limit([delta], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    extent = extent_km(extent_m)
    panels = [
        (ground_truth, "Ground truth BP (BurnP3+)", "viridis", bp_norm, "Burn probability"),
        (baseline, "Baseline BP (model)", "viridis", bp_norm, "Burn probability"),
        (scenario_bp, f"Scenario BP \u2014 {scenario_label}", "viridis", bp_norm, "Burn probability"),
        (delta, "\u0394BP (scenario \u2212 baseline)", "RdBu_r", delta_norm, "\u0394 Burn probability"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.6))
    for ax, (data, title, cmap, norm, cbar_label) in zip(axes, panels, strict=True):
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
            zone_labels,
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
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bp_patch_zoom(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_bp: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    *,
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
    pooled = np.concatenate([finite_values(ground_truth), finite_values(baseline), finite_values(scenario_bp)])
    bp_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
    delta_limit = symmetric_percentile_limit([delta], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    half = window // 2
    centers = hotspot_centers(delta, hotspot_block, count=patch_count, window=window)

    fig, axes = plt.subplots(len(centers), 4, figsize=(20.0, 5.0 * len(centers)), squeeze=False)
    for row, (center_row, center_col) in enumerate(centers):
        r0 = max(center_row - half, 0)
        c0 = max(center_col - half, 0)
        r1 = min(r0 + window, baseline.shape[0])
        c1 = min(c0 + window, baseline.shape[1])
        zone_window = zone_labels[r0:r1, c0:c1] if zone_labels is not None else None
        panels = [
            (ground_truth[r0:r1, c0:c1], "Ground truth BP (BurnP3+)", "viridis", bp_norm, "Burn probability"),
            (baseline[r0:r1, c0:c1], "Baseline BP", "viridis", bp_norm, "Burn probability"),
            (scenario_bp[r0:r1, c0:c1], f"Scenario BP \u2014 {scenario_label}", "viridis", bp_norm, "Burn probability"),
            (delta[r0:r1, c0:c1], "\u0394BP", "RdBu_r", delta_norm, "\u0394 Burn probability"),
        ]
        for col, (data, title, cmap, norm, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(data, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
            overlay_zone_boundaries(
                ax,
                zone_window,
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
        axes[row, 0].set_ylabel(f"Window #{row + 1}", labelpad=12)
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="Scenario name from scenario_prediction_index.csv.")
    parser.add_argument("--label", default=None, help="Human-readable scenario label for titles (defaults to the name).")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_hex16.yaml"))
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Defaults to baseline config data.raw_data_dir.")
    parser.add_argument("--patch_window", type=int, default=400, help="Side length (pixels) of patch zoom windows.")
    parser.add_argument("--hotspot_block", type=int, default=64, help="Block size for locating high-response windows.")
    parser.add_argument("--patch_count", type=int, default=3, help="Number of distinct high-response windows to render.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/<scenario>_bp.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label if args.label is not None else args.scenario
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / f"{args.scenario}_bp"
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else raw_data_dir_from_config(args.experiment_dir, endpoint=ENDPOINT)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)

    config = load_counterfactual_config(args.config)
    if not any(s.name == args.scenario for s in config.scenarios):
        raise KeyError(f"Scenario {args.scenario!r} not found in {args.config}.")

    ground_truth, baseline, scenario_bp, delta, extent, reference_profile = load_bp_response(
        prediction_dirs,
        args.hex_id,
        raw_data_dir,
        scenario=args.scenario,
    )
    zone_labels = None
    if args.zone_overlay:
        zone_labels = load_zone_labels(
            raw_data_dir,
            args.hex_id,
            reference_profile,
            support=prediction_footprint(prediction_dirs, args.hex_id, endpoint=ENDPOINT),
        )

    plot_bp_response_maps(
        ground_truth,
        baseline,
        scenario_bp,
        delta,
        extent,
        out_path=out_dir / f"{args.scenario}_bp_response_maps.png",
        scenario_label=label,
        suptitle=f"BP response \u2014 {label} (hex 16)",
        downsample=args.downsample,
        zone_labels=zone_labels,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
    )
    plot_bp_patch_zoom(
        ground_truth,
        baseline,
        scenario_bp,
        delta,
        out_path=out_dir / f"{args.scenario}_bp_patch_zoom.png",
        scenario_label=label,
        suptitle=f"High-response {args.patch_window}\u00d7{args.patch_window}-pixel windows \u2014 {label} (hex 16)",
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
        out_path=out_dir / f"{args.scenario}_bp_delta_distribution.png",
        xlabel="\u0394BP per pixel (scenario \u2212 baseline)",
        title=f"Per-pixel BP change \u2014 {label} (hex 16)",
        color=DELTA_COLOR,
    )

    values = finite_values(delta)
    print(f"{args.scenario}: n={values.size} mean_dBP={float(np.mean(values)):+.4f} median={float(np.median(values)):+.4f}")
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
