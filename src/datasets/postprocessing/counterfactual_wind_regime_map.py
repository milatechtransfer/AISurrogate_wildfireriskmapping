"""ROS response figures for the per-zone peak-wind regime counterfactual.

Generates scientist-facing figures for the ``wind_zone_peak_day`` scenario,
which transplants each weather zone's single highest-WindSpeed day onto every
day in the zone: the ground-truth/baseline/scenario/delta ROS maps, high-response
patch zoom-ins, the per-zone wind-speed shift driving the response, and the
hex-wide per-pixel ROS change distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.colors import Normalize, TwoSlopeNorm

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_hazard_map import (
    downsample_for_display,
    finite_values,
    prediction_dirs_from_index,
    prediction_raster_path,
    read_prediction,
    read_prediction_extent,
    restrict_to_support,
    symmetric_percentile_limit,
)
from src.datasets.postprocessing.fuel_barrier_geometry import parse_fuel_barrier_info

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

GT_RELATIVE_PATH = "results/burnP3Plus_OutputRateOfSpreadSummaryMap/fbpSummary-RateOfSpread-Average.tif"
SCENARIO = "wind_zone_peak_day"
ENDPOINT = "ros"
DELTA_COLOR = "#b2182b"


def _extent_km(extent_m: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent_m
    return ((left - left) / 1000.0, (right - left) / 1000.0, (bottom - bottom) / 1000.0, (top - bottom) / 1000.0)


def _raw_data_dir_from_config(experiment_dir: Path) -> Path:
    config_path = experiment_dir / "generated_configs" / f"baseline_{ENDPOINT}.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return Path(config["data"]["raw_data_dir"])


def load_ground_truth(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ma.MaskedArray:
    gt_path = raw_data_dir / f"hex{int(hex_id):02d}" / GT_RELATIVE_PATH
    gt, _ = load_spatial_raster(path=gt_path, reference_profile=reference_profile)
    return gt


def load_burnable_support(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ndarray:
    """Boolean mask of burnable pixels (non-fuel/water excluded) on the prediction grid."""
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_ma, _ = load_spatial_raster(path=paths.fuel_grid(hex_id), reference_profile=reference_profile)
    fuel_values = np.ma.asarray(fuel_ma).filled(-32768).astype(np.int32)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    return ~np.isin(fuel_values, fuel_info.nonfuel_ids)


def load_response(prediction_dirs: dict[tuple[str, str], Path], hex_id: str, raw_data_dir: Path):
    baseline_dir = prediction_dirs.get(("baseline", ENDPOINT))
    if baseline_dir is None:
        raise KeyError("Missing baseline ROS prediction directory.")
    baseline_path = prediction_raster_path(baseline_dir, hex_id)
    baseline = read_prediction(baseline_path)
    extent = read_prediction_extent(baseline_path)
    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()

    burnable = load_burnable_support(raw_data_dir, hex_id, reference_profile)
    support = burnable & ~np.ma.getmaskarray(baseline)

    ground_truth = load_ground_truth(raw_data_dir, hex_id, reference_profile)
    if ground_truth.shape != baseline.shape:
        raise ValueError(f"Ground-truth shape {ground_truth.shape} does not match prediction grid {baseline.shape}.")
    ground_truth = restrict_to_support(ground_truth, support)
    baseline = restrict_to_support(baseline, support)

    scenario_dir = prediction_dirs.get((SCENARIO, ENDPOINT))
    if scenario_dir is None:
        raise KeyError(f"Missing ROS prediction directory for scenario={SCENARIO!r}.")
    scenario_ros = read_prediction(prediction_raster_path(scenario_dir, hex_id))
    if scenario_ros.shape != baseline.shape:
        raise ValueError(f"Scenario ROS shape {scenario_ros.shape} does not match baseline grid {baseline.shape}.")
    scenario_ros = restrict_to_support(scenario_ros, support)
    delta = restrict_to_support(scenario_ros - baseline, support)
    return ground_truth, baseline, scenario_ros, delta, extent, reference_profile


def load_zone_labels(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ma.MaskedArray:
    firezones_path = raw_data_dir / f"hex{int(hex_id):02d}" / f"spatial/hex{int(hex_id):02d}_firezones.tif"
    zones, _ = load_spatial_raster(path=firezones_path, reference_profile=reference_profile)
    return zones


def plot_response_maps(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_ros: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    extent_m: tuple[float, float, float, float],
    *,
    out_path: Path,
    downsample: int,
) -> None:
    pooled = np.concatenate([finite_values(ground_truth), finite_values(baseline), finite_values(scenario_ros)])
    ros_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
    delta_limit = symmetric_percentile_limit([delta], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    extent = _extent_km(extent_m)

    panels = [
        (ground_truth, "Ground truth ROS (BurnP3+)", "viridis", ros_norm, "ROS (m/min)"),
        (baseline, "Baseline ROS (model)", "viridis", ros_norm, "ROS (m/min)"),
        (scenario_ros, "Scenario ROS \u2014 peak-wind regime", "viridis", ros_norm, "ROS (m/min)"),
        (delta, "\u0394ROS (scenario \u2212 baseline)", "RdBu_r", delta_norm, "\u0394ROS (m/min)"),
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
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, fontsize=14)
    fig.suptitle("ROS response to per-zone peak-wind regime transplant (hex 16)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _block_response(delta: np.ma.MaskedArray, block: int) -> np.ndarray:
    abs_delta = np.abs(np.ma.filled(delta, 0.0))
    valid = (~np.ma.getmaskarray(delta)).astype(np.float64)
    rows = (abs_delta.shape[0] // block) * block
    cols = (abs_delta.shape[1] // block) * block
    summed = abs_delta[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    counts = valid[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    return np.where(counts > 0, summed / np.maximum(counts, 1.0), 0.0)


def _hotspot_centers(delta: np.ma.MaskedArray, block: int, *, count: int, window: int) -> list[tuple[int, int]]:
    scores = _block_response(delta, block)
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


def plot_patch_zoom(
    baseline: np.ma.MaskedArray,
    scenario_ros: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    *,
    out_path: Path,
    window: int,
    hotspot_block: int,
    patch_count: int,
) -> None:
    pooled = np.concatenate([finite_values(baseline), finite_values(scenario_ros)])
    ros_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
    delta_limit = symmetric_percentile_limit([delta], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    half = window // 2

    centers = _hotspot_centers(delta, hotspot_block, count=patch_count, window=window)
    fig, axes = plt.subplots(len(centers), 3, figsize=(15.0, 5.0 * len(centers)), squeeze=False)
    for row, (center_row, center_col) in enumerate(centers):
        r0 = max(center_row - half, 0)
        c0 = max(center_col - half, 0)
        r1 = min(r0 + window, baseline.shape[0])
        c1 = min(c0 + window, baseline.shape[1])
        panels = [
            (baseline[r0:r1, c0:c1], "Baseline ROS", "viridis", ros_norm, "ROS (m/min)"),
            (scenario_ros[r0:r1, c0:c1], "Scenario ROS \u2014 peak-wind", "viridis", ros_norm, "ROS (m/min)"),
            (delta[r0:r1, c0:c1], "\u0394ROS", "RdBu_r", delta_norm, "\u0394ROS (m/min)"),
        ]
        for col, (data, title, cmap, norm, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(data, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label, fontsize=14)
        axes[row, 0].set_ylabel(f"Window #{row + 1}", labelpad=12)
    fig.suptitle(f"High-response {window}\u00d7{window}-pixel windows (hex 16)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_zone_wind_shift(edit_summary_path: Path, *, out_path: Path) -> None:
    summary = pd.read_csv(edit_summary_path)
    summary = summary[summary["scenario"] == SCENARIO].sort_values("zone")
    zones = summary["zone"].astype(int).to_numpy()
    positions = np.arange(zones.size)
    width = 0.4

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(positions - width / 2, summary["baseline_speed_mean"], width, label="Baseline zone-mean", color="0.6")
    ax.bar(positions + width / 2, summary["scenario_speed_mean"], width, label="Peak-wind day", color=DELTA_COLOR)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(zone) for zone in zones])
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Weather zone", fontsize=16)
    ax.set_ylabel("WindSpeed (km/h)", fontsize=16)
    ax.set_title("Input intervention: per-zone wind-speed shift driving the ROS response (hex 16)", fontsize=15)
    ax.legend(fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_distribution(delta: np.ma.MaskedArray, *, out_path: Path) -> None:
    values = finite_values(delta)
    limit = float(np.percentile(np.abs(values), 99.5))
    bins = np.linspace(-limit, limit, 201).tolist()
    mean = float(np.mean(values))
    frac_positive = float(np.mean(values > 0))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(values, bins=bins, histtype="step", linewidth=1.8, color=DELTA_COLOR)
    ax.axvline(mean, color=DELTA_COLOR, linestyle="--", linewidth=1.2, label=f"mean \u0394={mean:+.2f}")
    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel("\u0394ROS per pixel (scenario \u2212 baseline, m/min)")
    ax.set_ylabel("Pixel count")
    ax.set_yscale("log")
    ax.set_title(f"Per-pixel ROS change under peak-wind transplant (hex 16)\n{frac_positive:.1%} of pixels increase")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Defaults to baseline config data.raw_data_dir.")
    parser.add_argument("--patch_window", type=int, default=400, help="Side length (pixels) of patch zoom windows.")
    parser.add_argument("--hotspot_block", type=int, default=64, help="Block size for locating high-response windows.")
    parser.add_argument("--patch_count", type=int, default=3, help="Number of distinct high-response windows to render.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/wind_zone_peak.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / "wind_zone_peak"
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else _raw_data_dir_from_config(args.experiment_dir)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)
    ground_truth, baseline, scenario_ros, delta, extent, _ = load_response(prediction_dirs, args.hex_id, raw_data_dir)

    plot_response_maps(
        ground_truth,
        baseline,
        scenario_ros,
        delta,
        extent,
        out_path=out_dir / "wind_zone_peak_ros_response_maps.png",
        downsample=args.downsample,
    )
    plot_patch_zoom(
        baseline,
        scenario_ros,
        delta,
        out_path=out_dir / "wind_zone_peak_patch_zoom.png",
        window=args.patch_window,
        hotspot_block=args.hotspot_block,
        patch_count=args.patch_count,
    )
    plot_delta_distribution(delta, out_path=out_dir / "wind_zone_peak_ros_delta_distribution.png")

    edit_summary_path = args.experiment_dir / "wind_edit_summary.csv"
    if edit_summary_path.exists():
        plot_zone_wind_shift(edit_summary_path, out_path=out_dir / "wind_zone_peak_zone_wind_shift.png")

    values = finite_values(delta)
    print(f"{SCENARIO}: n={values.size} mean_dROS={float(np.mean(values)):+.3f} median={float(np.median(values)):+.3f}")
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
