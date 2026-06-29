"""Shared ROS response figures for weather-scenario counterfactuals.

Ground-truth/baseline/scenario/delta ROS maps and ground-truth-anchored
high-response patch zoom-ins, parametrised by scenario so the per-zone peak-wind
regime and uniform wind-direction scripts render identical panels without
duplicating the plotting code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm

from src.datasets.postprocessing.counterfactual_viz import (
    downsample_for_display,
    finite_values,
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

GT_RELATIVE_PATH = "results/burnP3Plus_OutputRateOfSpreadSummaryMap/fbpSummary-RateOfSpread-Average.tif"
ENDPOINT = "ros"


def load_ros_response(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    raw_data_dir: Path,
    *,
    scenario: str,
    endpoint: str = ENDPOINT,
):
    """Load baseline, scenario, and Δ ROS on the prediction grid for one scenario."""

    baseline_dir = prediction_dirs.get(("baseline", endpoint))
    if baseline_dir is None:
        raise KeyError(f"Missing baseline {endpoint.upper()} prediction directory.")
    baseline_path = prediction_raster_path(baseline_dir, hex_id)
    baseline = read_prediction(baseline_path)
    extent = read_prediction_extent(baseline_path)
    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()

    burnable = load_burnable_support(raw_data_dir, hex_id, reference_profile)
    support = burnable & ~np.ma.getmaskarray(baseline)

    ground_truth = load_ground_truth(raw_data_dir, hex_id, reference_profile, gt_relative_path=GT_RELATIVE_PATH)
    if ground_truth.shape != baseline.shape:
        raise ValueError(f"Ground-truth shape {ground_truth.shape} does not match prediction grid {baseline.shape}.")
    ground_truth = restrict_to_support(ground_truth, support)
    baseline = restrict_to_support(baseline, support)

    scenario_dir = prediction_dirs.get((scenario, endpoint))
    if scenario_dir is None:
        raise KeyError(f"Missing {endpoint.upper()} prediction directory for scenario={scenario!r}.")
    scenario_ros = read_prediction(prediction_raster_path(scenario_dir, hex_id))
    if scenario_ros.shape != baseline.shape:
        raise ValueError(f"Scenario ROS shape {scenario_ros.shape} does not match baseline grid {baseline.shape}.")
    scenario_ros = restrict_to_support(scenario_ros, support)
    delta = restrict_to_support(scenario_ros - baseline, support)
    return ground_truth, baseline, scenario_ros, delta, extent, reference_profile


def hotspot_centers(delta: np.ma.MaskedArray, block: int, *, count: int, window: int) -> list[tuple[int, int]]:
    """Centres of the ``count`` highest-response windows, suppressed to be distinct."""

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


def plot_ros_response_maps(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_ros: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    extent_m: tuple[float, float, float, float],
    *,
    out_path: Path,
    scenario_label: str,
    suptitle: str,
    downsample: int,
) -> None:
    """Ground-truth / baseline / scenario / Δ ROS maps across the whole hex."""

    pooled = np.concatenate([finite_values(ground_truth), finite_values(baseline), finite_values(scenario_ros)])
    ros_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
    delta_limit = symmetric_percentile_limit([delta], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    extent = extent_km(extent_m)

    panels = [
        (ground_truth, "Ground truth ROS (BurnP3+)", "viridis", ros_norm, "ROS (m/min)"),
        (baseline, "Baseline ROS (model)", "viridis", ros_norm, "ROS (m/min)"),
        (scenario_ros, f"Scenario ROS \u2014 {scenario_label}", "viridis", ros_norm, "ROS (m/min)"),
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
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ros_patch_zoom(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenario_ros: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    *,
    out_path: Path,
    scenario_label: str,
    suptitle: str,
    window: int,
    hotspot_block: int,
    patch_count: int,
) -> None:
    """Ground-truth/baseline/scenario/Δ ROS zoom-ins on the highest-response windows."""

    pooled = np.concatenate([finite_values(ground_truth), finite_values(baseline), finite_values(scenario_ros)])
    ros_norm = Normalize(vmin=0.0, vmax=float(np.percentile(pooled, 99.0)))
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
        panels = [
            (ground_truth[r0:r1, c0:c1], "Ground truth ROS (BurnP3+)", "viridis", ros_norm, "ROS (m/min)"),
            (baseline[r0:r1, c0:c1], "Baseline ROS", "viridis", ros_norm, "ROS (m/min)"),
            (scenario_ros[r0:r1, c0:c1], f"Scenario ROS \u2014 {scenario_label}", "viridis", ros_norm, "ROS (m/min)"),
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
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
