"""Local zoom panels around selected barrier-removal components."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch

from src.datasets.postprocessing.counterfactual_fuel import replace_nonfuel_components_with_adjacent_modal
from src.datasets.postprocessing.counterfactual_fuel_intervention_map import (
    FUEL_GROUP_COLOURS,
    FUEL_GROUP_LABELS,
    _categorical_codes,
)
from src.datasets.postprocessing.counterfactual_viz import (
    DEFAULT_ZONE_OVERLAY_ALPHA,
    DEFAULT_ZONE_OVERLAY_COLOR,
    DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    add_zone_overlay_args,
    finite_values,
    overlay_zone_boundaries,
    prediction_dirs_from_index,
    prediction_raster_path,
    prediction_reference_profile,
    read_prediction,
    values_and_valid,
)
from src.datasets.postprocessing.counterfactual_weather_maps import load_zone_labels
from src.datasets.postprocessing.fuel_barrier_geometry import load_grouped_fuel_on_prediction_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO = "remove_barriers_adjacent_modal"


class _SliceWindow(Protocol):
    """Structural type for window dataclasses exposing crop slices."""

    @property
    def row_slice(self) -> slice: ...

    @property
    def col_slice(self) -> slice: ...


@dataclass(frozen=True)
class NeighborhoodWindow:
    window_id: int
    rank: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    barrier_pixels: int
    barrier_density: float
    valid_fraction: float
    delta_hazard_mean: float
    delta_fi_mean: float
    score: float

    @property
    def row_slice(self) -> slice:
        return slice(self.row_min, self.row_max)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_min, self.col_max)


@dataclass(frozen=True)
class NeighborhoodSummary:
    scenario: str
    hex_id: str
    window_id: int
    rank: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    barrier_pixels: int
    barrier_density: float
    valid_fraction: float
    baseline_fi_mean: float
    scenario_fi_mean: float
    delta_fi_mean: float
    baseline_hazard_mean: float
    scenario_hazard_mean: float
    delta_hazard_mean: float
    score: float


def integral_image(values: np.ndarray) -> np.ndarray:
    """Return a padded integral image for fast rectangular sums."""

    return np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")


def window_sum(integral: np.ndarray, row_min: int, row_max: int, col_min: int, col_max: int) -> float:
    return float(integral[row_max, col_max] - integral[row_min, col_max] - integral[row_max, col_min] + integral[row_min, col_min])


def candidate_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if crop_size > length:
        return [0]
    starts = list(range(0, length - crop_size + 1, stride))
    final = length - crop_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _iou(a: NeighborhoodWindow, b: NeighborhoodWindow) -> float:
    row_overlap = max(0, min(a.row_max, b.row_max) - max(a.row_min, b.row_min))
    col_overlap = max(0, min(a.col_max, b.col_max) - max(a.col_min, b.col_min))
    intersection = row_overlap * col_overlap
    if intersection == 0:
        return 0.0
    area_a = (a.row_max - a.row_min) * (a.col_max - a.col_min)
    area_b = (b.row_max - b.row_min) * (b.col_max - b.col_min)
    return float(intersection / (area_a + area_b - intersection))


def select_neighborhood_windows(
    *,
    barrier_mask: np.ndarray,
    valid_mask: np.ndarray,
    delta_hazard: np.ndarray,
    delta_fi: np.ndarray,
    n_windows: int = 3,
    crop_size: int = 700,
    stride: int = 140,
    min_barrier_pixels: int = 1500,
    min_barrier_density: float = 0.01,
    max_barrier_density: float = 0.35,
    min_valid_fraction: float = 0.95,
    max_iou: float = 0.15,
    exclude_border_pixels: int = 100,
) -> list[NeighborhoodWindow]:
    """Select fixed-size windows that clearly contain edited barriers and response."""

    if barrier_mask.shape != valid_mask.shape or barrier_mask.shape != delta_hazard.shape or barrier_mask.shape != delta_fi.shape:
        raise ValueError("barrier_mask, valid_mask, delta_hazard, and delta_fi must have the same shape.")
    if n_windows < 1:
        raise ValueError("n_windows must be positive.")
    if crop_size < 1:
        raise ValueError("crop_size must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")

    h, w = barrier_mask.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    area = float(crop_h * crop_w)

    finite_hazard = valid_mask & np.isfinite(delta_hazard)
    finite_fi = valid_mask & np.isfinite(delta_fi)
    barrier_integral = integral_image(barrier_mask.astype(np.float64))
    valid_integral = integral_image(valid_mask.astype(np.float64))
    hazard_sum_integral = integral_image(np.where(finite_hazard, delta_hazard, 0.0))
    hazard_count_integral = integral_image(finite_hazard.astype(np.float64))
    fi_sum_integral = integral_image(np.where(finite_fi, delta_fi, 0.0))
    fi_count_integral = integral_image(finite_fi.astype(np.float64))

    candidates: list[NeighborhoodWindow] = []
    window_id = 0
    for row_min in candidate_starts(h, crop_h, stride):
        row_max = row_min + crop_h
        for col_min in candidate_starts(w, crop_w, stride):
            col_max = col_min + crop_w
            if exclude_border_pixels > 0 and (
                row_min < exclude_border_pixels
                or col_min < exclude_border_pixels
                or row_max > h - exclude_border_pixels
                or col_max > w - exclude_border_pixels
            ):
                continue
            barrier_pixels = int(window_sum(barrier_integral, row_min, row_max, col_min, col_max))
            if barrier_pixels < min_barrier_pixels:
                continue
            barrier_density = barrier_pixels / area
            if barrier_density < min_barrier_density or barrier_density > max_barrier_density:
                continue
            valid_pixels = window_sum(valid_integral, row_min, row_max, col_min, col_max)
            valid_fraction = valid_pixels / area
            if valid_fraction < min_valid_fraction:
                continue
            hazard_count = window_sum(hazard_count_integral, row_min, row_max, col_min, col_max)
            fi_count = window_sum(fi_count_integral, row_min, row_max, col_min, col_max)
            if hazard_count <= 0 or fi_count <= 0:
                continue
            delta_hazard_mean = window_sum(hazard_sum_integral, row_min, row_max, col_min, col_max) / hazard_count
            delta_fi_mean = window_sum(fi_sum_integral, row_min, row_max, col_min, col_max) / fi_count
            if delta_hazard_mean <= 0.0:
                continue

            density_preference = max(0.1, 1.0 - abs(barrier_density - 0.08) / 0.08)
            score = float(delta_hazard_mean * np.sqrt(barrier_pixels) * density_preference)
            window_id += 1
            candidates.append(
                NeighborhoodWindow(
                    window_id=window_id,
                    rank=0,
                    row_min=row_min,
                    row_max=row_max,
                    col_min=col_min,
                    col_max=col_max,
                    barrier_pixels=barrier_pixels,
                    barrier_density=float(barrier_density),
                    valid_fraction=float(valid_fraction),
                    delta_hazard_mean=float(delta_hazard_mean),
                    delta_fi_mean=float(delta_fi_mean),
                    score=score,
                )
            )

    if not candidates:
        raise ValueError("No neighborhood windows passed the selection criteria.")

    selected: list[NeighborhoodWindow] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(_iou(candidate, existing) <= max_iou for existing in selected):
            selected.append(candidate)
        if len(selected) >= n_windows:
            break
    if len(selected) < n_windows:
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= n_windows:
                break

    return [
        NeighborhoodWindow(
            window_id=window.window_id,
            rank=rank,
            row_min=window.row_min,
            row_max=window.row_max,
            col_min=window.col_min,
            col_max=window.col_max,
            barrier_pixels=window.barrier_pixels,
            barrier_density=window.barrier_density,
            valid_fraction=window.valid_fraction,
            delta_hazard_mean=window.delta_hazard_mean,
            delta_fi_mean=window.delta_fi_mean,
            score=window.score,
        )
        for rank, window in enumerate(selected, start=1)
    ]


def replacement_group_map_on_prediction_grid(grouped_fuel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return original non-fuel mask and its local-modal replacement group map."""

    edited, original_nonfuel, _, _ = replace_nonfuel_components_with_adjacent_modal(
        grouped_fuel,
        [0],
        scenario_name=SCENARIO,
    )
    replacement_map = np.full(grouped_fuel.shape, np.nan, dtype=np.float32)
    replacement_map[original_nonfuel] = edited[original_nonfuel]
    return original_nonfuel, replacement_map


def robust_norm(arrays: list[np.ndarray | np.ma.MaskedArray], *, low: float = 1.0, high: float = 99.0) -> Normalize:
    values = [finite_values(array) for array in arrays]
    values = [value for value in values if value.size > 0]
    if not values:
        return Normalize(vmin=0.0, vmax=1.0)
    pooled = np.concatenate(values)
    vmin = float(np.percentile(pooled, low))
    vmax = float(np.percentile(pooled, high))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def delta_norm(arrays: list[np.ndarray | np.ma.MaskedArray], *, percentile: float = 99.0) -> TwoSlopeNorm:
    values = [np.abs(finite_values(array)) for array in arrays]
    values = [value for value in values if value.size > 0]
    limit = float(np.percentile(np.concatenate(values), percentile)) if values else 1.0
    limit = max(limit, 1e-9)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def crop(data: np.ndarray | np.ma.MaskedArray, window: _SliceWindow) -> np.ma.MaskedArray:
    return np.ma.asarray(data)[window.row_slice, window.col_slice]


def _mask_overlay(ax: plt.Axes, mask: np.ndarray, *, colour: str = "#ff00ff", alpha: float = 0.35) -> None:
    overlay = np.ma.masked_where(~mask.astype(bool), np.ones(mask.shape, dtype=np.float32))
    cmap = matplotlib.colors.ListedColormap([colour])
    ax.imshow(overlay, cmap=cmap, interpolation="nearest", origin="upper", alpha=alpha)


def _norm_limits(norm: Normalize) -> str:
    return f"[{float(cast(float, norm.vmin)):.2g}, {float(cast(float, norm.vmax)):.2g}]"


def _replacement_legend(categories: list[int]) -> list[Patch]:
    return [
        Patch(
            facecolor=FUEL_GROUP_COLOURS.get(category, "#999999"),
            edgecolor="none",
            label=f"{category}: {FUEL_GROUP_LABELS.get(category, f'Group {category}')}",
        )
        for category in categories
    ]


def plot_neighborhood_grid(
    *,
    grouped_fuel: np.ndarray,
    replacement_map: np.ndarray,
    barrier_mask: np.ndarray,
    valid_mask: np.ndarray,
    baseline: np.ma.MaskedArray,
    scenario_values: np.ma.MaskedArray,
    delta: np.ma.MaskedArray,
    windows: list[NeighborhoodWindow],
    endpoint_label: str,
    sequential_label: str,
    delta_label: str,
    out_path: Path,
    row_labels: list[str] | None = None,
    zone_labels: np.ma.MaskedArray | None = None,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Plot intervention, replacement, baseline, scenario, and delta for selected windows."""

    sequential_norm = robust_norm(
        [
            np.ma.masked_where(~crop(valid_mask, window).filled(False).astype(bool), crop(array, window))
            for window in windows
            for array in (baseline, scenario_values)
        ],
        low=1.0,
        high=99.0,
    )
    diverging_norm = delta_norm(
        [np.ma.masked_where(~crop(valid_mask, window).filled(False).astype(bool), crop(delta, window)) for window in windows],
        percentile=99.0,
    )
    all_replacement_values = []
    for window in windows:
        values = finite_values(crop(replacement_map, window)).astype(np.int32)
        if values.size:
            all_replacement_values.extend(values.tolist())
    replacement_categories = sorted(set(map(int, all_replacement_values)))
    replacement_cmap = matplotlib.colors.ListedColormap(
        [FUEL_GROUP_COLOURS.get(category, "#999999") for category in replacement_categories]
    )
    replacement_cmap.set_bad(color="white", alpha=0.0)
    sequential_cmap = plt.get_cmap("viridis").copy()
    sequential_cmap.set_bad(color="#d9d9d9", alpha=1.0)
    delta_cmap = plt.get_cmap("RdBu_r").copy()
    delta_cmap.set_bad(color="#d9d9d9", alpha=1.0)

    n_rows = len(windows)
    fig = plt.figure(figsize=(18.6, max(5.0, 3.35 * n_rows + 0.6)))
    grid = fig.add_gridspec(
        n_rows,
        8,
        width_ratios=[1, 1, 1, 1, 0.045, 0.18, 1, 0.045],
        left=0.04,
        right=0.99,
        bottom=0.05,
        top=0.84,
        wspace=0.055,
        hspace=0.08,
    )
    grid_cols = [0, 1, 2, 3, 6]
    axes = np.array([[fig.add_subplot(grid[row_idx, grid_col]) for grid_col in grid_cols] for row_idx in range(n_rows)])
    sequential_cax = fig.add_subplot(grid[:, 4])
    delta_cax = fig.add_subplot(grid[:, 7])
    sequential_image = None
    delta_image = None
    for row_idx, window in enumerate(windows):
        row_label = row_labels[row_idx] if row_labels is not None and row_idx < len(row_labels) else f"Neighborhood {window.rank}"
        support_crop = crop(valid_mask, window).filled(False).astype(bool)
        barrier_crop = crop(barrier_mask, window).filled(False).astype(bool) & support_crop
        zone_crop = crop(zone_labels, window) if zone_labels is not None else None
        replacement_crop = np.ma.masked_where(~support_crop, crop(replacement_map, window))
        replacement_codes = _categorical_codes(replacement_crop, replacement_categories)
        support_context = np.where(support_crop, 1.0, 0.0)
        support_cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#f7f7f7"])

        panels = (
            ("Original barrier mask", None, None, None),
            ("Replacement fuel group", replacement_codes, replacement_cmap, None),
            (
                f"Baseline {endpoint_label}",
                np.ma.masked_where(~support_crop, crop(baseline, window)),
                sequential_cmap,
                sequential_norm,
            ),
            (
                f"Counterfactual {endpoint_label}",
                np.ma.masked_where(~support_crop, crop(scenario_values, window)),
                sequential_cmap,
                sequential_norm,
            ),
            (f"Δ{endpoint_label}", np.ma.masked_where(~support_crop, crop(delta, window)), delta_cmap, diverging_norm),
        )
        for col_idx, (title, array, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ax.imshow(support_context, cmap=support_cmap, interpolation="nearest", origin="upper")
                _mask_overlay(ax, barrier_crop, colour="#000000", alpha=0.85)
            elif col_idx == 1:
                ax.imshow(support_context, cmap=support_cmap, interpolation="nearest", origin="upper")
                ax.imshow(array, cmap=cmap, interpolation="nearest", origin="upper")
            else:
                image = ax.imshow(array, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
                if col_idx < 4:
                    sequential_image = image
                else:
                    delta_image = image
            overlay_zone_boundaries(
                ax,
                zone_crop,
                color=zone_overlay_color,
                linewidth=zone_overlay_linewidth,
                alpha=zone_overlay_alpha,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{row_label}\n" f"{window.barrier_pixels:,} barrier px\n" f"mean Δhazard {window.delta_hazard_mean:.2f}",
                    fontsize=8,
                )

    if sequential_image is not None:
        cbar = fig.colorbar(sequential_image, cax=sequential_cax)
        cbar.set_label(f"{sequential_label} shared scale {_norm_limits(sequential_norm)}")
    if delta_image is not None:
        cbar = fig.colorbar(delta_image, cax=delta_cax)
        cbar.set_label(f"{delta_label} centered scale {_norm_limits(diverging_norm)}")
    handles = [Patch(facecolor="#000000", edgecolor="none", alpha=0.85, label="Original non-fuel pixels")]
    handles.extend(_replacement_legend(replacement_categories))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        columnspacing=1.0,
        handlelength=1.0,
    )
    fig.suptitle(
        f"Local barrier-removal neighborhoods: intervention and {endpoint_label} response",
        y=0.965,
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _prediction_arrays(
    experiment_dir: Path,
    *,
    scenario: str,
    hex_id: str,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]:
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline_bp = read_prediction(prediction_raster_path(prediction_dirs[("baseline", "bp")], hex_id))
    scenario_bp = read_prediction(prediction_raster_path(prediction_dirs[(scenario, "bp")], hex_id))
    baseline_fi = read_prediction(prediction_raster_path(prediction_dirs[("baseline", "fi")], hex_id))
    scenario_fi = read_prediction(prediction_raster_path(prediction_dirs[(scenario, "fi")], hex_id))
    return baseline_bp, scenario_bp, baseline_fi, scenario_fi


def _mean_on_crop(data: np.ndarray | np.ma.MaskedArray, window: _SliceWindow) -> float:
    values = finite_values(crop(data, window))
    return float(np.mean(values)) if values.size else float("nan")


def write_local_neighborhood_panels(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    scenario: str = SCENARIO,
    hex_id: str = "16",
    n_windows: int = 2,
    crop_pixels: int = 700,
    stride_pixels: int = 140,
    min_barrier_pixels: int = 1500,
    exclude_border_pixels: int = 100,
    min_valid_fraction: float = 0.95,
    zone_overlay: bool = False,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline_bp, scenario_bp, baseline_fi, scenario_fi = _prediction_arrays(
        experiment_dir,
        scenario=scenario,
        hex_id=hex_id,
    )
    baseline_hazard = baseline_bp * baseline_fi
    scenario_hazard = scenario_bp * scenario_fi
    delta_fi = scenario_fi - baseline_fi
    delta_hazard = scenario_hazard - baseline_hazard

    grouped_fuel = load_grouped_fuel_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        reference_profile=prediction_reference_profile(prediction_dirs, hex_id),
        hex_id=hex_id,
    )
    original_nonfuel, replacement_map = replacement_group_map_on_prediction_grid(grouped_fuel)
    hazard_values, hazard_valid = values_and_valid(delta_hazard)
    fi_values, fi_valid = values_and_valid(delta_fi)
    valid_mask = hazard_valid & fi_valid & np.isfinite(hazard_values) & np.isfinite(fi_values)
    zone_labels = None
    if zone_overlay:
        zone_labels = load_zone_labels(raw_data_dir, hex_id, prediction_reference_profile(prediction_dirs, hex_id), support=valid_mask)
    selection_barrier_mask = original_nonfuel & valid_mask
    replacement_map = np.where(selection_barrier_mask, replacement_map, np.nan)
    windows = select_neighborhood_windows(
        barrier_mask=selection_barrier_mask,
        valid_mask=valid_mask,
        delta_hazard=hazard_values,
        delta_fi=fi_values,
        n_windows=n_windows,
        crop_size=crop_pixels,
        stride=stride_pixels,
        min_barrier_pixels=min_barrier_pixels,
        exclude_border_pixels=exclude_border_pixels,
        min_valid_fraction=min_valid_fraction,
    )

    out_dir = out_dir if out_dir is not None else experiment_dir / "figures" / "fuel_local_zoom"
    out_dir.mkdir(parents=True, exist_ok=True)
    fi_plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_local_neighborhood_fi.png"
    hazard_plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_local_neighborhood_hazard.png"
    plot_neighborhood_grid(
        grouped_fuel=grouped_fuel,
        replacement_map=replacement_map,
        barrier_mask=selection_barrier_mask,
        valid_mask=valid_mask,
        baseline=baseline_fi,
        scenario_values=scenario_fi,
        delta=delta_fi,
        windows=windows,
        endpoint_label="FI",
        sequential_label="Predicted FI",
        delta_label="ΔFI = counterfactual − baseline",
        out_path=fi_plot_path,
        zone_labels=zone_labels,
        zone_overlay_color=zone_overlay_color,
        zone_overlay_linewidth=zone_overlay_linewidth,
        zone_overlay_alpha=zone_overlay_alpha,
    )
    plot_neighborhood_grid(
        grouped_fuel=grouped_fuel,
        replacement_map=replacement_map,
        barrier_mask=selection_barrier_mask,
        valid_mask=valid_mask,
        baseline=baseline_hazard,
        scenario_values=scenario_hazard,
        delta=delta_hazard,
        windows=windows,
        endpoint_label="hazard",
        sequential_label="Predicted hazard = BP × FI",
        delta_label="Δhazard = counterfactual − baseline",
        out_path=hazard_plot_path,
        zone_labels=zone_labels,
        zone_overlay_color=zone_overlay_color,
        zone_overlay_linewidth=zone_overlay_linewidth,
        zone_overlay_alpha=zone_overlay_alpha,
    )

    rows = [
        asdict(
            NeighborhoodSummary(
                scenario=scenario,
                hex_id=hex_id,
                window_id=window.window_id,
                rank=window.rank,
                row_min=window.row_min,
                row_max=window.row_max,
                col_min=window.col_min,
                col_max=window.col_max,
                barrier_pixels=window.barrier_pixels,
                barrier_density=window.barrier_density,
                valid_fraction=window.valid_fraction,
                baseline_fi_mean=_mean_on_crop(baseline_fi, window),
                scenario_fi_mean=_mean_on_crop(scenario_fi, window),
                delta_fi_mean=_mean_on_crop(delta_fi, window),
                baseline_hazard_mean=_mean_on_crop(baseline_hazard, window),
                scenario_hazard_mean=_mean_on_crop(scenario_hazard, window),
                delta_hazard_mean=_mean_on_crop(delta_hazard, window),
                score=window.score,
            )
        )
        for window in windows
    ]
    summary_path = experiment_dir / f"counterfactual_{scenario}_local_neighborhood_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return fi_plot_path, hazard_plot_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local communication panels around selected barrier neighborhoods.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--n_windows", type=int, default=3)
    parser.add_argument("--crop_pixels", type=int, default=700)
    parser.add_argument("--stride_pixels", type=int, default=140)
    parser.add_argument("--min_barrier_pixels", type=int, default=1500)
    parser.add_argument("--exclude_border_pixels", type=int, default=100)
    parser.add_argument("--min_valid_fraction", type=float, default=0.95)
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/fuel_local_zoom.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fi_plot_path, hazard_plot_path, summary_path = write_local_neighborhood_panels(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        scenario=args.scenario,
        hex_id=str(args.hex_id).zfill(2),
        n_windows=max(1, int(args.n_windows)),
        crop_pixels=max(1, int(args.crop_pixels)),
        stride_pixels=max(1, int(args.stride_pixels)),
        min_barrier_pixels=max(1, int(args.min_barrier_pixels)),
        exclude_border_pixels=max(0, int(args.exclude_border_pixels)),
        min_valid_fraction=float(args.min_valid_fraction),
        zone_overlay=args.zone_overlay,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
        out_dir=args.out_dir,
    )
    print(f"Wrote local FI neighborhood panels: {fi_plot_path}")
    print(f"Wrote local hazard neighborhood panels: {hazard_plot_path}")
    print(f"Wrote local neighborhood summary: {summary_path}")


if __name__ == "__main__":
    main()
