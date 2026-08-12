"""Local zoom panels around selected fuel-intervention neighborhoods.

Fuel-intervention-specific companion to `counterfactual_response_maps.py`: instead
of generic high-|delta| hotspots, this selects fixed-size windows that clearly
contain evaluated fuel edits and a strong direction-aligned hazard/FI response, then
renders the edit mask, scenario fuel group, and baseline/scenario/delta FI and hazard
(BP x FI) maps side by side for each selected window.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    load_counterfactual_config,
    resolve_counterfactual_paths,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map import (
    FUEL_GROUP_COLOURS,
    FUEL_GROUP_LABELS,
    SCENARIO,
    _categorical_codes,
    burnable_fuel_support,
    group_raw_fuel,
    load_evaluated_fuel_pair,
    load_zone_labels_on_prediction_grid,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    DEFAULT_ZONE_OVERLAY_ALPHA,
    DEFAULT_ZONE_OVERLAY_COLOR,
    DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    EndpointResponse,
    add_zone_overlay_args,
    build_endpoint_response,
    delta_norm,
    finite_values,
    load_baseline_scenario_pair,
    overlay_zone_boundaries,
    prediction_dirs_from_index,
    prediction_reference_profile,
    robust_norm,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
    edit_pixels: int
    edit_density: float
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
    edit_pixels: int
    edit_density: float
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
    edit_mask: np.ndarray,
    valid_mask: np.ndarray,
    delta_hazard: np.ndarray,
    delta_fi: np.ndarray,
    response_direction: int | None,
    n_windows: int = 3,
    crop_size: int = 700,
    stride: int = 140,
    min_edit_pixels: int = 1500,
    min_edit_density: float = 0.01,
    max_edit_density: float = 0.35,
    min_valid_fraction: float = 0.95,
    max_iou: float = 0.15,
    exclude_border_pixels: int = 100,
) -> list[NeighborhoodWindow]:
    """Select windows containing evaluated edits and a strong hazard response.

    Support-changing edits use their expected response direction. Fuel substitutions
    that preserve burnable support use the absolute hazard response instead.
    """

    if edit_mask.shape != valid_mask.shape or edit_mask.shape != delta_hazard.shape or edit_mask.shape != delta_fi.shape:
        raise ValueError("edit_mask, valid_mask, delta_hazard, and delta_fi must have the same shape.")
    if response_direction not in {-1, 1, None}:
        raise ValueError("response_direction must be -1, 1, or None.")
    if n_windows < 1:
        raise ValueError("n_windows must be positive.")
    if crop_size < 1:
        raise ValueError("crop_size must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")

    h, w = edit_mask.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    area = float(crop_h * crop_w)

    finite_hazard = valid_mask & np.isfinite(delta_hazard)
    finite_fi = valid_mask & np.isfinite(delta_fi)
    edit_integral = integral_image(edit_mask.astype(np.float64))
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
            edit_pixels = int(window_sum(edit_integral, row_min, row_max, col_min, col_max))
            if edit_pixels < min_edit_pixels:
                continue
            edit_density = edit_pixels / area
            if edit_density < min_edit_density or edit_density > max_edit_density:
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
            hazard_response_strength = abs(delta_hazard_mean) if response_direction is None else response_direction * delta_hazard_mean
            if hazard_response_strength <= 0.0:
                continue

            density_preference = max(0.1, 1.0 - abs(edit_density - 0.08) / 0.08)
            score = float(hazard_response_strength * np.sqrt(edit_pixels) * density_preference)
            window_id += 1
            candidates.append(
                NeighborhoodWindow(
                    window_id=window_id,
                    rank=0,
                    row_min=row_min,
                    row_max=row_max,
                    col_min=col_min,
                    col_max=col_max,
                    edit_pixels=edit_pixels,
                    edit_density=float(edit_density),
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
            edit_pixels=window.edit_pixels,
            edit_density=window.edit_density,
            valid_fraction=window.valid_fraction,
            delta_hazard_mean=window.delta_hazard_mean,
            delta_fi_mean=window.delta_fi_mean,
            score=window.score,
        )
        for rank, window in enumerate(selected, start=1)
    ]


def crop(data: np.ndarray | np.ma.MaskedArray, window: _SliceWindow) -> np.ma.MaskedArray:
    return np.ma.asarray(data)[window.row_slice, window.col_slice]


def _mask_overlay(ax: plt.Axes, mask: np.ndarray, *, colour: str = "#ff00ff", alpha: float = 0.35) -> None:
    overlay = np.ma.masked_where(~mask.astype(bool), np.ones(mask.shape, dtype=np.float32))
    cmap = matplotlib.colors.ListedColormap([colour])
    ax.imshow(overlay, cmap=cmap, interpolation="nearest", origin="upper", alpha=alpha)


def _norm_limits(norm: Normalize) -> str:
    return f"[{float(cast(float, norm.vmin)):.2g}, {float(cast(float, norm.vmax)):.2g}]"


def _fuel_group_legend(categories: list[int]) -> list[Patch]:
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
    scenario_fuel_map: np.ndarray,
    edit_mask: np.ndarray,
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
    """Plot intervention, edited fuel, baseline, scenario, and delta for selected windows."""

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
    all_fuel_group_values = []
    for window in windows:
        values = finite_values(crop(scenario_fuel_map, window)).astype(np.int32)
        if values.size:
            all_fuel_group_values.extend(values.tolist())
    fuel_group_categories = sorted(set(map(int, all_fuel_group_values)))
    fuel_group_cmap = matplotlib.colors.ListedColormap([FUEL_GROUP_COLOURS.get(category, "#999999") for category in fuel_group_categories])
    fuel_group_cmap.set_bad(color="white", alpha=0.0)
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
        edit_crop = crop(edit_mask, window).filled(False).astype(bool) & support_crop
        zone_crop = crop(zone_labels, window) if zone_labels is not None else None
        scenario_fuel_crop = np.ma.masked_where(~support_crop, crop(scenario_fuel_map, window))
        fuel_group_codes = _categorical_codes(scenario_fuel_crop, fuel_group_categories)
        support_context = np.where(support_crop, 1.0, 0.0)
        support_cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#f7f7f7"])

        panels = (
            ("Evaluated edit mask", None, None, None),
            ("Scenario fuel group", fuel_group_codes, fuel_group_cmap, None),
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
            (f"\u0394{endpoint_label}", np.ma.masked_where(~support_crop, crop(delta, window)), delta_cmap, diverging_norm),
        )
        for col_idx, (title, array, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ax.imshow(support_context, cmap=support_cmap, interpolation="nearest", origin="upper")
                _mask_overlay(ax, edit_crop, colour="#000000", alpha=0.85)
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
                    f"{row_label}\n{window.edit_pixels:,} edited px\nmean \u0394hazard {window.delta_hazard_mean:.2f}",
                    fontsize=8,
                )

    if sequential_image is not None:
        cbar = fig.colorbar(sequential_image, cax=sequential_cax)
        cbar.set_label(f"{sequential_label} shared scale {_norm_limits(sequential_norm)}")
    if delta_image is not None:
        cbar = fig.colorbar(delta_image, cax=delta_cax)
        cbar.set_label(f"{delta_label} centered scale {_norm_limits(diverging_norm)}")
    handles = [Patch(facecolor="#000000", edgecolor="none", alpha=0.85, label="Edited fuel pixels")]
    handles.extend(_fuel_group_legend(fuel_group_categories))
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
        f"Local fuel-intervention neighborhoods: edit and {endpoint_label} response",
        y=0.965,
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _endpoint_response(
    experiment_dir: Path,
    *,
    scenario: str,
    hex_id: str,
    endpoint: str,
    nonfuel_ids: list[int],
) -> EndpointResponse:
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline, scenario_values = load_baseline_scenario_pair(
        prediction_dirs,
        hex_id,
        endpoint=endpoint,
        scenario=scenario,
    )
    baseline_fuel, scenario_fuel = load_evaluated_fuel_pair(
        experiment_dir=experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    return build_endpoint_response(
        baseline,
        scenario_values,
        baseline_support=burnable_fuel_support(baseline_fuel, nonfuel_ids),
        scenario_support=burnable_fuel_support(scenario_fuel, nonfuel_ids),
    )


def _hazard_response(
    bp_response: EndpointResponse,
    fi_response: EndpointResponse,
) -> EndpointResponse:
    if not np.array_equal(bp_response.baseline_support, fi_response.baseline_support) or not np.array_equal(
        bp_response.scenario_support,
        fi_response.scenario_support,
    ):
        raise ValueError("BP and FI fuel support differ.")

    baseline_values = bp_response.baseline.filled(0.0) * fi_response.baseline.filled(0.0)
    scenario_values = bp_response.scenario.filled(0.0) * fi_response.scenario.filled(0.0)
    return EndpointResponse(
        baseline=np.ma.masked_where(~bp_response.baseline_support, baseline_values),
        scenario=np.ma.masked_where(~bp_response.scenario_support, scenario_values),
        delta=np.ma.masked_where(~bp_response.response_support, scenario_values - baseline_values),
        baseline_support=bp_response.baseline_support,
        scenario_support=bp_response.scenario_support,
        response_support=bp_response.response_support,
    )


def _response_direction(
    edit_mask: np.ndarray,
    baseline_support: np.ndarray,
    scenario_support: np.ndarray,
) -> int | None:
    added_support = edit_mask & ~baseline_support & scenario_support
    removed_support = edit_mask & baseline_support & ~scenario_support
    if added_support.any() and removed_support.any():
        raise ValueError("Local zoom does not support interventions that both add and remove burnable support.")
    if added_support.any():
        return 1
    if removed_support.any():
        return -1
    return None


def _mean_on_crop(data: np.ndarray | np.ma.MaskedArray, window: _SliceWindow) -> float:
    values = finite_values(crop(data, window))
    return float(np.mean(values)) if values.size else float("nan")


def _effective_mean_on_crop(
    data: np.ma.MaskedArray,
    response_support: np.ndarray,
    window: _SliceWindow,
) -> float:
    effective = np.ma.masked_where(~response_support, data.filled(0.0))
    return _mean_on_crop(effective, window)


def write_local_neighborhood_panels(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    config_path: Path,
    scenario: str = SCENARIO,
    hex_id: str = "16",
    n_windows: int = 2,
    crop_pixels: int = 700,
    stride_pixels: int = 140,
    min_edit_pixels: int = 1500,
    exclude_border_pixels: int = 100,
    min_valid_fraction: float = 0.95,
    zone_overlay: bool = False,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    config = load_counterfactual_config(config_path)
    try:
        scenario_config = config.scenario(scenario)
    except KeyError as error:
        raise KeyError(f"Scenario {scenario!r} not found in {config_path}.") from error
    fuel_edit = scenario_config.fuel_edit()
    if fuel_edit is None:
        raise ValueError(f"Scenario {scenario!r} is not a fuel intervention.")
    configured_nonfuel_ids = fuel_edit.get("nonfuel_ids")
    if not isinstance(configured_nonfuel_ids, list | tuple) or not configured_nonfuel_ids:
        raise ValueError(f"Fuel scenario {scenario!r} must define nonfuel_ids.")
    nonfuel_ids = [int(value) for value in configured_nonfuel_ids]

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    bp_response = _endpoint_response(
        experiment_dir,
        scenario=scenario,
        hex_id=hex_id,
        endpoint="bp",
        nonfuel_ids=nonfuel_ids,
    )
    fi_response = _endpoint_response(
        experiment_dir,
        scenario=scenario,
        hex_id=hex_id,
        endpoint="fi",
        nonfuel_ids=nonfuel_ids,
    )
    hazard_response = _hazard_response(bp_response, fi_response)
    baseline_fi = fi_response.baseline
    scenario_fi = fi_response.scenario
    delta_fi = fi_response.delta
    baseline_hazard = hazard_response.baseline
    scenario_hazard = hazard_response.scenario
    delta_hazard = hazard_response.delta

    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    baseline_fuel, scenario_fuel = load_evaluated_fuel_pair(
        experiment_dir=experiment_dir,
        scenario=scenario,
        endpoint="bp",
        hex_id=hex_id,
    )
    baseline_fuel_values = np.asarray(baseline_fuel.filled(np.nan), dtype=np.float64)
    scenario_fuel_values = np.asarray(scenario_fuel.filled(np.nan), dtype=np.float64)
    footprint_mask = np.isfinite(baseline_fuel_values) & np.isfinite(scenario_fuel_values)
    edit_mask = footprint_mask & (baseline_fuel_values != scenario_fuel_values)
    response_direction = _response_direction(
        edit_mask,
        bp_response.baseline_support,
        bp_response.scenario_support,
    )
    scenario_fuel_map = np.where(edit_mask, group_raw_fuel(scenario_fuel_values), np.nan)
    hazard_values = np.asarray(delta_hazard.filled(np.nan), dtype=np.float64)
    fi_values = np.asarray(delta_fi.filled(np.nan), dtype=np.float64)
    zone_labels = None
    if zone_overlay:
        zone_labels = load_zone_labels_on_prediction_grid(
            raw_data_dir=raw_data_dir,
            reference_profile=reference_profile,
            hex_id=hex_id,
            support=footprint_mask,
        )
    windows = select_neighborhood_windows(
        edit_mask=edit_mask,
        valid_mask=footprint_mask,
        delta_hazard=hazard_values,
        delta_fi=fi_values,
        response_direction=response_direction,
        n_windows=n_windows,
        crop_size=crop_pixels,
        stride=stride_pixels,
        min_edit_pixels=min_edit_pixels,
        max_edit_density=1.0,
        exclude_border_pixels=exclude_border_pixels,
        min_valid_fraction=min_valid_fraction,
    )

    out_dir = out_dir if out_dir is not None else experiment_dir / "figures" / "fuel_local_zoom"
    out_dir.mkdir(parents=True, exist_ok=True)
    fi_plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_local_neighborhood_fi.png"
    hazard_plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_local_neighborhood_hazard.png"
    plot_neighborhood_grid(
        scenario_fuel_map=scenario_fuel_map,
        edit_mask=edit_mask,
        valid_mask=footprint_mask,
        baseline=baseline_fi,
        scenario_values=scenario_fi,
        delta=delta_fi,
        windows=windows,
        endpoint_label="FI",
        sequential_label="Predicted FI",
        delta_label="\u0394FI = counterfactual \u2212 baseline",
        out_path=fi_plot_path,
        zone_labels=zone_labels,
        zone_overlay_color=zone_overlay_color,
        zone_overlay_linewidth=zone_overlay_linewidth,
        zone_overlay_alpha=zone_overlay_alpha,
    )
    plot_neighborhood_grid(
        scenario_fuel_map=scenario_fuel_map,
        edit_mask=edit_mask,
        valid_mask=footprint_mask,
        baseline=baseline_hazard,
        scenario_values=scenario_hazard,
        delta=delta_hazard,
        windows=windows,
        endpoint_label="hazard",
        sequential_label="Predicted hazard = BP \u00d7 FI",
        delta_label="\u0394hazard = counterfactual \u2212 baseline",
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
                edit_pixels=window.edit_pixels,
                edit_density=window.edit_density,
                valid_fraction=window.valid_fraction,
                baseline_fi_mean=_effective_mean_on_crop(baseline_fi, fi_response.response_support, window),
                scenario_fi_mean=_effective_mean_on_crop(scenario_fi, fi_response.response_support, window),
                delta_fi_mean=_mean_on_crop(delta_fi, window),
                baseline_hazard_mean=_effective_mean_on_crop(
                    baseline_hazard,
                    hazard_response.response_support,
                    window,
                ),
                scenario_hazard_mean=_effective_mean_on_crop(
                    scenario_hazard,
                    hazard_response.response_support,
                    window,
                ),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_fuel.yaml"))
    parser.add_argument("--experiment_dir", type=Path, default=None, help="Overrides save_dir from --config.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Overrides raw_data_dir from --config.")
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--n_windows", type=int, default=3)
    parser.add_argument("--crop_pixels", type=int, default=700)
    parser.add_argument("--stride_pixels", type=int, default=140)
    parser.add_argument("--min_edit_pixels", type=int, default=1500)
    parser.add_argument("--exclude_border_pixels", type=int, default=100)
    parser.add_argument("--min_valid_fraction", type=float, default=0.95)
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/fuel_local_zoom.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_counterfactual_config(args.config)
    experiment_dir, raw_data_dir = resolve_counterfactual_paths(
        config,
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
    )
    fi_plot_path, hazard_plot_path, summary_path = write_local_neighborhood_panels(
        experiment_dir=experiment_dir,
        raw_data_dir=raw_data_dir,
        config_path=args.config,
        scenario=args.scenario,
        hex_id=normalize_hex_id(args.hex_id),
        n_windows=max(1, int(args.n_windows)),
        crop_pixels=max(1, int(args.crop_pixels)),
        stride_pixels=max(1, int(args.stride_pixels)),
        min_edit_pixels=max(1, int(args.min_edit_pixels)),
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
