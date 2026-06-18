"""Local maps for strong-wind anisotropic shadow stress tests."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from src.datasets.postprocessing.counterfactual_barrier_profile import (
    DISPLAY_SCENARIO,
    _load_barrier_layers_on_prediction_grid,
)
from src.datasets.postprocessing.counterfactual_hazard_map import (
    endpoint_prediction,
    finite_values,
    hazard_prediction,
    original_barrier_mask,
    pooled_positive_percentile,
    prediction_dirs_from_index,
    restrict_to_support,
    symmetric_percentile_limit,
    values_and_valid,
)
from src.datasets.postprocessing.counterfactual_local_zoom_panels import (
    candidate_starts,
    integral_image,
    window_sum,
)
from src.datasets.postprocessing.counterfactual_weather import raw_weather_path
from src.datasets.postprocessing.diagnose_bp_barrier_halo import ZONE_NODATA, compute_zone_wind_consistency

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SCENARIO = "wind_speed100_zone_consistent_direction"


@dataclass(frozen=True)
class LocalWindShadowWindow:
    scenario: str
    hex_id: str
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    barrier_pixels: int
    barrier_density: float
    valid_fraction: float
    abs_delta_hazard_mean: float
    score: float
    zone_id: int | None
    flow_bearing_deg: float

    @property
    def row_slice(self) -> slice:
        return slice(self.row_min, self.row_max)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_min, self.col_max)


def _zone_id_from_weather_zone(value: object) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def zone_flow_bearings(raw_data_dir: Path, hex_id: str) -> dict[int, float]:
    """Return physical downwind bearings keyed by integer fire-zone ID."""

    wind_df = compute_zone_wind_consistency(raw_weather_path(raw_data_dir, hex_id))
    rows: dict[int, float] = {}
    for item in wind_df.itertuples(index=False):
        zone_id = _zone_id_from_weather_zone(getattr(item, "zone"))
        flow_bearing = float(getattr(item, "dominant_direction_deg"))
        if zone_id is not None and np.isfinite(flow_bearing):
            rows[zone_id] = flow_bearing
    return rows


def majority_zone(firezones: np.ndarray, valid_mask: np.ndarray) -> int | None:
    zone_values = firezones[(firezones != ZONE_NODATA) & valid_mask]
    if zone_values.size == 0:
        return None
    values, counts = np.unique(zone_values.astype(np.int32), return_counts=True)
    return int(values[int(np.argmax(counts))])


def select_barrier_response_window(
    *,
    scenario: str,
    hex_id: str,
    barrier_mask: np.ndarray,
    valid_mask: np.ndarray,
    delta_hazard: np.ndarray,
    firezones: np.ndarray,
    flow_by_zone: dict[int, float],
    crop_size: int = 900,
    stride: int = 180,
    min_barrier_pixels: int = 1_000,
    min_valid_fraction: float = 0.80,
    max_barrier_density: float = 0.50,
    prefer_southern: bool = True,
) -> LocalWindShadowWindow:
    """Choose a local crop containing a prominent barrier and large local hazard response."""

    if barrier_mask.shape != valid_mask.shape or barrier_mask.shape != delta_hazard.shape or barrier_mask.shape != firezones.shape:
        raise ValueError("barrier_mask, valid_mask, delta_hazard, and firezones must share a shape.")
    if crop_size < 1:
        raise ValueError("crop_size must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")

    h, w = barrier_mask.shape
    crop_h = min(int(crop_size), h)
    crop_w = min(int(crop_size), w)
    area = float(crop_h * crop_w)

    finite_delta = valid_mask & np.isfinite(delta_hazard)
    abs_delta = np.where(finite_delta, np.abs(delta_hazard), 0.0)
    barrier_integral = integral_image(barrier_mask.astype(np.float64))
    valid_integral = integral_image(valid_mask.astype(np.float64))
    delta_sum_integral = integral_image(abs_delta.astype(np.float64))
    delta_count_integral = integral_image(finite_delta.astype(np.float64))

    candidates: list[LocalWindShadowWindow] = []
    for row_min in candidate_starts(h, crop_h, int(stride)):
        row_max = row_min + crop_h
        row_center_fraction = (row_min + 0.5 * crop_h) / max(float(h), 1.0)
        for col_min in candidate_starts(w, crop_w, int(stride)):
            col_max = col_min + crop_w
            barrier_pixels = int(window_sum(barrier_integral, row_min, row_max, col_min, col_max))
            if barrier_pixels < min_barrier_pixels:
                continue
            barrier_density = barrier_pixels / area
            if barrier_density > max_barrier_density:
                continue
            valid_fraction = window_sum(valid_integral, row_min, row_max, col_min, col_max) / area
            if valid_fraction < min_valid_fraction:
                continue
            delta_count = window_sum(delta_count_integral, row_min, row_max, col_min, col_max)
            if delta_count <= 0.0:
                continue
            abs_delta_hazard_mean = window_sum(delta_sum_integral, row_min, row_max, col_min, col_max) / delta_count
            density_preference = max(0.25, 1.0 - abs(barrier_density - 0.08) / 0.08)
            southern_preference = 0.75 + row_center_fraction if prefer_southern else 1.0
            score = float((abs_delta_hazard_mean + 1e-9) * np.sqrt(barrier_pixels) * density_preference * southern_preference)
            window_valid = valid_mask[row_min:row_max, col_min:col_max]
            zone_id = majority_zone(firezones[row_min:row_max, col_min:col_max], window_valid)
            flow_bearing = float(flow_by_zone.get(zone_id, np.nan)) if zone_id is not None else float("nan")
            candidates.append(
                LocalWindShadowWindow(
                    scenario=scenario,
                    hex_id=hex_id,
                    row_min=row_min,
                    row_max=row_max,
                    col_min=col_min,
                    col_max=col_max,
                    barrier_pixels=barrier_pixels,
                    barrier_density=float(barrier_density),
                    valid_fraction=float(valid_fraction),
                    abs_delta_hazard_mean=float(abs_delta_hazard_mean),
                    score=score,
                    zone_id=zone_id,
                    flow_bearing_deg=flow_bearing,
                )
            )

    if not candidates:
        raise ValueError("No local barrier windows passed the selection criteria.")
    return max(candidates, key=lambda item: item.score)


def _crop(data: np.ndarray | np.ma.MaskedArray, window: LocalWindShadowWindow) -> np.ma.MaskedArray:
    return np.ma.asarray(data)[window.row_slice, window.col_slice]


def _mask_bad_cmap(name: str, bad: str = "#eeeeee"):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=bad, alpha=1.0)
    return cmap


def _outline_barriers(ax: plt.Axes, barrier_crop: np.ndarray) -> None:
    if barrier_crop.any():
        ax.contour(barrier_crop.astype(np.float32), levels=[0.5], colors="black", linewidths=0.55, origin="upper")
        ax.contour(barrier_crop.astype(np.float32), levels=[0.5], colors="white", linewidths=0.20, origin="upper")


def _add_flow_arrow(ax: plt.Axes, flow_bearing_deg: float) -> None:
    if not np.isfinite(flow_bearing_deg):
        return
    theta = np.deg2rad(flow_bearing_deg)
    length = 0.15
    dx = float(np.sin(theta) * length)
    dy = float(np.cos(theta) * length)
    x0 = 0.16 if dx >= 0.0 else 0.84
    y0 = 0.16 if dy >= 0.0 else 0.84
    ax.annotate(
        "",
        xy=(x0 + dx, y0 + dy),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "cyan", "linewidth": 2.0},
    )
    ax.text(
        x0,
        y0 - 0.04 if dy >= 0.0 else y0 + 0.04,
        f"flow {flow_bearing_deg:.0f}°",
        transform=ax.transAxes,
        color="cyan",
        fontsize=8,
        weight="bold",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
    )


def plot_local_wind_shadow_map(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    scenario: str = DEFAULT_SCENARIO,
    hex_id: str = "16",
    crop_size: int = 900,
    stride: int = 180,
    percentile: float = 99.0,
    prefer_southern: bool = True,
) -> tuple[Path, Path]:
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    layers = _load_barrier_layers_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    barrier_mask = original_barrier_mask(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )

    baseline_bp = endpoint_prediction(prediction_dirs, "baseline", "bp", hex_id)
    scenario_bp = endpoint_prediction(prediction_dirs, scenario, "bp", hex_id)
    baseline_fi = endpoint_prediction(prediction_dirs, "baseline", "fi", hex_id)
    scenario_fi = endpoint_prediction(prediction_dirs, scenario, "fi", hex_id)
    baseline_hazard = hazard_prediction(prediction_dirs, "baseline", hex_id)
    scenario_hazard = hazard_prediction(prediction_dirs, scenario, hex_id)

    _, baseline_bp_valid = values_and_valid(baseline_bp)
    _, scenario_bp_valid = values_and_valid(scenario_bp)
    _, baseline_fi_valid = values_and_valid(baseline_fi)
    _, scenario_fi_valid = values_and_valid(scenario_fi)
    support = baseline_bp_valid & scenario_bp_valid & baseline_fi_valid & scenario_fi_valid

    baseline_bp = restrict_to_support(baseline_bp, support)
    scenario_bp = restrict_to_support(scenario_bp, support)
    baseline_fi = restrict_to_support(baseline_fi, support)
    scenario_fi = restrict_to_support(scenario_fi, support)
    baseline_hazard = restrict_to_support(baseline_hazard, support)
    scenario_hazard = restrict_to_support(scenario_hazard, support)
    delta_bp = scenario_bp - baseline_bp
    delta_fi = scenario_fi - baseline_fi
    delta_hazard = scenario_hazard - baseline_hazard

    flow_by_zone = zone_flow_bearings(raw_data_dir, hex_id)
    delta_hazard_values = np.asarray(np.ma.asarray(delta_hazard).filled(np.nan), dtype=np.float64)
    window = select_barrier_response_window(
        scenario=scenario,
        hex_id=hex_id,
        barrier_mask=barrier_mask,
        valid_mask=support,
        delta_hazard=delta_hazard_values,
        firezones=layers.firezones,
        flow_by_zone=flow_by_zone,
        crop_size=crop_size,
        stride=stride,
        prefer_southern=prefer_southern,
    )

    rows = (
        ("BP", baseline_bp, scenario_bp, delta_bp),
        ("FI", baseline_fi, scenario_fi, delta_fi),
        ("Hazard = BP x FI", baseline_hazard, scenario_hazard, delta_hazard),
    )
    barrier_crop = barrier_mask[window.row_slice, window.col_slice]
    sequential_cmap = _mask_bad_cmap("magma")
    delta_cmap = _mask_bad_cmap("RdBu_r")

    fig, axes = plt.subplots(len(rows), 3, figsize=(14.5, 12.5), squeeze=False, constrained_layout=True)
    for row_idx, (label, baseline, scenario_values, delta) in enumerate(rows):
        baseline_crop = _crop(baseline, window)
        scenario_crop = _crop(scenario_values, window)
        delta_crop = _crop(delta, window)
        vmax = pooled_positive_percentile([baseline_crop, scenario_crop], percentile=percentile)
        limit = symmetric_percentile_limit([delta_crop], percentile=percentile)
        sequential_norm = Normalize(vmin=0.0, vmax=vmax)
        diverging_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        panels = (
            ("Model baseline", baseline_crop, sequential_cmap, sequential_norm),
            (DISPLAY_SCENARIO.get(scenario, scenario), scenario_crop, sequential_cmap, sequential_norm),
            ("Δ scenario − baseline", delta_crop, delta_cmap, diverging_norm),
        )
        sequential_image = None
        delta_image = None
        for col_idx, (title, values_map, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#eeeeee")
            image = ax.imshow(values_map, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            _outline_barriers(ax, barrier_crop)
            _add_flow_arrow(ax, window.flow_bearing_deg)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                zone_text = f"zone {window.zone_id}" if window.zone_id is not None else "zone unknown"
                ax.set_ylabel(f"{label}\n{zone_text}", fontsize=10)
            if col_idx < 2:
                sequential_image = image
            else:
                delta_image = image
        if sequential_image is not None:
            cbar = fig.colorbar(sequential_image, ax=axes[row_idx, :2].tolist(), fraction=0.025, pad=0.02)
            cbar.set_label(f"{label} (shared baseline/scenario scale, p{percentile:g})")
        if delta_image is not None:
            cbar = fig.colorbar(delta_image, ax=axes[row_idx, 2], fraction=0.046, pad=0.03)
            cbar.set_label(f"Δ{label} (symmetric p{percentile:g})")

    scenario_label = DISPLAY_SCENARIO.get(scenario, scenario)
    fig.suptitle(
        (
            f"Hex{int(hex_id):02d} local anisotropic shadow stress test: {scenario_label}\n"
            f"crop rows {window.row_min}:{window.row_max}, cols {window.col_min}:{window.col_max}; "
            f"barrier density {window.barrier_density:.3f}, valid {window.valid_fraction:.3f}"
        ),
        y=1.02,
    )

    out_dir = experiment_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_local_anisotropic_shadow_bp_fi_hazard.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = experiment_dir / f"counterfactual_{scenario}_local_anisotropic_shadow_summary.csv"
    summary = asdict(window)
    for label, baseline, scenario_values, delta in rows:
        slug = label.lower().replace(" = ", "_").replace(" x ", "_").replace(" ", "_")
        baseline_crop = _crop(baseline, window)
        scenario_crop = _crop(scenario_values, window)
        delta_crop = _crop(delta, window)
        summary[f"{slug}_baseline_mean"] = float(np.mean(finite_values(baseline_crop)))
        summary[f"{slug}_scenario_mean"] = float(np.mean(finite_values(scenario_crop)))
        summary[f"{slug}_delta_mean"] = float(np.mean(finite_values(delta_crop)))
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return plot_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local strong-wind anisotropic shadow maps.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--crop_size", type=int, default=900)
    parser.add_argument("--stride", type=int, default=180)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--no_prefer_southern", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_path, summary_path = plot_local_wind_shadow_map(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        scenario=args.scenario,
        hex_id=str(args.hex_id).zfill(2),
        crop_size=args.crop_size,
        stride=args.stride,
        percentile=args.percentile,
        prefer_southern=not args.no_prefer_southern,
    )
    print(f"Wrote local anisotropic shadow map: {plot_path}")
    print(f"Wrote local anisotropic shadow summary: {summary_path}")


if __name__ == "__main__":
    main()
