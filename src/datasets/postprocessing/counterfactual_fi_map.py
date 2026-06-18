"""FI maps for the non-fuel-to-burnable counterfactual intervention."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_barrier_profile import _load_barrier_layers_on_prediction_grid
from src.datasets.postprocessing.counterfactual_hazard_map import (
    downsample_for_display,
    prediction_dirs_from_index,
    prediction_raster_path,
    read_prediction,
    read_prediction_extent,
    restrict_to_support,
    symmetric_percentile_limit,
    values_and_valid,
)
from src.datasets.postprocessing.diagnose_bp_barrier_halo import (
    DEFAULT_DIST_BIN_EDGES_M,
    compute_distance_fields,
    dist_bin_label,
    parse_fuel_barrier_info,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO = "remove_barriers_adjacent_modal"


@dataclass(frozen=True)
class FIMapSummary:
    scenario: str
    hex_id: str
    support_policy: str
    n_paired_pixels: int
    baseline_fi_mean: float
    scenario_fi_mean: float
    delta_fi_mean: float
    delta_fi_median: float
    delta_fi_p05: float
    delta_fi_p95: float
    delta_fi_min: float
    delta_fi_max: float
    delta_fi_abs_plot_limit: float
    frac_delta_positive: float
    replacement_pixels_total: int
    replacement_pixels_with_scenario_fi: int
    replacement_scenario_fi_mean: float
    replacement_scenario_fi_median: float
    replacement_scenario_fi_p95: float


def prediction_reference_profile(prediction_dirs: dict[tuple[str, str], Path], hex_id: str) -> dict:
    baseline_fi_dir = prediction_dirs.get(("baseline", "fi"))
    if baseline_fi_dir is None:
        raise KeyError("Missing baseline FI prediction directory; cannot define FI map grid.")
    with rasterio.open(prediction_raster_path(baseline_fi_dir, hex_id)) as src:
        return src.profile.copy()


def original_valid_fi_support(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
) -> np.ndarray:
    """Original finite raw FI burnable support aligned to the prediction grid."""

    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fi_ma, _ = load_spatial_raster(paths.output_fire_intensity(), reference_profile=reference_profile)
    fuel_ma, _ = load_spatial_raster(paths.fuel_grid(hex_id), reference_profile=reference_profile)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)

    fi_values = np.ma.asarray(fi_ma).filled(np.nan)
    fuel_values = np.ma.asarray(fuel_ma).filled(-32768).astype(np.int32)
    return np.isfinite(fi_values) & ~np.isin(fuel_values, fuel_info.nonfuel_ids)


def replacement_mask_on_prediction_grid(
    *,
    raw_data_dir: Path,
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
) -> tuple[np.ndarray, float, float]:
    """Original non-fuel pixels that the fuel scenario replaced with burnable fuel."""

    layers = _load_barrier_layers_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    return np.isin(layers.fuel, fuel_info.nonfuel_ids), layers.pixel_h_m, layers.pixel_w_m


def fi_delta(
    prediction_dirs: dict[tuple[str, str], Path],
    *,
    scenario: str,
    hex_id: str,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]:
    baseline_dir = prediction_dirs.get(("baseline", "fi"))
    scenario_dir = prediction_dirs.get((scenario, "fi"))
    if baseline_dir is None or scenario_dir is None:
        raise KeyError(f"Missing baseline/scenario FI predictions for scenario={scenario!r}.")
    baseline = read_prediction(prediction_raster_path(baseline_dir, hex_id))
    scenario_fi = read_prediction(prediction_raster_path(scenario_dir, hex_id))
    return baseline, scenario_fi, scenario_fi - baseline


def distance_binned_delta_map(
    delta_fi: np.ma.MaskedArray | np.ndarray,
    dist_m: np.ndarray,
    support_mask: np.ndarray,
    *,
    bin_edges_m: tuple[float, ...] = DEFAULT_DIST_BIN_EDGES_M,
) -> tuple[np.ma.MaskedArray, pd.DataFrame]:
    """Replace each support pixel by its distance-bin mean ΔFI."""

    delta_values, delta_valid = values_and_valid(delta_fi)
    valid = support_mask & delta_valid & np.isfinite(dist_m)
    binned = np.full(delta_values.shape, np.nan, dtype=np.float32)
    rows: list[dict] = []
    edges = [0.0, *bin_edges_m, np.inf]
    for bin_idx in range(len(edges) - 1):
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        if np.isinf(hi):
            in_bin = valid & (dist_m >= lo)
        else:
            in_bin = valid & (dist_m >= lo) & (dist_m < hi)
        if not in_bin.any():
            continue
        values = delta_values[in_bin]
        mean_value = float(np.mean(values))
        binned[in_bin] = mean_value
        rows.append(
            {
                "dist_bin_idx": bin_idx,
                "dist_bin": dist_bin_label(bin_idx, bin_edges_m),
                "dist_min_m": lo,
                "dist_max_m": hi,
                "n_pixels": int(values.size),
                "delta_fi_mean": mean_value,
                "delta_fi_median": float(np.median(values)),
                "delta_fi_p25": float(np.percentile(values, 25)),
                "delta_fi_p75": float(np.percentile(values, 75)),
            }
        )
    return np.ma.masked_invalid(binned), pd.DataFrame(rows)


def finite_values(data: np.ma.MaskedArray | np.ndarray) -> np.ndarray:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values[np.isfinite(values)]


def robust_sequential_norm(values: np.ndarray, *, low: float = 1.0, high: float = 99.0) -> Normalize:
    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.percentile(values, low))
    vmax = float(np.percentile(values, high))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def contour_replacement_overlay(
    ax: plt.Axes,
    replacement_mask: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    downsample: int,
    linewidth: float = 0.22,
    alpha: float = 0.45,
) -> None:
    mask = downsample_for_display(replacement_mask.astype(np.float32), downsample)
    if np.nanmax(mask) <= 0:
        return
    ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=linewidth,
        alpha=alpha,
        extent=extent,
        origin="upper",
    )


def plot_single_map(
    data: np.ma.MaskedArray,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
    norm: Normalize,
    replacement_mask: np.ndarray,
    extent: tuple[float, float, float, float],
    out_path: Path,
    downsample: int,
    overlay_replacements: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    image = ax.imshow(
        downsample_for_display(data, downsample),
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    if overlay_replacements:
        contour_replacement_overlay(ax, replacement_mask, extent=extent, downsample=downsample)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_fi_maps(
    *,
    continuous_delta: np.ma.MaskedArray,
    binned_delta: np.ma.MaskedArray,
    replacement_scenario_fi: np.ma.MaskedArray,
    replacement_mask: np.ndarray,
    extent: tuple[float, float, float, float],
    delta_norm: TwoSlopeNorm,
    replacement_norm: Normalize,
    out_path: Path,
    downsample: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 7.0), squeeze=False)
    panels = [
        (continuous_delta, "A. Continuous paired ΔFI", "RdBu_r", delta_norm, "ΔFI (scenario − baseline)", True),
        (binned_delta, "B. Distance-binned mean ΔFI", "RdBu_r", delta_norm, "Mean ΔFI by distance bin", True),
        (
            replacement_scenario_fi,
            "C. Scenario FI on replaced pixels",
            "magma",
            replacement_norm,
            "Scenario FI on newly burnable pixels",
            True,
        ),
    ]
    for ax, (data, title, cmap, norm, label, overlay) in zip(axes.ravel(), panels, strict=True):
        image = ax.imshow(
            downsample_for_display(data, downsample),
            cmap=cmap,
            norm=norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        if overlay:
            contour_replacement_overlay(ax, replacement_mask, extent=extent, downsample=downsample)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(label)

    fig.suptitle("Hex16 FI response to replacing non-fuel barriers with burnable fuel", y=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_fi_maps(
    *,
    scenario: str,
    hex_id: str,
    baseline_fi: np.ma.MaskedArray,
    scenario_fi: np.ma.MaskedArray,
    delta_fi: np.ma.MaskedArray,
    replacement_scenario_fi: np.ma.MaskedArray,
    replacement_mask: np.ndarray,
    delta_plot_limit: float,
) -> FIMapSummary:
    baseline = finite_values(baseline_fi)
    scenario_values = finite_values(scenario_fi)
    delta_values = finite_values(delta_fi)
    replacement_values = finite_values(replacement_scenario_fi)
    if delta_values.size == 0:
        raise ValueError("No finite paired ΔFI pixels to summarize.")
    if replacement_values.size == 0:
        raise ValueError("No finite scenario FI values on replaced pixels to summarize.")
    return FIMapSummary(
        scenario=scenario,
        hex_id=hex_id,
        support_policy="raw_valid_fi_burnable_for_delta; replaced_nonfuel_for_panel_c",
        n_paired_pixels=int(delta_values.size),
        baseline_fi_mean=float(np.mean(baseline)),
        scenario_fi_mean=float(np.mean(scenario_values)),
        delta_fi_mean=float(np.mean(delta_values)),
        delta_fi_median=float(np.median(delta_values)),
        delta_fi_p05=float(np.percentile(delta_values, 5)),
        delta_fi_p95=float(np.percentile(delta_values, 95)),
        delta_fi_min=float(np.min(delta_values)),
        delta_fi_max=float(np.max(delta_values)),
        delta_fi_abs_plot_limit=float(delta_plot_limit),
        frac_delta_positive=float(np.mean(delta_values > 0.0)),
        replacement_pixels_total=int(replacement_mask.sum()),
        replacement_pixels_with_scenario_fi=int(replacement_values.size),
        replacement_scenario_fi_mean=float(np.mean(replacement_values)),
        replacement_scenario_fi_median=float(np.median(replacement_values)),
        replacement_scenario_fi_p95=float(np.percentile(replacement_values, 95)),
    )


def write_fi_replacement_maps(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str = "16",
    scenario: str = SCENARIO,
    percentile: float = 99.5,
    downsample: int = 2,
) -> tuple[Path, list[Path]]:
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline_fi, scenario_fi, delta = fi_delta(prediction_dirs, scenario=scenario, hex_id=hex_id)
    support = original_valid_fi_support(raw_data_dir=raw_data_dir, prediction_dirs=prediction_dirs, hex_id=hex_id)
    replacement_mask, pixel_h_m, pixel_w_m = replacement_mask_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        prediction_dirs=prediction_dirs,
        hex_id=hex_id,
    )
    dist_m, _, _ = compute_distance_fields(replacement_mask, pixel_h_m, pixel_w_m, return_nearest_indices=False)

    paired_baseline_fi = restrict_to_support(baseline_fi, support)
    paired_scenario_fi = restrict_to_support(scenario_fi, support)
    paired_delta = restrict_to_support(delta, support)
    binned_delta, bin_summary = distance_binned_delta_map(
        paired_delta,
        dist_m,
        support,
        bin_edges_m=DEFAULT_DIST_BIN_EDGES_M,
    )
    replacement_scenario_fi = restrict_to_support(scenario_fi, replacement_mask)

    delta_plot_limit = symmetric_percentile_limit([paired_delta], percentile=percentile)
    delta_norm = TwoSlopeNorm(vmin=-delta_plot_limit, vcenter=0.0, vmax=delta_plot_limit)
    replacement_norm = robust_sequential_norm(finite_values(replacement_scenario_fi), low=1.0, high=99.0)

    baseline_fi_dir = prediction_dirs[("baseline", "fi")]
    extent = read_prediction_extent(prediction_raster_path(baseline_fi_dir, hex_id))

    out_dir = experiment_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_fi_maps.png"
    continuous_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_fi_delta_continuous.png"
    binned_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_fi_delta_distance_binned.png"
    replacement_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_scenario_fi_replaced_pixels.png"

    plot_combined_fi_maps(
        continuous_delta=paired_delta,
        binned_delta=binned_delta,
        replacement_scenario_fi=replacement_scenario_fi,
        replacement_mask=replacement_mask,
        extent=extent,
        delta_norm=delta_norm,
        replacement_norm=replacement_norm,
        out_path=combined_path,
        downsample=downsample,
    )
    plot_single_map(
        paired_delta,
        title="Continuous paired ΔFI with replacement overlay",
        cbar_label=f"ΔFI (clipped at p{percentile:g} abs.)",
        cmap="RdBu_r",
        norm=delta_norm,
        replacement_mask=replacement_mask,
        extent=extent,
        out_path=continuous_path,
        downsample=downsample,
    )
    plot_single_map(
        binned_delta,
        title="Distance-binned mean ΔFI with replacement overlay",
        cbar_label=f"Mean ΔFI by original-barrier distance bin (p{percentile:g} abs. scale)",
        cmap="RdBu_r",
        norm=delta_norm,
        replacement_mask=replacement_mask,
        extent=extent,
        out_path=binned_path,
        downsample=downsample,
    )
    plot_single_map(
        replacement_scenario_fi,
        title="Scenario FI on replaced non-fuel pixels",
        cbar_label="Scenario FI on newly burnable pixels",
        cmap="magma",
        norm=replacement_norm,
        replacement_mask=replacement_mask,
        extent=extent,
        out_path=replacement_path,
        downsample=downsample,
    )

    summary = summarize_fi_maps(
        scenario=scenario,
        hex_id=hex_id,
        baseline_fi=paired_baseline_fi,
        scenario_fi=paired_scenario_fi,
        delta_fi=paired_delta,
        replacement_scenario_fi=replacement_scenario_fi,
        replacement_mask=replacement_mask,
        delta_plot_limit=delta_plot_limit,
    )
    summary_path = experiment_dir / f"counterfactual_{scenario}_fi_map_summary.csv"
    bin_summary_path = experiment_dir / f"counterfactual_{scenario}_fi_distance_bin_summary.csv"
    pd.DataFrame([asdict(summary)]).to_csv(summary_path, index=False)
    bin_summary.to_csv(bin_summary_path, index=False)
    return summary_path, [combined_path, continuous_path, binned_path, replacement_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FI maps for the barrier-removal counterfactual.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--downsample", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path, plot_paths = write_fi_replacement_maps(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        hex_id=str(args.hex_id).zfill(2),
        scenario=args.scenario,
        percentile=args.percentile,
        downsample=max(1, int(args.downsample)),
    )
    print(f"Wrote FI map summary: {summary_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
