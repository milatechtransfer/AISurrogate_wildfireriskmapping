"""
Used to generate visuals for AAAI paper.
Compute the mean BP/FI/ROS delta for one counterfactual weather-scenario run.

Single-purpose script for a table row like:

    Intervention        delta_BP   delta_FI   delta_ROS
    QC to BC weather      0.0077      4159        1.78

Reads the already-materialized baseline/scenario prediction rasters for one
`experiment_dir` (as produced by `evaluate_counterfactual.py`, e.g.
`final_results/counterfactual_mean_weather_multi_output_hex16`) and reports, per
endpoint, the mean of (scenario - baseline) over all finite pixels of the
stitched hexel raster(s).

This intentionally does NOT apply fuel-based support masking (the
`nonfuel_ids`/`static_nonfuel_ids` restriction used by
`counterfactual_response_maps.py::load_endpoint_response`), because that
requires the raw fuel raster from `raw_data_dir`, which for a weather scenario
lives on the training cluster and is not part of the materialized experiment
output. Instead, each endpoint's delta is restricted to pixels where the
corresponding ground-truth raster under `<experiment_dir>/GT/<endpoint>.tif`
(`bp.tif`, `fi.tif`, `ros.tif`) is defined (not nodata/NaN), reprojected onto
the prediction grid -- the same "valid ground truth" restriction the full
counterfactual pipeline applies via `load_ground_truth`/`restrict_to_support`.
It is additionally clipped to the actual-fire-occurrence footprint for that hex
(`<experiment_dir>/GT/hex<hex_id>_actual.shp`, alongside a rasterized copy
`hex<hex_id>_actual.tif`), passed as `mask_path` directly to `load_spatial_raster`
-- the same pattern `load_target_grid_for_mask_scope` uses in
`src/datasets/postprocessing/utils.py` for the default "actual" mask scope.

Used to generate: Figure 4 and Table 4.

Usage:
    python -m src.paper_utils.weather_scenario_mean_delta \
        --experiment_dir final_results/counterfactual_mean_weather_multi_output_hex16 \
        --scenario bc_mean_weather_transplant \
        --hex_ids 16

Add `--out_path` to also render a 3-panel (BP/FI/ROS) response-delta figure for
the first requested hex_id, using the same rasters (RdBu_r diverging colour
scale, symmetric around zero).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    delta_norm,
    downsample_for_display,
    find_local_prediction_dir,
    finite_values,
    prediction_raster_path,
    read_prediction,
)

ENDPOINTS = ("bp", "fi", "ros")
DEFAULT_GT_SUBDIR = "GT"
PANEL_TITLES: dict[str, str] = {
    "bp": "Burn probability response",
    "fi": "Fire intensity response",
    "ros": "Rate of spread response",
}
PANEL_CBAR_LABELS: dict[str, str] = {
    "bp": "\u0394 burn probability",
    "fi": "\u0394 fire intensity (kW m$^{-1}$)",
    "ros": "\u0394 rate of spread (m min$^{-1}$)",
}


def ground_truth_path(experiment_dir: Path, endpoint: str, gt_dir: Path | None = None) -> Path:
    return (gt_dir if gt_dir is not None else experiment_dir / DEFAULT_GT_SUBDIR) / f"{endpoint}.tif"


def actual_fire_mask_path(experiment_dir: Path, hex_id: str, gt_dir: Path | None = None) -> Path:
    """Path to the actual-fire-occurrence mask shapefile for `hex_id`, e.g. GT/hex16_actual.shp.

    Matches `Paths.mask_grid_actual()`'s naming convention; the raster form
    (`hex16_actual.tif`) alongside it is a rasterized copy of the same footprint.
    """

    return (gt_dir if gt_dir is not None else experiment_dir / DEFAULT_GT_SUBDIR) / f"hex{hex_id}_actual.shp"


def load_ground_truth_valid_mask(
    experiment_dir: Path,
    hex_id: str,
    endpoint: str,
    reference_profile: dict,
    *,
    gt_dir: Path | None = None,
) -> np.ndarray:
    """Boolean mask, True where the ground-truth raster is defined (not nodata/NaN) within the
    actual-fire-occurrence footprint for `hex_id`, on the prediction grid."""

    gt_path = ground_truth_path(experiment_dir, endpoint, gt_dir)
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth raster for endpoint={endpoint!r}: {gt_path}")
    mask_path = actual_fire_mask_path(experiment_dir, hex_id, gt_dir)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing actual-fire mask for hex_id={hex_id!r}: {mask_path}")
    ground_truth, _ = load_spatial_raster(path=gt_path, reference_profile=reference_profile, mask_path=mask_path)
    invalid = np.ma.getmaskarray(ground_truth) | ~np.isfinite(np.ma.filled(ground_truth, np.nan))
    return ~invalid


def load_delta_raster(
    experiment_dir: Path,
    scenario: str,
    hex_id: str,
    endpoint: str,
    *,
    gt_dir: Path | None = None,
) -> np.ma.MaskedArray:
    """Read baseline/scenario prediction rasters for one hex_id/endpoint and return (scenario - baseline),
    restricted to pixels where the endpoint's ground-truth raster is defined (not nodata/NaN)."""

    baseline_dir = find_local_prediction_dir(experiment_dir, "baseline", hex_id, endpoint)
    scenario_dir = find_local_prediction_dir(experiment_dir, scenario, hex_id, endpoint)
    baseline_path = prediction_raster_path(baseline_dir, hex_id, target_name=endpoint)
    baseline = read_prediction(baseline_path)
    scenario_values = read_prediction(prediction_raster_path(scenario_dir, hex_id, target_name=endpoint))
    if scenario_values.shape != baseline.shape:
        raise ValueError(
            f"Scenario {endpoint.upper()} shape {scenario_values.shape} does not match "
            f"baseline grid {baseline.shape} for hex_id={hex_id!r}."
        )
    delta = scenario_values - baseline

    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()
    valid = load_ground_truth_valid_mask(experiment_dir, hex_id, endpoint, reference_profile, gt_dir=gt_dir)
    if valid.shape != delta.shape:
        raise ValueError(f"Ground-truth grid {valid.shape} does not match prediction grid {delta.shape} for endpoint={endpoint!r}.")
    return np.ma.masked_where(~valid | np.ma.getmaskarray(delta), delta)


def mean_delta_for_endpoint(experiment_dir: Path, scenario: str, hex_ids: list[str], endpoint: str, *, gt_dir: Path | None = None) -> float:
    """Pooled mean of (scenario - baseline) across all finite pixels of all `hex_ids`."""

    pooled = [finite_values(load_delta_raster(experiment_dir, scenario, hex_id, endpoint, gt_dir=gt_dir)) for hex_id in hex_ids]
    all_values = np.concatenate(pooled) if pooled else np.array([], dtype=np.float64)
    if all_values.size == 0:
        return float("nan")
    return float(np.mean(all_values))


def plot_response_figure(
    experiment_dir: Path,
    scenario: str,
    hex_id: str,
    out_path: Path,
    *,
    downsample: int = 1,
    gt_dir: Path | None = None,
) -> None:
    """Render a 1x3 BP/FI/ROS response-delta figure (RdBu_r, zero-centered) for one hex_id."""

    deltas = {endpoint: load_delta_raster(experiment_dir, scenario, hex_id, endpoint, gt_dir=gt_dir) for endpoint in ENDPOINTS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, endpoint in zip(axes, ENDPOINTS, strict=True):
        norm = delta_norm([deltas[endpoint]])
        image = ax.imshow(
            downsample_for_display(deltas[endpoint], downsample),
            cmap="RdBu_r",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(PANEL_TITLES[endpoint], fontsize=16)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cbar.set_label(PANEL_CBAR_LABELS[endpoint])
        # Fewer ticks + scientific-notation offset text (e.g. "1e-3") instead of many
        # long fixed-decimal labels, which overlap for these small delta magnitudes.
        cbar.locator = MaxNLocator(nbins=5)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar.formatter = formatter
        cbar.update_ticks()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        required=True,
        help="Materialized counterfactual run directory, e.g. final_results/counterfactual_mean_weather_multi_output_hex16.",
    )
    parser.add_argument("--scenario", type=str, required=True, help="Scenario name, e.g. bc_mean_weather_transplant.")
    parser.add_argument("--hex_ids", type=str, nargs="+", required=True, help="Hex id(s) to pool the mean delta over, e.g. 16.")
    parser.add_argument("--label", type=str, default=None, help="Optional row label to print, e.g. 'QC to BC weather'.")
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Optional output image path. If given, also renders a 3-panel BP/FI/ROS response-delta figure for the first --hex_ids entry.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Stride factor to downsample rasters before plotting (for speed/size).")
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=None,
        help="Directory with ground-truth rasters (bp.tif/fi.tif/ros.tif) used to restrict the response to valid pixels. Defaults to <experiment_dir>/GT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label or args.scenario

    deltas = {
        endpoint: mean_delta_for_endpoint(args.experiment_dir, args.scenario, args.hex_ids, endpoint, gt_dir=args.gt_dir)
        for endpoint in ENDPOINTS
    }

    print(f"{'Intervention':<24}{'delta_BP':>12}{'delta_FI':>12}{'delta_ROS':>12}")
    print(f"{label:<24}{deltas['bp']:>12.4f}{deltas['fi']:>12.0f}{deltas['ros']:>12.2f}")

    if args.out_path is not None:
        plot_response_figure(
            args.experiment_dir, args.scenario, args.hex_ids[0], args.out_path, downsample=args.downsample, gt_dir=args.gt_dir
        )
        print(f"Wrote response-delta figure: {args.out_path}")


if __name__ == "__main__":
    main()
