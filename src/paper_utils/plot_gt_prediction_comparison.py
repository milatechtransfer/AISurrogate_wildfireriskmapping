"""
Used to generate visuals for AAAI paper.
Plot Prediction / Ground Truth / Difference panels for BP, FI, and ROS from a
simple materialized folder structure (not a full evaluate_hexels/counterfactual
run), e.g.:

    final_plots/
        GT/
            bp.tif
            fi.tif
            ros.tif
            hex16_actual.shp (+ .dbf/.prj/.shx/.cpg, and a rasterized hex16_actual.tif)
        predictions/
            hexel_16_bp_predicted.tif
            hexel_16_fi_predicted.tif
            hexel_16_ros_predicted.tif

For each target (BP, FI, ROS), this produces one separate 1x3 figure via the
same `visualize_target_grids` renderer used elsewhere in the postprocessing
pipeline (`src/datasets/postprocessing/visualize_predictions.py`):

    Prediction (input support) | Ground Truth (finite targets) | Difference (target overlap)

Masking, per the requested convention:
  - Ground Truth is read via `load_spatial_raster` (`data_preparation/spatial/utils.py`),
    reprojected onto the prediction's grid and clipped to the actual-fire-occurrence footprint
    for the hex (`GT/hex<hex_id>_actual.shp`) in one call (`mask_path=...`) -- the same pattern
    `load_ground_truth_valid_mask` uses in `weather_scenario_mean_delta.py`. This also handles
    the GT raster living on a different grid/resolution than the prediction raster. The mask is
    only used to blank out (NaN) pixels outside the footprint -- no outline is drawn.
  - Prediction is then restricted to wherever that actual-masked Ground Truth is finite
    (non-NaN/nodata), rather than to the prediction's own valid pixels -- i.e. the GT's valid
    mask drives what's shown/compared for both panels, and the Difference panel is implicitly
    restricted to the same support (both are finite there by construction).
  - For FI and ROS, the color scale (both the shared Prediction/GT scale and
    the Difference scale) is robust-clipped to the 99th percentile of finite
    values, matching `effective_robust_plot_percentile`'s existing default for
    these two targets in `src/datasets/postprocessing/utils.py` (`target.name
    in {"fi", "ros"} -> 99.0`). BP keeps the full value range (no clipping).
  - Optionally (`--fill_nan_with_zero`), any remaining NaN/nodata gaps that fall
    *within* the actual-fire-occurrence footprint (rather than genuinely outside
    it) are filled with 0 instead of being left blank, for both Ground Truth and
    Prediction.

Used to generate Figure 2 with consistent masking.

Usage:
    python -m src.paper_utils.plot_gt_prediction_comparison \
        --dir final_plots --hex_id 16 --out_dir final_plots/comparison_figures \
        --fill_nan_with_zero
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features

from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.visualize_predictions import visualize_target_grids
from src.datasets.targets import TargetName, get_target_spec

TARGETS: list[TargetName] = ["bp", "fi", "ros"]

# Matches `effective_robust_plot_percentile` in `src/datasets/postprocessing/utils.py`:
# FI/ROS are robust-clipped to the 99th percentile; BP keeps the full range.
ROBUST_PLOT_PERCENTILE = {"bp": None, "fi": 99.0, "ros": 99.0}


def gt_raster_path(gt_dir: Path, target: str) -> Path:
    return gt_dir / f"{target}.tif"


def actual_mask_path(gt_dir: Path, hex_id: str) -> Path:
    """Path to the actual-fire-occurrence mask shapefile for `hex_id`, e.g. GT/hex16_actual.shp."""
    return gt_dir / f"hex{hex_id}_actual.shp"


def prediction_raster_path(predictions_dir: Path, hex_id: str, target: str) -> Path:
    return predictions_dir / f"hexel_{hex_id}_{target}_predicted.tif"


def read_prediction(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")
    with rasterio.open(path) as src:
        array = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            array = np.where(array == nodata, np.nan, array)
        return array, src.profile.copy()


def load_actual_mask(gt_dir: Path, hex_id: str, reference_profile: dict) -> np.ndarray:
    """Boolean array (True = inside the actual-fire-occurrence footprint) on `reference_profile`'s grid."""
    mask_path = actual_mask_path(gt_dir, hex_id)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing actual-fire mask for hex_id={hex_id!r}: {mask_path}")
    geometries = gpd.read_file(mask_path).to_crs(reference_profile["crs"]).geometry
    outside = rasterio.features.geometry_mask(
        geometries,
        out_shape=(reference_profile["height"], reference_profile["width"]),
        transform=reference_profile["transform"],
        invert=False,
    )
    return ~outside


def load_ground_truth_masked(gt_dir: Path, hex_id: str, target: str, reference_profile: dict) -> np.ndarray:
    """Ground-truth array (NaN elsewhere), reprojected onto `reference_profile`'s grid and clipped
    to the actual-fire-occurrence footprint for `hex_id`, via `load_spatial_raster`."""
    gt_path = gt_raster_path(gt_dir, target)
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth raster for target={target!r}: {gt_path}")
    mask_path = actual_mask_path(gt_dir, hex_id)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing actual-fire mask for hex_id={hex_id!r}: {mask_path}")
    ground_truth, _ = load_spatial_raster(path=gt_path, reference_profile=reference_profile, mask_path=mask_path)
    return np.ma.filled(ground_truth.astype("float64"), np.nan)


def plot_target_comparison(
    gt_dir: Path,
    predictions_dir: Path,
    hex_id: str,
    target: str,
    out_dir: Path,
    fill_nan_with_zero: bool = False,
) -> None:
    """Render one Prediction/Ground Truth/Difference figure for `target`, masked per module docstring.

    If `fill_nan_with_zero` is set, any remaining NaN/nodata gaps *within* the actual-fire-occurrence
    footprint (rather than genuinely outside it) are filled with 0 for both the Ground Truth and
    Prediction panels, instead of being left as NaN/blank.
    """
    target_spec = get_target_spec(target)

    pred_array, pred_profile = read_prediction(prediction_raster_path(predictions_dir, hex_id, target))
    gt_masked = load_ground_truth_masked(gt_dir, hex_id, target, pred_profile)
    if gt_masked.shape != pred_array.shape:
        raise ValueError(f"Ground truth grid {gt_masked.shape} does not match prediction grid {pred_array.shape} for target={target!r}.")

    if fill_nan_with_zero:
        actual_mask = load_actual_mask(gt_dir, hex_id, pred_profile)
        gt_masked = np.where(actual_mask & ~np.isfinite(gt_masked), 0.0, gt_masked)

    # Prediction restricted to wherever the actual-masked Ground Truth is finite (not the
    # prediction's own valid-pixel mask).
    pred_masked = np.where(np.isfinite(gt_masked), pred_array, np.nan)
    if fill_nan_with_zero:
        pred_masked = np.where(actual_mask & ~np.isfinite(pred_masked), 0.0, pred_masked)

    percentile = ROBUST_PLOT_PERCENTILE[target]
    visualize_target_grids(
        gt_grid=gt_masked,
        pred_grid=pred_masked,
        hex_id=hex_id,
        save_dir=str(out_dir),
        target_label=target_spec.label,
        target_name=target,
        value_percentile=percentile,
        diff_percentile=percentile,
        show_prediction_support_outline=False,
        gt_title="Reference Targets",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=str, required=True, help="Base folder containing GT/ and predictions/ subfolders.")
    parser.add_argument("--hex_id", type=str, default="16", help="Hex id to plot, e.g. '16'.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the 3 figures to (a 'predicted_hexels_plot' subfolder is created under it). "
        "Defaults to '<dir>/comparison_figures'.",
    )
    parser.add_argument(
        "--fill_nan_with_zero",
        action="store_true",
        help="Fill NaN/nodata gaps with 0 (instead of leaving them blank) wherever they fall inside the "
        "actual-fire-occurrence footprint, for both Ground Truth and Prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.dir)
    gt_dir = base_dir / "GT"
    predictions_dir = base_dir / "predictions"
    out_dir = Path(args.out_dir) if args.out_dir is not None else base_dir / "comparison_figures"

    for target in TARGETS:
        plot_target_comparison(gt_dir, predictions_dir, args.hex_id, target, out_dir, fill_nan_with_zero=args.fill_nan_with_zero)


if __name__ == "__main__":
    main()
