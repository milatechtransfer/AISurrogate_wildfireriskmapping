"""
Computes a national hazard map (``hazard = BP x min(FI, fi_cap)``, scaled and binned into
NRCan-style hazard classes) directly from the already-built national BP/FI rasters -- both
predicted and ground-truth -- rather than re-stitching a separate hazard hexel mosaic.

This is the hazard counterpart to ``generate_full_hexel_diff_map.py``: it consumes the same
already-mosaicked ``bp``/``fi`` national predicted rasters (from ``generate_full_hexel_map.py``)
and the same configured national GT rasters (``config.full_map.national_gt_raster_paths``), so
no new per-hexel/all-split reconstruction pipeline is required. Both rasters for each of
bp/fi are expected to already share one common grid (CRS/transform/shape) -- true by
construction, since ``generate_full_hexel_map.py`` mosaics predictions directly onto each
target's GT raster grid.

Usage:

    python -m src.full_map.generate_national_hazard_map --config path/to/config.yaml \
        --mosaic-dir experiments/full_map --output-dir experiments/full_map
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm

from src.config import Config
from src.datasets.postprocessing.hazard import (
    DEFAULT_FI_CAP,
    DEFAULT_HAZARD_BIN_THRESHOLDS,
    DEFAULT_SCALE_TO,
    validate_bin_thresholds,
)
from src.datasets.postprocessing.hazard_metrics import calculate_hazard_class_metrics, flatten_hazard_class_metrics

PREDICTED_HAZARD_FILENAME = "hazard_national_predicted_map.tif"
GROUND_TRUTH_HAZARD_FILENAME = "hazard_national_ground_truth_map.tif"
SUMMARY_JSON_FILENAME = "hazard_national_summary.json"


def load_config(path: str) -> Config:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _read_float32_nan(path: str) -> tuple[np.ndarray, dict, rasterio.coords.BoundingBox]:
    """Reads band 1 as float32 with nodata pixels converted to NaN in place (no masked-array
    copy), to keep peak memory as low as possible for national-scale (Canada-wide) rasters.
    """
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype="float32")
        nodata = src.nodata
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            arr[arr == nodata] = np.nan
        return arr, src.profile.copy(), src.bounds


def _require_common_grid(arrays: dict[str, tuple[np.ndarray, dict, rasterio.coords.BoundingBox]]) -> None:
    """Raises if the given (array, profile, bounds) grids don't all match shape/bounds.

    Hazard combines bp/fi pixelwise with no reprojection, so all rasters involved (predicted
    bp/fi, ground-truth bp/fi) must already share one grid.
    """
    names = list(arrays)
    reference_name = names[0]
    _, reference_profile, reference_bounds = arrays[reference_name]
    reference_shape = arrays[reference_name][0].shape
    for name in names[1:]:
        arr, profile, bounds = arrays[name]
        if arr.shape != reference_shape:
            raise ValueError(f"{name!r} raster shape {arr.shape} does not match {reference_name!r} shape {reference_shape}.")
        if bounds != reference_bounds:
            raise ValueError(f"{name!r} raster bounds {bounds} do not match {reference_name!r} bounds {reference_bounds}.")
        if profile.get("crs") != reference_profile.get("crs") or profile.get("transform") != reference_profile.get("transform"):
            raise ValueError(
                f"{name!r} raster CRS/transform ({profile.get('crs')}, {profile.get('transform')}) does not match "
                f"{reference_name!r} CRS/transform ({reference_profile.get('crs')}, {reference_profile.get('transform')})."
            )


def _raw_hazard_inplace(bp: np.ndarray, fi: np.ndarray, fi_cap: float | None) -> np.ndarray:
    """Computes ``bp * min(fi, fi_cap)`` in place, mutating and reusing ``bp``'s buffer as the
    output (and freeing ``fi``) to avoid allocating another national-sized array."""
    if fi_cap is not None:
        np.minimum(fi, np.float32(fi_cap), out=fi)
    bp *= fi  # bp now holds raw hazard
    return bp


def _scale_and_bin_inplace(raw: np.ndarray, denominator: float, scale_to: float, bin_edges: np.ndarray, invalid_class: int) -> np.ndarray:
    """Scales ``raw`` in place, then bins it into 1-based hazard classes (int32). ``raw`` is
    fully consumed (mutated into the scaled array) before the class array is allocated, so only
    one extra national-sized array (the int32 output) is created."""
    raw *= np.float32(scale_to / denominator)
    finite_mask = np.isfinite(raw)
    classes = np.searchsorted(bin_edges, raw, side="right").astype(np.int32)
    classes += 1
    classes[~finite_mask] = invalid_class
    return classes


def _raw_hazard_denominator(raw: np.ndarray) -> float:
    """Max finite, positive raw-hazard value in ``raw`` (float32-safe; avoids hazard.py's
    ``max_finite_hazard``, which upcasts to float64 -- expensive at national-raster scale)."""
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError("no finite hazard values found to compute a denominator")
    value = float(finite.max())
    if value <= 0.0:
        raise ValueError(f"maximum finite hazard must be > 0, got {value}")
    return value


def _compute_binned_hazard_inplace(
    bp: np.ndarray,
    fi: np.ndarray,
    fi_cap: float | None,
    denominator: float,
    scale_to: float,
    bin_edges: np.ndarray,
    invalid_class: int,
) -> np.ndarray:
    """Computes ``bin(scale(bp * min(fi, fi_cap)))`` in float32, mutating ``bp``/``fi`` in
    place to avoid allocating extra national-sized arrays (each is ~10GB at Canada-wide 100m
    resolution). Returns a new int32 class array; ``bp``/``fi`` are left in an unusable state
    (fully consumed) after this call.
    """
    raw = _raw_hazard_inplace(bp, fi, fi_cap)
    return _scale_and_bin_inplace(raw, denominator, scale_to, bin_edges, invalid_class)


def generate_national_hazard_maps(
    gt_bp_path: str,
    gt_fi_path: str,
    output_dir: str,
    predicted_bp_path: str | None = None,
    predicted_fi_path: str | None = None,
    fi_cap: float | None = DEFAULT_FI_CAP,
    scale_to: float = DEFAULT_SCALE_TO,
    bin_thresholds: list[float] | None = None,
    scale_denominator: float | None = None,
    invalid_class: int = 0,
    title: str | None = None,
    save_plots: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Builds a binned national hazard map for ground truth, and (if predicted bp/fi paths are
    given) for prediction too.

    ``predicted_bp_path``/``predicted_fi_path`` are optional: pass both to also compute the
    predicted hazard map and compare it against ground truth (the default, full behavior);
    leave both ``None`` to compute only the ground-truth hazard map, which needs nothing from
    the prediction pipeline (steps 1-2).

    ``scale_denominator``, if ``None``, is derived as the max finite raw hazard over the
    ground-truth bp/fi rasters (mirroring the hexel-level ``all_raw_ground_truth`` default),
    so ground truth and predictions are scaled/binned consistently without needing an explicit
    reference value.

    Memory: at Canada-wide 100m resolution (~55,000 x 46,000 px, ~10GB per float32 raster),
    this processes ground truth and prediction sequentially (never holding both sets of bp/fi
    rasters at once) and mutates arrays in place, so peak memory stays close to 2-3 national
    rasters' worth rather than growing with the number of hazard products/targets involved.

    Returns a summary dict with the resolved denominator and, when predictions are given, the
    hazard class metrics comparing the binned predicted map against the binned ground-truth map
    (via ``calculate_hazard_class_metrics``). Writes nothing and returns metrics read back from
    existing files if the relevant output(s) already exist and ``skip_existing`` is True.
    """
    compute_predicted = predicted_bp_path is not None and predicted_fi_path is not None
    if bool(predicted_bp_path) != bool(predicted_fi_path):
        raise ValueError("predicted_bp_path and predicted_fi_path must both be set, or both left unset (GT-only).")

    if bin_thresholds is None:
        bin_thresholds = list(DEFAULT_HAZARD_BIN_THRESHOLDS)
    bin_edges = validate_bin_thresholds(bin_thresholds)
    num_classes = len(bin_thresholds) + 1

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    pred_output_path = output_folder / PREDICTED_HAZARD_FILENAME
    gt_output_path = output_folder / GROUND_TRUTH_HAZARD_FILENAME
    summary_path = output_folder / SUMMARY_JSON_FILENAME

    required_outputs = [gt_output_path, *([pred_output_path] if compute_predicted else [])]
    if skip_existing and all(path.exists() for path in required_outputs):
        print(f"Skipping hazard national map: {required_outputs} already exist (--skip-existing).")
        with rasterio.open(gt_output_path) as gt_src:
            binned_gt = gt_src.read(1)

        denominator = scale_denominator
        if denominator is None and summary_path.exists():
            with open(summary_path) as handle:
                denominator = json.load(handle).get("denominator")
        if denominator is None:
            # Neither an explicit denominator nor a prior summary file is available (e.g. the
            # job was killed before writing hazard_national_summary.json) -- recompute it from
            # the ground-truth bp/fi rasters, the same way the non-skip path derives it.
            gt_bp, gt_profile, gt_bounds = _read_float32_nan(gt_bp_path)
            gt_fi, _, gt_fi_bounds = _read_float32_nan(gt_fi_path)
            _require_common_grid({"ground_truth_bp": (gt_bp, gt_profile, gt_bounds), "ground_truth_fi": (gt_fi, gt_profile, gt_fi_bounds)})
            raw_gt = _raw_hazard_inplace(gt_bp, gt_fi, fi_cap)
            del gt_bp, gt_fi
            gc.collect()
            denominator = _raw_hazard_denominator(raw_gt)
            del raw_gt
            gc.collect()

        if not compute_predicted:
            summary: dict[str, Any] = {"denominator": denominator, "metrics": None}
            with open(summary_path, "w") as handle:
                json.dump(summary, handle, indent=2)
            return summary
        with rasterio.open(pred_output_path) as pred_src:
            binned_pred = pred_src.read(1)
        metrics = calculate_hazard_class_metrics(binned_pred, binned_gt, invalid_class=invalid_class, num_classes=num_classes)
        summary = {"denominator": denominator, "metrics": flatten_hazard_class_metrics(metrics)}
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    # --- Ground truth phase: only gt bp/fi are held in memory at once. ---
    gt_bp, gt_profile, gt_bounds = _read_float32_nan(gt_bp_path)
    gt_fi, _, gt_fi_bounds = _read_float32_nan(gt_fi_path)
    _require_common_grid({"ground_truth_bp": (gt_bp, gt_profile, gt_bounds), "ground_truth_fi": (gt_fi, gt_profile, gt_fi_bounds)})

    if scale_denominator is not None:
        denominator = float(scale_denominator)
        binned_gt = _compute_binned_hazard_inplace(gt_bp, gt_fi, fi_cap, denominator, scale_to, bin_edges, invalid_class)
        del gt_bp, gt_fi
    else:
        raw_gt = _raw_hazard_inplace(gt_bp, gt_fi, fi_cap)
        del gt_fi
        gc.collect()
        denominator = _raw_hazard_denominator(raw_gt)
        binned_gt = _scale_and_bin_inplace(raw_gt, denominator, scale_to, bin_edges, invalid_class)
        del raw_gt
    gc.collect()

    out_profile = gt_profile.copy()
    out_profile.update(count=1, dtype="int32", nodata=invalid_class)

    with rasterio.open(gt_output_path, "w", **out_profile) as dst:
        dst.write(binned_gt, 1)
    print(f"Saved ground-truth national hazard map to {gt_output_path}")

    if save_plots:
        plot_title = f"{title} (Ground Truth)" if title else "National Hazard (Ground Truth)"
        _plot_hazard_classes(binned_gt, num_classes, invalid_class, gt_bp_path, str(gt_output_path.with_suffix(".png")), plot_title)

    flat_metrics: dict[str, float] | None = None
    if compute_predicted:
        # --- Prediction phase: gt bp/fi are already freed; only pred bp/fi are loaded now. ---
        pred_bp, pred_profile, pred_bounds = _read_float32_nan(predicted_bp_path)  # type: ignore[arg-type]
        pred_fi, _, pred_fi_bounds = _read_float32_nan(predicted_fi_path)  # type: ignore[arg-type]
        _require_common_grid(
            {
                "predicted_bp": (pred_bp, pred_profile, pred_bounds),
                "predicted_fi": (pred_fi, pred_profile, pred_fi_bounds),
                "ground_truth_bp": (binned_gt, gt_profile, gt_bounds),
            }
        )
        binned_pred = _compute_binned_hazard_inplace(pred_bp, pred_fi, fi_cap, denominator, scale_to, bin_edges, invalid_class)
        gc.collect()

        with rasterio.open(pred_output_path, "w", **out_profile) as dst:
            dst.write(binned_pred, 1)
        print(f"Saved predicted national hazard map to {pred_output_path}")

        metrics = calculate_hazard_class_metrics(binned_pred, binned_gt, invalid_class=invalid_class, num_classes=num_classes)
        flat_metrics = flatten_hazard_class_metrics(metrics)

        if save_plots:
            plot_title = f"{title} (Predicted)" if title else "National Hazard (Predicted)"
            _plot_hazard_classes(binned_pred, num_classes, invalid_class, gt_bp_path, str(pred_output_path.with_suffix(".png")), plot_title)

    summary = {"denominator": denominator, "metrics": flat_metrics}
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _plot_hazard_classes(
    binned: np.ndarray, num_classes: int, invalid_class: int, reference_raster_path: str, output_path: str, title: str | None
) -> None:
    cmap = copy.copy(plt.get_cmap("YlOrRd", num_classes))
    cmap.set_bad(color="black", alpha=0)
    masked = np.ma.masked_equal(binned, invalid_class)
    norm = BoundaryNorm(np.arange(0.5, num_classes + 1.5, 1.0), cmap.N)

    with rasterio.open(reference_raster_path) as ref_src:
        bounds = ref_src.bounds

    _, ax = plt.subplots(figsize=(16, 12), dpi=300)
    im = ax.imshow(masked, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, ticks=range(1, num_classes + 1))
    cbar.set_label("Hazard class", fontsize=12)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"Saved national hazard plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a binned national hazard map (BP x FI) for both prediction and ground truth.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file (reads config.full_map.national_gt_raster_paths / hazard_*)."
    )
    parser.add_argument(
        "--mosaic-dir",
        type=str,
        default=None,
        help="Directory containing bp_national_predicted_map.tif / fi_national_predicted_map.tif (from generate_full_hexel_map.py). "
        "Required unless --gt-only is passed.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Only compute the ground-truth hazard map (needs only config.full_map.national_gt_raster_paths, "
        "no predicted mosaics/--mosaic-dir required).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/full_map", help="Directory to write the predicted/ground-truth hazard maps into."
    )
    parser.add_argument("--save-plots", action="store_true", help="Also save a PNG plot per hazard map.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument(
        "--scale-denominator",
        type=float,
        default=None,
        help="Override config.full_map.hazard_scale_denominator. If unset (and the config value is also unset), "
        "the denominator is the max finite raw hazard over the ground-truth bp/fi rasters.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute the hazard maps even if they already exist in --output-dir. By default, existing "
        "outputs are skipped and their metrics are read back (resume behavior).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    gt_paths = config.full_map.national_gt_raster_paths or {}
    missing_gt = [target for target in ("bp", "fi") if target not in gt_paths]
    if missing_gt:
        raise ValueError(f"config.full_map.national_gt_raster_paths is missing required target(s) {missing_gt} for hazard.")

    predicted_bp_path = None
    predicted_fi_path = None
    if not args.gt_only:
        if not args.mosaic_dir:
            raise ValueError("--mosaic-dir is required unless --gt-only is passed.")
        mosaic_folder = Path(args.mosaic_dir)
        predicted_bp_path = mosaic_folder / "bp_national_predicted_map.tif"
        predicted_fi_path = mosaic_folder / "fi_national_predicted_map.tif"
        for path in (predicted_bp_path, predicted_fi_path):
            if not path.exists():
                raise FileNotFoundError(f"Predicted national mosaic not found: {path}. Run generate_full_hexel_map.py first.")

    scale_denominator = args.scale_denominator if args.scale_denominator is not None else config.full_map.hazard_scale_denominator

    summary = generate_national_hazard_maps(
        gt_bp_path=gt_paths["bp"],
        gt_fi_path=gt_paths["fi"],
        output_dir=args.output_dir,
        predicted_bp_path=str(predicted_bp_path) if predicted_bp_path else None,
        predicted_fi_path=str(predicted_fi_path) if predicted_fi_path else None,
        fi_cap=config.full_map.hazard_fi_cap,
        scale_to=config.full_map.hazard_scale_to,
        bin_thresholds=config.full_map.hazard_bin_thresholds,
        scale_denominator=scale_denominator,
        title=args.title,
        save_plots=args.save_plots,
        skip_existing=not args.force_recompute,
    )

    print("\n===== National Hazard Map Summary =====")
    print(f"Denominator: {summary['denominator']}")
    if summary["metrics"] is not None:
        print("Hazard class metrics (predicted vs ground truth):")
        for key, value in summary["metrics"].items():
            print(f"  {key}: {value}")
    else:
        print("Ground-truth-only run: no prediction comparison metrics.")


if __name__ == "__main__":
    main()
