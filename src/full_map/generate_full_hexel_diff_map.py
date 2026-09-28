"""
Compares each target's real-CRS national predicted mosaic (from ``generate_full_hexel_map.py``)
against that target's already-stitched national ground-truth raster
(``config.full_map.national_gt_raster_paths[target]``).
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import yaml
from matplotlib.colors import TwoSlopeNorm

from src.config import Config
from src.full_map.utils import mask_nodata
from src.metrics import compute_ccc, compute_normalized_mae, compute_spearman


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _read_masked(path: str) -> tuple[np.ma.MaskedArray, dict, rasterio.coords.BoundingBox]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        arr = mask_nodata(arr, nodata) if nodata is not None else np.ma.masked_invalid(arr)
        return arr, src.profile.copy(), src.bounds


def compute_national_diff(pred_path: str, gt_path: str) -> tuple[np.ma.MaskedArray, dict, dict[str, float]]:
    """Computes ``prediction - ground_truth`` over the shared national grid, plus summary
    comparison metrics restricted to pixels valid in both rasters."""
    pred_arr, pred_profile, pred_bounds = _read_masked(pred_path)
    gt_arr, gt_profile, gt_bounds = _read_masked(gt_path)

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Prediction mosaic shape {pred_arr.shape} does not match GT raster shape {gt_arr.shape}. Are they on the same grid?"
        )
    if pred_bounds != gt_bounds:
        raise ValueError(f"Prediction mosaic bounds {pred_bounds} do not match GT raster bounds {gt_bounds}. Are they on the same grid?")
    if pred_profile.get("crs") != gt_profile.get("crs") or pred_profile.get("transform") != gt_profile.get("transform"):
        raise ValueError(
            f"Prediction mosaic CRS/transform ({pred_profile.get('crs')}, {pred_profile.get('transform')}) does not match "
            f"GT raster CRS/transform ({gt_profile.get('crs')}, {gt_profile.get('transform')}). Are they on the same grid?"
        )

    valid = ~np.ma.getmaskarray(pred_arr) & ~np.ma.getmaskarray(gt_arr)
    # Cast to float before filling with NaN: integer-dtype rasters (e.g. ros GT, which uses a
    # uint8 array with nodata=255) cannot hold a NaN fill value. Use float32 (not float64) to
    # avoid doubling memory on national-sized (tens of thousands of pixels per side) rasters.
    pred_float = np.ma.filled(pred_arr.astype(np.float32), np.nan)
    gt_float = np.ma.filled(gt_arr.astype(np.float32), np.nan)
    diff = np.ma.masked_array(pred_float - gt_float, mask=~valid)

    out_profile = gt_profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=-9999.0)

    n_valid = int(valid.sum())
    metrics: dict[str, float] = {"n_valid_pixels": n_valid}

    min_valid = 2
    if n_valid < min_valid:
        metrics["ccc"] = float("nan")
        metrics["spearman"] = float("nan")
        metrics["normalized_mae"] = float("nan")
        return diff, out_profile, metrics

    # compute_ccc/compute_spearman/compute_normalized_mae expect batched (B, ...) tensors plus
    # a matching boolean mask; treat the whole national raster as a single sample.
    pred_tensor = torch.from_numpy(pred_arr.filled(0.0).astype(np.float32)).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_arr.filled(0.0).astype(np.float32)).unsqueeze(0)
    mask_tensor = torch.from_numpy(valid).unsqueeze(0)

    metrics["ccc"] = float(compute_ccc(pred_tensor, gt_tensor, mask=mask_tensor).item())
    metrics["spearman"] = float(compute_spearman(pred_tensor, gt_tensor, mask=mask_tensor).item())
    metrics["normalized_mae"] = float(compute_normalized_mae(pred_tensor, gt_tensor, mask=mask_tensor).item())

    return diff, out_profile, metrics


def save_diff_raster(diff: np.ma.MaskedArray, profile: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nodata = profile.get("nodata", -9999.0)
    write_array = diff.filled(nodata).astype("float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(write_array, 1)
    print(f"Saved national diff raster to {output_path}")


def save_diff_plot(diff: np.ma.MaskedArray, reference_raster_path: str, output_path: str, title: str | None, target_label: str) -> None:
    max_abs = max(float(np.abs(diff.compressed()).max()) if diff.count() else 1e-6, 1e-6)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    cmap = copy.copy(plt.get_cmap("RdBu_r"))
    cmap.set_bad(color="black", alpha=0)

    with rasterio.open(reference_raster_path) as ref_src:
        bounds = ref_src.bounds

    _, ax = plt.subplots(figsize=(16, 12), dpi=300)
    im = ax.imshow(diff, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label(f"{target_label} Error (Prediction - Ground Truth)", fontsize=12)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"Saved national diff plot to {output_path}")


# Default per-target labels used for the diff plot colorbar when none is given explicitly.
_DEFAULT_TARGET_LABELS = {"bp": "Burn Probability", "fi": "Fire Intensity", "ros": "Rate of Spread"}


def generate_national_diffs(
    mosaic_dir: str,
    national_gt_raster_paths: dict[str, str],
    output_dir: str,
    title: str | None = None,
    save_plots: bool = False,
    skip_existing: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Diffs every target's national predicted mosaic (``{target}_national_predicted_map.tif`` in
    ``mosaic_dir``, as written by ``generate_full_hexel_map.py``) against that target's
    configured GT raster.

    ``skip_existing``, if True, skips (re)computing a target's diff when
    ``{target}_national_diff_map.tif`` already exists in ``output_dir`` -- useful for resuming
    after a job was killed partway through the target loop. Skipped targets are omitted from
    the returned metrics (no need to recompute metrics for a diff that was already produced).

    Returns a mapping of target name -> summary comparison metrics (only for targets that were
    (re)computed this run).
    """
    mosaic_folder = Path(mosaic_dir)
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}
    for target_name, gt_raster_path in national_gt_raster_paths.items():
        pred_mosaic_path = mosaic_folder / f"{target_name}_national_predicted_map.tif"
        if not pred_mosaic_path.exists():
            print(f"Warning: no predicted mosaic found for target {target_name!r} at {pred_mosaic_path}; skipping its diff.")
            continue

        diff_output_path = output_folder / f"{target_name}_national_diff_map.tif"
        if skip_existing and diff_output_path.exists():
            print(f"Skipping {target_name!r}: {diff_output_path} already exists (--skip-existing).")
            continue

        diff, profile, metrics = compute_national_diff(str(pred_mosaic_path), gt_raster_path)
        save_diff_raster(diff, profile, str(diff_output_path))

        print(f"National diff summary metrics for target {target_name!r}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        all_metrics[target_name] = metrics

        if save_plots:
            plot_title = f"{title} ({target_name})" if title else target_name
            target_label = _DEFAULT_TARGET_LABELS.get(target_name, target_name)
            save_diff_plot(diff, gt_raster_path, str(output_folder / f"{target_name}_national_diff_map.png"), plot_title, target_label)

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff each target's real-CRS national predicted mosaic against its national GT raster.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file (reads config.full_map.national_gt_raster_paths)."
    )
    parser.add_argument(
        "--mosaic-dir",
        type=str,
        required=True,
        help="Directory containing {target}_national_predicted_map.tif files (from generate_full_hexel_map.py).",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/full_map", help="Directory to write one diff .tif per target into.")
    parser.add_argument("--save-plots", action="store_true", help="Also save a PNG diff plot per target.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title (target name is appended automatically).")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute every target's diff even if {target}_national_diff_map.tif already exists "
        "in --output-dir. By default, existing diffs are skipped (resume behavior).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.full_map.national_gt_raster_paths:
        raise ValueError("config.full_map.national_gt_raster_paths must be set (one entry per target) to compute the diff maps.")

    generate_national_diffs(
        mosaic_dir=args.mosaic_dir,
        national_gt_raster_paths=config.full_map.national_gt_raster_paths,
        output_dir=args.output_dir,
        title=args.title,
        save_plots=args.save_plots,
        skip_existing=not args.force_recompute,
    )


if __name__ == "__main__":
    main()
