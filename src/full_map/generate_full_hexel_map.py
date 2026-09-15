"""
Mosaics per-hexel predicted rasters (from ``generate_predictions.py``) into one true national
raster per target, using each hexel's real geographic footprint.

Unlike the previous schematic version of this script (which arranged a hand-picked subset of
hexels on a fake ``(row, col)`` grid with no real coordinates), this reprojects every predicted
hexel onto the same grid (CRS/transform/shape) as that target's configured national
ground-truth raster (``config.full_map.national_gt_raster_paths[target]``), so the resulting
mosaic is directly comparable to that GT raster (see ``generate_full_hexel_diff_map.py``). The
model here predicts multiple targets (e.g. bp/fi/ros), so one mosaic is produced per target.
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

from src.config import Config
from src.full_map.utils import (
    get_scale_settings,
    group_predicted_hexel_files_by_target,
    load_hexel_shapefile,
    mask_nodata,
    mosaic_predicted_hexels,
)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _plot_mosaic(mosaic: np.ndarray, profile: dict, reference_raster_path: str, output_path: str, title: str | None, scale: str) -> None:
    nodata = profile.get("nodata", -9999.0)
    masked = mask_nodata(mosaic, nodata)
    valid = masked.compressed()
    pos_min = float(valid[valid > 0].min()) if valid[valid > 0].size else 1e-6
    global_max = float(valid.max()) if valid.size else 1.0
    norm = get_scale_settings(scale, pos_min, global_max)

    with rasterio.open(reference_raster_path) as ref_src:
        bounds = ref_src.bounds

    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color="black", alpha=0)

    _, ax = plt.subplots(figsize=(16, 12), dpi=300)
    im = ax.imshow(masked, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label(f"Prediction ({scale.title()})", fontsize=12)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved mosaic plot to {output_path}")


def generate_national_mosaics(
    pred_root: str,
    pattern: str,
    shapefile_path: str,
    hexel_id_column: str,
    reference_raster_paths: dict[str, str],
    output_dir: str,
    title: str | None = None,
    scale: str = "linear",
    save_plots: bool = False,
    skip_existing: bool = True,
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Builds and saves one real-CRS national predicted-hexel mosaic per target.

    Args:
        pred_root: Directory containing per-split predicted hexels (e.g. ``config.save_dir``,
            with predictions under ``<split>/predicted_hexels/``).
        pattern: Glob pattern (relative to ``pred_root``) matching predicted hexel rasters,
            e.g. ``"*/predicted_hexels/hexel_*_predicted.tif"``.
        shapefile_path: Path to the national hexel-polygon shapefile.
        hexel_id_column: Column in the shapefile holding each polygon's hex_id.
        reference_raster_paths: Mapping of target name -> already-stitched national GT raster
            path; each defines the output grid (CRS/transform/shape) for that target's mosaic.
        output_dir: Directory to write ``{target}_national_predicted_map.tif`` (and, if
            ``save_plots``, ``{target}_national_predicted_map.png``) into.
        title: Optional plot title (per target name is appended automatically).
        scale: 'linear' or 'log' color scaling for the optional plots.
        save_plots: If True, also saves a real-coordinate PNG plot per target.
        skip_existing: If True, skip (re)building a target's mosaic when
            ``{target}_national_predicted_map.tif`` already exists in ``output_dir`` -- useful
            for resuming after a job was killed partway through the target loop, without
            re-mosaicking targets that already finished.

    Returns:
        Mapping of target name -> (mosaic array, rasterio profile). Targets skipped via
        ``skip_existing`` are read back from their existing ``.tif`` so callers still get a
        complete result set.
    """
    pred_folder = Path(pred_root)
    if not pred_folder.exists():
        raise FileNotFoundError(f"Prediction root directory not found: {pred_folder}")

    shapefile_gdf = load_hexel_shapefile(shapefile_path, hexel_id_column)
    files_by_target = group_predicted_hexel_files_by_target(pred_folder, pattern)
    if not files_by_target:
        raise FileNotFoundError(f"No predicted hexel rasters found under {pred_folder} matching pattern {pattern!r}.")

    results: dict[str, tuple[np.ndarray, dict]] = {}
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for target_name, file_map in files_by_target.items():
        if target_name not in reference_raster_paths:
            print(f"Warning: no reference/GT raster configured for target {target_name!r}; skipping its mosaic.")
            continue

        output_tif_path = output_folder / f"{target_name}_national_predicted_map.tif"
        if skip_existing and output_tif_path.exists():
            print(f"Skipping {target_name!r}: {output_tif_path} already exists (--skip-existing).")
            with rasterio.open(output_tif_path) as existing_src:
                results[target_name] = (existing_src.read(1), existing_src.profile.copy())
            continue

        reference_raster_path = reference_raster_paths[target_name]
        mosaic, profile = mosaic_predicted_hexels(
            file_map=file_map,
            shapefile_gdf=shapefile_gdf,
            hexel_id_column=hexel_id_column,
            reference_raster_path=reference_raster_path,
        )

        with rasterio.open(output_tif_path, "w", **profile) as dst:
            dst.write(mosaic, 1)
        print(f"Saved {target_name!r} national predicted mosaic to {output_tif_path}")

        if save_plots:
            plot_title = f"{title} ({target_name})" if title else target_name
            plot_path = output_folder / f"{target_name}_national_predicted_map.png"
            _plot_mosaic(mosaic, profile, reference_raster_path, str(plot_path), plot_title, scale)

        results[target_name] = (mosaic, profile)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic predicted hexel rasters onto the real national grid, per target.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (reads config.full_map.*).")
    parser.add_argument(
        "--pred-root", type=str, default=None, help="Directory containing per-split predicted hexels. Defaults to config.save_dir."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/predicted_hexels/hexel_*_predicted.tif",
        help="Glob pattern (relative to --pred-root) matching predicted hexel rasters.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/full_map", help="Directory to write one national mosaic .tif per target into."
    )
    parser.add_argument("--save-plots", action="store_true", help="Also save a PNG plot of each target's mosaic.")
    parser.add_argument("--scale", type=str, choices=["log", "linear"], default="linear", help="Color scaling for the optional plots.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title (target name is appended automatically).")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute every target's mosaic even if {target}_national_predicted_map.tif already exists "
        "in --output-dir. By default, existing mosaics are skipped and read back instead (resume behavior).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.full_map.national_shapefile_path:
        raise ValueError("config.full_map.national_shapefile_path must be set to mosaic predictions.")
    if not config.full_map.national_gt_raster_paths:
        raise ValueError("config.full_map.national_gt_raster_paths must be set (one entry per target) as each mosaic's reference grid.")

    generate_national_mosaics(
        pred_root=args.pred_root or config.save_dir,
        pattern=args.pattern,
        shapefile_path=config.full_map.national_shapefile_path,
        hexel_id_column=config.full_map.hexel_id_column,
        reference_raster_paths=config.full_map.national_gt_raster_paths,
        output_dir=args.output_dir,
        title=args.title,
        scale=args.scale,
        save_plots=args.save_plots,
        skip_existing=not args.force_recompute,
    )


if __name__ == "__main__":
    main()
