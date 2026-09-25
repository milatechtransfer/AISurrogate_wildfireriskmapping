"""
Mosaics per-hexel rasters into one true national raster per target, using each hexel's real
geographic footprint.

By default this mosaics *predicted* hexels (from ``generate_predictions.py``). Pass ``--gt`` to
mosaic *ground-truth* hexels instead -- stitching each hexel's raw GT raster
(``data_preparation.paths.Paths``) the same way, so the national GT raster is built exactly like
the predicted one instead of read directly from an already-stitched file. Once built, point
``config.full_map.national_gt_raster_paths`` at the produced ``{target}_national_gt_map.tif``
files.
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
    backfill_mosaic_from_reference,
    build_raw_hexel_file_map,
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
    shapefile_path: str,
    hexel_id_column: str,
    reference_raster_paths: dict[str, str],
    output_dir: str,
    pred_root: str | None = None,
    pattern: str | None = None,
    file_maps_by_target: dict[str, dict[int, Path]] | None = None,
    output_filename_template: str = "{target}_national_predicted_map.tif",
    title: str | None = None,
    scale: str = "linear",
    save_plots: bool = False,
    skip_existing: bool = True,
    backfill_from_reference: bool = False,
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Builds and saves one real-CRS national mosaic per target.

    Either pass ``pred_root``/``pattern`` (mosaics predicted hexels, scanned from disk -- the
    default/original behavior), or pass an already-built ``file_maps_by_target`` directly (e.g.
    via ``build_raw_hexel_file_map`` for ground-truth hexels) -- exactly one of the two must be
    given.

    Args:
        shapefile_path: Path to the national hexel-polygon shapefile.
        hexel_id_column: Column in the shapefile holding each polygon's hex_id.
        reference_raster_paths: Mapping of target name -> already-stitched national GT raster
            path; each defines the output grid (CRS/transform/shape) for that target's mosaic.
        output_dir: Directory to write ``output_filename_template``-named rasters (and, if
            ``save_plots``, the matching ``.png``) into.
        pred_root: Directory containing per-split predicted hexels (e.g. ``config.save_dir``,
            with predictions under ``<split>/predicted_hexels/``). Mutually exclusive with
            ``file_maps_by_target``.
        pattern: Glob pattern (relative to ``pred_root``) matching predicted hexel rasters,
            e.g. ``"*/predicted_hexels/hexel_*_predicted.tif"``. Required with ``pred_root``.
        file_maps_by_target: Mapping of target name -> {hex_id: path}, already built by the
            caller (e.g. ``build_raw_hexel_file_map`` per target for ground truth). Mutually
            exclusive with ``pred_root``/``pattern``.
        output_filename_template: Filename template (formatted with ``target=...``) for each
            target's output raster, e.g. ``"{target}_national_gt_map.tif"`` for ground truth.
        title: Optional plot title (per target name is appended automatically).
        scale: 'linear' or 'log' color scaling for the optional plots.
        save_plots: If True, also saves a real-coordinate PNG plot per target.
        skip_existing: If True, skip (re)building a target's mosaic when its output raster
            already exists in ``output_dir`` -- useful for resuming after a job was killed
            partway through the target loop, without re-mosaicking targets that already finished.
        backfill_from_reference: If True, fills any pixel still nodata after per-hexel
            stitching from ``reference_raster_paths[target]`` (e.g. a seamless, already-merged
            national raster) -- useful when per-hexel source rasters (e.g. raw ground-truth
            hexels via ``file_maps_by_target``) don't individually cover every pixel their
            official merged counterpart does. Per-hexel data always takes priority; this only
            fills gaps left after stitching, and only on freshly computed (non-skipped) mosaics.

    Returns:
        Mapping of target name -> (mosaic array, rasterio profile). Targets skipped via
        ``skip_existing`` are read back from their existing ``.tif`` so callers still get a
        complete result set.
    """
    pred_group_given = pred_root is not None and pattern is not None
    pred_group_partial = (pred_root is not None) != (pattern is not None)
    if pred_group_partial:
        raise ValueError("Both pred_root and pattern must be provided together.")
    if pred_group_given == (file_maps_by_target is not None):
        raise ValueError("Pass exactly one of (pred_root and pattern) or file_maps_by_target.")

    shapefile_gdf = load_hexel_shapefile(shapefile_path, hexel_id_column)
    if file_maps_by_target is None:
        pred_folder = Path(pred_root)  # type: ignore[arg-type]
        if not pred_folder.exists():
            raise FileNotFoundError(f"Prediction root directory not found: {pred_folder}")
        file_maps_by_target = group_predicted_hexel_files_by_target(pred_folder, pattern)  # type: ignore[arg-type]
        if not file_maps_by_target:
            raise FileNotFoundError(f"No predicted hexel rasters found under {pred_folder} matching pattern {pattern!r}.")

    # Single-target models are grouped under the key "default" (see
    # group_predicted_hexel_files_by_target); remap that to the sole configured target so its
    # mosaic isn't silently skipped below.
    if set(file_maps_by_target.keys()) == {"default"} and len(reference_raster_paths) == 1:
        (only_target,) = reference_raster_paths.keys()
        file_maps_by_target = {only_target: file_maps_by_target["default"]}

    results: dict[str, tuple[np.ndarray, dict]] = {}
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for target_name, file_map in file_maps_by_target.items():
        if target_name not in reference_raster_paths:
            print(f"Warning: no reference/GT raster configured for target {target_name!r}; skipping its mosaic.")
            continue

        output_tif_path = output_folder / output_filename_template.format(target=target_name)
        if skip_existing and output_tif_path.exists():
            print(f"Skipping {target_name!r}: {output_tif_path} already exists (--skip-existing).")
            with rasterio.open(output_tif_path) as existing_src:
                results[target_name] = (existing_src.read(1), existing_src.profile.copy())
            continue

        if not file_map:
            print(f"Warning: no hexel rasters found for target {target_name!r}; skipping its mosaic.")
            continue

        reference_raster_path = reference_raster_paths[target_name]
        mosaic, profile = mosaic_predicted_hexels(
            file_map=file_map,
            shapefile_gdf=shapefile_gdf,
            hexel_id_column=hexel_id_column,
            reference_raster_path=reference_raster_path,
        )

        if backfill_from_reference:
            mosaic = backfill_mosaic_from_reference(mosaic, profile["nodata"], reference_raster_path)

        with rasterio.open(output_tif_path, "w", **profile) as dst:
            dst.write(mosaic, 1)
        print(f"Saved {target_name!r} national mosaic to {output_tif_path}")

        if save_plots:
            plot_title = f"{title} ({target_name})" if title else target_name
            plot_path = output_tif_path.with_suffix(".png")
            _plot_mosaic(mosaic, profile, reference_raster_path, str(plot_path), plot_title, scale)

        results[target_name] = (mosaic, profile)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mosaic predicted (or, with --gt, ground-truth) hexel rasters onto the real national grid."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (reads config.full_map.*).")
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Mosaic ground-truth hexels (stitched from raw per-hexel rasters under --raw-data-dir) instead of "
        "predicted hexels. Writes {target}_national_gt_map.tif instead of {target}_national_predicted_map.tif.",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default=None,
        help="Directory containing per-split predicted hexels. Defaults to config.save_dir. Ignored with --gt.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/predicted_hexels/hexel_*_predicted.tif",
        help="Glob pattern (relative to --pred-root) matching predicted hexel rasters. Ignored with --gt.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help="Directory containing raw per-hexel data (hex{id}/...). Defaults to config.data.raw_data_dir. Only used with --gt.",
    )
    parser.add_argument(
        "--backfill-from-reference",
        action="store_true",
        help="Only used with --gt. Fills any pixel still nodata after per-hexel stitching from "
        "config.full_map.national_gt_raster_paths (the seamless, already-merged national raster) -- useful when "
        "per-hexel raw rasters don't individually cover every pixel their official merged counterpart does. "
        "Per-hexel data always takes priority; this only fills the remaining gaps.",
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
        help="Recompute every target's mosaic even if its output raster already exists in --output-dir. By "
        "default, existing mosaics are skipped and read back instead (resume behavior).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.full_map.national_shapefile_path:
        raise ValueError("config.full_map.national_shapefile_path must be set to mosaic hexels.")
    if not config.full_map.national_gt_raster_paths:
        raise ValueError("config.full_map.national_gt_raster_paths must be set (one entry per target) as each mosaic's reference grid.")

    common_kwargs = dict(
        shapefile_path=config.full_map.national_shapefile_path,
        hexel_id_column=config.full_map.hexel_id_column,
        reference_raster_paths=config.full_map.national_gt_raster_paths,
        output_dir=args.output_dir,
        title=args.title,
        scale=args.scale,
        save_plots=args.save_plots,
        skip_existing=not args.force_recompute,
    )

    if args.gt:
        raw_data_dir = args.raw_data_dir or config.data.raw_data_dir
        file_maps_by_target = {
            target: build_raw_hexel_file_map(raw_data_dir, target, config.data_prep.scenario_name)
            for target in config.full_map.national_gt_raster_paths
        }
        generate_national_mosaics(
            file_maps_by_target=file_maps_by_target,
            output_filename_template="{target}_national_gt_map.tif",
            backfill_from_reference=args.backfill_from_reference,
            **common_kwargs,
        )
    else:
        generate_national_mosaics(
            pred_root=args.pred_root or config.save_dir,
            pattern=args.pattern,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
