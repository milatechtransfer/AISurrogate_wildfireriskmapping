"""
Standalone plotting utility for an already-generated national mosaic (or diff) ``.tif``, e.g.
``bp_national_predicted_map.tif`` from ``generate_full_hexel_map.py`` or
``bp_national_diff_map.tif`` from ``generate_full_hexel_diff_map.py``.

Unlike the inline plotting built into those two scripts (which runs as part of generating the
mosaic/diff), this reads bounds/CRS/nodata directly from the raster file itself, so it can be
used to (re-)plot any saved output on its own -- no config or reference raster needed.

A full Canada-wide 100m raster is ~55,000 x 46,000 px (~10GB as float32), which is easily
enough to crash a local machine/VS Code if read at full resolution just to make a PNG. Use
``--downsample`` (or ``--max-dim``) to have GDAL decode a much smaller array directly, instead
of loading the full-resolution raster into memory first.

Usage:
    python -m src.full_map.visualize_mosaic --tif experiments/full_map/bp_national_predicted_map.tif \\
        --output experiments/full_map/bp_national_predicted_map.png --scale log --title "Burn Probability" \\
        --max-dim 2000

    # Diff rasters (pred - gt) are centered at 0 with a diverging colormap instead:
    python -m src.full_map.visualize_mosaic --tif experiments/full_map/bp_national_diff_map.tif \\
        --output experiments/full_map/bp_national_diff_map.png --diff --title "Burn Probability Diff" \\
        --max-dim 2000
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm
from rasterio.enums import Resampling

from src.full_map.utils import get_scale_settings


def _read_downsampled(tif_path: str, downsample: int, max_dim: int | None) -> tuple[np.ndarray, float, "rasterio.coords.BoundingBox"]:
    """Reads band 1 of `tif_path`, letting GDAL decode directly at a reduced resolution so the
    full-resolution raster is never materialized in memory.

    `max_dim` (if set) takes priority: the read shape is chosen so the larger raster dimension
    is at most `max_dim` pixels. Otherwise `downsample` (an integer decimation factor, e.g. 10
    means read at 1/10th resolution) is used directly. Nearest-neighbor resampling is used so
    nodata pixels aren't blended into valid ones.
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        bounds = src.bounds
        height, width = src.height, src.width

        factor = max(1, int(np.ceil(max(height, width) / max_dim))) if max_dim is not None else max(1, downsample)

        out_shape = (max(1, height // factor), max(1, width // factor))
        band = src.read(1, out_shape=out_shape, resampling=Resampling.nearest)

    return band, nodata, bounds


def plot_raster(
    tif_path: str,
    output_path: str,
    title: str | None = None,
    scale: str = "linear",
    diff: bool = False,
    cmap_name: str | None = None,
    downsample: int = 1,
    max_dim: int | None = 2000,
) -> None:
    """Plots a single-band raster (national mosaic or diff map) using its own embedded bounds,
    CRS, and nodata value.

    Args:
        tif_path: Path to the raster to plot (e.g. a `{target}_national_predicted_map.tif` or
            `{target}_national_diff_map.tif`).
        output_path: Where to save the PNG.
        title: Optional plot title.
        scale: 'linear' or 'log' color scaling for prediction rasters. Ignored if `diff=True`.
        diff: If True, plots as a diff map: a red/blue diverging colormap centered at 0
            (matches `generate_full_hexel_diff_map.py`'s plotting) instead of `scale`-based
            prediction scaling.
        cmap_name: Optional explicit matplotlib colormap name, overriding the default
            ('RdBu_r' for diffs, 'viridis' for predictions).
        downsample: Integer decimation factor (e.g. 10 reads at 1/10th resolution). Ignored if
            `max_dim` is set. Use 1 to read at full resolution (memory-heavy for national
            rasters).
        max_dim: If set (default), the larger raster dimension is capped to this many pixels by
            choosing an appropriate decimation factor -- this is what actually keeps memory low
            for large national rasters, and is plenty of resolution for a full-map PNG. Set to
            `None` and use `downsample` directly for finer control, or `downsample=1` for a
            full-resolution read.
    """
    band, nodata, bounds = _read_downsampled(tif_path, downsample=downsample, max_dim=max_dim)

    masked = np.ma.masked_equal(band, nodata)
    valid = masked.compressed()
    if valid.size == 0:
        raise ValueError(f"No valid (non-nodata) pixels found in {tif_path}.")

    cmap = copy.copy(plt.get_cmap(cmap_name or ("RdBu_r" if diff else "viridis")))
    cmap.set_bad(color="black", alpha=0)

    norm: Normalize
    if diff:
        abs_max = float(np.abs(valid).max()) or 1e-6
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    else:
        pos_valid = valid[valid > 0]
        pos_min = float(pos_valid.min()) if pos_valid.size else 1e-6
        global_max = float(valid.max())
        norm = get_scale_settings(scale, pos_min, global_max)

    _, ax = plt.subplots(figsize=(16, 12), dpi=300)
    im = ax.imshow(masked, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Difference (Prediction - GT)" if diff else f"Prediction ({scale.title()})", fontsize=12)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved national mosaic or diff .tif on its own, without re-running the pipeline.")
    parser.add_argument("--tif", type=str, required=True, help="Path to the national mosaic or diff .tif to plot.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output PNG to.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--scale", type=str, choices=["log", "linear"], default="linear", help="Color scaling (ignored if --diff).")
    parser.add_argument("--diff", action="store_true", help="Plot as a diff map (diverging colormap centered at 0) instead of --scale.")
    parser.add_argument("--cmap", type=str, default=None, help="Optional matplotlib colormap name override.")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2000,
        help="Cap the larger raster dimension to this many pixels when reading (default 2000) -- keeps memory low "
        "for national rasters. Set to 0 to disable and use --downsample directly instead.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Integer decimation factor to read at (e.g. 10 = 1/10th resolution). Only used if --max-dim=0.",
    )
    args = parser.parse_args()

    plot_raster(
        tif_path=args.tif,
        output_path=args.output,
        title=args.title,
        scale=args.scale,
        diff=args.diff,
        cmap_name=args.cmap,
        downsample=args.downsample,
        max_dim=args.max_dim if args.max_dim > 0 else None,
    )


if __name__ == "__main__":
    main()
