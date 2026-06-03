"""Visualize a national grid GeoTIFF with its conflict mask overlaid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Maximum pixels along the longer axis used for display.  Full-res national
# grids are enormous; downsampling on load is orders of magnitude faster.
_MAX_DISPLAY_PX = 3000


def _decimated_read(src: rasterio.DatasetReader, max_px: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read a rasterio dataset downsampled to at most *max_px* on its longer axis."""
    scale = min(1.0, max_px / max(src.width, src.height))
    out_h = max(1, int(src.height * scale))
    out_w = max(1, int(src.width * scale))
    data = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.average)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return data, extent


def visualize_national_grid_with_conflict(
    grid_tif: str | Path,
    conflict_mask_tif: str | Path,
    output_path: str | Path | None = None,
    grid_cmap: str = "viridis",
    title: str | None = None,
    max_display_px: int = _MAX_DISPLAY_PX,
) -> None:
    """Plot a national grid raster with conflicting pixels highlighted on top.

    Pixels where overlapping hexels had different values (conflict mask = 1)
    are drawn in red over the grid.  Areas with no data are transparent.

    Args:
        grid_tif: Path to the national grid GeoTIFF (e.g. national_elevation.tif).
        conflict_mask_tif: Path to the corresponding conflict mask GeoTIFF
            (e.g. national_elevation_conflict_mask.tif).
        output_path: If given, save the figure to this path instead of
            displaying it interactively.
        grid_cmap: Matplotlib colormap name used for the grid values.
        title: Figure title.  Derived from the grid filename when *None*.
        max_display_px: Downsample so the longer axis is at most this many
            pixels.  Reduces memory and render time dramatically for large
            national grids.  Set to 0 to read at full resolution.
    """
    with rasterio.open(grid_tif) as src:
        nodata = src.nodata
        n_full_pixels = src.width * src.height  # for conflict % denominator
        extent: tuple[float, float, float, float]
        if max_display_px > 0:
            grid_data, extent = _decimated_read(src, max_display_px)
        else:
            grid_data = src.read(1)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        grid_data = grid_data.astype(np.float32)

    with rasterio.open(conflict_mask_tif) as src2:
        # Count conflicts at full resolution for accurate stats.
        n_conflict = int(src2.read(1).sum())
        if max_display_px > 0:
            conflict_data, _ = _decimated_read(src2, max_display_px)
        else:
            conflict_data = src2.read(1)

    # Mask nodata (NaN or explicit nodata value).
    grid_masked = np.where(grid_data == nodata, np.nan, grid_data) if nodata is not None and not np.isnan(nodata) else grid_data.copy()

    # Conflict overlay: threshold at 0.5 so averaged-down pixels still show.
    conflict_overlay = np.where(conflict_data >= 0.5, 1.0, np.nan)

    n_valid_full = n_full_pixels - int((grid_data == nodata).sum()) if nodata is not None else n_full_pixels
    conflict_pct = 100.0 * n_conflict / n_valid_full if n_valid_full > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- base grid ---
    im = ax.imshow(
        grid_masked,
        cmap=grid_cmap,
        extent=extent,
        origin="upper",
        interpolation="none",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Grid value")

    # --- conflict overlay (red, semi-transparent) ---
    conflict_cmap = ListedColormap(["red"])
    ax.imshow(
        conflict_overlay,
        cmap=conflict_cmap,
        extent=extent,
        origin="upper",
        interpolation="none",
        alpha=0.6,
        vmin=0,
        vmax=1,
    )

    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label=f"Conflict pixels ({n_conflict:,} / {conflict_pct:.2f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig_title = title or Path(grid_tif).stem.replace("_", " ").title()
    ax.set_title(fig_title, fontsize=13)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a national grid GeoTIFF with its conflict mask overlaid.")
    parser.add_argument("--grid_tif", help="Path to the national grid GeoTIFF.")
    parser.add_argument("--conflict_mask_tif", help="Path to the conflict mask GeoTIFF.")
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save figure to this path (PNG/PDF/SVG) instead of displaying it.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        metavar="CMAP",
        help="Matplotlib colormap for the grid values (default: viridis).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title (default: derived from the grid filename).",
    )
    parser.add_argument(
        "--max-px",
        type=int,
        default=_MAX_DISPLAY_PX,
        metavar="N",
        help=(
            f"Downsample so the longer axis is at most N pixels before plotting "
            f"(default: {_MAX_DISPLAY_PX}).  Use 0 for full resolution (slow)."
        ),
    )
    args = parser.parse_args()

    visualize_national_grid_with_conflict(
        grid_tif=args.grid_tif,
        conflict_mask_tif=args.conflict_mask_tif,
        output_path=args.output,
        grid_cmap=args.cmap,
        title=args.title,
        max_display_px=args.max_px,
    )


if __name__ == "__main__":
    main()
