"""PNG map pictures of prediction rasters, saved next to the GeoTIFFs (e.g. ``bp/hex01_bp.png``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Continuous layers are drawn up to this percentile so a few extreme pixels do not wash out the map.
CONTINUOUS_VMAX_PERCENTILE = 99.0
PICTURE_DPI = 130


def plot_prediction_map(
    grid: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
    num_classes: int | None = None,
) -> Path:
    """Save ``grid`` as a map picture; ``num_classes`` draws a 1..N class raster (0 = no data) with a discrete legend."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    from src.datasets.postprocessing.visualize_predictions import _add_mask_outline

    plt.switch_backend("Agg")
    data = np.asarray(grid, dtype=float)
    if num_classes is not None:
        data = np.where((data >= 1) & (data <= num_classes), data, np.nan)
    valid = np.isfinite(data)
    if not valid.any():
        raise ValueError("the raster has no valid pixels")

    height, width = data.shape
    aspect = width / max(height, 1)
    fig, ax = plt.subplots(figsize=(min(10.0, max(5.0, 6.0 * aspect + 1.5)), 6.0), constrained_layout=True)
    try:
        if num_classes is not None:
            cmap = plt.get_cmap("YlOrRd", num_classes).copy()
            norm = BoundaryNorm(np.arange(0.5, num_classes + 1.5), cmap.N)
            cmap.set_bad((1.0, 1.0, 1.0, 0.0))
            image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, interpolation="nearest")
            colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(1, num_classes + 1), shrink=0.85)
            labels = [str(value) for value in range(1, num_classes + 1)]
            labels[0], labels[-1] = "1 (lowest)", f"{num_classes} (highest)"
            colorbar.ax.set_yticklabels(labels)
        else:
            values = data[valid]
            vmin = float(values.min())
            vmax = float(np.percentile(values, CONTINUOUS_VMAX_PERCENTILE))
            if vmax <= vmin:
                vmax = float(values.max()) if values.max() > vmin else vmin + 1e-6
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad((1.0, 1.0, 1.0, 0.0))
            image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            extend = "max" if values.max() > vmax else "neither"
            colorbar = fig.colorbar(image, ax=ax, extend=extend, shrink=0.85)
        colorbar.set_label(colorbar_label)
        _add_mask_outline(ax, valid, color="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_axis_off()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=PICTURE_DPI)
    finally:
        plt.close(fig)
    return path
