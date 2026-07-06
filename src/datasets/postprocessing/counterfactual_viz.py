"""Shared IO and plotting helpers for counterfactual map scripts.

Prediction-grid raster IO (reading materialized hexel predictions, aligning to a
reference grid) and small plotting/colour utilities reused across the
counterfactual figure scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import LineCollection


def prediction_dirs_from_index(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Read scenario/endpoint prediction directories from the materialization index."""

    index_path = experiment_dir / "scenario_prediction_index.csv"
    index = pd.read_csv(index_path)
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def prediction_raster_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "predicted_hexels" / f"hexel_{int(hex_id):02d}_predicted.tif"


def read_prediction(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def read_prediction_extent(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        bounds = src.bounds
    return bounds.left, bounds.right, bounds.bottom, bounds.top


def prediction_reference_profile(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    *,
    baseline_endpoint: str = "bp",
) -> dict:
    """Reference raster profile defining the prediction grid for a hexel."""

    baseline_dir = prediction_dirs.get(("baseline", baseline_endpoint))
    if baseline_dir is None:
        raise KeyError(f"Missing baseline {baseline_endpoint.upper()} prediction directory; cannot define reference grid.")
    with rasterio.open(prediction_raster_path(baseline_dir, hex_id)) as src:
        return src.profile.copy()


def finite_values(data: np.ma.MaskedArray | np.ndarray) -> np.ndarray:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values[np.isfinite(values)]


def values_and_valid(data: np.ma.MaskedArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(np.ma.asarray(data).filled(np.nan), dtype=np.float64)
    return values, np.isfinite(values)


def symmetric_percentile_limit(deltas: list[np.ma.MaskedArray], percentile: float = 99.5) -> float:
    """Symmetric color limit from pooled absolute delta values."""

    values = [np.abs(finite_values(delta)) for delta in deltas]
    values = [value for value in values if value.size > 0]
    if not values:
        return 1.0
    limit = float(np.percentile(np.concatenate(values), percentile))
    return max(limit, 1e-9)


def downsample_for_display(data: np.ma.MaskedArray | np.ndarray, factor: int) -> np.ma.MaskedArray:
    """Stride-downsample a raster for plotting only."""

    if factor <= 1:
        return np.ma.asarray(data)
    return np.ma.asarray(data)[::factor, ::factor]


def restrict_to_support(data: np.ma.MaskedArray | np.ndarray, support_mask: np.ndarray) -> np.ma.MaskedArray:
    """Mask an array outside a boolean analysis support mask."""

    arr = np.ma.asarray(data)
    if arr.shape != support_mask.shape:
        raise ValueError(f"Support mask shape {support_mask.shape} does not match data shape {arr.shape}.")
    return np.ma.masked_where(~support_mask | np.ma.getmaskarray(arr), arr)


def zone_boundary_segments(
    zone_labels: np.ma.MaskedArray | np.ndarray,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Line segments tracing borders between differing (valid) firezone labels.

    Returns an ``(n, 2, 2)`` array of ``[(x0, y0), (x1, y1)]`` segments in the data
    coordinates of an ``imshow(origin="upper")`` axis: pass the same ``extent`` used
    for the base image, or ``None`` for pixel-index coordinates.  Borders touching
    masked/no-data pixels are skipped, so only firezone-firezone boundaries are drawn.
    """

    labels = np.ma.filled(np.ma.asarray(zone_labels), -1).astype(np.int64)
    valid = ~np.ma.getmaskarray(np.ma.asarray(zone_labels))
    height, width = labels.shape
    left, right, bottom, top = extent if extent is not None else (-0.5, width - 0.5, height - 0.5, -0.5)

    def to_x(edge: np.ndarray) -> np.ndarray:
        return left + edge / width * (right - left)

    def to_y(edge: np.ndarray) -> np.ndarray:
        return top + edge / height * (bottom - top)

    parts: list[np.ndarray] = []
    vertical = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    rows, cols = np.nonzero(vertical)
    if rows.size:
        x = to_x(cols + 1.0)
        start = np.column_stack([x, to_y(rows.astype(np.float64))])
        end = np.column_stack([x, to_y(rows + 1.0)])
        parts.append(np.stack([start, end], axis=1))
    horizontal = valid[:-1, :] & valid[1:, :] & (labels[:-1, :] != labels[1:, :])
    rows, cols = np.nonzero(horizontal)
    if rows.size:
        y = to_y(rows + 1.0)
        start = np.column_stack([to_x(cols.astype(np.float64)), y])
        end = np.column_stack([to_x(cols + 1.0), y])
        parts.append(np.stack([start, end], axis=1))
    if not parts:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def overlay_zone_boundaries(
    ax: plt.Axes,
    zone_labels: np.ma.MaskedArray | np.ndarray | None,
    *,
    extent: tuple[float, float, float, float] | None = None,
    color: str = "#111111",
    linewidth: float = 0.8,
    alpha: float = 0.9,
) -> None:
    """Overlay firezone boundaries on a hex-map axis (no-op when ``zone_labels`` is None)."""

    if zone_labels is None:
        return
    segments = zone_boundary_segments(zone_labels, extent=extent)
    if segments.shape[0] == 0:
        return
    ax.add_collection(LineCollection(list(segments), colors=color, linewidths=linewidth, alpha=alpha, zorder=5))


def cumulative_abs_share(delta: np.ma.MaskedArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concentration curve of an output delta.

    Returns ``(pixel_fraction, cumulative_abs_share)`` for finite pixels ranked by
    descending ``|delta|``: at ``pixel_fraction == 0.1`` the share is the fraction of
    total absolute change contributed by the most-changed 10% of pixels.  A curve near
    the diagonal means changes are spread out; a curve bowed to the top-left means a few
    pixels dominate.
    """

    values = np.abs(finite_values(delta))
    if values.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    values = np.sort(values)[::-1]
    pixel_fraction = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    total = float(values.sum())
    cumulative = np.cumsum(values) / total if total > 0.0 else np.zeros_like(pixel_fraction)
    return pixel_fraction, cumulative


def abs_share_at(pixel_fraction: np.ndarray, cumulative_share: np.ndarray, top_fraction: float) -> float:
    """Cumulative absolute-change share contributed by the top ``top_fraction`` of pixels."""

    if pixel_fraction.size == 0:
        return 0.0
    index = int(np.searchsorted(pixel_fraction, top_fraction, side="left"))
    index = min(index, cumulative_share.size - 1)
    return float(cumulative_share[index])


def plot_delta_histogram(
    delta: np.ma.MaskedArray | np.ndarray,
    *,
    out_path: Path,
    xlabel: str,
    title: str,
    color: str = "#b2182b",
    percentile: float = 99.5,
    bins: int = 201,
) -> None:
    """Reusable per-pixel output-delta diagnostic for any counterfactual intervention.

    Left panel: log-count histogram of the delta with mean and zero markers.  Right
    panel: cumulative absolute-change concentration curve (see ``cumulative_abs_share``)
    showing whether the response is across-the-board or driven by a few pixels.
    """

    values = finite_values(delta)
    fig, (ax_hist, ax_conc) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    if values.size == 0:
        for ax in (ax_hist, ax_conc):
            ax.text(0.5, 0.5, "no finite delta", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(title)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    limit = max(float(np.percentile(np.abs(values), percentile)), 1e-9)
    mean = float(np.mean(values))
    frac_positive = float(np.mean(values > 0))
    ax_hist.hist(values, bins=np.linspace(-limit, limit, bins).tolist(), histtype="step", linewidth=1.8, color=color)
    ax_hist.axvline(mean, color=color, linestyle="--", linewidth=1.2, label=f"mean \u0394={mean:+.3g}")
    ax_hist.axvline(0.0, color="0.4", linewidth=1.0)
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Pixel count")
    ax_hist.set_yscale("log")
    ax_hist.set_title(f"{100.0 * frac_positive:g}% of pixels increase")
    ax_hist.legend(loc="upper left")

    pixel_fraction, cumulative = cumulative_abs_share(delta)
    ax_conc.plot([0.0, 1.0], [0.0, 1.0], color="0.6", linestyle=":", linewidth=1.0, label="uniform")
    ax_conc.plot(pixel_fraction, cumulative, color=color, linewidth=1.8)
    annotations = "  ".join(f"top {int(f * 100)}%: {abs_share_at(pixel_fraction, cumulative, f):.0%}" for f in (0.01, 0.05, 0.10))
    ax_conc.set_xlim(0.0, 1.0)
    ax_conc.set_ylim(0.0, 1.0)
    ax_conc.set_xlabel("Top fraction of pixels (ranked by |\u0394|)")
    ax_conc.set_ylabel("Cumulative share of total |\u0394|")
    ax_conc.set_title(annotations)
    ax_conc.legend(loc="lower right")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
