"""Shared IO and plotting helpers for counterfactual map scripts.

Prediction-grid raster IO (reading materialized hexel predictions, aligning to a
reference grid) and firezone boundary overlay utilities reused across the
counterfactual figure scripts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

DEFAULT_ZONE_OVERLAY_COLOR = "#111111"
DEFAULT_ZONE_OVERLAY_LINEWIDTH = 1.4
DEFAULT_ZONE_OVERLAY_ALPHA = 0.9


@dataclass(frozen=True)
class EndpointResponse:
    baseline: np.ma.MaskedArray
    scenario: np.ma.MaskedArray
    delta: np.ma.MaskedArray
    baseline_support: np.ndarray
    scenario_support: np.ndarray
    response_support: np.ndarray


def build_endpoint_response(
    baseline: np.ma.MaskedArray | np.ndarray,
    scenario: np.ma.MaskedArray | np.ndarray,
    *,
    baseline_support: np.ndarray | None = None,
    scenario_support: np.ndarray | None = None,
) -> EndpointResponse:
    """Build a symmetric counterfactual response on baseline/scenario support.

    Values outside each scenario's support contribute zero to the response. For fuel
    edits this makes newly burnable pixels contribute ``+scenario`` and newly
    non-burnable pixels contribute ``-baseline``.
    """

    baseline_values, baseline_valid = values_and_valid(baseline)
    scenario_values, scenario_valid = values_and_valid(scenario)
    if baseline_values.shape != scenario_values.shape:
        raise ValueError(f"Baseline/scenario shapes differ: {baseline_values.shape} vs {scenario_values.shape}.")
    paired = baseline_valid & scenario_valid

    if baseline_support is None and scenario_support is None:
        supported_baseline = paired
        supported_scenario = paired
    elif baseline_support is None or scenario_support is None:
        raise ValueError("baseline_support and scenario_support must be provided together.")
    else:
        if baseline_support.shape != paired.shape or scenario_support.shape != paired.shape:
            raise ValueError("Prediction and support masks must have the same shape.")
        supported_baseline = paired & np.asarray(baseline_support, dtype=bool)
        supported_scenario = paired & np.asarray(scenario_support, dtype=bool)

    response_support = supported_baseline | supported_scenario
    baseline_effective = np.where(supported_baseline, baseline_values, 0.0)
    scenario_effective = np.where(supported_scenario, scenario_values, 0.0)
    return EndpointResponse(
        baseline=np.ma.masked_where(~supported_baseline, baseline_values),
        scenario=np.ma.masked_where(~supported_scenario, scenario_values),
        delta=np.ma.masked_where(~response_support, scenario_effective - baseline_effective),
        baseline_support=supported_baseline,
        scenario_support=supported_scenario,
        response_support=response_support,
    )


def prediction_dirs_from_index(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Read scenario/endpoint prediction directories from the materialization index."""

    index_path = experiment_dir / "scenario_prediction_index.csv"
    index = pd.read_csv(index_path)
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def prediction_raster_path(prediction_dir: Path, hex_id: str, target_name: str | None = None) -> Path:
    """Path to a stitched prediction raster inside a scenario/endpoint prediction directory.

    Single-output models write one unsuffixed raster per hexel
    (``hexel_XX_predicted.tif``). A multi-output model (e.g. one checkpoint jointly
    predicting ``bp``/``fi``/``ros``) writes one target-suffixed raster per target
    into the same directory (``hexel_XX_<target_name>_predicted.tif``). When
    ``target_name`` is given, the suffixed path is preferred and falls back to the
    legacy unsuffixed path if it doesn't exist, so both setups resolve correctly
    without knowing in advance which kind of model produced a given endpoint's
    predictions.
    """
    predicted_dir = prediction_dir / "predicted_hexels"
    if target_name:
        suffixed_path = predicted_dir / f"hexel_{int(hex_id):02d}_{target_name}_predicted.tif"
        if suffixed_path.exists():
            return suffixed_path
    return predicted_dir / f"hexel_{int(hex_id):02d}_predicted.tif"


def find_local_prediction_dir(experiment_dir: Path, scenario: str, hex_id: str, endpoint: str) -> Path:
    """Locate a scenario/endpoint prediction directory by scanning the local `predictions/` tree.

    Materialized runs store predictions under `predictions/<scenario>/<subdir>/predicted_hexels/`,
    where `<subdir>` is usually the endpoint name but may instead be a shared subdir when several
    endpoints are deduplicated onto the same multi-output checkpoint run (e.g. `bp`, `fi`, and
    `ros` all resolving to a `predictions/<scenario>/bp/` directory). Rather than trusting
    `scenario_prediction_index.csv` (which stores the original, possibly remote/cluster, absolute
    paths), this scans the locally materialized `predictions/<scenario>/` subdirectories for the
    one that actually contains this endpoint's stitched raster.
    """
    scenario_dir = experiment_dir / "predictions" / scenario
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"No predictions directory for scenario={scenario!r} under {experiment_dir}.")
    for candidate in sorted(scenario_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if prediction_raster_path(candidate, hex_id, target_name=endpoint).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a {endpoint!r} prediction raster for scenario={scenario!r}, hex_id={hex_id!r} under {scenario_dir}."
    )


def read_prediction(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


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
    with rasterio.open(prediction_raster_path(baseline_dir, hex_id, target_name=baseline_endpoint)) as src:
        return src.profile.copy()


def load_baseline_scenario_pair(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    *,
    endpoint: str,
    scenario: str,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Read the baseline and scenario prediction rasters for one endpoint."""

    baseline_dir = prediction_dirs.get(("baseline", endpoint))
    scenario_dir = prediction_dirs.get((scenario, endpoint))
    if baseline_dir is None:
        raise KeyError(f"Missing baseline {endpoint.upper()} prediction directory.")
    if scenario_dir is None:
        raise KeyError(f"Missing {endpoint.upper()} prediction directory for scenario={scenario!r}.")
    baseline = read_prediction(prediction_raster_path(baseline_dir, hex_id, target_name=endpoint))
    scenario_values = read_prediction(prediction_raster_path(scenario_dir, hex_id, target_name=endpoint))
    if scenario_values.shape != baseline.shape:
        raise ValueError(f"Scenario {endpoint.upper()} shape {scenario_values.shape} does not match baseline grid {baseline.shape}.")
    return baseline, scenario_values


def read_prediction_extent(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        bounds = src.bounds
    return bounds.left, bounds.right, bounds.bottom, bounds.top


def prediction_footprint(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    *,
    endpoint: str,
    scenario: str = "baseline",
) -> np.ndarray:
    """Valid raster footprint before scenario-specific map masks are applied."""

    prediction_dir = prediction_dirs.get((scenario, endpoint))
    if prediction_dir is None:
        raise KeyError(f"Missing {scenario} {endpoint.upper()} prediction directory; cannot define prediction footprint.")
    return ~np.ma.getmaskarray(read_prediction(prediction_raster_path(prediction_dir, hex_id, target_name=endpoint)))


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


def robust_norm(arrays: list[np.ndarray | np.ma.MaskedArray], *, low: float = 1.0, high: float = 99.0) -> Normalize:
    """Sequential colour scale spanning the `[low, high]` percentiles of pooled values."""

    values = [finite_values(array) for array in arrays]
    values = [value for value in values if value.size > 0]
    if not values:
        return Normalize(vmin=0.0, vmax=1.0)
    pooled = np.concatenate(values)
    vmin = float(np.percentile(pooled, low))
    vmax = float(np.percentile(pooled, high))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def delta_norm(deltas: list[np.ma.MaskedArray], percentile: float = 99.5) -> TwoSlopeNorm:
    """Zero-centered diverging colour scale spanning `symmetric_percentile_limit(deltas)`."""

    limit = symmetric_percentile_limit(deltas, percentile=percentile)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def downsample_for_display(data: np.ma.MaskedArray | np.ndarray, factor: int) -> np.ma.MaskedArray:
    """Stride-downsample a raster for plotting only."""

    arr = np.ma.asarray(data)
    if factor <= 1:
        return arr
    row_indices = np.arange(0, arr.shape[0], factor)
    col_indices = np.arange(0, arr.shape[1], factor)
    if row_indices.size > 0 and row_indices[-1] != arr.shape[0] - 1:
        row_indices = np.append(row_indices, arr.shape[0] - 1)
    if col_indices.size > 0 and col_indices[-1] != arr.shape[1] - 1:
        col_indices = np.append(col_indices, arr.shape[1] - 1)
    return arr[np.ix_(row_indices, col_indices)]


def restrict_to_support(data: np.ma.MaskedArray | np.ndarray, support_mask: np.ndarray) -> np.ma.MaskedArray:
    """Mask an array outside a boolean analysis support mask."""

    arr = np.ma.asarray(data)
    if arr.shape != support_mask.shape:
        raise ValueError(f"Support mask shape {support_mask.shape} does not match data shape {arr.shape}.")
    return np.ma.masked_where(~support_mask | np.ma.getmaskarray(arr), arr)


def add_zone_overlay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--zone_overlay", action="store_true", help="Draw firezone boundary overlays on map panels.")
    parser.add_argument(
        "--zone_overlay_color",
        default=DEFAULT_ZONE_OVERLAY_COLOR,
        help="Firezone boundary overlay color.",
    )
    parser.add_argument(
        "--zone_overlay_linewidth",
        type=float,
        default=DEFAULT_ZONE_OVERLAY_LINEWIDTH,
        help="Firezone boundary overlay line width.",
    )
    parser.add_argument(
        "--zone_overlay_alpha",
        type=float,
        default=DEFAULT_ZONE_OVERLAY_ALPHA,
        help="Firezone boundary overlay alpha.",
    )


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
    color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Overlay firezone boundaries on a hex-map axis (no-op when ``zone_labels`` is None)."""

    if zone_labels is None:
        return
    segments = zone_boundary_segments(zone_labels, extent=extent)
    if segments.shape[0] == 0:
        return
    ax.add_collection(LineCollection(list(segments), colors=color, linewidths=linewidth, alpha=alpha, zorder=5, clip_on=False))


EDIT_OVERLAY_STYLES = ("none", "contour", "hatch")
DEFAULT_EDIT_OVERLAY_COLOR = "#000000"
DEFAULT_EDIT_OVERLAY_LINEWIDTH = 0.6
DEFAULT_EDIT_OVERLAY_ALPHA = 0.9
DEFAULT_EDIT_OVERLAY_HATCH = "///"


def add_edit_overlay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--edit_overlay",
        choices=EDIT_OVERLAY_STYLES,
        default="none",
        help="Mark directly-edited fuel pixels on map panels: a thin contour around the edit, or a hatch fill.",
    )
    parser.add_argument("--edit_overlay_color", default=DEFAULT_EDIT_OVERLAY_COLOR, help="Edit-mask overlay color.")
    parser.add_argument(
        "--edit_overlay_linewidth",
        type=float,
        default=DEFAULT_EDIT_OVERLAY_LINEWIDTH,
        help="Edit-mask contour line width (style=contour only).",
    )
    parser.add_argument("--edit_overlay_alpha", type=float, default=DEFAULT_EDIT_OVERLAY_ALPHA, help="Edit-mask overlay alpha.")
    parser.add_argument(
        "--edit_overlay_hatch",
        default=DEFAULT_EDIT_OVERLAY_HATCH,
        help="Hatch pattern for the edit-mask overlay (style=hatch only).",
    )


def overlay_edit_mask(
    ax: plt.Axes,
    edit_mask: np.ndarray | None,
    *,
    style: str,
    extent: tuple[float, float, float, float] | None = None,
    color: str = DEFAULT_EDIT_OVERLAY_COLOR,
    linewidth: float = DEFAULT_EDIT_OVERLAY_LINEWIDTH,
    alpha: float = DEFAULT_EDIT_OVERLAY_ALPHA,
    hatch: str = DEFAULT_EDIT_OVERLAY_HATCH,
) -> None:
    """Mark directly-edited fuel pixels on a map axis (no-op when ``style`` is ``"none"`` or the mask is empty).

    ``style="contour"`` traces the boundary of the edited region (reusing the same pixel-grid
    boundary tracer as :func:`overlay_zone_boundaries`); ``style="hatch"`` fills it with a hatch
    pattern instead, leaving the underlying map colours visible.
    """

    if style == "none" or edit_mask is None or not np.any(edit_mask):
        return
    if style == "contour":
        overlay_zone_boundaries(ax, edit_mask.astype(np.int64), extent=extent, color=color, linewidth=linewidth, alpha=alpha)
    elif style == "hatch":
        contour_set = ax.contourf(
            edit_mask.astype(np.float64),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[hatch],
            extent=extent,
            origin="upper",
        )
        contour_set.set_edgecolor(color)
        contour_set.set_linewidth(0.0)
        contour_set.set_alpha(alpha)
    else:
        raise ValueError(f"Unknown edit overlay style {style!r}; expected one of {EDIT_OVERLAY_STYLES}.")


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


def pixel_fraction_for_share(pixel_fraction: np.ndarray, cumulative_share: np.ndarray, target_share: float) -> float:
    """Smallest top-pixel fraction whose cumulative absolute-change share reaches ``target_share``."""

    if cumulative_share.size == 0:
        return 0.0
    index = int(np.searchsorted(cumulative_share, target_share, side="left"))
    index = min(index, pixel_fraction.size - 1)
    return float(pixel_fraction[index])


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
    """Reusable per-pixel output-delta histogram for any counterfactual intervention.

    Log-count histogram of the delta with mean and zero markers.
    """

    values = finite_values(delta)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    if values.size == 0:
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
    ax.hist(values, bins=np.linspace(-limit, limit, bins).tolist(), histtype="step", linewidth=1.8, color=color)
    ax.axvline(mean, color=color, linestyle="--", linewidth=1.2, label=f"mean \u0394={mean:+.3g}")
    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.set_yscale("log")
    ax.set_title(f"{100.0 * frac_positive:g}% of pixels increase")
    ax.legend(loc="upper left")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_concentration(
    delta: np.ma.MaskedArray | np.ndarray,
    *,
    out_path: Path,
    title: str,
    color: str = "#b2182b",
) -> None:
    """Reusable |delta| concentration (cumulative-share) curve for any counterfactual intervention.

    Shows whether the response is spread across-the-board or driven by a few
    pixels: at ``pixel_fraction == 0.1`` the y-value is the fraction of total
    absolute change contributed by the most-changed 10% of pixels.
    """

    pixel_fraction, cumulative = cumulative_abs_share(delta)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.6", linestyle=":", linewidth=1.0, label="uniform")
    ax.plot(pixel_fraction, cumulative, color=color, linewidth=1.8)
    annotations = "  ".join(f"top {int(f * 100)}%: {abs_share_at(pixel_fraction, cumulative, f):.0%}" for f in (0.01, 0.05, 0.10))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Top fraction of pixels (ranked by |\u0394|)")
    ax.set_ylabel("Cumulative share of total |\u0394|")
    ax.set_title(annotations)
    ax.legend(loc="lower right")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
