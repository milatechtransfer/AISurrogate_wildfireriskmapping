import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.logger import CometLogger


def _target_suffix(target_name: str | None) -> str:
    return f"_{target_name}" if target_name else ""


def as_float_array_with_nan(grid: np.ndarray) -> np.ndarray:
    masked = np.ma.masked_invalid(np.ma.asarray(grid).astype("float32"))
    return np.asarray(masked.filled(np.nan), dtype=np.float32)


def _validated_support_mask(
    mask: np.ndarray | None, expected_shape: tuple[int, ...], *, name: str, plot_label: str, hex_id: str | int
) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != expected_shape:
        raise ValueError(f"{name} shape {mask.shape} does not match {plot_label} shape {expected_shape} for hex {hex_id}.")
    return mask


def _valid_pair_values(gt_grid: np.ndarray, pred_grid: np.ndarray, hex_id: str | int) -> tuple[np.ndarray, np.ndarray]:
    gt_arr = as_float_array_with_nan(gt_grid)
    pred_arr = as_float_array_with_nan(pred_grid)
    valid_mask = np.isfinite(gt_arr) & np.isfinite(pred_arr)
    if not np.any(valid_mask):
        raise ValueError(f"No finite overlapping target/prediction pixels available for hex {str(hex_id).zfill(2)}.")
    return gt_arr[valid_mask], pred_arr[valid_mask]


def _percentile_or_extreme(values: np.ndarray, percentile: float | None, use_abs: bool = False) -> float:
    if percentile is not None and not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}.")

    values = values[np.isfinite(values)]
    if use_abs:
        values = np.abs(values)

    if values.size == 0:
        return float("nan")

    if percentile is None:
        return float(np.nanmax(values) if use_abs else np.nanmax(values))
    return float(np.nanpercentile(values, percentile))


def _target_prediction_diff_grids(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    hex_id: str | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_arr = as_float_array_with_nan(gt_grid)
    pred_arr = as_float_array_with_nan(pred_grid)
    pred_mask = np.isfinite(pred_arr)
    gt_mask = np.isfinite(gt_arr)
    overlap_mask = gt_mask & pred_mask
    if not np.any(pred_mask):
        raise ValueError(f"No finite prediction pixels available for hex {str(hex_id).zfill(2)}.")
    if not np.any(overlap_mask):
        raise ValueError(f"No finite overlapping target/prediction pixels available for hex {str(hex_id).zfill(2)}.")

    pred_plot = np.where(pred_mask, pred_arr, np.nan)
    gt_plot = np.where(pred_mask, gt_arr, np.nan)
    diff_plot = np.where(overlap_mask, pred_arr - gt_arr, np.nan)
    return gt_plot, pred_plot, diff_plot, pred_mask, overlap_mask


def _add_mask_outline(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float = 0.9,
    max_contour_dim: int = 1200,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] < 2 or mask.shape[1] < 2 or not np.any(mask) or np.all(mask):
        return

    stride = max(1, int(np.ceil(max(mask.shape) / max_contour_dim)))
    if stride == 1:
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
        )
        return

    y = np.arange(0, mask.shape[0], stride)
    x = np.arange(0, mask.shape[1], stride)
    if y[-1] != mask.shape[0] - 1:
        y = np.append(y, mask.shape[0] - 1)
    if x[-1] != mask.shape[1] - 1:
        x = np.append(x, mask.shape[1] - 1)
    contour_mask = mask[np.ix_(y, x)]
    if contour_mask.shape[0] < 2 or contour_mask.shape[1] < 2 or not np.any(contour_mask) or np.all(contour_mask):
        return
    ax.contour(
        x,
        y,
        contour_mask.astype(float),
        levels=[0.5],
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )


def _draw_support_background(ax: plt.Axes, support_mask: np.ndarray) -> None:
    background = np.ones((*support_mask.shape, 4), dtype=np.float32)
    background[support_mask] = (0.92, 0.92, 0.92, 1.0)
    ax.imshow(background, origin="upper")


def visualize_target_grids(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    hex_id: str,
    save_dir: str,
    experiment_logger: CometLogger | None = None,
    target_label: str = "Burn Probability",
    target_name: str | None = None,
    value_percentile: float | None = None,
    diff_percentile: float | None = None,
    filename_suffix: str = "",
    actual_support_mask: np.ndarray | None = None,
    buffer_support_mask: np.ndarray | None = None,
    prediction_support_label: str = "finite targets",
    show_prediction_support_outline: bool = True,
    show_suptitle: bool = False,
    show_caption: bool = False,
    diff_title: str = "Difference",
    gt_title: str = "Ground Truth (finite targets)",
):
    """
    Visualizes Ground Truth, Prediction, and Difference (GT - Prediction), side-by-side.
    """

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    suffix = _target_suffix(target_name)
    out_path = os.path.join(out_dir, f"hexel_{hex_id}{suffix}_predicted{filename_suffix}.png")

    gt_grid, pred_grid, diff_grid, pred_mask, overlap_mask = _target_prediction_diff_grids(
        gt_grid=gt_grid,
        pred_grid=pred_grid,
        hex_id=hex_id,
    )
    actual_support_mask = _validated_support_mask(
        actual_support_mask, pred_mask.shape, name="actual_support_mask", plot_label="plot", hex_id=hex_id
    )
    buffer_support_mask = _validated_support_mask(
        buffer_support_mask, pred_mask.shape, name="buffer_support_mask", plot_label="plot", hex_id=hex_id
    )
    # Shared scale for GT and Prediction
    paired_values = np.concatenate([gt_grid[np.isfinite(gt_grid)], pred_grid[np.isfinite(pred_grid)]])
    if paired_values.size == 0:
        raise ValueError(f"No finite values available for value scale for hex {hex_id}.")
    shared_vmin = float(np.nanmin(paired_values))
    shared_vmax = _percentile_or_extreme(paired_values, value_percentile)
    if shared_vmax <= shared_vmin:
        shared_vmax = shared_vmin + 1e-6

    # Symmetric scale for difference around 0
    diff_abs_max = _percentile_or_extreme(diff_grid[overlap_mask], diff_percentile, use_abs=True)
    if not np.isfinite(diff_abs_max) or diff_abs_max == 0.0:
        diff_abs_max = 1e-6
    diff_norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0.0, vmax=diff_abs_max)
    value_cmap = plt.get_cmap("viridis").copy()
    value_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    diff_cmap = plt.get_cmap("RdBu_r").copy()
    diff_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    if show_suptitle:
        fig.suptitle(f"{target_label} Prediction — Hex {hex_id}", fontsize=16)

    # --- Prediction ---
    _draw_support_background(axes[0], pred_mask)
    im2 = axes[0].imshow(
        pred_grid,
        cmap=value_cmap,
        origin="upper",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[0].set_title(f"Prediction ({prediction_support_label})")
    if show_prediction_support_outline:
        _add_mask_outline(axes[0], pred_mask, color="black", linewidth=0.5)
    if buffer_support_mask is not None:
        _add_mask_outline(axes[0], buffer_support_mask, color="black", linewidth=0.8)
    if actual_support_mask is not None:
        _add_mask_outline(axes[0], actual_support_mask, color="red", linewidth=0.8)
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    # --- Ground Truth ---
    _draw_support_background(axes[1], pred_mask)
    axes[1].imshow(
        gt_grid,
        cmap=value_cmap,
        origin="upper",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[1].set_title(gt_title)
    if show_prediction_support_outline:
        _add_mask_outline(axes[1], pred_mask, color="black", linewidth=0.5)
    if buffer_support_mask is not None:
        _add_mask_outline(axes[1], buffer_support_mask, color="black", linewidth=0.8)
    if actual_support_mask is not None:
        _add_mask_outline(axes[1], actual_support_mask, color="red", linewidth=0.8)
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")

    # --- Difference ---
    _draw_support_background(axes[2], pred_mask)
    im3 = axes[2].imshow(
        diff_grid,
        cmap=diff_cmap,
        origin="upper",
        norm=diff_norm,
    )
    axes[2].set_title(diff_title)
    if show_prediction_support_outline:
        _add_mask_outline(axes[2], pred_mask, color="black", linewidth=0.5)
    if buffer_support_mask is not None:
        _add_mask_outline(axes[2], buffer_support_mask, color="black", linewidth=0.8)
    if actual_support_mask is not None:
        _add_mask_outline(axes[2], actual_support_mask, color="red", linewidth=0.8)
    axes[2].set_xlabel("Easting (m)")
    axes[2].set_ylabel("Northing (m)")

    # Shared colorbar for first two plots only
    cbar_shared = fig.colorbar(
        im2,
        ax=axes[:2],
        shrink=0.85,
        pad=0.02,
        extend="max" if value_percentile is not None and value_percentile < 100.0 else "neither",
    )
    cbar_shared.set_label(target_label)

    # Separate colorbar for difference plot only
    cbar_diff = fig.colorbar(
        im3,
        ax=axes[2],
        shrink=0.85,
        pad=0.02,
        extend="both" if diff_percentile is not None and diff_percentile < 100.0 else "neither",
    )
    cbar_diff.set_label("Difference")
    if show_caption:
        caption_parts = []
        if buffer_support_mask is not None:
            caption_parts.append("black outline = buffer boundary")
        elif show_prediction_support_outline:
            caption_parts.append(f"Black outline = prediction {prediction_support_label}")
        if actual_support_mask is not None:
            caption_parts.append("red outline = actual hex boundary")
        caption_parts.append(f"light gray = prediction {prediction_support_label} without finite target/difference value")
        fig.text(0.5, -0.03, "; ".join(caption_parts) + ".", ha="center", fontsize=9)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")
    if experiment_logger:
        experiment_logger.log_image(
            image_path=out_path,
            name=f"predicted_hexel_{hex_id}{suffix}",
        )

    plt.close(fig)


def visualize_hexel_iou(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
    hex_id: str,
    save_dir: str,
    percentile: float,
    target_label: str = "Burn Probability",
    target_name: str | None = None,
    actual_support_mask: np.ndarray | None = None,
    buffer_support_mask: np.ndarray | None = None,
):
    """
    Visualize target and prediction maps, their binary TopK hotspots,
    contours of those hotspots, and the overlay map.
    """
    top_pct = round((1.0 - percentile) * 100.0, 2)
    top_pct_str = f"{top_pct:g}"

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hexel_{hex_id}{_target_suffix(target_name)}_top_{top_pct_str}perc_iou.png")

    gt_grid = as_float_array_with_nan(gt_grid)
    pred_grid = as_float_array_with_nan(pred_grid)
    valid_mask = np.isfinite(gt_grid) & np.isfinite(pred_grid)
    if not np.any(valid_mask):
        raise ValueError(f"No finite overlapping target/prediction pixels available for hex {hex_id}.")
    actual_support_mask = _validated_support_mask(
        actual_support_mask, valid_mask.shape, name="actual_support_mask", plot_label="IoU plot", hex_id=hex_id
    )
    buffer_support_mask = _validated_support_mask(
        buffer_support_mask, valid_mask.shape, name="buffer_support_mask", plot_label="IoU plot", hex_id=hex_id
    )

    gt_grid = np.where(valid_mask, gt_grid, np.nan)
    pred_grid = np.where(valid_mask, pred_grid, np.nan)
    inferred_vmax = float(np.nanmax(np.concatenate([gt_grid[valid_mask], pred_grid[valid_mask]])))
    if not np.isfinite(inferred_vmax) or inferred_vmax <= 0.0:
        inferred_vmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
    fig.suptitle(f"Top {top_pct_str}% {target_label} Hotspots - Hex {hex_id}", fontsize=20)

    # get topK contours for visualization
    _ = axes[0, 0].imshow(pred_grid, cmap="viridis", origin="upper", vmin=0, vmax=inferred_vmax)
    axes[0, 0].contour(np.nan_to_num(pred_bin), levels=[0.5], colors="white", linewidths=0.4, alpha=0.7)
    axes[0, 0].set_title(f"Prediction with Top {top_pct_str}% Contours")

    im2 = axes[0, 1].imshow(gt_grid, cmap="viridis", origin="upper", vmin=0, vmax=inferred_vmax)
    axes[0, 1].contour(np.nan_to_num(gt_bin), levels=[0.5], colors="white", linewidths=0.4, alpha=0.7)
    axes[0, 1].set_title(f"Ground Truth with Top {top_pct_str}% Contours")

    fig.colorbar(im2, ax=axes[0, 1], label=target_label, shrink=0.8)

    # overlap visuals
    h, w = gt_grid.shape
    rgb_overlap = np.ones((h, w, 3)) * 0.95

    nan_mask = ~valid_mask
    p_bool = pred_bin.astype(bool) & valid_mask
    t_bool = gt_bin.astype(bool) & valid_mask

    rgb_overlap[p_bool & ~t_bool] = [1.0, 0.0, 0.0]
    rgb_overlap[~p_bool & t_bool] = [0.0, 0.0, 1.0]
    rgb_overlap[p_bool & t_bool] = [1.0, 0.0, 1.0]
    rgb_overlap[nan_mask] = [1.0, 1.0, 1.0]

    axes[1, 0].imshow(rgb_overlap, origin="upper")
    axes[1, 0].set_title(f"Top {top_pct_str}% IoU Overlap Composite")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        if buffer_support_mask is not None:
            _add_mask_outline(ax, buffer_support_mask, color="black", linewidth=0.8)
        if actual_support_mask is not None:
            _add_mask_outline(ax, actual_support_mask, color="red", linewidth=0.8)

    legend_elements: list[Artist] = [
        Patch(facecolor="magenta", edgecolor="black", label="Intersection"),
        Patch(facecolor="red", edgecolor="black", label="Prediction Only"),
        Patch(facecolor="blue", edgecolor="black", label="Ground Truth Only"),
    ]
    if buffer_support_mask is not None:
        legend_elements.append(Line2D([0], [0], color="black", linewidth=1.0, label="Buffer boundary"))
    if actual_support_mask is not None:
        legend_elements.append(Line2D([0], [0], color="red", linewidth=1.0, label="Actual boundary"))
    axes[1, 0].legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=10)
    axes[1, 1].axis("off")

    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hexbin_distribution(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    hex_id: str | int,
    save_dir: str,
    experiment_logger: CometLogger | None = None,
    target_label: str = "Burn Probability",
    probability_scale: bool = True,
    target_name: str | None = None,
) -> None:
    """
    Generates and saves a 2D hex binning histogram comparing predictions vs. target values.
    """
    hex_id_str = str(hex_id).zfill(2)

    gt_vals, pred_vals = _valid_pair_values(gt_grid=gt_grid, pred_grid=pred_grid, hex_id=hex_id)

    max_limit = get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=probability_scale)

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(gt_vals, pred_vals, gridsize=100, cmap="inferno_r", bins="log", mincnt=1)

    ax.plot([0, max_limit], [0, max_limit], color="red", linestyle="--", linewidth=2, label="Perfect Alignment")

    ax.set_title(f"{target_label} Distribution: Preds vs Targets (GT) - Hex {hex_id_str}")
    ax.set_xlabel(f"Ground Truth {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    fig.colorbar(hb, ax=ax, label="Log(Count of Pixels)")
    ax.legend()

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    suffix = _target_suffix(target_name)
    out_hexbin_path = os.path.join(out_dir, f"hexbin_hex_{hex_id_str}{suffix}.png")

    plt.savefig(out_hexbin_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if experiment_logger is not None:
        experiment_logger.log_image(out_hexbin_path, name=f"hexbin_hex_{hex_id_str}{suffix}")


def plot_histogram_distribution(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    hex_id: str | int,
    save_dir: str,
    experiment_logger: CometLogger | None = None,
    num_bins: int = 100,
    target_label: str = "Burn Probability",
    probability_scale: bool = True,
    target_name: str | None = None,
) -> None:
    """
    Generates and saves an overlaid 1D histogram comparing the global distributions
    of predictions and targets on a log. scale.
    """
    hex_id_str = str(hex_id).zfill(2)

    gt_vals, pred_vals = _valid_pair_values(gt_grid=gt_grid, pred_grid=pred_grid, hex_id=hex_id)

    max_val = get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=probability_scale)

    shared_bins = np.linspace(0.0, max_val, num=num_bins)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.hist(gt_vals, bins=shared_bins.tolist(), color="blue", alpha=0.5, log=True, label="Ground Truth")
    ax.hist(pred_vals, bins=shared_bins.tolist(), color="orange", alpha=0.5, log=True, label="Prediction")

    ax.set_title(f"{target_label} Distribution (Log Scale) - Hex {hex_id_str}", fontsize=14)
    ax.set_xlabel(target_label, fontsize=12)
    ax.set_ylabel("Pixel Count (Log Scale)", fontsize=12)

    ax.legend(fontsize=12)

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    suffix = _target_suffix(target_name)
    out_hist_path = os.path.join(out_dir, f"hist_hex_{hex_id_str}{suffix}.png")

    plt.savefig(out_hist_path, bbox_inches="tight")
    plt.close(fig)

    if experiment_logger is not None:
        experiment_logger.log_image(out_hist_path, name=f"hist_dist_hex_{hex_id_str}{suffix}")


def get_distribution_axis_limit(gt_vals: np.ndarray, pred_vals: np.ndarray, probability_scale: bool) -> float:
    fallback = 0.15 if probability_scale else 1.0
    if len(gt_vals) == 0 or len(pred_vals) == 0:
        return fallback

    gt_vals = as_float_array_with_nan(gt_vals)
    pred_vals = as_float_array_with_nan(pred_vals)
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    pred_vals = pred_vals[np.isfinite(pred_vals)]
    if len(gt_vals) == 0 or len(pred_vals) == 0:
        return fallback

    actual_max = float(max(np.max(gt_vals), np.max(pred_vals)))
    if not np.isfinite(actual_max) or actual_max <= 0.0:
        return fallback

    axis_limit = actual_max * 1.05
    if probability_scale:
        axis_limit = min(1.0, axis_limit)
    return axis_limit
