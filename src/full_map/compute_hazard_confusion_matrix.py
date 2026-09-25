"""
Computes confusion-matrix-based stats (exact-match accuracy, within-1/within-2 accuracy, mean
absolute class error, macro IoU/F1, and the full confusion matrix) between two already-generated
national hazard class rasters -- typically ``hazard_national_predicted_map.tif`` and
``hazard_national_ground_truth_map.tif`` from ``generate_national_hazard_map.py``.

Both rasters are read in row-block windows and the confusion matrix (tiny: ``num_classes**2``
ints) is accumulated incrementally, so this never materializes a full national-sized array of
either raster's pixels -- unlike the classification rasters themselves (~5GB each as int32 at
Canada-wide 100m resolution), peak memory here stays close to a couple of row-block windows.

Usage:
    python -m src.full_map.compute_hazard_confusion_matrix \
        --pred-tif experiments/full_map/hazard_national_predicted_map.tif \
        --gt-tif experiments/full_map/hazard_national_ground_truth_map.tif \
        --output-dir experiments/full_map --save-plot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

from src.datasets.postprocessing.hazard_metrics import build_confusion_matrix, flatten_hazard_class_metrics, metrics_from_confusion

SUMMARY_JSON_FILENAME = "hazard_confusion_matrix_summary.json"
PLOT_FILENAME = "hazard_confusion_matrix.png"


def _row_block_windows(height: int, width: int, block_rows: int) -> list[Window]:
    return [Window(0, row_off, width, min(block_rows, height - row_off)) for row_off in range(0, height, block_rows)]


def accumulate_confusion_matrix(
    pred_tif: str,
    gt_tif: str,
    *,
    invalid_class: int = 0,
    num_classes: int = 13,
    block_rows: int = 4096,
) -> np.ndarray:
    """Accumulates the (num_classes x num_classes) confusion matrix over ``pred_tif``/``gt_tif``
    by reading both rasters in matching row-block windows, so only a small (block_rows x width)
    slice of each is ever in memory at once.
    """
    with rasterio.open(pred_tif) as pred_src, rasterio.open(gt_tif) as gt_src:
        if pred_src.shape != gt_src.shape:
            raise ValueError(f"pred/gt raster shapes must match, got {pred_src.shape} and {gt_src.shape}")
        if pred_src.bounds != gt_src.bounds:
            raise ValueError(f"pred/gt raster bounds must match, got {pred_src.bounds} and {gt_src.bounds}")
        if pred_src.crs != gt_src.crs or pred_src.transform != gt_src.transform:
            raise ValueError(
                f"pred/gt raster CRS/transform must match, got ({pred_src.crs}, {pred_src.transform}) and "
                f"({gt_src.crs}, {gt_src.transform})"
            )

        height, width = pred_src.shape
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for window in _row_block_windows(height, width, block_rows):
            pred_block = pred_src.read(1, window=window)
            gt_block = gt_src.read(1, window=window)
            confusion += build_confusion_matrix(pred_block, gt_block, invalid_class=invalid_class, num_classes=num_classes)
    return confusion


def _plot_confusion_matrix(confusion: np.ndarray, output_path: str, title: str | None) -> None:
    num_classes = confusion.shape[0]
    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(row_sums > 0, confusion / row_sums, 0.0)

    _, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([str(c) for c in range(1, num_classes + 1)])
    ax.set_yticklabels([str(c) for c in range(1, num_classes + 1)])
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="white", fontsize=6)
    ax.set_title(title or "Hazard class confusion matrix (row-normalized)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"Saved confusion matrix plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute confusion matrix and ordinal accuracy stats between two hazard class rasters.")
    parser.add_argument("--pred-tif", type=str, required=True, help="Path to the predicted hazard class raster.")
    parser.add_argument("--gt-tif", type=str, required=True, help="Path to the ground-truth hazard class raster.")
    parser.add_argument("--output-dir", type=str, default="experiments/full_map", help="Directory to write the summary JSON/plot into.")
    parser.add_argument("--num-classes", type=int, default=13, help="Number of hazard classes (must match how the rasters were binned).")
    parser.add_argument("--invalid-class", type=int, default=0, help="Class value marking invalid/nodata pixels (ignored in stats).")
    parser.add_argument(
        "--block-rows", type=int, default=4096, help="Number of raster rows read per window while accumulating the confusion matrix."
    )
    parser.add_argument("--save-plot", action="store_true", help="Also save a row-normalized confusion matrix heatmap PNG.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = parser.parse_args()

    confusion = accumulate_confusion_matrix(
        args.pred_tif,
        args.gt_tif,
        invalid_class=args.invalid_class,
        num_classes=args.num_classes,
        block_rows=args.block_rows,
    )
    metrics = metrics_from_confusion(confusion)
    flat_metrics = flatten_hazard_class_metrics(metrics)

    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    summary = {"metrics": flat_metrics, "confusion_matrix": confusion.tolist()}
    with open(output_folder / SUMMARY_JSON_FILENAME, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved confusion matrix summary to {output_folder / SUMMARY_JSON_FILENAME}")

    if args.save_plot:
        _plot_confusion_matrix(confusion, str(output_folder / PLOT_FILENAME), args.title)

    print("\n===== Hazard Confusion Matrix Summary =====")
    print(f"Total valid pixels: {int(confusion.sum())}")
    for key in ("exact_accuracy", "within_1_accuracy", "within_2_accuracy", "mean_absolute_class_error", "macro_iou", "macro_f1"):
        print(f"  {key}: {flat_metrics[key]}")


if __name__ == "__main__":
    main()
