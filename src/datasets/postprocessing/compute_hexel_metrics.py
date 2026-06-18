"""
Compute per-hexel evaluation metrics (CCC, Spearman, Norm-MAE, AUC-IoU-top10)
from already-predicted TIF files in an experiment directory.

Usage:
    uv run python -m src.datasets.postprocessing.compute_hexel_metrics \\
        --exp_dir experiments/fi_full_config_v3_... \\
        --raw_data_dir /path/to/canada_bp3+_2026_MILA \\
        --target fi

    # Auto-detect target from experiment directory name:
    uv run python -m src.datasets.postprocessing.compute_hexel_metrics \\
        --exp_dir experiments/fi_full_config_v3_... \\
        --raw_data_dir /path/to/canada_bp3+_2026_MILA

    # Compare multiple experiments side-by-side:
    uv run python -m src.datasets.postprocessing.compute_hexel_metrics \\
        --exp_dir experiments/fi_baseline experiments/fi_wind_vector \\
        --raw_data_dir /path/to/canada_bp3+_2026_MILA \\
        --target fi
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import torch

from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.utils import calculate_hexel_metrics_pytorch
from src.metrics import compute_auc_iou, compute_ccc, compute_normalized_mae, compute_spearman

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TEST_HEXELS = ["01", "12", "16", "39", "49"]

_GT_RELATIVE_PATHS: dict[str, str] = {
    "fi": "results/burnP3Plus_OutputFireIntensitySummaryMap/fbpSummary-FireIntensity-Average.tif",
    "ros": "results/burnP3Plus_OutputRateOfSpreadSummaryMap/fbpSummary-RateOfSpread-Average.tif",
    "bp": "results/burnP3Plus_OutputBurnProbability/burnProbability-sn2.tif",
}

METRIC_FNS: dict[str, Callable[..., Any]] = {
    "ccc": compute_ccc,
    "spearman": compute_spearman,
    "normalized_mae": compute_normalized_mae,
    "auc_iou_top10": partial(compute_auc_iou, k_values=(0.01, 0.10), steps=10),
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _infer_target(exp_dir: str) -> str | None:
    """Try to infer target (fi / ros / bp) from the experiment directory name."""
    name = Path(exp_dir).name.lower()
    for target in ("fi", "ros", "bp"):
        pattern = rf"(?:^|[_/]){target}(?:[_/]|$)"
        if re.search(pattern, name):
            return target
    return None


def _load_pred(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, profile


def _compute_metrics_for_experiment(
    exp_dir: Path,
    raw_data_dir: Path,
    target: str,
    hex_ids: list[str],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Returns {hex_id: {metric_name: value}} for all available hexels.
    Missing pred TIFs are silently skipped.
    """
    gt_rel = _GT_RELATIVE_PATHS[target]
    results: dict[str, dict[str, float]] = {}

    for hex_id in hex_ids:
        pred_path = exp_dir / "predicted_hexels" / f"hexel_{hex_id}_predicted.tif"
        gt_path = raw_data_dir / f"hex{hex_id}" / gt_rel

        if not pred_path.exists():
            print(f"  [skip] hex{hex_id}: predicted TIF not found ({pred_path})")
            continue
        if not gt_path.exists():
            print(f"  [skip] hex{hex_id}: GT TIF not found ({gt_path})")
            continue

        pred, pred_profile = _load_pred(pred_path)
        gt_ma, _ = load_spatial_raster(path=gt_path, reference_profile=pred_profile)
        gt = gt_ma.filled(np.nan).astype(np.float32)

        if pred.shape != gt.shape:
            print(f"  [skip] hex{hex_id}: shape mismatch after reproject (pred={pred.shape}, gt={gt.shape})")
            continue

        metrics = calculate_hexel_metrics_pytorch(gt, pred, device, METRIC_FNS)
        results[hex_id] = metrics

    return results


def _print_results_table(
    exp_label: str,
    hex_metrics: dict[str, dict[str, float]],
    metric_names: list[str],
) -> None:
    col_w = 12
    hex_col = 8

    header = f"{'Hex':<{hex_col}}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    sep = "-" * len(header)
    print(f"\n  {exp_label}")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    all_vals: dict[str, list[float]] = {m: [] for m in metric_names}
    for hex_id, metrics in sorted(hex_metrics.items()):
        row = f"{'hex' + hex_id:<{hex_col}}" + "".join(f"{metrics.get(m, float('nan')):>{col_w}.4f}" for m in metric_names)
        print(f"  {row}")
        for m in metric_names:
            v = metrics.get(m, float("nan"))
            if not np.isnan(v):
                all_vals[m].append(v)

    print(f"  {sep}")
    mean_row = f"{'Mean':<{hex_col}}" + "".join(
        f"{np.mean(all_vals[m]):>{col_w}.4f}" if all_vals[m] else f"{'N/A':>{col_w}}" for m in metric_names
    )
    print(f"  {mean_row}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-hexel metrics from predicted TIFs vs. ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp_dir",
        nargs="+",
        required=True,
        help="One or more experiment directories containing a predicted_hexels/ subfolder.",
    )
    parser.add_argument(
        "--raw_data_dir",
        required=True,
        help="Root raw data directory (contains hex01/, hex12/, …).",
    )
    parser.add_argument(
        "--target",
        choices=list(_GT_RELATIVE_PATHS.keys()),
        default=None,
        help="Target variable (fi / ros / bp). Auto-detected from exp_dir name if omitted.",
    )
    parser.add_argument(
        "--hex_ids",
        nargs="+",
        default=DEFAULT_TEST_HEXELS,
        help=f"Hexel IDs to evaluate (default: {' '.join(DEFAULT_TEST_HEXELS)}).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for metric computation (default: cpu).",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save results as a CSV file.",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    raw_data_dir = Path(args.raw_data_dir)
    metric_names = list(METRIC_FNS.keys())

    csv_rows: list[dict] = []

    for exp_dir_str in args.exp_dir:
        exp_dir = Path(exp_dir_str)

        # Resolve target
        target = args.target
        if target is None:
            target = _infer_target(exp_dir_str)
        if target is None:
            print(f"ERROR: Could not infer target for '{exp_dir_str}'. Use --target.", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Experiment : {exp_dir.name}")
        print(f"Target     : {target.upper()}")
        print(f"Hexels     : {', '.join(args.hex_ids)}")
        print(f"{'='*60}")

        hex_metrics = _compute_metrics_for_experiment(
            exp_dir=exp_dir,
            raw_data_dir=raw_data_dir,
            target=target,
            hex_ids=args.hex_ids,
            device=device,
        )

        if not hex_metrics:
            print("  No results found.")
            continue

        _print_results_table(exp_label=exp_dir.name, hex_metrics=hex_metrics, metric_names=metric_names)

        for hex_id, metrics in hex_metrics.items():
            csv_rows.append({"exp": exp_dir.name, "target": target, "hex_id": hex_id, **metrics})

    if args.output_csv and csv_rows:
        import csv

        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["exp", "target", "hex_id"] + metric_names
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved CSV → {out_path}")


if __name__ == "__main__":
    main()
