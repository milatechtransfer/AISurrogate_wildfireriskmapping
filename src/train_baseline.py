"""
Train a tabular baseline (XGBoost / RandomForest / LinearRegression) on
spatial-only channels PLUS the per-pixel iROS fuel curve vector (no encoder --
raw curve values used directly, per the paper's contribution), evaluated with
the same metric functions and target denormalization convention used by the
UNet Trainer, at both patch level and region (stitched hexel) level.

Supports BP (min_max out_norm), FI, and ROS (log_standard out_norm) targets via
--target. Trains and predicts in NORMALIZED space (matching the U-Net's
training convention), denormalizing only at evaluation time for metric
computation.

Usage:
    python -m src.train_baseline --config=configs/bp_spatial_only_xgb.yaml --target=bp
    python -m src.train_baseline --config=configs/fi_spatial_only_xgb.yaml --target=fi
    python -m src.train_baseline --config=configs/ros_spatial_only_xgb.yaml --target=ros
    python -m src.train_baseline --config=configs/bp_spatial_only_xgb.yaml --target=bp \
        --pixels_per_patch=256 --max_train_batches=5 --max_eval_batches=5  # smoke test

Note: tqdm progress bars are written to stderr; milestone prints go to stdout.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import yaml
import joblib
from scipy.stats import spearmanr
from tqdm import tqdm
from xgboost import XGBRegressor

from src.config import Config
from src.datasets.dataset import get_train_val_dataloader, get_test_dataloader
from src.datasets.utils import get_dataset_dimensions, apply_bp_nodata_zero_range
from src.datasets.targets import get_target_specs
from data_preparation.spatial.utils import get_range_output, read_split_hex_ids, get_output_log_stats_cached
from src.utils import AVAILABLE_METRICS, seed_everything
from src.datasets.postprocessing.stitch_hexel import stitch_windows
from src.datasets.postprocessing.utils import calculate_hexel_metrics_pytorch, load_target_grid_for_mask_scope, load_spatial_raster
from data_preparation.paths import Paths


# Subset of metrics shown in the terminal region-level table (full data always
# in metrics.json). Keeps the printed table readable regardless of target.
DISPLAY_METRICS = ["mae", "normalized_mae", "spearman", "bias", "ccc", "auc_iou_full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a tabular baseline (XGBoost/RF/linear) on spatial + iROS fuel curve inputs."
    )
    parser.add_argument("--config", type=str, default="configs/bp_spatial_only_xgb.yaml", help="Path to YAML config file.")
    parser.add_argument("--target", type=str, default="bp", choices=["bp", "fi", "ros"], help="Which target to train the baseline on.")
    parser.add_argument("--pixels_per_patch", type=int, default=4096, help="Training subsample size per patch (0 = use all valid pixels).")
    parser.add_argument("--max_train_batches", type=int, default=0, help="If >0, cap number of training batches read during tabularization (for smoke tests).")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="If >0, cap number of val/test batches evaluated (for smoke tests).")
    return parser.parse_args()


def load_config(path: str) -> Config:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# ---------- Timing ----------

@contextmanager
def timed(label: str, timings: dict):
    """Times a block, prints '======= <label> Time ========', stores seconds in timings[label]."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    timings[label] = elapsed
    print(f"======= {label} Time ========\n{elapsed:.1f}s")


# ---------- Target transform params (mirrors Trainer._configure_metric_target_transform,
# generalized across min_max (BP) and log_standard (FI/ROS)) ----------

def get_grid_source_params(config: Config):
    source_map = {s.name: s for s in config.data.input_sources}
    if "grid" not in source_map:
        raise ValueError("Config data.input_sources must include a 'grid' source for this baseline.")
    return source_map["grid"].params


def get_target_transform_params(config: Config, target_name: str):
    """
    Returns (out_norm, target_min, target_max, target_log_mean, target_log_std)
    needed to denormalize predictions for the given target.
    """
    target_spec = get_target_specs(target_name)[0]
    grid_params = get_grid_source_params(config)
    out_norm = grid_params.out_norm
    train_hex_ids = read_split_hex_ids(os.path.join(config.data.root_dir, config.data.train_split))

    target_min, target_max = 0.0, 1.0
    target_log_mean = getattr(grid_params, "target_log_mean", None)
    target_log_std = getattr(grid_params, "target_log_std", None)

    if out_norm == "min_max":
        target_max, target_min = get_range_output(
            root_dir=config.data.raw_data_dir, output_type=target_spec.output_type, allowed_hex_ids=train_hex_ids,
        )
        target_max, target_min = apply_bp_nodata_zero_range(
            target_name=target_spec.name, max_value=target_max, min_value=target_min,
            bp_nodata_as_zero=grid_params.bp_nodata_as_zero,
        )
    elif out_norm == "log_standard":
        if target_log_mean is None or target_log_std is None:
            target_log_mean, target_log_std = get_output_log_stats_cached(
                root_dir=config.data.root_dir, output_type=target_spec.output_type,
                allowed_hex_ids=train_hex_ids, raw_data_dir=config.data.raw_data_dir,
            )
        if target_log_mean is None or target_log_std is None:
            raise ValueError(f"target_log_mean/std unavailable for target={target_name!r} with out_norm='log_standard'.")
    else:
        raise ValueError(f"Unsupported out_norm={out_norm!r} for this baseline. Supported: 'min_max', 'log_standard'.")

    return out_norm, target_min, target_max, target_log_mean, target_log_std


def denormalize_target(y_norm: np.ndarray, out_norm: str, target_min: float, target_max: float,
                        target_log_mean: float | None, target_log_std: float | None) -> np.ndarray:
    if out_norm == "min_max":
        return y_norm * (target_max - target_min) + target_min
    if out_norm == "log_standard":
        return np.clip(np.expm1(y_norm * target_log_std + target_log_mean), 0.0, None)
    raise ValueError(f"Unsupported out_norm={out_norm!r}.")


def clip_normalized_prediction(preds_norm: np.ndarray, out_norm: str) -> np.ndarray:
    """min_max predictions are naturally bounded [0,1] (mirrors get_predicted_hexel's
    clip); log_standard predictions have no such natural bound, so left unclipped."""
    if out_norm == "min_max":
        return np.clip(preds_norm, 0.0, 1.0)
    return preds_norm


def predict_real_scale(model, x_flat: np.ndarray, out_norm: str, target_min: float, target_max: float,
                        target_log_mean: float | None, target_log_std: float | None) -> np.ndarray:
    """Predict in normalized space, clip if applicable, then denormalize to real units."""
    preds_norm = clip_normalized_prediction(model.predict(x_flat), out_norm)
    return denormalize_target(preds_norm, out_norm, target_min, target_max, target_log_mean, target_log_std)


# ---------- Mask handling ----------

def resolve_mask(masks: np.ndarray, valid_mask_threshold: float) -> np.ndarray:
    if masks.dtype == bool:
        return masks
    return masks > valid_mask_threshold


# ---------- Combine spatial grid + fuel curve channels ----------

def combine_inputs(inputs_np: np.ndarray, fuel_curve_np: np.ndarray | None) -> np.ndarray:
    """
    inputs_np: (B, C, H, W) spatial channels.
    fuel_curve_np: (B, L, H, W) per-pixel iROS curve values, or None if not configured.
    Returns concatenated (B, C+L, H, W). No encoder -- raw curve values used directly.
    """
    if fuel_curve_np is None:
        return inputs_np
    return np.concatenate([inputs_np, fuel_curve_np], axis=1)


# ---------- Batch -> tabular rows ----------

def tabularize_loader(loader, valid_mask_threshold, pixels_per_patch=None, rng=None, max_batches=0):
    """Trains directly on normalized target space, matching the U-Net's training convention."""
    X_parts, y_parts = [], []
    n_batches = max_batches if max_batches else len(loader)
    progress = tqdm(enumerate(loader), total=n_batches, desc="Tabularizing", leave=True)

    for i, batch in progress:
        if max_batches and i >= max_batches:
            break

        inputs, targets, masks = batch["grid"]
        fuel_curve = batch.get("fuel_curve")

        inputs_np = inputs.numpy()
        fuel_curve_np = fuel_curve.numpy() if fuel_curve is not None else None
        combined_np = combine_inputs(inputs_np, fuel_curve_np)

        targets_np = targets.numpy()
        masks_np = masks.numpy()

        B, C, H, W = combined_np.shape
        x = combined_np.transpose(0, 2, 3, 1).reshape(-1, C)
        y = targets_np.reshape(-1)  # normalized space, no denormalization
        m = resolve_mask(masks_np.reshape(-1), valid_mask_threshold)

        if pixels_per_patch:
            valid_idx = np.where(m)[0]
            n_keep = min(pixels_per_patch * B, len(valid_idx))
            if n_keep < len(valid_idx):
                keep = rng.choice(valid_idx, size=n_keep, replace=False)
                x, y = x[keep], y[keep]
            else:
                x, y = x[valid_idx], y[valid_idx]
        else:
            x, y = x[m], y[m]

        X_parts.append(x.astype(np.float32))
        y_parts.append(y.astype(np.float32))

        progress.set_postfix({"rows": sum(p.shape[0] for p in X_parts)})

    if not X_parts:
        raise RuntimeError("No batches were tabularized. Check max_batches/loader length.")

    return np.concatenate(X_parts), np.concatenate(y_parts)


# ---------- Per-patch summaries (real-scale units) ----------

def patch_summary(values: np.ndarray, mask: np.ndarray, top_fraction: float | None = None) -> float:
    valid = values[mask]
    if valid.size == 0:
        return float("nan")
    if top_fraction is None:
        return float(valid.mean())
    k = max(1, int(np.ceil(valid.size * top_fraction)))
    return float(np.sort(valid)[-k:].mean())


# ---------- Patch-level metrics + scalar-summary hexel rank agreement ----------

def evaluate_patchwise_and_hexel(
    model, loader, out_norm, target_min, target_max, target_log_mean, target_log_std,
    valid_mask_threshold, metric_names, top_fraction: float = 0.10, max_batches: int = 0
):
    metric_fns = {k: AVAILABLE_METRICS[k] for k in metric_names}
    running = {name: 0.0 for name in metric_fns}
    running_count = 0

    hex_pred_mean: dict[int, list[float]] = {}
    hex_target_mean: dict[int, list[float]] = {}
    hex_pred_top: dict[int, list[float]] = {}
    hex_target_top: dict[int, list[float]] = {}

    n_batches = max_batches if max_batches else len(loader)
    progress = tqdm(enumerate(loader), total=n_batches, desc="Evaluating", leave=True)

    for i, batch in progress:
        if max_batches and i >= max_batches:
            break

        inputs, targets, masks = batch["grid"]
        fuel_curve = batch.get("fuel_curve")
        patch_metadata = batch.get("patch_metadata")
        if patch_metadata is None or "hex_id" not in patch_metadata:
            raise ValueError(
                "Batch is missing patch_metadata['hex_id']; check config.data.include_patch_metadata=true."
            )
        hex_ids = patch_metadata["hex_id"].numpy()

        inputs_np = inputs.numpy()
        fuel_curve_np = fuel_curve.numpy() if fuel_curve is not None else None
        combined_np = combine_inputs(inputs_np, fuel_curve_np)

        targets_np, masks_np = targets.numpy(), masks.numpy()
        B, C, H, W = combined_np.shape

        x_flat = combined_np.transpose(0, 2, 3, 1).reshape(-1, C)
        preds_real_flat = predict_real_scale(model, x_flat, out_norm, target_min, target_max, target_log_mean, target_log_std)
        preds = preds_real_flat.reshape(B, H, W)[:, None, :, :]  # (B, 1, H, W)

        targets_real = denormalize_target(targets_np, out_norm, target_min, target_max, target_log_mean, target_log_std)
        mask_resolved = resolve_mask(masks_np, valid_mask_threshold)

        # ---------- Patch-level metrics ----------
        preds_t = torch.from_numpy(preds).float()
        targets_t = torch.from_numpy(targets_real).float()
        masks_t = torch.from_numpy(mask_resolved).float()
        for name, fn in metric_fns.items():
            val = fn(preds_t, targets_t, masks_t)
            running[name] += val.item() * B
        running_count += B

        # ---------- Per-patch summaries, grouped by hex_id ----------
        for b in range(B):
            hid = int(hex_ids[b])
            pv = preds[b, 0]
            tv = targets_real[b, 0]
            mv = mask_resolved[b, 0].astype(bool)

            hex_pred_mean.setdefault(hid, []).append(patch_summary(pv, mv))
            hex_target_mean.setdefault(hid, []).append(patch_summary(tv, mv))
            hex_pred_top.setdefault(hid, []).append(patch_summary(pv, mv, top_fraction))
            hex_target_top.setdefault(hid, []).append(patch_summary(tv, mv, top_fraction))

        progress.set_postfix({"pixels_seen": running_count, "hexels_seen": len(hex_pred_mean)})

    if running_count == 0:
        raise RuntimeError("No batches were evaluated. Check max_batches/loader length.")

    patch_metrics = {name: total / running_count for name, total in running.items()}

    hex_ids_sorted = sorted(hex_pred_mean.keys())
    hex_pred_mean_agg = np.array([np.nanmean(hex_pred_mean[h]) for h in hex_ids_sorted])
    hex_target_mean_agg = np.array([np.nanmean(hex_target_mean[h]) for h in hex_ids_sorted])
    hex_pred_top_agg = np.array([np.nanmean(hex_pred_top[h]) for h in hex_ids_sorted])
    hex_target_top_agg = np.array([np.nanmean(hex_target_top[h]) for h in hex_ids_sorted])

    if len(hex_ids_sorted) >= 2:
        hex_mean_spearman = float(spearmanr(hex_pred_mean_agg, hex_target_mean_agg).correlation)
        hex_top_spearman = float(spearmanr(hex_pred_top_agg, hex_target_top_agg).correlation)
    else:
        hex_mean_spearman = float("nan")
        hex_top_spearman = float("nan")

    hexel_metrics = {
        "n_hexels": len(hex_ids_sorted),
        "hex_mean_spearman": hex_mean_spearman,
        f"hex_top{int(top_fraction * 100)}_spearman": hex_top_spearman,
    }

    return patch_metrics, hexel_metrics


# ---------- Region-level (stitched full-hexel raster) evaluation ----------

def evaluate_region_level(model, loader, out_norm, target_min, target_max, target_log_mean, target_log_std,
                           valid_mask_threshold, config, metric_functions, target_name: str, device="cpu", max_batches: int = 0):
    """
    Stitches per-patch predictions (denormalized to real units) into full hexel
    rasters via stitch_windows, compares against the true raw raster (loaded via
    load_target_grid_for_mask_scope), and computes the full metric suite once per
    hexel via calculate_hexel_metrics_pytorch.
    """
    hex_data = defaultdict(lambda: {"preds": [], "locations": [], "masks": []})
    target_spec = get_target_specs(target_name)[0]

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        inputs, targets, masks = batch["grid"]
        fuel_curve = batch.get("fuel_curve")
        patch_metadata = batch.get("patch_metadata")
        if patch_metadata is None or "row" not in patch_metadata or "col" not in patch_metadata:
            raise ValueError(
                "Batch is missing patch_metadata['row']/['col']; requires the "
                "dataset.py patch_metadata extension (see PR)."
            )
        hex_ids = patch_metadata["hex_id"].numpy()
        rows = patch_metadata["row"].numpy()
        cols = patch_metadata["col"].numpy()

        inputs_np = inputs.numpy()
        fuel_curve_np = fuel_curve.numpy() if fuel_curve is not None else None
        combined_np = combine_inputs(inputs_np, fuel_curve_np)
        masks_np = masks.numpy()
        B, C, H, W = combined_np.shape

        x_flat = combined_np.transpose(0, 2, 3, 1).reshape(-1, C)
        preds_real_flat = predict_real_scale(model, x_flat, out_norm, target_min, target_max, target_log_mean, target_log_std)
        preds_real = preds_real_flat.reshape(B, H, W)
        mask_resolved = resolve_mask(masks_np, valid_mask_threshold)[:, 0]  # (B, H, W)

        for b in range(B):
            hid = str(int(hex_ids[b])).zfill(2)
            hex_data[hid]["preds"].append(preds_real[b])
            hex_data[hid]["locations"].append((int(rows[b]), int(cols[b])))
            hex_data[hid]["masks"].append(mask_resolved[b])

    per_hexel_metrics = {}
    for hid, data in hex_data.items():
        paths = Paths(hex_id=hid, root_dir=config.data.raw_data_dir)

        gt_grid_raw, profile = load_spatial_raster(
            path=getattr(paths, target_spec.path_method)(),
            mask_path=paths.mask_grid(hex_id=hid, mask_scope="actual"),
        )
        pred_grid = stitch_windows(data["preds"], data["locations"], data["masks"], gt_grid_raw.shape, mode="mean")

        gt_grid, pred_grid = load_target_grid_for_mask_scope(
            paths=paths,
            target=target_spec,
            pred_grid=pred_grid,
            profile=profile,
            mask_scope="actual",
            hex_id=hid,
            bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero,
        )

        per_hexel_metrics[hid] = calculate_hexel_metrics_pytorch(
            gt_grid=gt_grid, pred_grid=pred_grid, device=device, metric_functions=metric_functions
        )

    all_metric_names = next(iter(per_hexel_metrics.values())).keys()
    aggregated = {
        name: float(np.mean([m[name] for m in per_hexel_metrics.values()]))
        for name in all_metric_names
    }

    return aggregated, per_hexel_metrics


# ---------- Pretty printing ----------

def print_region_metrics_table(aggregated: dict, per_hexel: dict, split_name: str, display_metrics: list[str] | None = None) -> None:
    """Aligned table: hexels as rows, selected metrics as columns. Full metrics stay in metrics.json."""
    metric_names = [m for m in (display_metrics or list(aggregated.keys())) if m in aggregated]
    hex_ids = sorted(per_hexel.keys())

    col_width = 12
    header = f"{'hex_id':<8}" + "".join(f"{name[:col_width]:>{col_width}}" for name in metric_names)
    print(f"\n======= {split_name} metrics (region-level) ========")
    print(header)
    print("-" * len(header))

    for hid in hex_ids:
        row = f"{hid:<8}" + "".join(f"{per_hexel[hid][name]:>{col_width}.4f}" for name in metric_names)
        print(row)

    print("-" * len(header))
    agg_row = f"{'mean':<8}" + "".join(f"{aggregated[name]:>{col_width}.4f}" for name in metric_names)
    print(agg_row)
    print()


def print_metrics(title: str, metrics: dict) -> None:
    print(f"======= {title} ========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


# ---------- Model dispatch ----------

def build_baseline_model(model_config):
    architecture = model_config.architecture.strip().lower().replace("-", "_")
    params = getattr(model_config, "params", None) or {}
    if architecture == "xgboost":
        return XGBRegressor(**params)
    if architecture in {"random_forest", "rf"}:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    if architecture in {"linear_regression", "linear"}:
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    raise ValueError(f"Unknown baseline architecture '{model_config.architecture}'.")


# ---------- Per-split evaluation (patch + region), deduplicating val/test blocks ----------

def run_split_evaluation(
    model, loader, split_name: str, out_norm, target_min, target_max, target_log_mean, target_log_std,
    valid_mask_threshold, config, region_metric_functions, target_name: str, max_eval_batches: int, timings: dict,
):
    with timed(f"{split_name} Eval", timings):
        patch_metrics, hex_rank_metrics = evaluate_patchwise_and_hexel(
            model, loader, out_norm, target_min, target_max, target_log_mean, target_log_std,
            valid_mask_threshold, config.metrics, max_batches=max_eval_batches,
        )
    print_metrics(f"{split_name} metrics (patch)", patch_metrics)
    print_metrics(f"{split_name} metrics (hexel, scalar-summary rank agreement)", hex_rank_metrics)

    with timed(f"{split_name} Region Eval", timings):
        region_agg, region_per_hexel = evaluate_region_level(
            model, loader, out_norm, target_min, target_max, target_log_mean, target_log_std,
            valid_mask_threshold, config, region_metric_functions, target_name=target_name,
            max_batches=max_eval_batches,
        )
    print_region_metrics_table(region_agg, region_per_hexel, split_name, DISPLAY_METRICS)

    return patch_metrics, hex_rank_metrics, region_agg, region_per_hexel


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = getattr(config, "seed", 42)
    seed_everything(seed=seed, deterministic=getattr(config, "deterministic", True))
    rng = np.random.default_rng(seed)

    timings: dict[str, float] = {}

    # ---------- Data ----------
    train_loader, val_loader = get_train_val_dataloader(
        config=config.data, modelling_approach=config.modelling_approach, seed=seed
    )
    spatial_channels, aux_dims = get_dataset_dimensions(train_loader.dataset)
    fuel_curve_len = aux_dims.get("fuel_curve", 0) if aux_dims else 0
    total_channels = spatial_channels + fuel_curve_len
    print(f"Detected Data Dimensions: Spatial={spatial_channels}, FuelCurve={fuel_curve_len}, Total={total_channels}")

    out_norm, target_min, target_max, target_log_mean, target_log_std = get_target_transform_params(config, args.target)
    print(f"======= Target transform ({args.target}) ========\nout_norm={out_norm}, min={target_min}, max={target_max}, "
          f"log_mean={target_log_mean}, log_std={target_log_std}")

    valid_mask_threshold = config.data.valid_mask_threshold
    pixels_per_patch = args.pixels_per_patch if args.pixels_per_patch > 0 else None

    # ---------- Tabularize ----------
    with timed("Tabularize", timings):
        X_train, y_train = tabularize_loader(
            train_loader, valid_mask_threshold,
            pixels_per_patch=pixels_per_patch, rng=rng, max_batches=args.max_train_batches,
        )
    print(f"======= Train rows ========\n{X_train.shape[0]:,} pixels, {X_train.shape[1]} channels")

    # ---------- Fit ----------
    model = build_baseline_model(config.model)
    print(f"[Baseline] Fitting {config.model.architecture} for target={args.target}...")
    with timed("Fit", timings):
        model.fit(X_train, y_train)

    # ---------- Evaluate: val + test (patch + region) ----------
    region_metric_functions = {k: AVAILABLE_METRICS[k] for k in config.metrics}

    val_metrics, val_hex_metrics, val_region_agg, val_region_per_hexel = run_split_evaluation(
        model, val_loader, "Val", out_norm, target_min, target_max, target_log_mean, target_log_std,
        valid_mask_threshold, config, region_metric_functions, args.target, args.max_eval_batches, timings,
    )

    test_loader = get_test_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=seed)
    test_metrics, test_hex_metrics, test_region_agg, test_region_per_hexel = run_split_evaluation(
        model, test_loader, "Test", out_norm, target_min, target_max, target_log_mean, target_log_std,
        valid_mask_threshold, config, region_metric_functions, args.target, args.max_eval_batches, timings,
    )

    total_time = sum(timings.values())
    print(f"======= Total Time ========\n{total_time:.1f}s")

    # ---------- Save ----------
    os.makedirs(config.save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(config.save_dir, f"baseline_model_{args.target}.joblib"))
    with open(os.path.join(config.save_dir, f"metrics_{args.target}.json"), "w") as f:
        json.dump(
            {
                "target": args.target,
                "out_norm": out_norm,
                "val": val_metrics,
                "val_hexel_rank_agreement": val_hex_metrics,
                "test": test_metrics,
                "test_hexel_rank_agreement": test_hex_metrics,
                "val_region_aggregated": val_region_agg,
                "val_region_per_hexel": val_region_per_hexel,
                "test_region_aggregated": test_region_agg,
                "test_region_per_hexel": test_region_per_hexel,
                "feature_dims": {"spatial": spatial_channels, "fuel_curve": fuel_curve_len, "total": total_channels},
                "timing": {**timings, "total_s": total_time},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()