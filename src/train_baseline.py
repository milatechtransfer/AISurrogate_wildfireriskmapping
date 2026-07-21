"""
Train a tabular baseline (XGBoost / RandomForest / LinearRegression) on
spatial-only channels, evaluated with the same metric functions and target
denormalization convention used by the UNet Trainer, at both patch level and
hexel level. Includes phase timing and tqdm progress bars.

Usage:
    python -m src.train_baseline --config=configs/bp_spatial_only_xgb.yaml
    python -m src.train_baseline --config=configs/bp_spatial_only_xgb.yaml \
        --pixels_per_patch=256 --max_train_batches=5 --max_eval_batches=5  # smoke test

Note: tqdm progress bars are written to stderr; milestone prints go to stdout.
When running via SLURM, check both the .out and .err log files.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
import joblib
from scipy.stats import spearmanr
from tqdm import tqdm
from xgboost import XGBRegressor
from collections import defaultdict


from src.config import Config
from src.datasets.dataset import get_train_val_dataloader, get_test_dataloader
from src.datasets.utils import get_dataset_dimensions, apply_bp_nodata_zero_range
from src.datasets.targets import get_target_specs
from data_preparation.spatial.utils import get_range_output, read_split_hex_ids
from src.utils import AVAILABLE_METRICS, seed_everything
from src.datasets.postprocessing.stitch_hexel import stitch_windows
from src.datasets.postprocessing.utils import calculate_hexel_metrics_pytorch, load_target_grid_for_mask_scope, load_spatial_raster
from data_preparation.paths import Paths
from src.datasets.targets import get_target_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a tabular baseline (XGBoost/RF/linear) on spatial-only inputs."
    )
    parser.add_argument("--config", type=str, default="configs/bp_spatial_only_xgb.yaml", help="Path to YAML config file.")
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


# ---------- Target denormalization (mirrors Trainer._configure_metric_target_transform, min_max only) ----------

def get_grid_source_params(config: Config):
    source_map = {s.name: s for s in config.data.input_sources}
    if "grid" not in source_map:
        raise ValueError("Config data.input_sources must include a 'grid' source for this baseline.")
    return source_map["grid"].params


def get_bp_real_scale_range(config: Config) -> tuple[float, float]:
    target_spec = get_target_specs("bp")[0]
    train_hex_ids = read_split_hex_ids(os.path.join(config.data.root_dir, config.data.train_split))
    target_max, target_min = get_range_output(
        root_dir=config.data.raw_data_dir,
        output_type=target_spec.output_type,
        allowed_hex_ids=train_hex_ids,
    )
    grid_params = get_grid_source_params(config)
    target_max, target_min = apply_bp_nodata_zero_range(
        target_name=target_spec.name,
        max_value=target_max,
        min_value=target_min,
        bp_nodata_as_zero=grid_params.bp_nodata_as_zero,
    )
    return target_min, target_max


def denormalize_min_max(y_norm: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    return y_norm * (target_max - target_min) + target_min


# ---------- Mask handling ----------
# NOTE: assumes masks may be boolean or a continuous valid-fraction; verified
# working against real data in the smoke test (job 10163234).

def resolve_mask(masks: np.ndarray, valid_mask_threshold: float) -> np.ndarray:
    if masks.dtype == bool:
        return masks
    return masks > valid_mask_threshold


# ---------- Batch -> tabular rows ----------

def tabularize_loader(loader, target_min, target_max, valid_mask_threshold, pixels_per_patch=None, rng=None, max_batches=0):
    X_parts, y_parts = [], []
    n_batches = max_batches if max_batches else len(loader)
    progress = tqdm(enumerate(loader), total=n_batches, desc="Tabularizing", leave=True)

    for i, batch in progress:
        if max_batches and i >= max_batches:
            break

        inputs, targets, masks = batch["grid"]
        inputs_np = inputs.numpy()
        targets_np = targets.numpy()
        masks_np = masks.numpy()

        B, C, H, W = inputs_np.shape
        x = inputs_np.transpose(0, 2, 3, 1).reshape(-1, C)
        y = denormalize_min_max(targets_np.reshape(-1), target_min, target_max)
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


# ---------- Per-patch summaries (real-scale BP units; parallels HexSummaryLoss._patch_summaries
# but operates in denormalized space rather than sigmoid-probability space, to stay consistent
# with the rest of this script's metric reporting) ----------

def patch_summary(values: np.ndarray, mask: np.ndarray, top_fraction: float | None = None) -> float:
    valid = values[mask]
    if valid.size == 0:
        return float("nan")
    if top_fraction is None:
        return float(valid.mean())
    k = max(1, int(np.ceil(valid.size * top_fraction)))
    return float(np.sort(valid)[-k:].mean())


# ---------- Patch-level metrics + hexel-level aggregation ----------

def evaluate_patchwise_and_hexel(
    model, loader, target_min, target_max, valid_mask_threshold, metric_names, top_fraction: float = 0.10, max_batches: int = 0
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
        patch_metadata = batch.get("patch_metadata")
        if patch_metadata is None or "hex_id" not in patch_metadata:
            raise ValueError(
                "Batch is missing patch_metadata['hex_id']; check config.data.include_patch_metadata=true."
            )
        hex_ids = patch_metadata["hex_id"].numpy()

        inputs_np, targets_np, masks_np = inputs.numpy(), targets.numpy(), masks.numpy()
        B, C, H, W = inputs_np.shape

        x_flat = inputs_np.transpose(0, 2, 3, 1).reshape(-1, C)
        preds_flat = model.predict(x_flat)
        preds = preds_flat.reshape(B, H, W)[:, None, :, :]  # (B, 1, H, W)

        targets_real = denormalize_min_max(targets_np, target_min, target_max)
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

def evaluate_region_level(model, loader, valid_mask_threshold, config, metric_functions, device="cpu", max_batches: int = 0):
    """
    Region-level (whole-hexel) evaluation: gather per-patch predictions with their
    (row, col) placement, stitch into full hexel rasters via stitch_windows (the
    same function the U-Net eval pipeline uses), compare against the true raw BP
    raster, and compute the full metric suite once per hexel via
    calculate_hexel_metrics_pytorch -- matching the "Region-Level Evaluation"
    reporting convention used for the U-Net.

    Predictions are stitched directly in real-scale units: mean-overlap averaging
    commutes with the affine min_max denormalization, so stitching real-scale
    values first and denormalizing (already done, since our predictions are
    trained/predicted in real-scale space) is equivalent to stitching normalized
    values and denormalizing after -- no double-transform risk.

    Ground truth is loaded via load_target_grid_for_mask_scope (not a raw
    load_spatial_raster call) so bp_nodata_as_zero and mask-scope handling exactly
    match the U-Net's own evaluate_and_visualize_hexels pipeline -- otherwise the
    two ground truths could silently diverge on nodata pixels.
    """

    hex_data = defaultdict(lambda: {"preds": [], "locations": [], "masks": []})
    target_spec = get_target_specs("bp")[0]

    for i,batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        inputs, targets, masks = batch["grid"]
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
        masks_np = masks.numpy()
        B, C, H, W = inputs_np.shape

        x_flat = inputs_np.transpose(0, 2, 3, 1).reshape(-1, C)
        preds_flat = model.predict(x_flat).reshape(B, H, W)
        mask_resolved = resolve_mask(masks_np, valid_mask_threshold)[:, 0]  # (B, H, W)

        for b in range(B):
            hid = str(int(hex_ids[b])).zfill(2)
            hex_data[hid]["preds"].append(preds_flat[b])
            hex_data[hid]["locations"].append((int(rows[b]), int(cols[b])))
            hex_data[hid]["masks"].append(mask_resolved[b])

    per_hexel_metrics = {}
    for hid, data in hex_data.items():
        paths = Paths(hex_id=hid, root_dir=config.data.raw_data_dir)

        # Load once for shape/profile so we can stitch to the correct raster size.
        gt_grid_raw, profile = load_spatial_raster(
            path=paths.output_burn_prob(),
            mask_path=paths.mask_grid(hex_id=hid, mask_scope="actual"),
        )
        pred_grid = stitch_windows(data["preds"], data["locations"], data["masks"], gt_grid_raw.shape, mode="mean")

        # Re-load ground truth via the same helper the U-Net eval pipeline uses,
        # so bp_nodata_as_zero and mask-scope handling match exactly.
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

    # Aggregate across hexels: mean per metric, matching the single-row-per-config
    # convention in the comparison table.
    all_metric_names = next(iter(per_hexel_metrics.values())).keys()
    aggregated = {
        name: float(np.mean([m[name] for m in per_hexel_metrics.values()]))
        for name in all_metric_names
    }

    return aggregated, per_hexel_metrics


# ---------- Model dispatch ----------
# Mirrors src.models.factory.build_model's dispatch pattern, kept as a separate
# function since these estimators don't satisfy the nn.Module/autograd contract
# that Trainer/factory.py assume.

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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = getattr(config, "seed", 42)
    seed_everything(seed=seed, deterministic=getattr(config, "deterministic", True))
    rng = np.random.default_rng(seed)

    # ---------- Data ----------
    train_loader, val_loader = get_train_val_dataloader(
        config=config.data, modelling_approach=config.modelling_approach, seed=seed
    )
    spatial_channels, _ = get_dataset_dimensions(train_loader.dataset)
    print(f"Detected Data Dimensions: Spatial={spatial_channels}")

    target_min, target_max = get_bp_real_scale_range(config)
    print(f"======= BP real-scale range ========\nmin={target_min}, max={target_max}")

    valid_mask_threshold = config.data.valid_mask_threshold
    pixels_per_patch = args.pixels_per_patch if args.pixels_per_patch > 0 else None

    # ---------- Tabularize ----------
    t0 = time.time()
    X_train, y_train = tabularize_loader(
        train_loader, target_min, target_max, valid_mask_threshold,
        pixels_per_patch=pixels_per_patch, rng=rng, max_batches=args.max_train_batches,
    )
    tabularize_time = time.time() - t0
    print(f"======= Tabularize Time ========\n{tabularize_time:.1f}s")
    print(f"======= Train rows ========\n{X_train.shape[0]:,} pixels, {X_train.shape[1]} channels")

    # ---------- Fit ----------
    model = build_baseline_model(config.model)
    print(f"[Baseline] Fitting {config.model.architecture}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"======= Fit Time ========\n{fit_time:.1f}s")

    # ---------- Evaluate: val (patch-level) ----------
    t0 = time.time()
    val_metrics, val_hex_metrics = evaluate_patchwise_and_hexel(
        model, val_loader, target_min, target_max, valid_mask_threshold, config.metrics,
        max_batches=args.max_eval_batches,
    )
    val_eval_time = time.time() - t0
    print(f"======= Val Eval Time ========\n{val_eval_time:.1f}s")
    print("======= Val metrics (patch) ========")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")
    print("======= Val metrics (hexel, scalar-summary rank agreement) ========")
    for k, v in val_hex_metrics.items():
        print(f"{k}: {v}")

    # ---------- Evaluate: test (patch-level) ----------
    test_loader = get_test_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=seed)
    t0 = time.time()
    test_metrics, test_hex_metrics = evaluate_patchwise_and_hexel(
        model, test_loader, target_min, target_max, valid_mask_threshold, config.metrics,
        max_batches=args.max_eval_batches,
    )
    test_eval_time = time.time() - t0
    print(f"======= Test Eval Time ========\n{test_eval_time:.1f}s")
    print("======= Test metrics (patch) ========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print("======= Test metrics (hexel, scalar-summary rank agreement) ========")
    for k, v in test_hex_metrics.items():
        print(f"{k}: {v}")

    # ---------- Evaluate: val (region-level, stitched full-hexel rasters) ----------
    region_metric_functions = {k: AVAILABLE_METRICS[k] for k in config.metrics}

    t0 = time.time()
    val_region_agg, val_region_per_hexel = evaluate_region_level(
        model, val_loader, valid_mask_threshold, config, region_metric_functions
    )
    val_region_time = time.time() - t0
    print(f"======= Val Region Eval Time ========\n{val_region_time:.1f}s")
    print("======= Val metrics (region-level, aggregated) ========")
    for k, v in val_region_agg.items():
        print(f"{k}: {v:.4f}")
    print("======= Val metrics (region-level, per-hexel) ========")
    for hid, m in val_region_per_hexel.items():
        print(f"hex{hid}: {m}")

    # ---------- Evaluate: test (region-level, stitched full-hexel rasters) ----------
    t0 = time.time()
    test_region_agg, test_region_per_hexel = evaluate_region_level(
        model, test_loader, valid_mask_threshold, config, region_metric_functions
    )
    test_region_time = time.time() - t0
    print(f"======= Test Region Eval Time ========\n{test_region_time:.1f}s")
    print("======= Test metrics (region-level, aggregated) ========")
    for k, v in test_region_agg.items():
        print(f"{k}: {v:.4f}")
    print("======= Test metrics (region-level, per-hexel) ========")
    for hid, m in test_region_per_hexel.items():
        print(f"hex{hid}: {m}")

    total_time = (
        tabularize_time + fit_time + val_eval_time + test_eval_time + val_region_time + test_region_time
    )
    print(f"======= Total Time ========\n{total_time:.1f}s")

    # ---------- Save ----------
    os.makedirs(config.save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(config.save_dir, "baseline_model.joblib"))
    with open(os.path.join(config.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "val": val_metrics,
                "val_hexel_rank_agreement": val_hex_metrics,
                "test": test_metrics,
                "test_hexel_rank_agreement": test_hex_metrics,
                "val_region_aggregated": val_region_agg,
                "val_region_per_hexel": val_region_per_hexel,
                "test_region_aggregated": test_region_agg,
                "test_region_per_hexel": test_region_per_hexel,
                "timing": {
                    "tabularize_s": tabularize_time,
                    "fit_s": fit_time,
                    "val_eval_s": val_eval_time,
                    "test_eval_s": test_eval_time,
                    "val_region_eval_s": val_region_time,
                    "test_region_eval_s": test_region_time,
                    "total_s": total_time,
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()