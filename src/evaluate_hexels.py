"""
End-to-end script for evaluation of one hexel
"""

import argparse
import glob
import json
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from data_preparation.paths import MASK_SCOPE_CHOICES
from src.config import Config, GridParams, apply_run_id_overrides
from src.datasets.dataset import MultiSourceDataset, get_test_dataloader
from src.datasets.postprocessing.utils import evaluate_and_visualize_hexels, print_and_log_eval_metrics
from src.datasets.utils import get_dataset_dimensions
from src.trainer import Trainer
from src.utils import (
    seed_everything,
    visualize_model_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate UNet baseline.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/bp_common_input_pipeline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="Boolean flag to visualize some random predictions vs. targets",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Boolean flag to save the visualization figure.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Compute stitched hexel metrics without writing predicted hexels or plots.",
    )
    parser.add_argument(
        "--skip_hexel_plots",
        action="store_true",
        help="Write stitched predicted hexel rasters and metrics, but skip per-hexel PNG diagnostics.",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Do not save patch-level test_predictions.npy.",
    )
    parser.add_argument(
        "--fast_eval",
        action="store_true",
        help="Enable speed-focused eval defaults: metrics-only and skip writing prediction arrays/plots.",
    )
    parser.add_argument(
        "--tif_only",
        action="store_true",
        help="Skip all metric computation (patch- and hexel-level) and plots. Still stitches and saves predicted hexel .tif rasters.",
    )
    parser.add_argument(
        "--robust_plot_percentile",
        type=float,
        default=None,
        help="Also save target/prediction/diff plots clipped to this percentile, e.g. 99 writes *_p99.png.",
    )
    parser.add_argument(
        "--stitch_mode",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to combine overlapping patch predictions when reconstructing hexels.",
    )
    parser.add_argument(
        "--mask_scope",
        choices=MASK_SCOPE_CHOICES,
        default="actual",
        help="Mask scope for stitched evaluation/inference artifacts. Non-actual scopes require matching patch metadata.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="SLURM array task ID (or run index) used to derive the run-specific seed, save_dir, and "
        "Comet experiment name matching the corresponding training run (see run_files/train_no_tmp_copy_array.sh).",
    )
    return parser.parse_args()


def load_config(path: str) -> Config:
    """
    load yaml config
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def main(
    *,
    args: argparse.Namespace | None = None,
    config: Config | None = None,
    patch_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    metadata_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, float]:
    args = args or parse_args()
    if getattr(args, "fast_eval", False):
        args.metrics_only = True
        args.skip_hexel_plots = True
        args.no_save_predictions = True
    config = config or load_config(args.config)

    if getattr(args, "tif_only", False):
        args.metrics_only = False
        args.skip_hexel_plots = True

    if args.run_id is not None:
        run_seed = apply_run_id_overrides(config, args.run_id)
        print(f"[run_id={args.run_id}] Overriding seed={run_seed}, save_dir={config.save_dir}")

    # ---------- Set Seed ----------
    seed = getattr(config, "seed", 42)
    deterministic = getattr(config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    config.logger.enabled = False

    print("\n[Evaluation] Loading test set...")
    # NOTE: If we need the stats on a particular hexel then modify the test_indices.csv in the config file with
    # meta_hex_{hex_id}.csv file
    start_time = time.time()

    # ---------- Memory tracking ----------
    # NOTE: continuous in-process tracking, not interval polling -- short
    # eval runs don't give coarse polling enough samples to reliably catch
    # a peak, unlike BurnP3+'s multi-hour runs where 30s sstat polling was
    # validated against a sustained multi-minute plateau.
    _peak_rss_bytes = 0
    _stop_sampler = threading.Event()

    def _sample_memory(interval: float = 0.5) -> None:
        nonlocal _peak_rss_bytes
        proc = psutil.Process()
        while not _stop_sampler.is_set():
            try:
                total = proc.memory_info().rss
                total += sum(c.memory_info().rss for c in proc.children(recursive=True))
            except psutil.Error:
                pass
            else:
                _peak_rss_bytes = max(_peak_rss_bytes, total)
            _stop_sampler.wait(interval)

    sampler_thread = threading.Thread(target=_sample_memory, daemon=True)
    sampler_thread.start()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---------- Dataloader / normalization-range setup ----------
    # NOTE: this is where GridSource scans raw training-hexel rasters under
    # raw_data_dir to compute min_max normalization ranges -- on a network
    # filesystem (e.g. a shared /network/projects/... mount) with the real ~44-hexel
    # training set, this step alone can take on the order of a minute, and
    # was previously invisible (folded silently into "Total Evaluation Time"
    # with no separate line to attribute it to). With the normalization
    # caching PR in place, this should drop sharply whenever dataset_norm_stats.json
    # is present, since GridSource's cached helpers skip the raw raster scan.
    dataloader_start_time = time.time()
    test_loader = get_test_dataloader(
        config=config.data,
        modelling_approach=config.modelling_approach,
        seed=seed,
        patch_transform=patch_transform,
        metadata_filter=metadata_filter,
    )
    dataloader_time = time.time() - dataloader_start_time

    # Get all data sources from the test dataset
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(test_loader.dataset)
    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Auxiliary={auxiliary_input_dims}")

    # iROS stats come from the checkpoint (registered buffers), not re-computed at eval time.
    trainer_init_start_time = time.time()
    trainer = Trainer(config, spatial_input_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims)
    trainer_init_time = time.time() - trainer_init_start_time
    if getattr(args, "tif_only", False):
        trainer.metric_functions = {}

    # ---------- Load best checkpoint ----------
    # Try best.pth first, fall back to last.pth if needed
    model_ckpt = None
    checkpoint_start_time = time.time()
    try:
        print(f"\n[Checkpoint] Loading {config.evaluation.checkpoint_filename} for evaluation...")
        model_ckpt = trainer.load_model(filename=config.evaluation.checkpoint_filename)
    except (FileNotFoundError, AttributeError):
        raise ValueError("[Checkpoint] checkpoint file not found or invalid...")  # noqa: B904
    checkpoint_time = time.time() - checkpoint_start_time

    if model_ckpt is not None:
        print(f"[Checkpoint] Loaded epoch={model_ckpt.get('epoch', 'N/A')} Checkpoint Metrics={model_ckpt.get('metric_value', 'N/A')}")

    # ---------- Evaluation ----------

    source_map = {s.name: s for s in config.data.input_sources}
    grid_source = source_map.get("grid") if "grid" in source_map else None
    grid_features = None
    out_norm = "min_max"  # default fallback, prevent mypy crash
    if grid_source and isinstance(grid_source.params, GridParams):
        grid_features = grid_source.params.feature_names_list
        out_norm = grid_source.params.out_norm

    preds_start_time = time.time()
    test_metrics, test_predictions = trainer.test(test_loader, return_predictions=True)
    preds_time = time.time() - preds_start_time
    peak_mps_driver_gb = test_metrics.pop("_peak_mps_driver_allocated_gb", None)

    if args.visualize_predictions and isinstance(test_predictions, np.ndarray):
        # get the channel mapping dict if it exists
        json_pattern = os.path.join(config.data.root_dir, "feature_channel_map_*.json")
        json_files = glob.glob(json_pattern)

        channel_map = None
        if json_files:
            with open(json_files[0]) as f:
                channel_map = json.load(f)

        # save path for visualization figure (if True)
        viz_save_path = None
        if args.save_visualizations:
            viz_save_path = os.path.join(config.save_dir, "inference_samples_examples.png")

        visualize_model_predictions(
            test_loader=test_loader,
            test_predictions=test_predictions,
            save_path=viz_save_path,
            channel_map=channel_map,
            feature_names_list=grid_features,
        )

    if not args.no_save_predictions:
        np.save(os.path.join(config.save_dir, "test_predictions.npy"), test_predictions)

    hexel_metrics: dict[str, float] = {}

    if isinstance(test_predictions, np.ndarray):
        if not isinstance(test_loader.dataset, MultiSourceDataset):
            raise TypeError(f"Expected MultiSourceDataset, got {type(test_loader.dataset).__name__}.")
        hexel_metrics = evaluate_and_visualize_hexels(
            test_predictions=test_predictions,
            config=config,
            out_norm=out_norm,
            device=trainer.device,
            experiment_logger=None,
            metric_functions=None if getattr(args, "tif_only", False) else trainer.metric_functions,
            stitch_mode=args.stitch_mode,
            save_artifacts=True if getattr(args, "tif_only", False) else not args.metrics_only,
            save_plots=False if getattr(args, "tif_only", False) else not args.skip_hexel_plots,
            robust_plot_percentile=args.robust_plot_percentile
            if args.robust_plot_percentile is not None
            else config.evaluation.robust_plot_percentile,
            mask_scope=args.mask_scope,
            test_metadata=test_loader.dataset.metadata,
        )

        # print metrics in terminal and log into comet
        print_and_log_eval_metrics(test_metrics=test_metrics, hexel_metrics=hexel_metrics, experiment_logger=trainer.logger)

        # persist metrics to disk as a single-row CSV so multi-run results can be
        # aggregated later (see src/aggregate_multirun_results.py)
        test_metrics_row: dict[str, float | int | str] = {
            "run_id": args.run_id if args.run_id is not None else "",
            "seed": config.seed,
            "save_dir": config.save_dir,
        }
        test_metrics_row.update({f"test_patch_{k}": v for k, v in (test_metrics if isinstance(test_metrics, dict) else {}).items()})
        test_metrics_row.update({f"test_hexel/{k}": v for k, v in hexel_metrics.items()})

        os.makedirs(config.save_dir, exist_ok=True)
        pd.DataFrame([test_metrics_row]).to_csv(os.path.join(config.save_dir, "test_metrics.csv"), index=False)

    # ---------- Stop memory tracking & report ----------
    _stop_sampler.set()
    sampler_thread.join(timeout=2.0)
    peak_rss_gb = _peak_rss_bytes / 1024**3

    total_eval_time = time.time() - start_time
    print(f"=======Total Evaluation Time {round(total_eval_time, 3)}s========")
    print(f"=======Dataloader Setup Time {round(dataloader_time, 3)}s========")
    print(f"=======Trainer Init Time {round(trainer_init_time, 3)}s========")
    print(f"=======Checkpoint Load Time {round(checkpoint_time, 3)}s========")
    print(f"=======Prediction Time {round(preds_time, 3)}s========")
    print(f"=======Eval Time Excl. Dataloader Setup {round(total_eval_time - dataloader_time, 3)}s========")
    print(f"=======Peak Host RSS {round(peak_rss_gb, 3)} GB========")
    if torch.cuda.is_available():
        peak_gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        peak_gpu_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"=======Peak GPU Reserved {round(peak_gpu_reserved_gb, 3)} GB========")
        print(f"=======Peak GPU Allocated {round(peak_gpu_allocated_gb, 3)} GB========")
    elif torch.backends.mps.is_available():
        # NOTE: MPS has no reserved-vs-allocated distinction like CUDA's
        # caching allocator -- driver_allocated_memory() is the closest
        # analog to CUDA's "reserved" (total claimed from the OS/driver),
        # current_allocated_memory() is the live tensor footprint at call
        # time, not a tracked peak, so it's reported as a point-in-time
        # figure taken right after the run rather than a true running max.
        if peak_mps_driver_gb is not None:
            print(f"=======Peak MPS Driver Allocated (per-batch, pre-flush) {round(peak_mps_driver_gb, 3)} GB========")
        if hasattr(torch.mps, "driver_allocated_memory"):
            mps_driver_gb = torch.mps.driver_allocated_memory() / 1024**3
            print(f"=======MPS Driver Allocated (post-run snapshot) {round(mps_driver_gb, 3)} GB========")
        if hasattr(torch.mps, "current_allocated_memory"):
            mps_current_gb = torch.mps.current_allocated_memory() / 1024**3
            print(f"=======MPS Current Allocated {round(mps_current_gb, 3)} GB========")

    return hexel_metrics


if __name__ == "__main__":
    main()
