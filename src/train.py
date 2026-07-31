"""
End-to-end script for running training and evaluation
"""

import argparse
import os
import signal
from types import FrameType

import numpy as np
import pandas as pd
import yaml

from src.config import Config, GridParams, apply_run_id_overrides
from src.datasets.dataset import get_test_dataloader, get_train_val_dataloader
from src.datasets.postprocessing.utils import evaluate_and_visualize_hexels, print_and_log_eval_metrics
from src.datasets.utils import get_dataset_dimensions
from src.trainer import Trainer
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate UNet baseline.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/multi_output_spatial_weather.yaml",
        help="Path to YAML config file.",
    )
    # logging is enabled by default, unless you pass --no_log_test_predicted_hexels
    parser.add_argument(
        "--no_log_test_predicted_hexels",
        dest="log_test_predicted_hexels",
        action="store_false",
        default=True,
        help="Disable saving predicted hexels to comet (default: True)",
    )
    # logging is enabled by default, unless you pass --no_log_val_predicted_hexels
    parser.add_argument(
        "--no_log_val_predicted_hexels",
        dest="log_val_predicted_hexels",
        action="store_false",
        default=True,
        help="Disable saving predicted hexels to comet (default: True)",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="SLURM array task ID (or run index) used to derive a run-specific seed, save_dir, and "
        "Comet experiment name for parallel multi-seed runs (see run_files/train_no_tmp_copy_array.sh).",
    )
    return parser.parse_args()


def _handle_sigterm(signum: int, _frame: FrameType | None) -> None:
    """
    Log receipt of SLURM's pre-timeout SIGTERM (see `--signal=B:TERM@300` in
    run_files/train_no_tmp_copy.sh). The Trainer already checkpoints at each epoch
    boundary and `--requeue` causes SLURM to resubmit the job, so no extra cleanup
    is required here beyond a clear log message before the process is killed.
    """
    print(f"[Signal] Received {signal.Signals(signum).name}; job is being preempted/timed out and will be requeued.")


def load_config(path: str) -> Config:
    """
    load yaml config
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)

    args = parse_args()
    config = load_config(args.config)

    if args.run_id is not None:
        run_seed = apply_run_id_overrides(config, args.run_id)
        print(f"[run_id={args.run_id}] Overriding seed={run_seed}, save_dir={config.save_dir}")

    # ---------- Set Seed ----------
    seed = getattr(config, "seed", 42)
    deterministic = getattr(config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    # ---------- Data ----------
    train_loader, val_loader = get_train_val_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=seed)

    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(train_loader.dataset)
    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Auxiliary={auxiliary_input_dims}")

    # ---------- Training ----------
    trainer = Trainer(
        config=config,
        spatial_input_channels=spatial_channels,
        auxiliary_input_dims=auxiliary_input_dims,
        train_dataset=train_loader.dataset,
    )

    trainer.run_training(
        train_loader=train_loader,
        val_loader=val_loader,
    )
    # ---------- Load best checkpoint ----------
    # Try best.pth first, fall back to last.pth if needed
    best_ckpt = None
    try:
        print("\n[Checkpoint] Loading best.pth for evaluation...")
        best_ckpt = trainer.load_model(filename="best.pth")
    except FileNotFoundError:
        print("[Checkpoint] best.pth not found, falling back to last.pth...")
        try:
            best_ckpt = trainer.load_model(filename="last.pth")
        except FileNotFoundError:
            print("[Checkpoint] No checkpoint found (last.pth missing)...")
            return

    if best_ckpt is not None:
        print(f"[Checkpoint] Loaded epoch={best_ckpt.get('epoch', 'N/A')} Checkpoint Metrics={best_ckpt.get('metric_value', 'N/A')}")

    # ---------- Evaluation ----------
    print("\n[Evaluation] Running on test set...")
    test_loader = get_test_dataloader(
        config=config.data,
        modelling_approach=config.modelling_approach,
        seed=seed,
    )
    test_metrics, test_predictions = trainer.test(test_loader, return_predictions=True)

    hexel_metrics = {}

    if args.log_test_predicted_hexels:
        source_map = {s.name: s for s in config.data.input_sources}
        grid_source = source_map.get("grid") if "grid" in source_map else None
        out_norm = "min_max"  # default fallback, prevent mypy crash
        if grid_source and isinstance(grid_source.params, GridParams):
            out_norm = grid_source.params.out_norm

        if isinstance(test_predictions, np.ndarray):  # for mypy
            hexel_metrics = evaluate_and_visualize_hexels(
                test_predictions=test_predictions,
                config=config,
                out_norm=out_norm,
                device=trainer.device,
                experiment_logger=trainer.logger,
                metric_functions=trainer.metric_functions,
            )

    # print metrics in terminal and log into comet
    print_and_log_eval_metrics(test_metrics=test_metrics, hexel_metrics=hexel_metrics, experiment_logger=trainer.logger)

    # persist test (best checkpoint) metrics to disk as a single-row CSV so multi-run
    # results can be aggregated later (mirrors src/evaluate_hexels.py's test_metrics.csv)
    test_metrics_row: dict[str, float | int | str] = {
        "run_id": args.run_id if args.run_id is not None else "",
        "seed": seed,
        "save_dir": config.save_dir,
    }
    test_metrics_row.update({f"test_patch_{k}": v for k, v in (test_metrics if isinstance(test_metrics, dict) else {}).items()})
    test_metrics_row.update({f"test_hexel/{k}": v for k, v in hexel_metrics.items()})

    os.makedirs(config.save_dir, exist_ok=True)
    pd.DataFrame([test_metrics_row]).to_csv(os.path.join(config.save_dir, "test_metrics.csv"), index=False)

    # ---------- Validation Evaluation (optional) ----------
    if args.log_val_predicted_hexels:
        print("\n[Evaluation] Running on validation set...")
        val_metrics, val_predictions = trainer.test(val_loader, return_predictions=True)

        val_hexel_metrics = {}

        source_map = {s.name: s for s in config.data.input_sources}
        grid_source = source_map.get("grid") if "grid" in source_map else None
        out_norm = "min_max"  # default fallback, prevent mypy crash
        if grid_source and isinstance(grid_source.params, GridParams):
            out_norm = grid_source.params.out_norm

        if isinstance(val_predictions, np.ndarray):  # for mypy
            val_hexel_metrics = evaluate_and_visualize_hexels(
                test_predictions=val_predictions,
                config=config,
                out_norm=out_norm,
                device=trainer.device,
                experiment_logger=trainer.logger,
                metric_functions=trainer.metric_functions,
                split_csv=config.data.val_split,
                save_dir_suffix="val",
            )

        print_and_log_eval_metrics(
            test_metrics=val_metrics,
            hexel_metrics=val_hexel_metrics,
            experiment_logger=trainer.logger,
            split_label="Val",
            metric_prefix="val_hexel",
        )

        # persist validation (best checkpoint) metrics to disk as a single-row CSV,
        # mirroring the test_metrics.csv logic above.
        val_metrics_row: dict[str, float | int | str] = {
            "run_id": args.run_id if args.run_id is not None else "",
            "seed": seed,
            "save_dir": config.save_dir,
        }
        val_metrics_row.update({f"val_patch_{k}": v for k, v in (val_metrics if isinstance(val_metrics, dict) else {}).items()})
        val_metrics_row.update({f"val_hexel/{k}": v for k, v in val_hexel_metrics.items()})

        os.makedirs(config.save_dir, exist_ok=True)
        pd.DataFrame([val_metrics_row]).to_csv(os.path.join(config.save_dir, "val_results.csv"), index=False)


if __name__ == "__main__":
    main()
