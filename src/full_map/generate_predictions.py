"""
Generates model predictions for *every* hexel in the prepared dataset (train + val + test
splits combined), as the first step of the full-Canada map vis tool.

For each split, this:
  * runs inference over the full split (no shuffling),
  * stitches patch-level predictions into per-hexel rasters and writes them to disk as
    ``.tif`` (no diagnostic plots, no Comet logging), and
  * writes one wide-format ``hexel_metrics.csv`` per split (one row per hexel, metric names as
    columns), so per-hexel accuracy can be inspected without touching Comet.

The resulting ``<split>/predicted_hexels/hexel_{hex_id}[_target]_predicted.tif`` rasters are the
inputs consumed by ``generate_full_hexel_map.py``'s real-CRS mosaicking step.
"""

import argparse
import os
import re

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data_preparation.paths import MASK_SCOPE_CHOICES
from src.config import Config, GridParams
from src.datasets.dataset import MultiSourceDataset, build_dataset
from src.datasets.postprocessing.utils import evaluate_and_visualize_hexels
from src.datasets.utils import get_dataset_dimensions
from src.trainer import Trainer
from src.utils import seed_everything, seed_worker

# hexel_metrics keys look like "hex12/mse", "hex12/bp_mse", "buffer_mse", or the aggregate
# "all/mse" (skipped here since we only want per-hexel rows).
_HEXEL_METRIC_KEY_RE = re.compile(r"^hex(?P<hex_id>\w+)/(?P<metric>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predicted hexel rasters + per-hexel metrics for all splits (train/val/test).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
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
        default=None,
        help="Mask scope for stitched prediction artifacts. Defaults to config.data_prep.mask_scope.",
    )
    parser.add_argument(
        "--report_firezone_metrics",
        action="store_true",
        help="Also break down hexel-level metrics by firezone ID. Defaults to config.evaluation.report_firezone_metrics.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="SLURM array task ID (or run index) used to derive the run-specific seed/save_dir, matching training.",
    )
    return parser.parse_args()


def load_config(path: str) -> Config:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def reshape_hexel_metrics_to_wide(hexel_metrics: dict[str, float]) -> pd.DataFrame:
    """Reshape the flat ``hex{id}/{metric}`` -> value dict returned by
    ``evaluate_and_visualize_hexels`` into a wide per-hexel DataFrame: one row per hex_id,
    metric names as columns. The aggregate ``all/...`` keys are dropped since those summarize
    across hexels rather than describing one.
    """
    per_hex: dict[str, dict[str, float]] = {}
    for key, value in hexel_metrics.items():
        match = _HEXEL_METRIC_KEY_RE.match(key)
        if match is None:
            continue
        hex_id = match.group("hex_id")
        metric = match.group("metric")
        per_hex.setdefault(hex_id, {})[metric] = value

    if not per_hex:
        return pd.DataFrame(columns=["hex_id"])

    rows = [{"hex_id": hex_id, **metrics} for hex_id, metrics in per_hex.items()]
    df = pd.DataFrame(rows)
    # keep a stable column order: hex_id first, then metrics sorted for readability
    metric_cols = sorted(c for c in df.columns if c != "hex_id")
    return df[["hex_id", *metric_cols]]


def build_split_dataloader(config: Config, csv_name: str) -> DataLoader:
    """Builds a non-shuffled DataLoader for an arbitrary split csv (mirrors
    ``src.datasets.dataset.get_test_dataloader`` but works for any of train/val/test)."""
    dataset = build_dataset(config.data, csv_name=csv_name, modelling_approach=config.modelling_approach)
    g = torch.Generator()
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )


def main(*, args: argparse.Namespace | None = None, config: Config | None = None) -> None:
    args = args or parse_args()
    config = config or load_config(args.config)

    if args.run_id is not None:
        from src.config import apply_run_id_overrides

        run_seed = apply_run_id_overrides(config, args.run_id)
        print(f"[run_id={args.run_id}] Overriding seed={run_seed}, save_dir={config.save_dir}")

    if getattr(args, "report_firezone_metrics", False):
        config.evaluation.report_firezone_metrics = True

    # Full-map predictions are never sent to Comet, regardless of what the config says.
    config.logger.enabled = False

    seed = getattr(config, "seed", 42)
    deterministic = getattr(config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    source_map = {s.name: s for s in config.data.input_sources}
    grid_source = source_map.get("grid")
    out_norm = "min_max"
    if grid_source is not None and isinstance(grid_source.params, GridParams):
        out_norm = grid_source.params.out_norm

    splits = [
        ("train", config.data.train_split),
        ("val", config.data.val_split),
        ("test", config.data.test_split),
    ]

    # Build one loader up-front only to size the model; splits are hexel-disjoint so any of
    # them is representative of the dataset's channel/dimension layout.
    sizing_loader = build_split_dataloader(config, config.data.train_split)
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(sizing_loader.dataset)
    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Auxiliary={auxiliary_input_dims}")

    trainer = Trainer(config, spatial_input_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims)
    print(f"[Checkpoint] Loading {config.evaluation.checkpoint_filename} for full-map generation...")
    trainer.load_model(filename=config.evaluation.checkpoint_filename)

    mask_scope = args.mask_scope or config.data_prep.mask_scope or None

    for split_name, split_csv in splits:
        print(f"\n[FullMap] Generating predictions for split={split_name!r} ({split_csv})...")
        loader = build_split_dataloader(config, split_csv)
        if not isinstance(loader.dataset, MultiSourceDataset):
            raise TypeError(f"Expected MultiSourceDataset, got {type(loader.dataset).__name__}.")
        _, split_predictions = trainer.test(loader, return_predictions=True)

        hexel_metrics = evaluate_and_visualize_hexels(
            test_predictions=split_predictions,
            config=config,
            out_norm=out_norm,
            device=trainer.device,
            experiment_logger=None,
            metric_functions=trainer.metric_functions,
            stitch_mode=args.stitch_mode,
            save_artifacts=True,
            save_plots=False,
            mask_scope=mask_scope,
            split_csv=split_csv,
            save_dir_suffix=split_name,
            test_metadata=loader.dataset.metadata,
        )

        split_save_dir = os.path.join(config.save_dir, split_name)
        os.makedirs(split_save_dir, exist_ok=True)
        wide_metrics_df = reshape_hexel_metrics_to_wide(hexel_metrics)
        metrics_path = os.path.join(split_save_dir, "hexel_metrics.csv")
        wide_metrics_df.to_csv(metrics_path, index=False)
        print(f"[FullMap] Wrote {len(wide_metrics_df)} hexel metric rows to {metrics_path}")


if __name__ == "__main__":
    main()
