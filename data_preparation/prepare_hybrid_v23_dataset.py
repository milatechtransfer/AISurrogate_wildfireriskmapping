"""Create a lightweight native-grid dataset view for mechanistic hybrid v2.3."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.utils import process_fire_size_df

FIRE_SIZE_TABLE = "df_fire_fru_processed.csv"
FIRE_SIZE_GLOBAL_FILL_TABLE = "fire_size_global_fill.csv"
FIRE_SIZE_STATS = "fire_size_log_stats.json"
EXCLUDED_GRIDCODES = {36}
QUANTILES = (0.1, 0.5, 0.9)


def _training_firezone_ids(source_root: Path) -> set[int]:
    train_split = pd.read_csv(source_root / "train_indices.csv")
    ignition_count = pd.read_csv(source_root / "ignition_count_processed.csv")
    required_split_columns = {"hex_id"}
    required_count_columns = {"hex_id", "GRIDCODE"}
    if not required_split_columns <= set(train_split.columns):
        raise ValueError(
            f"{source_root / 'train_indices.csv'} is missing columns {sorted(required_split_columns - set(train_split.columns))}."
        )
    if not required_count_columns <= set(ignition_count.columns):
        raise ValueError(
            f"{source_root / 'ignition_count_processed.csv'} is missing columns "
            f"{sorted(required_count_columns - set(ignition_count.columns))}."
        )
    train_hex_ids = set(pd.to_numeric(train_split["hex_id"], errors="raise").astype(int))
    train_rows = ignition_count[pd.to_numeric(ignition_count["hex_id"], errors="raise").astype(int).isin(train_hex_ids)]
    firezone_ids = set(pd.to_numeric(train_rows["GRIDCODE"], errors="raise").astype(int)) - EXCLUDED_GRIDCODES
    if not firezone_ids:
        raise ValueError("No training fire-zone IDs were resolved for the v2.3 fire-size reference.")
    return firezone_ids


def build_v23_fire_size_artifacts(source_root: Path, destination_root: Path) -> dict[str, object]:
    source_table = pd.read_csv(source_root / FIRE_SIZE_TABLE)
    processed = process_fire_size_df(
        source_table,
        normalization="none",
        add_synthetic_zone_36=False,
        excluded_gridcodes=EXCLUDED_GRIDCODES,
    )
    processed = processed.reset_index(drop=True)
    train_firezone_ids = _training_firezone_ids(source_root)
    training_rows = processed[processed["GRIDCODE"].astype(int).isin(train_firezone_ids)].copy()
    if training_rows.empty:
        raise ValueError("No fire-size observations match the v2.3 training fire zones.")

    zone_quantiles = training_rows.groupby("GRIDCODE")["LOG_SIZE_HA"].quantile(QUANTILES).unstack()
    if zone_quantiles.empty or zone_quantiles.isna().any().any():
        raise ValueError("Could not compute complete q10/q50/q90 fire-size values for the training zones.")
    reference_values = zone_quantiles.to_numpy(dtype=np.float64).reshape(-1)
    neural_mean = float(reference_values.mean())
    neural_std = float(reference_values.std(ddof=0))
    if not np.isfinite(neural_mean) or not np.isfinite(neural_std) or neural_std <= 0.0:
        raise ValueError(f"Invalid v2.3 fire-size neural statistics: mean={neural_mean}, std={neural_std}.")

    global_quantiles = training_rows["LOG_SIZE_HA"].quantile(QUANTILES)
    stats: dict[str, object] = {
        "contract": "log10_1p_hectares",
        "feature_name": "LOG_SIZE_HA",
        "quantiles": list(QUANTILES),
        "reference_weighting": "equal_training_zone_quantiles",
        "excluded_gridcodes": sorted(EXCLUDED_GRIDCODES),
        "training_firezone_count": int(len(zone_quantiles)),
        "neural_mean": neural_mean,
        "neural_std": neural_std,
        "global_fill_log_size_ha": {f"q{round(quantile * 100):02d}": float(global_quantiles.loc[quantile]) for quantile in QUANTILES},
    }

    destination_root.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination_root / FIRE_SIZE_TABLE, index=False)
    training_rows.to_csv(destination_root / FIRE_SIZE_GLOBAL_FILL_TABLE, index=False)
    (destination_root / FIRE_SIZE_STATS).write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def prepare_v23_dataset(source_root: Path, destination_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if source_root == destination_root:
        raise ValueError("The v2.3 destination root must differ from the immutable v2.2 source root.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")

    destination_root.mkdir(parents=True, exist_ok=True)
    source_numpy = source_root / "numpy_files"
    destination_numpy = destination_root / "numpy_files"
    if not source_numpy.is_dir():
        raise FileNotFoundError(f"Source patch directory does not exist: {source_numpy}")
    if destination_numpy.is_symlink():
        if destination_numpy.resolve() != source_numpy.resolve():
            raise RuntimeError(f"{destination_numpy} points to {destination_numpy.resolve()}, expected {source_numpy}.")
    elif destination_numpy.exists():
        raise RuntimeError(f"{destination_numpy} exists and is not a symlink; refusing to replace it.")
    else:
        destination_numpy.symlink_to(source_numpy, target_is_directory=True)

    excluded_names = {"numpy_files", FIRE_SIZE_TABLE, "fire_size_norm_params.json", FIRE_SIZE_GLOBAL_FILL_TABLE, FIRE_SIZE_STATS}
    for source_path in source_root.iterdir():
        if source_path.name in excluded_names:
            continue
        if source_path.is_file():
            shutil.copy2(source_path, destination_root / source_path.name)

    legacy_norm = destination_root / "fire_size_norm_params.json"
    if legacy_norm.is_file() or legacy_norm.is_symlink():
        legacy_norm.unlink()

    stats = build_v23_fire_size_artifacts(source_root, destination_root)
    for split_name in ("train_indices.csv", "val_indices.csv", "test_indices.csv"):
        split = pd.read_csv(destination_root / split_name)
        missing = [filename for filename in split["filename"] if not (destination_root / str(filename)).is_file()]
        if missing:
            raise FileNotFoundError(f"{split_name} references {len(missing)} missing patches; first={missing[0]}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()

    stats = prepare_v23_dataset(args.source_root, args.destination_root)
    print(f"Prepared hybrid v2.3 dataset at {args.destination_root}: mean={stats['neural_mean']:.15g}, std={stats['neural_std']:.15g}")


if __name__ == "__main__":
    main()
