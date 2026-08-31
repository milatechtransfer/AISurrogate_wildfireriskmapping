"""Create a lightweight dataset view using z-scored raw-hectare q10/q50/q90 inputs.

If ``fire_size_raw_hectares_zscore_params.json`` already exists at ``destination_root``, its
mean/std are loaded and applied as-is (no refitting, no training-zone resolution) and only
``df_fire_fru_processed.csv`` is regenerated from it. Otherwise, statistics are fit from
``source_root``'s training zones and saved to that file.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from data_preparation.utils import (
    RAW_FIRE_SIZE_ZSCORE_PARAMS,
    load_raw_fire_size_zscore_params,
    process_raw_fire_size_zscore_df,
)

FIRE_SIZE_TABLE = "df_fire_fru_processed.csv"


def _training_firezone_ids(source_root: Path) -> set[int]:
    train_split = pd.read_csv(source_root / "train_indices.csv")
    ignition_count = pd.read_csv(source_root / "ignition_count_processed.csv")
    if "hex_id" not in train_split:
        raise ValueError(f"{source_root / 'train_indices.csv'} is missing the hex_id column.")
    required_count_columns = {"hex_id", "GRIDCODE"}
    if not required_count_columns <= set(ignition_count.columns):
        missing = sorted(required_count_columns - set(ignition_count.columns))
        raise ValueError(f"{source_root / 'ignition_count_processed.csv'} is missing columns {missing}.")

    train_hex_ids = set(pd.to_numeric(train_split["hex_id"], errors="raise").astype(int))
    train_rows = ignition_count[pd.to_numeric(ignition_count["hex_id"], errors="raise").astype(int).isin(train_hex_ids)]
    firezone_ids = set(pd.to_numeric(train_rows["GRIDCODE"], errors="raise").astype(int))
    if not firezone_ids:
        raise ValueError("No training fire-zone IDs were resolved for raw-hectare z-score fitting.")
    return firezone_ids


def prepare_raw_hectares_zscore_dataset(source_root: Path, destination_root: Path) -> tuple[float, float]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if source_root == destination_root:
        raise ValueError("The destination root must differ from the source dataset root.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")

    source_numpy = source_root / "numpy_files"
    if not source_numpy.is_dir():
        raise FileNotFoundError(f"Source patch directory does not exist: {source_numpy}")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_numpy = destination_root / "numpy_files"
    if destination_numpy.is_symlink():
        if destination_numpy.resolve() != source_numpy.resolve():
            raise RuntimeError(f"{destination_numpy} points to {destination_numpy.resolve()}, expected {source_numpy}.")
    elif destination_numpy.exists():
        raise RuntimeError(f"{destination_numpy} exists and is not a symlink; refusing to replace it.")
    else:
        destination_numpy.symlink_to(source_numpy, target_is_directory=True)

    excluded_names = {"numpy_files", FIRE_SIZE_TABLE, "fire_size_norm_params.json", RAW_FIRE_SIZE_ZSCORE_PARAMS}
    for source_path in source_root.iterdir():
        if source_path.name not in excluded_names and source_path.is_file():
            shutil.copy2(source_path, destination_root / source_path.name)

    legacy_params_path = destination_root / "fire_size_norm_params.json"
    if legacy_params_path.is_file() or legacy_params_path.is_symlink():
        legacy_params_path.unlink()
    params_path = destination_root / RAW_FIRE_SIZE_ZSCORE_PARAMS
    # If frozen params already exist at the destination, reuse them (load-and-apply) instead
    # of refitting: skip deleting the file and skip resolving training fire-zone IDs, which is
    # only needed to fit new statistics.
    reuse_existing_params = params_path.is_file()
    train_firezone_ids = None if reuse_existing_params else _training_firezone_ids(source_root)
    processed = process_raw_fire_size_zscore_df(
        pd.read_csv(source_root / FIRE_SIZE_TABLE),
        train_firezone_ids=train_firezone_ids,
        norm_params_path=params_path,
    )
    processed.to_csv(destination_root / FIRE_SIZE_TABLE, index=False)
    mean, std = load_raw_fire_size_zscore_params(params_path)

    for split_name in ("train_indices.csv", "val_indices.csv", "test_indices.csv"):
        split = pd.read_csv(destination_root / split_name)
        missing = [filename for filename in split["filename"] if not (destination_root / str(filename)).is_file()]
        if missing:
            raise FileNotFoundError(f"{split_name} references {len(missing)} missing patches; first={missing[0]}")
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()

    reused_params = (args.destination_root / RAW_FIRE_SIZE_ZSCORE_PARAMS).is_file()
    mean, std = prepare_raw_hectares_zscore_dataset(args.source_root, args.destination_root)
    mode = "reused existing" if reused_params else "fitted new"
    print(f"Prepared raw-hectare z-score dataset at {args.destination_root} ({mode} params): mean={mean:.15g}, std={std:.15g}")


if __name__ == "__main__":
    main()
