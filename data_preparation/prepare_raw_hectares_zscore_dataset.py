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


def regenerate_fire_size_csv(source_df: Path, output_df: Path, params_path: Path) -> tuple[float, float]:
    """Regenerate only the z-scored fire-size CSV: reads ``source_df``, writes ``output_df``.

    Nothing else on disk is read, written, or touched (no numpy_files symlink, no copying
    other dataset files). ``params_path`` must already contain frozen mean/std from an earlier
    ``prepare_raw_hectares_zscore_dataset`` run; they are loaded and applied as-is, with no
    refitting and no training-zone resolution.
    """
    source_df = Path(source_df)
    output_df = Path(output_df)
    params_path = Path(params_path)
    if not params_path.is_file():
        raise FileNotFoundError(
            f"{params_path} does not exist. This mode only applies already-frozen z-score params; "
            "run this script with --source-root/--destination-root first to fit and save them."
        )
    processed = process_raw_fire_size_zscore_df(pd.read_csv(source_df), norm_params_path=params_path)
    output_df.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_df, index=False)
    return load_raw_fire_size_zscore_params(params_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Prepared dataset root to build a full raw-hectare z-score dataset view from.")
    parser.add_argument("--destination-root", type=Path, help="Destination root for the full dataset view (used with --source-root).")
    parser.add_argument("--source-df", type=Path, help="Input fire-size CSV to regenerate in isolation; nothing else is touched.")
    parser.add_argument("--output-df", type=Path, help="Output path for the regenerated fire-size CSV (used with --source-df).")
    parser.add_argument(
        "--params-path",
        type=Path,
        help="Path to an existing fire_size_raw_hectares_zscore_params.json with frozen mean/std to apply "
        "(required with --source-df/--output-df).",
    )
    args = parser.parse_args()

    if args.source_df or args.output_df or args.params_path:
        if args.source_root or args.destination_root:
            parser.error("--source-df/--output-df/--params-path cannot be combined with --source-root/--destination-root.")
        if not (args.source_df and args.output_df and args.params_path):
            parser.error("--source-df, --output-df, and --params-path must all be provided together.")
        mean, std = regenerate_fire_size_csv(args.source_df, args.output_df, args.params_path)
        print(f"Regenerated {args.output_df} using frozen params from {args.params_path}: mean={mean:.15g}, std={std:.15g}")
        return

    if not (args.source_root and args.destination_root):
        parser.error("Provide either --source-root and --destination-root, or --source-df, --output-df, and --params-path.")
    reused_params = (args.destination_root / RAW_FIRE_SIZE_ZSCORE_PARAMS).is_file()
    mean, std = prepare_raw_hectares_zscore_dataset(args.source_root, args.destination_root)
    mode = "reused existing" if reused_params else "fitted new"
    print(f"Prepared raw-hectare z-score dataset at {args.destination_root} ({mode} params): mean={mean:.15g}, std={std:.15g}")


if __name__ == "__main__":
    main()
