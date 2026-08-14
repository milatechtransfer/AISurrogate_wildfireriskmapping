"""Create a lightweight centered-crop metadata view over prepared context patches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def prepare_context_crop_view(source_dir: Path, save_dir: Path, target_crop_h: int, target_crop_w: int) -> None:
    source_dir = source_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    numpy_source = source_dir / "numpy_files"
    numpy_destination = save_dir / "numpy_files"
    if not numpy_source.is_dir():
        raise FileNotFoundError(f"Prepared patch directory not found: {numpy_source}")
    if numpy_destination.exists() and numpy_destination.resolve() != numpy_source:
        raise FileExistsError(f"Destination already contains a different numpy_files entry: {numpy_destination}")
    if not numpy_destination.exists():
        numpy_destination.symlink_to(numpy_source, target_is_directory=True)

    split_names = {"train_indices.csv", "val_indices.csv", "test_indices.csv"}
    for source_path in source_dir.iterdir():
        if not source_path.is_file() or source_path.name in split_names or source_path.name.startswith("meta_hex_"):
            continue
        destination = save_dir / source_path.name
        if not destination.exists():
            destination.symlink_to(source_path.resolve())

    for split_name in sorted(split_names):
        source_path = source_dir / split_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Source split not found: {source_path}")
        frame = pd.read_csv(source_path)
        required_columns = {"row", "col", "input_win_h", "input_win_w", "target_crop_h", "target_crop_w"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{source_path} is missing context metadata columns: {sorted(missing_columns)}")

        input_sizes = frame[["input_win_h", "input_win_w"]].drop_duplicates()
        source_crops = frame[["target_crop_h", "target_crop_w"]].drop_duplicates()
        if len(input_sizes) != 1 or len(source_crops) != 1:
            raise ValueError(f"{source_path} contains mixed patch geometry.")
        input_h, input_w = (int(value) for value in input_sizes.iloc[0])
        source_crop_h, source_crop_w = (int(value) for value in source_crops.iloc[0])
        if target_crop_h > source_crop_h or target_crop_w > source_crop_w:
            raise ValueError(
                f"Requested crop {(target_crop_h, target_crop_w)} exceeds source prediction crop {(source_crop_h, source_crop_w)}."
            )
        if (source_crop_h - target_crop_h) % 2 or (source_crop_w - target_crop_w) % 2:
            raise ValueError("Source and requested prediction crops must have the same parity.")
        if target_crop_h > input_h or target_crop_w > input_w:
            raise ValueError("Requested prediction crop exceeds input dimensions.")

        frame["row"] = frame["row"].astype(int) + (source_crop_h - target_crop_h) // 2
        frame["col"] = frame["col"].astype(int) + (source_crop_w - target_crop_w) // 2
        frame["target_crop_h"] = target_crop_h
        frame["target_crop_w"] = target_crop_w
        frame.to_csv(save_dir / split_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--target_crop_h", type=int, required=True)
    parser.add_argument("--target_crop_w", type=int, required=True)
    args = parser.parse_args()
    prepare_context_crop_view(args.source_dir, args.save_dir, args.target_crop_h, args.target_crop_w)
    print(f"Prepared centered crop view at {os.fspath(args.save_dir)}")


if __name__ == "__main__":
    main()
