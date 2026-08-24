import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.hexel_loader import FUEL_GRID_CHOICES, IGNITION_WEIGHTING_CHOICES, load_spatial_features_per_hexel
from data_preparation.paths import MASK_SCOPE_CHOICES, prepared_mask_scope
from data_preparation.spatial import NODATA
from data_preparation.utils import find_hex_ids, get_processed_hex_ids
from src.datasets.context_crop import centered_crop_slices


def save_split_hexel_windows(
    valid_window: np.ndarray, out_dir: str, win_id: int, season: str, cause: str, hex_id: str, format: str = "npy"
) -> str:
    """Save a hexel window"""
    filename = f"numpy_files/hex_{hex_id}_{str(win_id)}_{season}_{cause}.{format}"
    numpy_file_path = os.path.join(out_dir, filename)
    Path(numpy_file_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(numpy_file_path, valid_window) if format == "npy" else np.savez_compressed(numpy_file_path, arr=valid_window)
    return filename


def get_split_hexel_window(
    season_cause_stacked_feats: np.ndarray,
    season_cause_mask: np.ndarray,
    season_cause_mapping: dict | None,
    out_dir: str,
    root_dir: str,
    hex_id: str,
    win_h: int = 128,
    win_w: int = 128,
    target_crop_h: int | None = None,
    target_crop_w: int | None = None,
    overlap_ratio: float = 0.2,
    mask_scope: str = "actual",
):
    """
    Split the hexel using sliding windows for inp to the model
    Inputs:
        season_cause_stacked_feats(np.ndarray): Stacked array of all the features of the shape (num_season_cause, H, W, num_feats)
        season_cause_mask(np.ndarray): A bool array where True means to ignore the pixel (num_season_cause, H,W)
    """
    scope = prepared_mask_scope(mask_scope)
    print("============Splitting the hexel==================")
    num_season_cause, H, W, _ = season_cause_stacked_feats.shape
    if (target_crop_h is None) != (target_crop_w is None):
        raise ValueError("target_crop_h and target_crop_w must either both be set or both be omitted.")
    crop_h = win_h if target_crop_h is None else target_crop_h
    crop_w = win_w if target_crop_w is None else target_crop_w
    crop_rows, crop_cols = centered_crop_slices(win_h, win_w, crop_h, crop_w)
    context_h = crop_rows.start or 0
    context_w = crop_cols.start or 0
    stride_h = max(1, int(crop_h * (1 - overlap_ratio)))
    stride_w = max(1, int(crop_w * (1 - overlap_ratio)))

    def trailing_padding(length: int, crop_size: int, stride: int) -> int:
        if length <= crop_size:
            return crop_size - length
        return (stride - (length - crop_size) % stride) % stride

    target_pad_h = trailing_padding(H, crop_h, stride_h)
    target_pad_w = trailing_padding(W, crop_w, stride_w)
    season_cause_stacked_feats_padded = np.pad(
        season_cause_stacked_feats,
        ((0, 0), (context_h, context_h + target_pad_h), (context_w, context_w + target_pad_w), (0, 0)),
        mode="constant",
        constant_values=NODATA,
    )
    season_cause_mask_padded = np.pad(
        season_cause_mask,
        ((0, 0), (context_h, context_h + target_pad_h), (context_w, context_w + target_pad_w)),
        mode="constant",
        constant_values=1.0,
    )

    window_area = crop_h * crop_w
    num_total_windows, num_valid_windows = 0.0, 0.0
    valid_coords = []
    for i in range(num_season_cause):
        stacked_feats = season_cause_stacked_feats_padded[i]
        mask = season_cause_mask_padded[i]
        if season_cause_mapping is None:
            season, cause = "all", "all"
        else:
            season, cause = season_cause_mapping[i]
        for row in range(0, H + target_pad_h - crop_h + 1, stride_h):
            for col in range(0, W + target_pad_w - crop_w + 1, stride_w):
                num_total_windows += 1
                mask_window = mask[
                    row + context_h : row + context_h + crop_h,
                    col + context_w : col + context_w + crop_w,
                ]
                true_count = np.count_nonzero(~mask_window)
                valid_ratio = true_count / window_area

                num_valid_windows += 1
                window_data = stacked_feats[row : row + win_h, col : col + win_w, :]
                filename = save_split_hexel_windows(window_data, out_dir, int(num_valid_windows), season, cause, hex_id)
                valid_coords.append(
                    [
                        str(Path(filename)),
                        season,
                        cause,
                        str(hex_id),
                        num_valid_windows,
                        row,
                        col,
                        valid_ratio,
                        scope,
                        win_h,
                        win_w,
                        crop_h,
                        crop_w,
                    ]
                )
    df_coords = pd.DataFrame(valid_coords)
    df_coords.columns = [
        "filename",
        "season",
        "cause",
        "hex_id",
        "window_id",
        "row",
        "col",
        "valid_ratio",
        "mask_scope",
        "input_win_h",
        "input_win_w",
        "target_crop_h",
        "target_crop_w",
    ]
    df_coords.to_csv(os.path.join(out_dir, f"meta_hex_{hex_id}.csv"), index=False)
    print(f"=====Hexel data Saved at {out_dir} ========")


def generate_data_samples(
    root_dir: str,
    modelling_approach: int,
    save_dir: str | None,
    win_h: int = 128,
    win_w: int = 128,
    target_crop_h: int | None = None,
    target_crop_w: int | None = None,
    overlap_ratio: float = 0.2,
    is_array_job: bool = False,
    task_id: int = 0,
    num_tasks: int = 1,
    mask_scope: str = "actual",
    ignition_weighting: str = "distribution",
    fuel_representation: str = "raw",
    overwrite: bool = False,
):
    scope = prepared_mask_scope(mask_scope)
    if save_dir:
        out_dir = save_dir
    else:
        suffix = "" if scope == "actual" else f"_{scope}"
        out_dir = os.path.join(root_dir, f"data_samples_approach_{modelling_approach}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "numpy_files"), exist_ok=True)
    completed_hex_ids = [] if overwrite else get_processed_hex_ids(out_dir)

    if is_array_job:
        # 1. Get all Hex IDs
        hex_ids = find_hex_ids(root_dir)
        # Sort them to ensure every worker sees the same order
        hex_ids = sorted(list(hex_ids))

        # 2. Split the work
        if num_tasks > 1:
            # Python list slicing magic: start at task_id, take every Nth item
            my_hexels = hex_ids[task_id::num_tasks]
            print(f"[Worker {task_id}/{num_tasks}] Processing {len(my_hexels)} hexels out of {len(hex_ids)} total.")
            hex_ids = my_hexels
    else:
        hex_ids = find_hex_ids(root_dir)

    for hex_id in hex_ids:
        if hex_id in completed_hex_ids:
            print(f"==========Skipping because completed hex{hex_id}=============")
            continue
        print(f"======Working on Hex ID: {hex_id}==========")
        stacked_feats, mask, season_cause_mapping = load_spatial_features_per_hexel(
            root_dir=root_dir,
            hex_id=hex_id,
            feature_channel_map_path=os.path.join(out_dir, f"feature_channel_map_{modelling_approach}.json"),
            modelling_approach=modelling_approach,
            mask_scope=scope,
            ignition_weighting=ignition_weighting,
            fuel_representation=fuel_representation,
        )
        if (stacked_feats is None) or (mask is None):
            print(f"================Failed for hex {hex_id}===================")
            continue
        get_split_hexel_window(
            season_cause_stacked_feats=stacked_feats,
            season_cause_mask=mask,
            season_cause_mapping=season_cause_mapping,
            out_dir=out_dir,
            root_dir=root_dir,
            hex_id=hex_id,
            win_h=win_h,
            win_w=win_w,
            target_crop_h=target_crop_h,
            target_crop_w=target_crop_w,
            overlap_ratio=overlap_ratio,
            mask_scope=scope,
        )
        print(f"======Processed Hex ID: {hex_id}==========")


def main():
    parser = argparse.ArgumentParser(description="Generate data samples from each hexel")

    parser.add_argument("--root_dir", type=str, help="data root directory", required=True)
    parser.add_argument("--save_dir", type=str, help="save data directory", default=None)
    parser.add_argument("--modelling_approach", type=int, help="Either 1 or 2", default=2)
    parser.add_argument("--win_h", type=int, help="Height of the window", default=128)
    parser.add_argument("--win_w", type=int, help="Height of the window", default=128)
    parser.add_argument("--target_crop_h", type=int, default=None, help="Height of the centered prediction crop.")
    parser.add_argument("--target_crop_w", type=int, default=None, help="Width of the centered prediction crop.")
    parser.add_argument("--overlap_ratio", type=float, help="Overlap ratio between windows", default=0.2)
    parser.add_argument("--is_array_job", action="store_true", help="Boolean to indicate if using SLURM job array")
    parser.add_argument("--task_id", type=int, default=0, help="SLURM array ID")
    parser.add_argument("--num_tasks", type=int, default=1, help="Total number of array tasks")
    parser.add_argument("--mask_scope", choices=MASK_SCOPE_CHOICES, default="actual", help="Mask scope for generated patch rasters.")
    parser.add_argument(
        "--ignition_weighting",
        choices=IGNITION_WEIGHTING_CHOICES,
        default="distribution",
        help="'distribution' (default) for zone-area-weighted 2-channel ignition or 'max' for the original max-aggregation (1 channel).",
    )
    parser.add_argument(
        "--fuel_grid_representation",
        choices=FUEL_GRID_CHOICES,
        default="raw",
        help="'raw' (default) for raw fuel class values (use with iROS curves) or 'group' to group similar classes for one-hot encoding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess every hex even if its meta_hex_*.csv already exists (overwrites patches in place).",
    )
    args = parser.parse_args()

    generate_data_samples(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        modelling_approach=args.modelling_approach,
        win_h=args.win_h,
        win_w=args.win_w,
        target_crop_h=args.target_crop_h,
        target_crop_w=args.target_crop_w,
        overlap_ratio=args.overlap_ratio,
        is_array_job=args.is_array_job,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
        mask_scope=args.mask_scope,
        ignition_weighting=args.ignition_weighting,
        fuel_representation=args.fuel_grid_representation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
