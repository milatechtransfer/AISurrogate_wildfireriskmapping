import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data_preparation.paths import Paths

feature_names = ["fuel_grid", "elevation_grid", "ignition_grid", "firezones_grid", "bp_out_grid", "fi_out_grid", "ros_out_grid"]
FIRE_SIZE_FEATURE_COLS = ["GRIDCODE", "SIZE_HA"]


def find_file_path(filename: str, *search_dirs: Path) -> Path:
    """Searches for a file in multiple directories and returns the path if found."""
    for d in search_dirs:
        p = Path(d) / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename} in {', '.join(str(d) for d in search_dirs)}")


def process_fire_size_df(df_fire_size: pd.DataFrame) -> pd.DataFrame:
    # Validate that required columns are present before selecting them
    missing_cols = [col for col in FIRE_SIZE_FEATURE_COLS if col not in df_fire_size.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s) in fire size DataFrame: {missing_cols}. "
            f"Expected columns: {FIRE_SIZE_FEATURE_COLS}. "
            f"Available columns: {list(df_fire_size.columns)}"
        )
    df_fire_size = df_fire_size[FIRE_SIZE_FEATURE_COLS]  # Remove unnamed column
    zone_36 = {"GRIDCODE": 36, "SIZE_HA": 0}  # Consulted with experts and concluded that imputing with 0 is most reasonable
    if 36 not in df_fire_size["GRIDCODE"].values:
        df_fire_size = pd.concat(
            [df_fire_size, pd.DataFrame([zone_36])],
            ignore_index=True,
        )  # Only append synthetic zone 36 if it is not already present, and avoid duplicate indices

    df_fire_size["LOG_SIZE_HA"] = np.log10(df_fire_size["SIZE_HA"] + 1)
    min_val = df_fire_size["LOG_SIZE_HA"].min()
    max_val = df_fire_size["LOG_SIZE_HA"].max()
    if max_val == min_val:
        # Avoid division by zero when all LOG_SIZE_HA values are identical
        df_fire_size["NORM_LOG_SIZE_HA"] = 0.0
    else:
        df_fire_size["NORM_LOG_SIZE_HA"] = (df_fire_size["LOG_SIZE_HA"] - min_val) / ((max_val - min_val) + 1e-5)
    return df_fire_size


def aggregate_csv_by_pattern(root_dir: Path, pattern: str, load_function: Callable | None = None) -> pd.DataFrame:
    """
    Orchestrates the finding, loading, merging files of a certain pattern across all hex folders
    """
    # 1. Locate all files to load using specific pattern
    files = sorted(root_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern '{pattern}' in {root_dir}...")

    # 2. Load (with option to use feature specific loader function) and stack all dataframes
    data_frames = []
    for f in files:
        df = load_function(f) if load_function else pd.read_csv(f)
        if df is not None:
            data_frames.append(df)

    full_df = pd.concat(data_frames, ignore_index=True)

    return full_df


def read_ids_from_csv(csv_path: str | Path, column_name: str = "Description") -> list[int]:
    """
    Read a CSV file and return the column as a list of ints (for seasons and causes csvs)
    """
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise ValueError(f"Missing {column_name} column in {csv_path}. " f"Found columns: {list(df.columns)}")

    return df[column_name].dropna().astype(int).tolist()


def find_hex_ids(root_dir: str) -> list:
    hex_ids = []
    try:
        with os.scandir(root_dir) as entries:
            for entry in entries:
                # Check if it's a directory AND starts with 'hex'
                if entry.is_dir() and entry.name.startswith("hex"):
                    hex_ids.append(entry.name[3:])
    except FileNotFoundError:
        print(f"Directory not found: {root_dir}")
        return []

    return hex_ids


def get_min_max_hex_prob_df(root_dir: str) -> list:
    """create a list of burn prob dist for stratified sampling"""
    from data_preparation.spatial import load_spatial_raster

    all_hex_ids = find_hex_ids(root_dir)
    hex_min_max_bp = []
    for hex_id in all_hex_ids:
        all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
        out_grid, _ = load_spatial_raster(all_paths.output_burn_prob())
        out_grid_ravel = out_grid.ravel()
        area_burnt = len(out_grid_ravel[out_grid_ravel != 0.0]) / len(out_grid_ravel)
        hex_min_max_bp.append([hex_id, np.nanmin(out_grid), np.min(out_grid[out_grid != 0.0]), np.nanmax(out_grid), area_burnt])
    return hex_min_max_bp


def get_stratified_data_split(data_dir: str):
    """Stratified sampling for the valid data split"""
    hex_min_max_bp = get_min_max_hex_prob_df(data_dir)
    df_min_max = pd.DataFrame(hex_min_max_bp, columns=["hex_id", "min_prob", "min_except_0", "max_prob", "area_burnt"])
    df_min_max["area_bin"] = pd.qcut(df_min_max["area_burnt"], q=2, labels=["LowArea", "HighArea"])
    df_min_max["max_bin"] = pd.qcut(df_min_max["max_prob"], q=2, labels=["LowMax", "HighMax"])

    df_min_max["strat_key"] = df_min_max["area_bin"].astype(str) + "_" + df_min_max["max_bin"].astype(str)

    train_val, test = train_test_split(df_min_max, test_size=5, stratify=df_min_max["strat_key"], random_state=42)
    train, val = train_test_split(train_val, test_size=5, stratify=train_val["strat_key"], random_state=42)

    print(f"Total: {len(df_min_max)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"List of val ids {list(val["hex_id"])}")
    print(f"List of test ids {list(test["hex_id"])}")


def get_processed_hex_ids(folder_path: str) -> list:
    """
    Finds all hex_ids from the metadata df files
    """
    folder = Path(folder_path)
    hex_ids = []
    for file_path in folder.glob("meta_hex_*.csv"):
        filename_no_ext = file_path.stem
        extracted_id = filename_no_ext.removeprefix("meta_hex_")

        hex_ids.append(extracted_id)
    return hex_ids


def plot_split_window_hexel(windows, channel_index=0, max_cols=5, figsize=(15, 15)):
    """
    Plots a list/array of 3D windows in a subplot grid.

    Args:
        windows (list or np.ndarray): List of windows. Shape (N, H, W, C).
        channel_index (int): The channel to visualize (e.g., 0 for Red/Band1).
        max_cols (int): Maximum number of columns in the grid.
        figsize (tuple): Figure size (width, height).
    """
    num_windows = len(windows)

    if num_windows == 0:
        print("No windows to plot.")
        return

    # Calculate grid dimensions
    num_cols = min(num_windows, max_cols)
    num_rows = math.ceil(num_windows / num_cols)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Flatten axes for easy iteration (handle case where axes is not a list)
    axes = [axes] if num_windows == 1 else axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]

        if i < num_windows:
            window = windows[i]

            # Extract specific channel
            if window.ndim == 3:  # noqa: SIM108
                # Shape (H, W, C) -> Extract channel
                img_data = window[:, :, channel_index]
            else:
                # Fallback if window is already 2D
                img_data = window

            # Plot
            im = ax.imshow(img_data, cmap="gray")  # noqa: F841
            ax.set_title(f"Window {i}")

        # Hide axis ticks for all subplots (cleaner look)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
