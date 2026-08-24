"""
Script for aggregating and preparing the sequential weather table
and processing the fire size distribution table.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.tabular.weather import load_weather_list, preprocess_weather_list
from data_preparation.utils import aggregate_csv_by_pattern, find_file_path, process_fire_size_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _resolve_norm_params_path(filename: str, *, search_dirs: tuple[Path, ...], default_dir: Path) -> Path | None:
    """Resolve a normalization-params file path.

    If ``filename`` is empty, returns None (normalization params are not persisted).
    Otherwise returns the first existing ``<dir>/<filename>`` across ``search_dirs`` so a
    pre-fitted file (inference/eval-only mode) is picked up regardless of whether it lives in
    ``root_dir`` or ``save_dir``. When no existing file is found, returns ``default_dir/filename``
    so scalers can be fitted and saved there (training/first-run mode).
    """
    if not filename:
        return None
    for d in search_dirs:
        candidate = Path(d) / filename
        if candidate.exists():
            return candidate
    return default_dir / filename


def _hex_id_from_weather_path(path: Path) -> str:
    return path.parent.parent.name.removeprefix("hex").zfill(2)


def _load_raw_weather_with_hex_id(path: Path) -> pd.DataFrame:
    df = load_weather_list(str(path), normalize_weatherlist=False)
    df.insert(0, "hex_id", int(_hex_id_from_weather_path(path)))
    return df


def _train_hex_ids(train_split_path: Path | None) -> set[str] | None:
    if train_split_path is None:
        return None
    if not train_split_path.exists():
        raise FileNotFoundError(
            f"Training split {train_split_path} not found. Refusing to fall back to full-table tabular "
            "preprocessing stats, which would fit scalers on held-out (val/test) rows and leak. Provide a "
            "valid --train_split_file, or pass --train_split_file='' to explicitly opt into full-table stats."
        )
    train_df = pd.read_csv(train_split_path)
    if "hex_id" not in train_df.columns:
        raise KeyError(f"Column 'hex_id' not found in training split {train_split_path}.")
    hex_ids = {str(value).zfill(2) for value in train_df["hex_id"].dropna().astype(int).astype(str)}
    if not hex_ids:
        raise ValueError(f"Training split {train_split_path} contains no hex IDs.")
    return hex_ids


def _train_firezone_ids(
    *,
    data_dir: Path,
    train_split_path: Path | None,
    modelling_approach: str,
    zone_channel_key: str = "firezones_grid",
    filename_col: str = "filename",
    valid_mask_threshold: float = 0.01,
) -> set[int] | None:
    if train_split_path is None:
        return None
    if not train_split_path.exists():
        raise FileNotFoundError(
            f"Training split {train_split_path} not found. Refusing to fall back to full-table fire-size "
            "normalization stats, which would fit scalers on held-out (val/test) rows and leak. Provide a "
            "valid --train_split_file, or pass --train_split_file='' to explicitly opt into full-table stats."
        )

    train_df = pd.read_csv(train_split_path)
    if "valid_ratio" in train_df.columns:
        train_df = train_df[train_df["valid_ratio"] > valid_mask_threshold].copy()
    if filename_col not in train_df.columns:
        raise KeyError(f"Column {filename_col!r} not found in training split {train_split_path}.")
    if train_df.empty:
        raise ValueError(f"Training split {train_split_path} has no rows after valid_ratio filtering.")

    channel_map_path = data_dir / f"feature_channel_map_{modelling_approach}.json"
    with channel_map_path.open() as handle:
        channel_map = json.load(handle)
    if zone_channel_key not in channel_map:
        raise ValueError(f"Missing zone channel {zone_channel_key!r} in {channel_map_path}.")
    zone_channel = int(channel_map[zone_channel_key][0])

    zones: set[int] = set()
    for filename in train_df[filename_col].drop_duplicates():
        patch_path = data_dir / str(filename)
        if not patch_path.exists():
            raise FileNotFoundError(f"Training patch referenced by {train_split_path} does not exist: {patch_path}")
        patch = np.load(patch_path, mmap_mode="r")
        if zone_channel >= patch.shape[2]:
            raise ValueError(f"Zone channel {zone_channel} is out of bounds for {patch_path} with shape {patch.shape}.")
        zone_grid = np.asarray(patch[:, :, zone_channel])
        finite = np.isfinite(zone_grid) & (zone_grid > 0)
        if finite.any():
            rounded = np.rint(zone_grid[finite])
            if not np.allclose(zone_grid[finite], rounded, atol=1e-3):
                bad_value = float(zone_grid[finite][np.argmax(np.abs(zone_grid[finite] - rounded))])
                raise ValueError(f"Zone channel {zone_channel_key!r} in {patch_path} contains non-integer zone id {bad_value}.")
            zones.update(int(value) for value in np.unique(rounded.astype(np.int64)))
    if not zones:
        raise ValueError(f"No positive finite fire-zone IDs found in training split {train_split_path}.")
    return zones


def build_weather_table(
    root_dir: Path,
    save_path: Path,
    pattern: str = "hex*/tabular/hex*_DailyWeather.csv",
    train_split_path: Path | None = None,
    norm_params_path: Path | None = None,
):
    """
    Aggregates weather CSVs into a single table and applies global preprocessing.

    The output table retains a ``hex_id`` column identifying which hexel each row came from. This lets
    downstream consumers (e.g. ``SpatializedTabularSource`` with ``hex_id_col`` set) aggregate weather
    features per ``(hex_id, WeatherZone)``, so a zone spanning
    multiple hexels never mixes rows across hexels that live in different train/val/test splits.

    Parameters
    ----------
    norm_params_path:
        Path to a JSON file for normalization parameters.
        - If the file exists: parameters are loaded and applied (inference mode).
        - If the file does not exist: parameters are fitted and saved for reuse.
        - If None: parameters are fitted but not saved.
    """
    df_weather = aggregate_csv_by_pattern(root_dir=root_dir, pattern=pattern, load_function=_load_raw_weather_with_hex_id)
    # When pre-fitted normalization parameters exist we are in inference/eval-only mode:
    # preprocess_weather_list loads and applies them and ignores fit_mask entirely, so the
    # train split (and the hex IDs derived from it) is unused. Skip reading it to avoid
    # coupling the weather path to a train split that may be empty or absent in eval-only runs.
    fit_mask = None
    if not (norm_params_path is not None and Path(norm_params_path).exists()):
        train_hex_ids = _train_hex_ids(train_split_path)
        if train_hex_ids is not None:
            hex_id_strs = df_weather["hex_id"].astype(int).astype(str).str.zfill(2)
            fit_mask = hex_id_strs.isin(train_hex_ids)
    df_weather = preprocess_weather_list(
        df_weather,
        fit_mask=fit_mask,
        norm_params_path=norm_params_path,
    )
    df_weather.to_csv(save_path, index=False)
    logger.info(f"Aggregated weather data saved to {save_path} with shape {df_weather.shape} and columns: {df_weather.columns.tolist()}")


def process_fire_size_distribution_table(
    input_path: Path,
    output_path: Path,
    train_firezone_ids: set[int] | None = None,
    norm_params_path: Path | None = None,
):
    """
    Processes the fire size distribution table.
    Args:
        input_path (Path): Path to the existing fire size distribution CSV file.
        output_path (Path): Path to save the processed fire size distribution CSV file.
        train_firezone_ids (set[int] | None): GRIDCODE (fire-zone) values of the training split;
            the normalization is fit only on these zones to avoid leakage. None fits on all rows.
            Ignored when ``norm_params_path`` points to an existing file.
        norm_params_path:
            Path to a JSON file for normalization parameters.
            - If the file exists: parameters are loaded and applied (inference mode).
            - If the file does not exist: parameters are fitted and saved for reuse.
            - If None: parameters are fitted but not saved.
    """
    df_fire_size = pd.read_csv(input_path)
    df_fire_size_processed = process_fire_size_df(
        df_fire_size,
        train_firezone_ids=train_firezone_ids,
        norm_params_path=norm_params_path,
    )
    df_fire_size_processed.to_csv(output_path, index=False)
    logger.info(
        f"Processed fire size distribution data saved to {output_path} with shape {df_fire_size_processed.shape} and columns: {df_fire_size_processed.columns.tolist()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compile raw weather CSVs into a single lookup table and processing fire size distribution table."
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Path to raw data root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save directory")
    parser.add_argument(
        "--weather_output_file",
        type=str,
        help="File name (ends in .csv) to be used to save the aggregated and processed weather table",
        default="weather_table.csv",
    )
    parser.add_argument(
        "--fire_size_input_file",
        type=str,
        help="File name (ends in .csv) of existing fire size distribution table",
        default="df_fire_fru.csv",
    )
    parser.add_argument(
        "--fire_size_output_file",
        type=str,
        help="File name (ends in .csv) to be used to save the processed distribution table",
        default="df_fire_fru_processed.csv",
    )
    parser.add_argument(
        "--train_split_file",
        type=str,
        default="train_indices.csv",
        help="Training split CSV in save_dir used to fit tabular preprocessing stats. Raises if the file is missing "
        "(to prevent silent full-table leakage); pass an empty string to explicitly opt into full-table stats.",
    )
    parser.add_argument("--modelling_approach", type=str, default="1", help="Modelling approach used to read feature_channel_map_<N>.json")
    parser.add_argument(
        "--weather_norm_params_file",
        type=str,
        default="weather_norm_params.json",
        help="File name for weather normalization parameters JSON in save_dir. "
        "If the file exists it is loaded (inference mode); otherwise scalers are fitted and saved.",
    )
    parser.add_argument(
        "--fire_size_norm_params_file",
        type=str,
        default="fire_size_norm_params.json",
        help="File name for fire size normalization parameters JSON in save_dir. "
        "If the file exists it is loaded (inference mode); otherwise parameters are fitted and saved.",
    )

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    weather_save_path = save_dir / args.weather_output_file
    fire_size_save_path = save_dir / args.fire_size_output_file

    weather_norm_params_path = _resolve_norm_params_path(
        args.weather_norm_params_file, search_dirs=(root_dir, save_dir), default_dir=save_dir
    )
    fire_size_norm_params_path = _resolve_norm_params_path(
        args.fire_size_norm_params_file, search_dirs=(root_dir, save_dir), default_dir=save_dir
    )

    # A norm-params file existing on disk means that source is in inference / eval-only mode:
    # its scalers are loaded from the file and the train split is not needed to fit them. Each
    # source is gated independently so a missing/empty train split can't break a source that
    # already has its parameters saved.
    weather_needs_fit = not (weather_norm_params_path is not None and weather_norm_params_path.exists())
    fire_size_needs_fit = not (fire_size_norm_params_path is not None and fire_size_norm_params_path.exists())
    any_fit_needed = weather_needs_fit or fire_size_needs_fit
    train_split_path = save_dir / args.train_split_file if (any_fit_needed and args.train_split_file) else None
    fire_size_read_path = find_file_path(
        args.fire_size_input_file, root_dir, save_dir
    )  # Allows for flexibility in where the fire size distribution file is located, since it was generated by Yan and saved in the data_sampling_approach_XYZ folder and I don't want to assume it's in the sampling folder or the raw data folder. It will look in both and use the one it finds first.

    build_weather_table(
        root_dir=root_dir,
        save_path=weather_save_path,
        train_split_path=train_split_path,
        norm_params_path=weather_norm_params_path,
    )
    train_firezone_ids = (
        _train_firezone_ids(
            data_dir=save_dir,
            train_split_path=train_split_path,
            modelling_approach=str(args.modelling_approach),
        )
        if fire_size_needs_fit
        else None
    )
    process_fire_size_distribution_table(
        input_path=fire_size_read_path,
        output_path=fire_size_save_path,
        train_firezone_ids=train_firezone_ids,
        norm_params_path=fire_size_norm_params_path,
    )


if __name__ == "__main__":
    main()
