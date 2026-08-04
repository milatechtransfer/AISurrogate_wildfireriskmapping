import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig
from src.datasets.sources import DataSource
from src.datasets.transforms import get_transforms
from src.datasets.utils import AVAILABLE_DATA_SOURCES, SPATIALIZED_TABULAR_SOURCE_NAMES, get_data_source_class
from src.utils import seed_worker


class MultiSourceDataset(Dataset):
    """
    Composable dataset that handles data loading from multiple sources.

    Extraction: Delegated to DataSource objects.
    I/O Optimization: Files are memory-mapped once per __getitem__ and shared via context to prevent redundant reads.
    """

    def __init__(
        self,
        csv_name: str,
        root_dir: str,
        filename_col: str = "filename",
        valid_mask_threshold: float = 0.01,
        sources: dict[str, DataSource] | None = None,
        grid_transform=None,
        include_patch_metadata: bool = False,
        patch_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
        metadata_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        """
        Args:
            csv_name (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            val_mask_threshold (float): The threshold for how much valid data should be present in a data sample
            input_sources (dict[str, DataSource]): A dictionary mapping output keys
            (example: 'grid', 'tabular_weather') to their respective data sources (example: GridSource, TabularSource)
        """

        self.csv_name = csv_name
        self.root_dir = root_dir
        self.metadata_path = os.path.join(self.root_dir, self.csv_name)
        self.valid_mask_threshold = valid_mask_threshold
        self.filename_col = filename_col
        self.grid_transform = grid_transform
        self.include_patch_metadata = include_patch_metadata
        self.patch_transform = patch_transform
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at: {self.metadata_path}")
        metadata_df = pd.read_csv(self.metadata_path)
        # Filter by valid_ratio if the column exists
        if "valid_ratio" in metadata_df.columns:
            metadata_df = metadata_df[metadata_df["valid_ratio"] > self.valid_mask_threshold].copy()
        if metadata_filter is not None:
            metadata_df = metadata_filter(metadata_df).copy()
        if filename_col not in metadata_df.columns:
            raise KeyError(f"Column '{filename_col}' not found in {csv_name}")
        self.metadata = metadata_df
        self.records = self.metadata.to_dict("records")  # Convert to list of dicts for O(1) access performance
        self.sources = sources if sources is not None else {}

    def get_patch_info(self, idx: int):
        row = self.records[idx]
        patch_info = row.copy()
        patch_info["idx"] = idx
        patch_info["file_path"] = os.path.join(self.root_dir, row[self.filename_col])
        return patch_info

    def __getitem__(self, idx):
        patch_info = self.get_patch_info(idx)
        data = np.load(patch_info["file_path"], mmap_mode="r")
        patch_info["data"] = self.patch_transform(data, patch_info) if self.patch_transform is not None else data
        sample = {}
        spatialized_inputs = []
        for name, source in self.sources.items():
            source_sample = source.get_sample(patch_info)
            if name in SPATIALIZED_TABULAR_SOURCE_NAMES:
                spatialized_inputs.append(source_sample)
            elif name == "grid" and len(source_sample) == 4:
                # iROS mode: GridSource returns (input_arr, fuel_curve_arr, output_arr, mask)
                inputs, fuel_curve_arr, targets, masks = source_sample
                sample["grid"] = (inputs, targets, masks)
                sample["fuel_curve"] = fuel_curve_arr
            else:
                sample[name] = source_sample
        if spatialized_inputs:
            if "grid" not in sample:
                raise ValueError("Spatialized tabular sources require a 'grid' source to append channels.")
            inputs, targets, masks = sample["grid"]
            sample["grid"] = (torch.cat([inputs, *spatialized_inputs], dim=0), targets, masks)
        if self.grid_transform is not None:
            if "grid" not in sample:
                raise ValueError("Configured grid transform requires a 'grid' source.")
            inputs, targets, masks = sample["grid"]
            if "fuel_curve" in sample:
                # Transform iROS jointly with spatial inputs so geometric augmentations stay consistent.
                n_spatial = inputs.shape[0]
                combined = torch.cat([inputs, sample["fuel_curve"]], dim=0)
                combined, targets, masks = self.grid_transform(combined, targets, masks)
                sample["grid"] = (combined[:n_spatial], targets, masks)
                sample["fuel_curve"] = combined[n_spatial:]
            else:
                sample["grid"] = self.grid_transform(inputs, targets, masks)
        if self.include_patch_metadata:
            if "hex_id" not in patch_info:
                raise KeyError("Patch metadata requested but split row does not contain 'hex_id'.")
            patch_metadata = {"hex_id": torch.tensor(int(patch_info["hex_id"]), dtype=torch.long)}
            has_row_col = "row" in patch_info and "col" in patch_info
            if has_row_col:
                patch_metadata["row"] = torch.tensor(int(patch_info["row"]), dtype=torch.long)
                patch_metadata["col"] = torch.tensor(int(patch_info["col"]), dtype=torch.long)
            sample["patch_metadata"] = patch_metadata
        return sample

    def __len__(self):
        return len(self.records)


def build_dataset(
    config: DataConfig,
    csv_name: str,
    modelling_approach: str = "1",
    *,
    patch_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    metadata_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> MultiSourceDataset:
    """
    Args:
        config (DataConfig): Contains information for multi source dataset instantiation
        csv_name (str): Name of CSV file that contains split index information
        modelling_approach (str): The approach used for modelling
    Returns:
        MultiSourceDataset: containing the metadata and all the sources specified in DataConfig
    """
    # Build sources
    sources: dict[str, DataSource] = {}

    grid_transform = None
    for source_conf in config.input_sources:
        if source_conf.name not in AVAILABLE_DATA_SOURCES:
            raise ValueError(f"Invalid source name '{source_conf.name} in config. Supported sources are: {AVAILABLE_DATA_SOURCES}")

        # Setup transforms
        is_train = "train" in csv_name.lower()
        transform = None
        if source_conf.name == "grid" and is_train:
            grid_transform = get_transforms(source_conf)

        # Instantiate each data source class
        source_class = get_data_source_class(source_conf.name)
        source_kwargs: dict[str, Any] = {
            "root_dir": config.root_dir,
            "params": source_conf.params,
            "modelling_approach": modelling_approach,
            "transform": transform,
        }
        if source_conf.name == "grid":
            source_kwargs["raw_data_dir"] = config.raw_data_dir
            source_kwargs["train_split_csv_name"] = config.train_split
        sources[source_conf.name] = source_class(**source_kwargs)

    dataset = MultiSourceDataset(
        csv_name=csv_name,
        root_dir=config.root_dir,
        filename_col=config.filename_col,
        valid_mask_threshold=config.valid_mask_threshold,
        sources=sources,
        grid_transform=grid_transform,
        include_patch_metadata=config.include_patch_metadata,
        patch_transform=patch_transform,
        metadata_filter=metadata_filter,
    )
    return dataset


def get_train_val_dataloader(config: DataConfig, modelling_approach: str = "1", seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """
    Creates and returns a DataLoader with deterministic shuffling
    """
    batch_size = config.batch_size
    num_workers = config.num_workers
    train_split = config.train_split
    val_split = config.val_split

    g = torch.Generator()
    g.manual_seed(seed)
    train_dataset = build_dataset(config, csv_name=train_split, modelling_approach=modelling_approach)
    val_dataset = build_dataset(config, csv_name=val_split, modelling_approach=modelling_approach)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g, pin_memory=True
    )
    return train_dataloader, val_dataloader


def get_test_dataloader(
    config: DataConfig,
    modelling_approach: str = "1",
    seed: int = 42,
    *,
    patch_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    metadata_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> DataLoader:
    """
    Creates and returns a test DataLoader with deterministic shuffling
    """
    batch_size = config.batch_size
    num_workers = config.num_workers
    test_split = config.test_split

    g = torch.Generator()
    test_dataset = build_dataset(
        config,
        csv_name=test_split,
        modelling_approach=modelling_approach,
        patch_transform=patch_transform,
        metadata_filter=metadata_filter,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    return test_dataloader
