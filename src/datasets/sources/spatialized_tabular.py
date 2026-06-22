import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import SpatializedTabularParams
from src.datasets.sources.base import DataSource


def _integer_zone_ids_from_series(zone_ids: pd.Series, *, column_name: str, source_name: str) -> pd.Series:
    """Coerce a zone-id series to a nullable ``Int64`` series, validating that values are integers.

    Missing entries are preserved as ``pd.NA``. Raises ``ValueError`` if any present value is
    non-numeric, non-finite, or not (approximately) an integer.
    """
    numeric_zone_ids = pd.to_numeric(zone_ids, errors="coerce")
    invalid_numeric_mask = zone_ids.notna() & numeric_zone_ids.isna()
    if invalid_numeric_mask.any():
        bad_value = zone_ids[invalid_numeric_mask].iloc[0]
        raise ValueError(f"Zone column {column_name!r} in {source_name!r} contains non-numeric zone id {bad_value!r}.")

    present_mask = numeric_zone_ids.notna()
    present_values = numeric_zone_ids[present_mask].to_numpy(dtype=np.float64)
    non_finite_mask = ~np.isfinite(present_values)
    if non_finite_mask.any():
        bad_value = float(present_values[non_finite_mask][0])
        raise ValueError(f"Zone column {column_name!r} in {source_name!r} contains non-finite zone id {bad_value}.")

    rounded_values = np.rint(present_values)
    if not np.allclose(present_values, rounded_values, atol=1e-3):
        bad_value = float(present_values[np.argmax(np.abs(present_values - rounded_values))])
        raise ValueError(f"Zone column {column_name!r} in {source_name!r} contains non-integer zone id {bad_value}.")

    integer_zone_ids = pd.Series(pd.NA, index=zone_ids.index, dtype="Int64")
    integer_zone_ids.loc[present_mask] = rounded_values.astype(np.int64)
    return integer_zone_ids


@lru_cache(maxsize=64)
def _train_zone_ids(
    root_dir: str,
    train_split_csv_name: str,
    zone_channel_key: str,
    modelling_approach: str,
    filename_col: str,
    valid_mask_threshold: float,
) -> tuple[int, ...]:
    metadata_path = Path(root_dir) / train_split_csv_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training split not found for spatialized-tabular imputation stats: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if "valid_ratio" in metadata.columns:
        metadata = metadata[metadata["valid_ratio"] > valid_mask_threshold].copy()
    if filename_col not in metadata.columns:
        raise KeyError(f"Column {filename_col!r} not found in training split {metadata_path}.")
    if metadata.empty:
        raise ValueError(f"Training split {metadata_path} has no rows after valid_ratio filtering; cannot compute imputation stats.")

    with (Path(root_dir) / f"feature_channel_map_{modelling_approach}.json").open() as handle:
        channel_feature_map = json.load(handle)
    if zone_channel_key not in channel_feature_map:
        raise ValueError(f"Missing zone channel {zone_channel_key!r} in feature channel map. Available keys: {list(channel_feature_map)}")
    zone_channel = int(channel_feature_map[zone_channel_key][0])

    zones: set[int] = set()
    for rel_path in metadata[filename_col].drop_duplicates():
        patch_path = Path(root_dir) / str(rel_path)
        if not patch_path.exists():
            raise FileNotFoundError(f"Training patch referenced by {metadata_path} does not exist: {patch_path}")
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
        raise ValueError(f"No finite positive zones found in training split {metadata_path}; cannot compute imputation stats.")
    return tuple(sorted(zones))


def train_global_fill_values(
    *,
    root_dir: str | Path,
    csv_name: str,
    feature_names_list: list[str],
    zone_id_col: str,
    zone_channel_key: str,
    train_split_csv_name: str,
    filename_col: str,
    valid_mask_threshold: float,
    modelling_approach: str = "1",
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Compute imputation fill values from rows whose zones occur in the training split."""

    root = Path(root_dir)
    df = pd.read_csv(root / csv_name)
    missing_columns = [col for col in [zone_id_col, *feature_names_list] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {csv_name!r}: {missing_columns}")

    train_zones = _train_zone_ids(
        str(root),
        train_split_csv_name,
        zone_channel_key,
        str(modelling_approach),
        filename_col,
        float(valid_mask_threshold),
    )
    integer_zone_ids = _integer_zone_ids_from_series(df[zone_id_col], column_name=zone_id_col, source_name=csv_name)
    train_rows = integer_zone_ids.isin(train_zones)
    if not train_rows.any():
        raise ValueError(f"No rows in {csv_name!r} match zones from training split {train_split_csv_name!r}: {list(train_zones)[:20]}")
    fill_values = df.loc[train_rows, feature_names_list].mean(axis=0).to_numpy(dtype=np.float32)
    if not np.isfinite(fill_values).all():
        raise ValueError(f"Training-derived imputation stats for {csv_name!r} contain non-finite values.")
    return fill_values, train_zones


def write_train_global_fill_stats(
    *,
    root_dir: str | Path,
    output_path: str | Path,
    source_name: str,
    csv_name: str,
    feature_names_list: list[str],
    zone_id_col: str,
    zone_channel_key: str,
    train_split_csv_name: str,
    filename_col: str,
    valid_mask_threshold: float,
    modelling_approach: str = "1",
) -> None:
    fill_values, train_zones = train_global_fill_values(
        root_dir=root_dir,
        csv_name=csv_name,
        feature_names_list=feature_names_list,
        zone_id_col=zone_id_col,
        zone_channel_key=zone_channel_key,
        train_split_csv_name=train_split_csv_name,
        filename_col=filename_col,
        valid_mask_threshold=valid_mask_threshold,
        modelling_approach=modelling_approach,
    )
    stats = {
        "source_name": source_name,
        "csv_name": csv_name,
        "feature_names_list": feature_names_list,
        "zone_id_col": zone_id_col,
        "zone_channel_key": zone_channel_key,
        "train_split_csv_name": train_split_csv_name,
        "train_zone_ids": list(train_zones),
        "global_fill": fill_values.astype(float).tolist(),
        "note": "global_fill was computed from rows whose zones occur in the training split.",
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(stats, handle, indent=2)


class SpatializedTabularSource(DataSource):
    """Rasterize zone-level tabular features into patch-aligned image channels."""

    def __init__(
        self,
        root_dir: str,
        params: SpatializedTabularParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
        train_split_csv_name: str | None = None,
        filename_col: str = "filename",
        valid_mask_threshold: float = 0.01,
    ):
        self.root_dir = root_dir
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform

        self.csv_name = params.csv_name
        self.feature_names_list = params.feature_names_list
        self.zone_id_col = params.fire_weather_zone_id_col
        self.zone_channel_key = params.zone_channel_key
        self.aggregation = params.aggregation.lower()
        self.include_missing_firezone_mask = params.include_missing_firezone_mask
        self.missing_value_strategy = params.missing_value_strategy.lower()
        self.imputation_stats_path = params.imputation_stats_path
        self.shuffle_lut = params.shuffle_lut
        self.shuffle_seed = params.shuffle_seed

        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
        if self.zone_channel_key not in channel_feature_map:
            raise ValueError(
                f"Missing zone channel {self.zone_channel_key!r} in feature channel map. Available keys: {list(channel_feature_map)}"
            )
        self.zone_channel = channel_feature_map[self.zone_channel_key][0]

        df = pd.read_csv(os.path.join(self.root_dir, self.csv_name))
        missing_columns = [col for col in [self.zone_id_col, *self.feature_names_list] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {self.csv_name!r}: {missing_columns}")

        self.train_zones = self._resolve_train_zones(
            train_split_csv_name=train_split_csv_name,
            filename_col=filename_col,
            valid_mask_threshold=valid_mask_threshold,
        )

        self.lut = self._build_lut(df, self.train_zones)
        if not self.lut:
            raise ValueError(
                f"Spatialized tabular LUT is empty for {self.csv_name!r}. Check zone column {self.zone_id_col!r} and selected features."
            )

        self._validate_missing_value_strategy()
        self.global_fill = self._global_fill(df, self.train_zones)
        if self.shuffle_lut:
            self._shuffle_lut_values()

    def _integer_zone_ids_from_csv(self, zone_ids: pd.Series) -> pd.Series:
        return _integer_zone_ids_from_series(zone_ids, column_name=self.zone_id_col, source_name=self.csv_name)

    def _train_zones_from_stats(self, stats_path: str | Path) -> tuple[int, ...] | None:
        path = Path(stats_path)
        if not path.is_absolute():
            path = Path(self.root_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Configured imputation_stats_path does not exist: {path}")
        with path.open() as handle:
            stats = json.load(handle)
        raw_zones = stats.get("train_zone_ids")
        if raw_zones is None:
            return None
        try:
            zones = tuple(sorted({int(zone) for zone in raw_zones}))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Imputation stats {path} contain invalid train_zone_ids.") from error
        if not zones:
            raise ValueError(f"Imputation stats {path} contain an empty train_zone_ids list.")
        return zones

    def _resolve_train_zones(
        self,
        *,
        train_split_csv_name: str | None,
        filename_col: str,
        valid_mask_threshold: float,
    ) -> tuple[int, ...]:
        # The LUT and imputation fill must be derived only from training zones so that
        # held-out rows never leak into the rasterized features.  Prefer a persisted,
        # train-derived artifact (avoids scanning every training patch); otherwise infer
        # the training zones from the training split.
        if self.imputation_stats_path:
            persisted_zones = self._train_zones_from_stats(self.imputation_stats_path)
            if persisted_zones is not None:
                return persisted_zones
        if train_split_csv_name is not None:
            return _train_zone_ids(
                self.root_dir,
                train_split_csv_name,
                self.zone_channel_key,
                self.modelling_approach,
                filename_col,
                float(valid_mask_threshold),
            )
        raise ValueError(
            "SpatializedTabularSource requires train_split_csv_name (or imputation_stats_path "
            "containing 'train_zone_ids') so the LUT and imputation fill are computed only from "
            "training zones and never from held-out rows."
        )

    def _build_lut(self, df: pd.DataFrame, train_zones: tuple[int, ...]) -> dict[int, np.ndarray]:
        aggregations = {
            "mean": "mean",
            "median": "median",
            "min": "min",
            "max": "max",
        }
        if self.aggregation not in aggregations:
            raise ValueError(f"Unsupported spatialized tabular aggregation {self.aggregation!r}. Supported values: {sorted(aggregations)}")

        normalized_df = df.copy()
        zone_id_key = "__spatialized_tabular_zone_id"
        normalized_df[zone_id_key] = self._integer_zone_ids_from_csv(df[self.zone_id_col])
        normalized_df = normalized_df[normalized_df[zone_id_key].isin(train_zones)]
        if normalized_df.empty:
            raise ValueError(
                f"No rows in {self.csv_name!r} match the training zones {list(train_zones)[:20]}; cannot build a leak-safe LUT."
            )
        grouped = normalized_df.groupby(zone_id_key, dropna=True)[self.feature_names_list].agg(aggregations[self.aggregation])
        grouped = grouped.dropna(how="any")
        return {int(zone): row.to_numpy(dtype=np.float32) for zone, row in grouped.iterrows()}

    def _validate_missing_value_strategy(self) -> None:
        if self.missing_value_strategy not in {"global_mean", "zero", "raise"}:
            raise ValueError(
                f"Unsupported missing_value_strategy={self.missing_value_strategy!r}. Supported values: ['global_mean', 'zero', 'raise']."
            )

    def _global_fill_from_stats(self, stats_path: str | Path) -> np.ndarray:
        """Load the precomputed global-fill vector from a JSON imputation-stats file.

        Resolves ``stats_path`` relative to ``root_dir`` when not absolute, checks that the file's
        ``feature_names_list`` matches this source, and returns the ``global_fill`` array. Raises if
        the file is missing, feature names disagree, or the values are the wrong shape or non-finite.
        """
        path = Path(stats_path)
        if not path.is_absolute():
            path = Path(self.root_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Configured imputation_stats_path does not exist: {path}")
        with path.open() as handle:
            stats = json.load(handle)
        stats_features = list(stats.get("feature_names_list", []))
        if stats_features != self.feature_names_list:
            raise ValueError(
                f"Imputation stats {path} feature_names_list does not match source {self.csv_name!r}: "
                f"{stats_features} != {self.feature_names_list}"
            )
        values = np.asarray(stats.get("global_fill"), dtype=np.float32)
        if values.shape != (len(self.feature_names_list),) or not np.isfinite(values).all():
            raise ValueError(f"Imputation stats {path} contain invalid global_fill values.")
        return values

    def _global_fill(self, df: pd.DataFrame, train_zones: tuple[int, ...]) -> np.ndarray:
        """Compute the per-feature fill vector used to impute pixels whose zone is absent from the LUT.

        Returns zeros unless ``missing_value_strategy == "global_mean"``. When a precomputed
        ``imputation_stats_path`` is configured it is used directly; otherwise the fill is the mean of
        each feature over rows belonging to ``train_zones`` only (train-split to avoid leakage). Raises
        if no rows match the training zones or the resulting means are non-finite.
        """
        if self.missing_value_strategy != "global_mean":
            return np.zeros(len(self.feature_names_list), dtype=np.float32)
        if self.imputation_stats_path:
            return self._global_fill_from_stats(self.imputation_stats_path)
        integer_zone_ids = self._integer_zone_ids_from_csv(df[self.zone_id_col])
        train_rows = integer_zone_ids.isin(train_zones)
        if not train_rows.any():
            raise ValueError(f"No rows in {self.csv_name!r} match the training zones {list(train_zones)[:20]}.")
        values = df.loc[train_rows, self.feature_names_list].mean(axis=0).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Training-derived imputation stats for {self.csv_name!r} contain non-finite values.")
        return values

    def _shuffle_lut_values(self) -> None:
        zones = sorted(self.lut)
        values = [self.lut[zone].copy() for zone in zones]
        rng = np.random.default_rng(self.shuffle_seed)
        permutation = rng.permutation(len(values))
        self.lut = {zone: values[permutation[index]] for index, zone in enumerate(zones)}

    def _initial_features(self, height: int, width: int) -> np.ndarray:
        if self.missing_value_strategy == "global_mean":
            fill_value = self.global_fill
        elif self.missing_value_strategy == "zero":
            fill_value = np.zeros(len(self.feature_names_list), dtype=np.float32)
        else:
            fill_value = np.full(len(self.feature_names_list), np.nan, dtype=np.float32)
        return np.broadcast_to(fill_value, (height, width, len(self.feature_names_list))).copy()

    def _zone_ids(self, zone_grid: np.ndarray, finite_zone_mask: np.ndarray) -> np.ndarray:
        zone_int_grid = np.zeros(zone_grid.shape, dtype=np.int64)
        if not finite_zone_mask.any():
            return zone_int_grid

        finite_zone_values = zone_grid[finite_zone_mask]
        rounded_zone_values = np.rint(finite_zone_values)
        if not np.allclose(finite_zone_values, rounded_zone_values, atol=1e-3):
            bad_value = float(finite_zone_values[np.argmax(np.abs(finite_zone_values - rounded_zone_values))])
            raise ValueError(f"Zone channel {self.zone_channel_key!r} contains non-integer zone id {bad_value}.")

        zone_int_grid[finite_zone_mask] = rounded_zone_values.astype(np.int64)
        return zone_int_grid

    def get_sample(self, patch_info: dict):
        data = patch_info["data"] if "data" in patch_info else np.load(patch_info["file_path"], mmap_mode="r")
        zone_grid = np.asarray(data[:, :, self.zone_channel])
        height, width = zone_grid.shape
        features = self._initial_features(height, width)

        finite_zone_mask = np.isfinite(zone_grid) & (zone_grid > 0)
        zone_int_grid = self._zone_ids(zone_grid=zone_grid, finite_zone_mask=finite_zone_mask)
        matched_mask = np.zeros((height, width), dtype=bool)
        for zone in np.unique(zone_int_grid[finite_zone_mask]):
            zone_features = self.lut.get(int(zone))
            if zone_features is None:
                continue
            pixel_mask = finite_zone_mask & (zone_int_grid == zone)
            features[pixel_mask] = zone_features
            matched_mask[pixel_mask] = True

        missing_firezone_mask = ~matched_mask
        if self.missing_value_strategy == "raise" and missing_firezone_mask.any():
            missing_zones = sorted({int(z) for z in zone_int_grid[finite_zone_mask & missing_firezone_mask]})
            raise ValueError(f"Spatialized tabular source {self.csv_name!r} has missing LUT zones: {missing_zones[:20]}")

        if self.include_missing_firezone_mask:
            features = np.concatenate([features, missing_firezone_mask[:, :, None].astype(np.float32)], axis=-1)

        return torch.from_numpy(features.astype(np.float32)).permute(2, 0, 1)

    def input_dim(self):
        return len(self.feature_names_list) + int(self.include_missing_firezone_mask)
