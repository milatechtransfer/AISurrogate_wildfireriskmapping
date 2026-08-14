import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from src.config import SpatializedTabularParams
from src.datasets.sources.base import DataSource

# Sentinel used as the hex-id component of the LUT key when hex_id_col is not configured, so the LUT
# key type stays a uniform tuple[int, int] regardless of whether hex-scoping is enabled.
_NO_HEX_ID = -1


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


class SpatializedTabularSource(DataSource):
    """Rasterize zone-level tabular features into patch-aligned image channels."""

    def __init__(
        self,
        root_dir: str,
        params: SpatializedTabularParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
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
        self.global_fill_csv_name = params.global_fill_csv_name
        self.shuffle_lut = params.shuffle_lut
        self.shuffle_seed = params.shuffle_seed
        self.hex_id_col = params.hex_id_col
        self.quantiles = params.quantiles

        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
        if self.zone_channel_key not in channel_feature_map:
            raise ValueError(
                f"Missing zone channel {self.zone_channel_key!r} in feature channel map. Available keys: {list(channel_feature_map)}"
            )
        self.zone_channel = channel_feature_map[self.zone_channel_key][0]

        df = pd.read_csv(os.path.join(self.root_dir, self.csv_name))
        required_columns = [self.zone_id_col, *self.feature_names_list]
        if self.hex_id_col is not None:
            required_columns.append(self.hex_id_col)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {self.csv_name!r}: {missing_columns}")

        self._validate_missing_value_strategy()
        self.lut = self._build_lut(df)
        if not self.lut:
            raise ValueError(
                f"Spatialized tabular LUT is empty for {self.csv_name!r}. Check zone column {self.zone_id_col!r} and selected features."
            )
        global_fill_df = df
        if self.global_fill_csv_name is not None and self.missing_value_strategy == "global_mean":
            global_fill_df = pd.read_csv(os.path.join(self.root_dir, self.global_fill_csv_name))
            missing_fill_columns = [column for column in self.feature_names_list if column not in global_fill_df.columns]
            if missing_fill_columns:
                raise ValueError(f"Missing columns in global-fill CSV {self.global_fill_csv_name!r}: {missing_fill_columns}")
            if (
                self.hex_id_col is not None
                and self.missing_value_strategy == "global_mean"
                and self.hex_id_col not in global_fill_df.columns
            ):
                raise ValueError(f"Missing hex id column {self.hex_id_col!r} in global-fill CSV {self.global_fill_csv_name!r}.")
        self.global_fill, self.global_fill_by_hex = self._global_fill(global_fill_df)
        if self.shuffle_lut:
            self._shuffle_lut_values()

    def _integer_zone_ids_from_csv(self, zone_ids: pd.Series) -> pd.Series:
        return _integer_zone_ids_from_series(zone_ids, column_name=self.zone_id_col, source_name=self.csv_name)

    def _build_lut(self, df: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
        """Aggregate every (hex, zone) or zone present in the tabular source into a feature-vector LUT.

        Built from all rows (not just training zones) so that zones absent from the training patches
        still resolve to their own aggregated values instead of an imputed fill; the missing-firezone
        fill/mask then applies only to zones with no row at all.

        The LUT is always keyed by ``(hex_id, zone)`` tuples. When ``hex_id_col`` is set, each patch
        only ever pulls features aggregated from rows belonging to its own hexel (see ``get_sample``),
        so a zone that spans multiple hexels never mixes rows across hexels — and therefore never
        mixes rows across train/val/test splits either, since each hexel belongs to exactly one split.

        When ``hex_id_col`` is not set, ``hex_id`` is replaced by a shared sentinel so the LUT is
        effectively keyed by zone alone, pooling rows across every hexel that reports that zone.
        Whether that is leak-free depends on the source and the split: it holds for exogenous
        covariates under spatial (zone/hex) splits, whereas a temporal split or a target-derived
        covariate could leak. Callers are responsible for ensuring the configured columns (and
        whether ``hex_id_col`` is set) are appropriate for the split in use.
        """
        aggregations = {
            "mean": "mean",
            "median": "median",
            "min": "min",
            "max": "max",
        }
        zone_ids = self._integer_zone_ids_from_csv(df[self.zone_id_col])
        if self.hex_id_col is not None:
            hex_ids = pd.to_numeric(df[self.hex_id_col], errors="coerce").astype("Int64")
            invalid_hex_mask = df[self.hex_id_col].notna() & hex_ids.isna()
            if invalid_hex_mask.any():
                bad_value = df[self.hex_id_col][invalid_hex_mask].iloc[0]
                raise ValueError(f"Hex id column {self.hex_id_col!r} in {self.csv_name!r} contains non-numeric hex id {bad_value!r}.")
        else:
            hex_ids = pd.Series(_NO_HEX_ID, index=df.index, dtype="Int64")

        if self.quantiles is not None:
            lookup: dict[tuple[int, int], np.ndarray] = {}
            grouped = df.assign(_hex_id=hex_ids, _zone_id=zone_ids).groupby(["_hex_id", "_zone_id"], dropna=True)
            for (hex_id, zone), frame in grouped:
                values: list[float] = []
                for feature_name in self.feature_names_list:
                    feature_values = frame[feature_name].dropna()
                    if feature_values.empty:
                        values = []
                        break
                    values.extend(float(feature_values.quantile(quantile)) for quantile in self.quantiles)
                if values:
                    lookup[(int(hex_id), int(zone))] = np.asarray(values, dtype=np.float32)
            return lookup
        if self.aggregation not in aggregations:
            raise ValueError(f"Unsupported spatialized tabular aggregation {self.aggregation!r}. Supported values: {sorted(aggregations)}")

        grouped = df.groupby([hex_ids, zone_ids], dropna=True)[self.feature_names_list].agg(aggregations[self.aggregation])
        grouped = grouped.dropna(how="any")
        return {(int(hex_id), int(zone)): row.to_numpy(dtype=np.float32) for (hex_id, zone), row in grouped.iterrows()}

    def _validate_missing_value_strategy(self) -> None:
        if self.missing_value_strategy not in {"global_mean", "zero", "raise"}:
            raise ValueError(
                f"Unsupported missing_value_strategy={self.missing_value_strategy!r}. Supported values: ['global_mean', 'zero', 'raise']."
            )

    def _global_fill(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Per-feature fill vector(s) for pixels whose firezone has no row in the tabular source.

        Returns a ``(global_fill, fill_by_hex)`` pair. Both are all-zeros unless
        ``missing_value_strategy == "global_mean"``, in which case they hold the mean of each feature
        over rows. ``fill_by_hex`` is populated only when ``hex_id_col`` is configured, with the mean
        computed separately per hex_id, so a patch's missing-zone fill is only ever derived from rows
        belonging to its own hexel — never pooled across hexels, which would otherwise leak data
        across train/val/test splits. ``global_fill`` (mean over all rows) is used when ``hex_id_col``
        is not configured. Raises if any resulting mean is non-finite.
        """
        if self.missing_value_strategy != "global_mean":
            zeros: np.ndarray = np.zeros(self._feature_dim(), dtype=np.float32)
            return zeros, {}

        fill_by_hex: dict[int, np.ndarray] = {}
        if self.hex_id_col is not None:
            hex_ids = pd.to_numeric(df[self.hex_id_col], errors="coerce").astype("Int64")
            for hex_id, frame in df.groupby(hex_ids):
                values = self._fill_values(frame)
                if not np.isfinite(values).all():
                    raise ValueError(f"Imputation fill for {self.csv_name!r} and hex_id={hex_id} contains non-finite values.")
                fill_by_hex[int(hex_id)] = values
            return np.zeros(self._feature_dim(), dtype=np.float32), fill_by_hex

        values = self._fill_values(df)
        if not np.isfinite(values).all():
            raise ValueError(f"Imputation fill for {self.csv_name!r} contains non-finite values.")
        return values, fill_by_hex

    def _fill_values(self, df: pd.DataFrame) -> np.ndarray:
        if self.quantiles is None:
            return df[self.feature_names_list].mean(axis=0).to_numpy(dtype=np.float32)
        values = [
            float(df[feature_name].dropna().quantile(quantile)) for feature_name in self.feature_names_list for quantile in self.quantiles
        ]
        return np.asarray(values, dtype=np.float32)

    def _shuffle_lut_values(self) -> None:
        zones = sorted(self.lut)
        values = [self.lut[zone].copy() for zone in zones]
        rng = np.random.default_rng(self.shuffle_seed)
        permutation = rng.permutation(len(values))
        self.lut = {zone: values[permutation[index]] for index, zone in enumerate(zones)}

    def _initial_features(self, height: int, width: int, hex_id: int) -> np.ndarray:
        if self.missing_value_strategy == "global_mean":
            if self.hex_id_col is not None:
                fill_value = self.global_fill_by_hex.get(hex_id)
                if fill_value is None:
                    raise ValueError(
                        f"No global-mean fill found for hex_id={hex_id} in {self.csv_name!r} (or {self.global_fill_csv_name!r})."
                    )
            else:
                fill_value = self.global_fill
        elif self.missing_value_strategy == "zero":
            fill_value = np.zeros(self._feature_dim(), dtype=np.float32)
        else:
            fill_value = np.full(self._feature_dim(), np.nan, dtype=np.float32)
        return np.broadcast_to(fill_value, (height, width, self._feature_dim())).copy()

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

        hex_id = _NO_HEX_ID
        if self.hex_id_col is not None:
            if "hex_id" not in patch_info:
                raise KeyError(
                    f"Spatialized tabular source {self.csv_name!r} is configured with hex_id_col={self.hex_id_col!r} "
                    "but patch_info does not contain 'hex_id'."
                )
            hex_id = int(patch_info["hex_id"])

        features = self._initial_features(height, width, hex_id)

        finite_zone_mask = np.isfinite(zone_grid) & (zone_grid > 0)
        zone_int_grid = self._zone_ids(zone_grid=zone_grid, finite_zone_mask=finite_zone_mask)
        matched_mask = np.zeros((height, width), dtype=bool)
        for zone in np.unique(zone_int_grid[finite_zone_mask]):
            zone_key = (hex_id, int(zone))
            zone_features = self.lut.get(zone_key)
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
        return self._feature_dim() + int(self.include_missing_firezone_mask)

    def _feature_dim(self) -> int:
        feature_multiplier = len(self.quantiles) if self.quantiles is not None else 1
        return len(self.feature_names_list) * feature_multiplier

    def output_feature_names(self) -> list[str]:
        if self.quantiles is None:
            names = list(self.feature_names_list)
        else:
            names = [
                f"{feature_name}_q{round(quantile * 100):02d}" for feature_name in self.feature_names_list for quantile in self.quantiles
            ]
        if self.include_missing_firezone_mask:
            names.append("missing_firezone_mask")
        return names
