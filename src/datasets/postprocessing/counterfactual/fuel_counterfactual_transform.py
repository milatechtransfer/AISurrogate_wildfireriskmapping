"""In-memory fuel counterfactual transform for prepared patch datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio

from data_preparation.paths import Paths
from data_preparation.spatial.fuel import load_fuel_grid
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.counterfactual_fuel import FUEL_NODATA, apply_fuel_edit
from src.datasets.postprocessing.counterfactual.fire_polygon_mask import build_fire_polygon_mask, fire_polygon_mask_csv_path


@dataclass(frozen=True)
class PatchFuelWindow:
    hex_id: str
    row: int
    col: int


def fuel_intervention_raster_path(
    prediction_dir: Path,
    hex_id: str,
    variant: Literal["baseline", "scenario"],
) -> Path:
    return prediction_dir / "fuel_intervention" / f"hexel_{int(hex_id):02d}_{variant}_fuel.tif"


def _write_fuel_raster(data: np.ndarray, profile: dict, path: Path) -> None:
    write_profile = profile.copy()
    write_profile.update(dtype="int16", count=1, compress="lzw", nodata=FUEL_NODATA)
    values = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size and not np.equal(finite_values, np.rint(finite_values)).all():
        raise ValueError("Fuel rasters must contain integer-valued categorical fuel IDs.")
    dtype_limits = np.iinfo(np.int16)
    if finite_values.size and (finite_values.min() < dtype_limits.min or finite_values.max() > dtype_limits.max):
        raise ValueError(
            f"Fuel raster IDs must fit in the int16 range [{dtype_limits.min}, {dtype_limits.max}]; "
            f"found finite values from {finite_values.min():g} to {finite_values.max():g}."
        )
    write_values = np.full(values.shape, FUEL_NODATA, dtype=np.int16)
    write_values[finite] = np.rint(finite_values).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **write_profile) as dst:
        dst.write(write_values, 1)


def _write_intervention_rasters(
    *,
    baseline_fuel: np.ndarray,
    scenario_fuel: np.ndarray,
    profile: dict,
    prediction_dir: Path,
    hex_id: str,
) -> None:
    _write_fuel_raster(
        baseline_fuel,
        profile,
        fuel_intervention_raster_path(prediction_dir, hex_id, "baseline"),
    )
    _write_fuel_raster(
        scenario_fuel,
        profile,
        fuel_intervention_raster_path(prediction_dir, hex_id, "scenario"),
    )


class FuelCounterfactualTransform:
    """Dataset patch transform that overlays a per-hexel fuel-edit scenario.

    Instances are built once per scenario via `from_metadata`, which loads each
    hexel's raw fuel raster directly from `raw_data_dir` and applies the edit once
    on the full grid. `__call__` slices the corresponding window from that edited
    grid and substitutes the patch's fuel channel at data-loading time.
    """

    def __init__(
        self,
        *,
        fuel_channel: int,
        filename_col: str,
        edited_hexels: dict[str, np.ndarray],
        patch_windows: dict[str, PatchFuelWindow],
        summary: pd.DataFrame,
        components: pd.DataFrame,
    ) -> None:
        self.fuel_channel = fuel_channel
        self.filename_col = filename_col
        self.edited_hexels = edited_hexels
        self.patch_windows = patch_windows
        self.summary = summary
        self.components = components

    @classmethod
    def from_metadata(
        cls,
        *,
        metadata: pd.DataFrame,
        fuel_channel: int,
        scenario: ScenarioConfig,
        raw_data_dir: Path,
        filename_col: str = "filename",
        mask_scope: str = "actual",
        prediction_dir: Path | None = None,
    ) -> FuelCounterfactualTransform:
        """Precompute the edited fuel channel for every hexel referenced in `metadata`.

        Each hexel's raw fuel raster is loaded directly from `raw_data_dir` (the same
        raster patches were generated from) and the scenario's edit is applied once on
        the full grid. `metadata` is only used to enumerate the (hex_id, row, col)
        patch windows that need the edited fuel channel; patch files are never read
        here. When `prediction_dir` is supplied, the exact baseline and edited fuel
        rasters are also persisted for downstream analysis.
        """
        params = dict(scenario.fuel_edit() or {})
        mode = str(params.pop("mode", "nonfuel_to_burnable_local_adjacent_modal"))
        nonfuel_ids = params.pop("nonfuel_ids", None)
        fire_polygons = params.pop("fire_polygons", None)
        if fire_polygons is not None and not isinstance(fire_polygons, dict):
            raise ValueError(f"Fuel scenario {scenario.name!r} must define fire_polygons as a mapping.")
        if not isinstance(nonfuel_ids, list | tuple) or not nonfuel_ids:
            raise ValueError(f"Fuel scenario {scenario.name!r} must define nonfuel_ids.")
        if "hex_id" not in metadata.columns:
            raise ValueError("Patch metadata is missing required column 'hex_id'.")

        normalized_hex_ids = metadata["hex_id"].astype(str).map(normalize_hex_id)
        edited_hexels: dict[str, np.ndarray] = {}
        patch_windows: dict[str, PatchFuelWindow] = {}
        summary_rows: list[dict[str, Any]] = []
        component_frames: list[pd.DataFrame] = []

        for hex_id in sorted(normalized_hex_ids.unique()):
            hex_metadata = metadata.loc[normalized_hex_ids == hex_id].drop_duplicates(filename_col)
            paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
            # Reproject onto the elevation grid's reference profile, mirroring
            # load_spatial_features_per_hexel's patch-generation path. Without this,
            # the fuel raster reprojects to an unrelated default CRS/grid, so patch
            # (row, col) windows no longer index the same pixels.
            _, reference_profile = load_spatial_raster(
                path=paths.elevation_grid(hex_id),
                mask_path=paths.mask_grid(hex_id, mask_scope=mask_scope),
            )
            fuel_grid = load_fuel_grid(
                root_dir=str(raw_data_dir),
                hex_id=hex_id,
                reference_profile=reference_profile,
                mask_scope=mask_scope,
            )
            profile = reference_profile
            baseline = np.ma.filled(fuel_grid.astype(np.float32), np.nan)

            hex_params = dict(params)
            polygon_mask = None
            if fire_polygons is not None:
                polygon_mask = build_fire_polygon_mask(
                    params=fire_polygons,
                    reference_profile=reference_profile,
                    hex_id=hex_id,
                )
                hex_params["edit_mask"] = polygon_mask.mask

            result = apply_fuel_edit(
                baseline,
                [int(value) for value in nonfuel_ids],
                mode=mode,
                scenario_name=scenario.name,
                params=hex_params,
            )
            edited_hexels[hex_id] = np.asarray(result.fuel, dtype=np.float32)

            for _, item in hex_metadata.iterrows():
                key = Path(str(item[filename_col])).as_posix()
                if key in patch_windows:
                    raise ValueError(f"Duplicate patch filename in counterfactual metadata: {key}")
                patch_windows[key] = PatchFuelWindow(hex_id=hex_id, row=int(item["row"]), col=int(item["col"]))

            if prediction_dir is not None:
                _write_intervention_rasters(
                    baseline_fuel=baseline,
                    scenario_fuel=result.fuel,
                    profile=profile,
                    prediction_dir=prediction_dir,
                    hex_id=hex_id,
                )
                if polygon_mask is not None:
                    csv_path = fire_polygon_mask_csv_path(prediction_dir, hex_id)
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    polygon_mask.summary.to_csv(csv_path, index=False)

            summary = asdict(result.report)
            summary["hex_id"] = hex_id
            if polygon_mask is not None:
                summary["fire_polygon_fires"] = polygon_mask.n_fires
                summary["fire_polygon_mask_pixels"] = polygon_mask.masked_pixels
                summary["fire_polygon_source_crs"] = polygon_mask.source_crs
                summary["fire_polygon_buffer_m"] = polygon_mask.buffer_m
            summary_rows.append(summary)
            if not result.components.empty:
                components = result.components.copy()
                components.insert(0, "hex_id", hex_id)
                components.insert(0, "scenario_name", scenario.name)
                component_frames.append(components)

        return cls(
            fuel_channel=fuel_channel,
            filename_col=filename_col,
            edited_hexels=edited_hexels,
            patch_windows=patch_windows,
            summary=pd.DataFrame(summary_rows),
            components=pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame(),
        )

    def __call__(self, data: np.ndarray, patch_info: dict[str, Any]) -> np.ndarray:
        """Return a copy of `data` with its fuel channel replaced by the precomputed edit."""
        key = Path(str(patch_info[self.filename_col])).as_posix()
        patch_window = self.patch_windows.get(key)
        if patch_window is None:
            raise KeyError(f"No counterfactual fuel channel was prepared for patch {key}.")
        edited_hexel = self.edited_hexels[patch_window.hex_id]
        height, width = data.shape[:2]
        row_end = patch_window.row + height
        col_end = patch_window.col + width
        edited_channel = edited_hexel[patch_window.row : row_end, patch_window.col : col_end]
        if edited_channel.shape != (height, width):
            # Patches near a hex's edge extend past the raw raster's true extent (the
            # patch-generation pipeline pads with NODATA before windowing); pad the same way.
            pad_height = height - edited_channel.shape[0]
            pad_width = width - edited_channel.shape[1]
            if pad_height < 0 or pad_width < 0:
                raise ValueError(f"Edited fuel slice for patch {key} has shape {edited_channel.shape}; expected {(height, width)}.")
            edited_channel = np.pad(edited_channel, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=np.nan)
        # Patch generation masks NODATA as the union of every source grid's own mask
        # (fuel, elevation, ignition, firezones - see hexel_loader.stack_sample), so a
        # pixel can be NODATA in the patch even where the raw fuel raster has a valid
        # value. Preserve the original fuel channel's NaNs so the substituted channel
        # stays consistent with every other channel in the patch.
        original_channel = data[:, :, self.fuel_channel]
        edited_channel = np.where(np.isnan(original_channel), np.nan, edited_channel)
        edited = np.array(data, copy=True)
        edited[:, :, self.fuel_channel] = edited_channel.astype(edited.dtype, copy=False)
        return edited
