"""In-memory fuel counterfactual transform for prepared patch datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.counterfactual_fuel import FUEL_NODATA, apply_fuel_edit


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
    write_profile.update(dtype="float32", count=1, compress="lzw", nodata=float(FUEL_NODATA))
    values = np.asarray(data, dtype=np.float32)
    write_values = np.where(np.isfinite(values), values, float(FUEL_NODATA)).astype(np.float32)
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
            fuel_grid, profile = load_spatial_raster(
                path=paths.fuel_grid(hex_id),
                mask_path=paths.mask_grid(hex_id, mask_scope=mask_scope),
            )
            baseline = np.ma.filled(fuel_grid.astype(np.float32), np.nan)

            result = apply_fuel_edit(
                baseline,
                [int(value) for value in nonfuel_ids],
                mode=mode,
                scenario_name=scenario.name,
                params=params,
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

            summary = asdict(result.report)
            summary["hex_id"] = hex_id
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
        edited = np.array(data, copy=True)
        edited[:, :, self.fuel_channel] = edited_channel.astype(edited.dtype, copy=False)
        return edited
