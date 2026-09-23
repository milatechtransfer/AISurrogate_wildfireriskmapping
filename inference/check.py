"""
Check a project's input files against a model bundle before predicting.

For every hexel folder it reports missing or unreadable files, rasters without a coordinate system,
rasters that do not cover the prediction mask, cell sizes the model was not trained on, fuel codes the
model has no fuel curve for, fire zones without weather or fire-size data, and malformed tables. Fuel codes
missing from the model's fuel table are defined from the project's FuelTypes/FuelCodeCrosswalk tables when
possible (see ``inference.fuels``).

Errors stop (or would silently corrupt) a prediction and must be fixed. Warnings describe inputs the
model can handle but whose results deserve a second look. ``predict`` runs these checks first.
With ``--outputs`` it also checks the BurnP3+ output rasters that ``evaluate`` compares against.

Example:
    python -m inference.check --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project [--outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioError
from rasterio.features import geometry_mask
from rasterio.mask import mask as mask_raster
from rasterio.warp import calculate_default_transform

from data_preparation.paths import Paths
from data_preparation.spatial.ignition import _IGN_GRID_PATTERN
from data_preparation.spatial.utils import fire_cause_label_mapping, fire_cause_mapping
from data_preparation.tabular.utils import check_weather_list, weather_column_aliases, weather_column_names
from data_preparation.utils import FIRE_SIZE_COLUMN_ALIASES, FIRE_SIZE_FEATURE_COLS, find_hex_ids
from inference.bundle import (
    FIRE_SIZE_SOURCE,
    MASK_SCOPES,
    NO_MASK_SCOPE,
    RESOURCE_FIRE_SIZE_TABLE,
    RESOURCE_FUEL_CURVES,
    WEATHER_SOURCE,
    BundleError,
    ModelBundle,
    load_bundle,
    resolve_mask_scope,
    resolve_scenario_name,
)
from inference.fuels import CODE_COL, describe_codes, read_project_fuel_codes, resolve_fuel_curves
from src.datasets.fuel_utils import _FEATURE_COLUMN, FUEL_CURVE_ENCODINGS, read_curves
from src.datasets.targets import get_target_spec

# error: must be fixed; warning: worth reviewing; note: information the user should be aware of.
Level = Literal["error", "warning", "note"]

# Relative deviation of the DEM cell size (after reprojection) from the model's cell size.
# Training DEMs were 98.6-101.5 m after reprojection to ESRI:102002.
RESOLUTION_ERROR_TOLERANCE = 0.10
RESOLUTION_WARNING_TOLERANCE = 0.03
_GERUNDS = {"predict": "predicting", "evaluate": "evaluating"}

# Warn when less than this fraction of the mask has valid data in a raster.
MIN_VALID_COVERAGE = 0.9
# Zone-level synthetic row appended to every fire-size table by process_fire_size_df.
IMPUTED_FIRE_SIZE_ZONES = frozenset({36})
MAX_LISTED = 12

# Plausible ranges of the weather columns; values outside are reported as warnings.
WEATHER_VALUE_RANGES: dict[str, tuple[float | None, float | None]] = {
    "Temperature": (-50.0, 50.0),
    "RelativeHumidity": (0.0, 100.0),
    "WindSpeed": (0.0, None),
    "WindDirection": (0.0, 360.0),
    "Precipitation": (0.0, None),
    "FineFuelMoistureCode": (0.0, 101.0),
    "DuffMoistureCode": (0.0, None),
    "DroughtCode": (0.0, None),
    "InitialSpreadIndex": (0.0, None),
    "BuildupIndex": (0.0, None),
    "FireWeatherIndex": (0.0, None),
}


@dataclass
class Finding:
    level: Level
    message: str
    hexel: str | None = None  # e.g. "hex07"; None for project-level findings
    path: str | None = None  # relative to the project folder when inside it

    def format(self) -> str:
        location = f"{self.path}: " if self.path else ""
        return f"{self.level.upper():<8} {location}{self.message}"

    def describe(self) -> str:
        """One line naming where the problem is, for logs and error messages."""
        return f"{self.path or self.hexel or 'project'}: {self.message}"


@dataclass
class CheckReport:
    project_dir: str
    bundle: str
    mask_scope: str
    hexels: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    action: str = "predict"  # what the checked project is being prepared for, used in the summary line

    @property
    def errors(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "warning"]

    @property
    def notes(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "note"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok, "num_errors": len(self.errors), "num_warnings": len(self.warnings)}

    def format(self, levels: tuple[Level, ...] = ("error", "warning", "note")) -> str:
        lines = [f"Checked {len(self.hexels)} hexel(s) in {self.project_dir} for model {self.bundle} (mask: {self.mask_scope})", ""]
        project_findings = [f for f in self.findings if f.hexel is None and f.level in levels]
        if project_findings:
            lines.append("Project")
            lines += [f"  {finding.format()}" for finding in project_findings]
        for hexel in self.hexels:
            hexel_findings = [f for f in self.findings if f.hexel == hexel and f.level != "note"]
            shown = [f for f in self.findings if f.hexel == hexel and f.level in levels]
            num_errors = sum(f.level == "error" for f in hexel_findings)
            num_warnings = len(hexel_findings) - num_errors
            status = "OK" if not hexel_findings else f"{_plural(num_errors, 'error')}, {_plural(num_warnings, 'warning')}"
            lines.append(f"{hexel}: {status}")
            lines += [f"  {finding.format()}" for finding in shown]
        lines.append("")
        if self.errors:
            lines.append(
                f"Result: {_plural(len(self.errors), 'error')}, {_plural(len(self.warnings), 'warning')}. "
                f"Fix the errors before {_GERUNDS[self.action]}."
            )
        elif self.warnings:
            lines.append(f"Result: ready to {self.action}, with {_plural(len(self.warnings), 'warning')} worth reviewing.")
        else:
            lines.append(f"Result: ready to {self.action}, no problems found.")
        return "\n".join(lines)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _listing(values: Any, limit: int = MAX_LISTED) -> str:
    values = list(values)
    shown = ", ".join(str(value) for value in values[:limit])
    return shown + (f", ... ({len(values) - limit} more)" if len(values) > limit else "")


def resolve_hex_ids(requested: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    """Match requested hexels ("7", "07", "hex07") to folder IDs; returns (found, not_found)."""
    found, not_found = [], []
    for value in requested:
        value = str(value).strip()
        hex_id = value[3:] if value.lower().startswith("hex") else value
        matches = [
            candidate
            for candidate in available
            if candidate == hex_id or (hex_id.isdigit() and candidate.isdigit() and int(candidate) == int(hex_id))
        ]
        if matches:
            found.append(matches[0])
        else:
            not_found.append(value)
    return found, not_found


class _Collector:
    def __init__(self, report: CheckReport, project_dir: Path, hexel: str | None = None) -> None:
        self.report = report
        self.project_dir = project_dir
        self.hexel = hexel

    def _add(self, level: Level, message: str, path: Path | None) -> None:
        shown = None
        if path is not None:
            try:
                shown = Path(path).resolve().relative_to(self.project_dir).as_posix()
            except ValueError:
                shown = str(path)
        self.report.findings.append(Finding(level=level, message=message, hexel=self.hexel, path=shown))

    def error(self, message: str, path: Path | None = None) -> None:
        self._add("error", message, path)

    def warning(self, message: str, path: Path | None = None) -> None:
        self._add("warning", message, path)

    def note(self, message: str, path: Path | None = None) -> None:
        self._add("note", message, path)


@dataclass(frozen=True)
class ModelRequirements:
    """What a bundle needs from each hexel."""

    crs: str
    resolution_m: float
    uses_weather: bool
    uses_fire_size: bool
    uses_ignition_distribution: bool
    known_fuel_codes: frozenset[int] | None  # None: the model accepts any fuel code
    multi_season_fuel_codes: frozenset[int]
    targets: tuple[str, ...] = ()
    fuel_curves: pd.DataFrame | None = field(default=None, compare=False, repr=False)  # the model's fuel curve table
    fuel_feature_col: str = "ROS"

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> ModelRequirements:
        manifest = bundle.manifest
        grid = manifest.grid_params()
        known_codes: frozenset[int] | None = None
        multi_season: frozenset[int] = frozenset()
        curve_table: pd.DataFrame | None = None
        feature_col = "ROS"
        curves_path = bundle.optional_resource_path(RESOURCE_FUEL_CURVES)
        if "fuel_grid" in grid.feature_names_list and grid.fuel_feats_encoding in FUEL_CURVE_ENCODINGS and curves_path is not None:
            feature_col = _FEATURE_COLUMN[grid.fuel_feats_encoding]
            curves = read_curves(ros_csv_path=curves_path, feature_col=feature_col)
            known_codes = frozenset(int(code) for code in curves)
            multi_season = frozenset(int(code) for code, states in curves.items() if len(states) > 1)
            curve_table = pd.read_csv(curves_path)
        sources = manifest.input_source_names()
        return cls(
            crs=manifest.inputs.crs,
            resolution_m=manifest.inputs.resolution_m,
            uses_weather=WEATHER_SOURCE in sources,
            uses_fire_size=FIRE_SIZE_SOURCE in sources,
            uses_ignition_distribution=manifest.data_prep.ignition_weighting == "distribution",
            known_fuel_codes=known_codes,
            multi_season_fuel_codes=multi_season,
            targets=tuple(manifest.target_names),
            fuel_curves=curve_table,
            fuel_feature_col=feature_col,
        )


def _read_csv(path: Path, what: str, out: _Collector) -> pd.DataFrame | None:
    if not path.is_file():
        out.error(f"Missing {what}.", path)
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        out.error(f"Cannot read {what}: {exc}", path)
        return None
    df.columns = [str(column).strip() for column in df.columns]
    if df.empty:
        out.error(f"{what[0].upper()}{what[1:]} has no rows.", path)
        return None
    return df


def check_fire_size_table(path: Path, out: _Collector) -> set[int] | None:
    """Validate a fire-size table; return its fire-zone IDs (GRIDCODE) or None if unusable."""
    df = _read_csv(path, "fire-size table", out)
    if df is None:
        return None
    df = df.rename(columns={old: new for old, new in FIRE_SIZE_COLUMN_ALIASES.items() if old in df.columns and new not in df.columns})
    missing = [column for column in FIRE_SIZE_FEATURE_COLS if column not in df.columns]
    if missing:
        out.error(f"Fire-size table is missing column(s) {missing}. Found: {list(df.columns)}.", path)
        return None
    zones = pd.to_numeric(df["GRIDCODE"], errors="coerce")
    bad_zones = df["GRIDCODE"][df["GRIDCODE"].notna() & (zones.isna() | (zones != zones.round()))]
    if not bad_zones.empty:
        out.error(f"GRIDCODE must hold integer fire-zone IDs; found {_listing(bad_zones.unique())}.", path)
        return None
    sizes = pd.to_numeric(df["SIZE_HA"], errors="coerce")
    non_numeric = df["SIZE_HA"][df["SIZE_HA"].notna() & sizes.isna()]
    if not non_numeric.empty:
        out.error(f"SIZE_HA must be numbers (hectares); found {_listing(non_numeric.unique())}.", path)
        return None
    if (sizes < 0).any():
        out.error(f"SIZE_HA has {int((sizes < 0).sum())} negative value(s).", path)
    if sizes.isna().any():
        out.warning(f"SIZE_HA is empty in {int(sizes.isna().sum())} row(s); they are ignored.", path)
    return {int(zone) for zone in zones.dropna()} | set(IMPUTED_FIRE_SIZE_ZONES)


@dataclass
class _Raster:
    path: Path
    crs: Any
    res_m: float  # cell size after reprojection to the model CRS
    data: np.ma.MaskedArray | None  # values inside the mask (None when not read)


class _HexelCheck:
    def __init__(
        self,
        report: CheckReport,
        project_dir: Path,
        hex_id: str,
        requirements: ModelRequirements,
        mask_scope: str,
        scenario_name: str | None,
        fire_size_zones: set[int] | None,
        inputs: bool = True,
        outputs: bool = False,
    ) -> None:
        self.hex_id = hex_id
        self.out = _Collector(report, project_dir, hexel=f"hex{hex_id}")
        self.paths = Paths(hex_id=hex_id, root_dir=project_dir)
        self.requirements = requirements
        self.mask_scope = mask_scope
        self.scenario_name = scenario_name
        self.fire_size_zones = fire_size_zones
        self.inputs = inputs
        self.outputs = outputs

    def run(self) -> None:
        if not self.hex_id.isdigit():
            self.out.error("Hexel folders must be named 'hex' followed by digits, e.g. hex07.", self.paths.base_dir)
            return
        mask = self._load_mask()
        if self.inputs:
            self._check_inputs(mask)
        if self.outputs:
            self._check_outputs(mask)

    def _check_inputs(self, mask: gpd.GeoDataFrame | None) -> None:
        dem = self._check_raster(self.paths.elevation_grid(self.hex_id), "DEM", mask)
        if dem is not None:
            self._check_resolution(dem)
        fuel = self._check_raster(self.paths.fuel_grid(self.hex_id, scenario_name=self.scenario_name), "fuel raster", mask)
        zones_raster = self._check_raster(self.paths.firezones_grid(self.hex_id), "fire-zone raster", mask)
        for raster in (fuel, zones_raster):
            if dem is not None and raster is not None:
                self._compare_resolution(raster, dem)
        if fuel is not None:
            self._check_fuel_codes(fuel)
        zones = self._fire_zones(zones_raster) if zones_raster is not None else None
        grids = self._check_ignition_grids(mask, dem)
        # The ignition distribution weights the ignition grids and also blends season-dependent fuel curves.
        zone_names = self._check_firezones_table(zones) if self.requirements.uses_ignition_distribution else None
        distribution = None
        if self.requirements.uses_ignition_distribution or self.requirements.multi_season_fuel_codes:
            distribution = self._check_ignition_distribution(zones, zone_names, grids)
        if self.requirements.multi_season_fuel_codes:
            self._check_greenup(distribution)
        if self.requirements.uses_weather:
            self._check_weather(zones)
        if self.fire_size_zones is not None and zones:
            missing = sorted(set(zones) - self.fire_size_zones)
            if missing:
                self.out.warning(
                    f"Fire zone(s) {_listing(missing)} are not in the fire-size table; the model uses the "
                    "fire-size distribution of the whole table there.",
                    self.paths.firezones_grid(self.hex_id),
                )

    def _check_outputs(self, mask: gpd.GeoDataFrame | None) -> None:
        """BurnP3+ result rasters that evaluate compares the predictions with."""
        for name in self.requirements.targets:
            spec = get_target_spec(name)
            path = getattr(self.paths, spec.path_method)(scenario_name=self.scenario_name)
            self._check_raster(path, f"BurnP3+ {spec.label.lower()} output", mask, read=False)

    # --- spatial -----------------------------------------------------------------------------

    def _load_mask(self) -> gpd.GeoDataFrame | None:
        if self.mask_scope == NO_MASK_SCOPE:
            return None
        path = self.paths.mask_grid(self.hex_id, mask_scope=self.mask_scope)
        if not path.is_file():
            self.out.error(
                f"Missing the {self.mask_scope!r} mask shapefile (defines the area to predict). For a study area without "
                f"a hexel mask, use --mask_scope {NO_MASK_SCOPE} (the whole raster extent).",
                path,
            )
            return None
        missing_parts = [path.with_suffix(ext).name for ext in (".shx", ".dbf", ".prj") if not path.with_suffix(ext).is_file()]
        if missing_parts:
            self.out.error(f"Shapefile is incomplete: {', '.join(missing_parts)} missing next to it.", path)
            return None
        try:
            mask = gpd.read_file(path)
        except Exception as exc:  # noqa: BLE001 - any GDAL/fiona failure means an unreadable file
            self.out.error(f"Cannot read the mask shapefile: {exc}", path)
            return None
        if mask.crs is None:
            self.out.error("Mask shapefile has no coordinate reference system (invalid .prj).", path)
            return None
        mask = mask[~(mask.geometry.isna() | mask.geometry.is_empty)]
        if mask.empty or float(mask.geometry.area.sum()) <= 0:
            self.out.error("Mask shapefile contains no polygon.", path)
            return None
        return mask

    def _check_raster(self, path: Path, what: str, mask: gpd.GeoDataFrame | None, read: bool = True) -> _Raster | None:
        if not path.is_file():
            self.out.error(f"Missing {what}.", path)
            return None
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    self.out.error(f"{what[0].upper()}{what[1:]} has no coordinate reference system.", path)
                    return None
                transform, _, _ = calculate_default_transform(src.crs, self.requirements.crs, src.width, src.height, *src.bounds)
                raster = _Raster(path=path, crs=src.crs, res_m=float(abs(transform.a)), data=None)
                if mask is None:
                    if self.mask_scope == NO_MASK_SCOPE and read:
                        return self._read_whole_raster(src, raster, what)
                    return raster
                shapes = list(mask.to_crs(src.crs).geometry)
                if not read:
                    left, bottom, right, top = mask.to_crs(src.crs).total_bounds
                    if left >= src.bounds.right or right <= src.bounds.left or bottom >= src.bounds.top or top <= src.bounds.bottom:
                        self.out.error(f"{what[0].upper()}{what[1:]} does not overlap the {self.mask_scope!r} mask.", path)
                        return None
                    return raster
                try:
                    data, window_transform = mask_raster(src, shapes, crop=True, filled=False, indexes=1)
                except ValueError:
                    self.out.error(f"{what[0].upper()}{what[1:]} does not overlap the {self.mask_scope!r} mask.", path)
                    return None
        except RasterioError as exc:
            self.out.error(f"Cannot read {what}: {exc}", path)
            return None

        inside = geometry_mask(shapes, out_shape=data.shape, transform=window_transform, invert=True)
        valid = inside & ~np.ma.getmaskarray(data)
        if not inside.any() or not valid.any():
            self.out.error(f"{what[0].upper()}{what[1:]} has no valid data inside the {self.mask_scope!r} mask.", path)
            return None
        coverage = valid.sum() / inside.sum()
        if coverage < MIN_VALID_COVERAGE:
            self.out.warning(f"{100 * (1 - coverage):.0f}% of the mask area is nodata in the {what}; those cells get no prediction.", path)
        raster.data = np.ma.masked_array(data.data, mask=~valid)
        return raster

    def _read_whole_raster(self, src: rasterio.io.DatasetReader, raster: _Raster, what: str) -> _Raster | None:
        data = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(data)
        if np.issubdtype(data.dtype, np.floating):
            valid &= np.isfinite(data.data)
        if not valid.any():
            self.out.error(f"{what[0].upper()}{what[1:]} has no valid data.", raster.path)
            return None
        raster.data = np.ma.masked_array(data.data, mask=~valid)
        return raster

    def _check_resolution(self, dem: _Raster) -> None:
        expected = self.requirements.resolution_m
        deviation = abs(dem.res_m - expected) / expected
        message = (
            f"DEM cells are {dem.res_m:.1f} m after reprojection to {self.requirements.crs}; the model was trained on "
            f"~{expected:g} m cells (all rasters are aligned to the DEM grid)."
        )
        path = self.paths.elevation_grid(self.hex_id)
        if deviation > RESOLUTION_ERROR_TOLERANCE:
            self.out.error(message + f" Resample the inputs to {expected:g} m.", path)
        elif deviation > RESOLUTION_WARNING_TOLERANCE:
            self.out.warning(message, path)

    def _compare_resolution(self, raster: _Raster, dem: _Raster) -> None:
        if abs(raster.res_m - dem.res_m) / dem.res_m > RESOLUTION_WARNING_TOLERANCE:
            self.out.warning(
                f"Cell size ({raster.res_m:.1f} m) differs from the DEM ({dem.res_m:.1f} m); it will be resampled "
                "(nearest neighbour) onto the DEM grid.",
                raster.path,
            )

    def _integer_values(self, raster: _Raster, what: str, path: Path) -> np.ndarray | None:
        if raster.data is None:  # not read, e.g. the mask could not be loaded (already reported)
            return None
        values = raster.data.compressed()
        if np.issubdtype(values.dtype, np.floating) and not np.allclose(values, np.rint(values), atol=1e-3):
            self.out.error(f"{what[0].upper()}{what[1:]} must hold integer codes; found non-integer values.", path)
            return None
        return np.rint(values).astype(np.int64)

    def _check_fuel_codes(self, fuel: _Raster) -> None:
        path = fuel.path
        values = self._integer_values(fuel, "fuel raster", path)
        if values is None:
            return
        codes, counts = np.unique(values, return_counts=True)
        cells = {int(code): int(count) for code, count in zip(codes, counts, strict=True)}
        curves = self.requirements.fuel_curves
        known = self.requirements.known_fuel_codes
        if curves is None or known is None:
            return
        tables = self.paths.fuel_crosswalk_table(self.hex_id)
        try:
            project_codes = read_project_fuel_codes(self.paths, self.hex_id)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            self.out.error(f"Could not read the fuel tables: {exc}", self.paths.fuel_table(self.hex_id))
            project_codes = None
        resolution = resolve_fuel_curves(
            curves,
            project_codes,
            codes=cells,
            feature_col=self.requirements.fuel_feature_col,
            project_tables=f"hex{self.hex_id}_FuelTypes.csv / hex{self.hex_id}_FuelCodeCrosswalk.csv",
        )
        if resolution.unresolved:
            listing = "; ".join(f"{code} ({cells[code]:,} cells): {reason}" for code, reason in sorted(resolution.unresolved.items()))
            known_codes = sorted(int(code) for code in curves[CODE_COL].unique())
            self.out.error(
                f"Fuel code(s) the model has no fuel curve for: {listing}. "
                f"Known FBP codes: {_listing(known_codes, limit=len(known_codes))}. Recode these cells, declare them nodata, "
                "or define them in the project's fuel tables.",
                path,
            )
        if resolution.derived:
            self.out.note(
                f"Fuel code(s) not in the model's fuel table, with curves computed from the project's fuel tables "
                f"and the FBP equations: {describe_codes(resolution.derived, cells)}.",
                tables,
            )
        if resolution.unverified:
            self.out.warning(
                "Fuel code(s) the project defines as a fuel whose curve cannot be computed here, so the model's curve "
                f"for the code is used; check that the fuel raster uses the model's codes: {describe_codes(resolution.unverified, cells)}.",
                tables,
            )
        if resolution.replaced:
            self.out.warning(
                f"Fuel code(s) the project defines differently from the model's fuel table; the project's definition "
                f"is used: {describe_codes(resolution.replaced, cells)}. Check that the fuel raster uses the same codes.",
                tables,
            )

    def _fire_zones(self, zones_raster: _Raster) -> Counter[int] | None:
        path = self.paths.firezones_grid(self.hex_id)
        values = self._integer_values(zones_raster, "fire-zone raster", path)
        if values is None:
            return None
        non_positive = int((values <= 0).sum())
        if non_positive:
            self.out.warning(f"{non_positive:,} cell(s) inside the mask have fire-zone ID <= 0 and get no zone data.", path)
        return Counter(int(zone) for zone in values[values > 0])

    def _check_ignition_grids(self, mask: gpd.GeoDataFrame | None, dem: _Raster | None) -> set[tuple[str, str]]:
        folder = self.paths.ignition_prob_dir()
        if not folder.is_dir():
            self.out.error("Missing ignition_grids folder (hexNN_ignGrid_<H|N>_<season>.tif).", folder)
            return set()
        grids: set[tuple[str, str]] = set()
        ignored = []
        known_causes = set(fire_cause_mapping.values())
        for path in sorted(folder.glob("*.tif")):
            match = _IGN_GRID_PATTERN.search(path.name)
            if match is None or match.group(1) not in known_causes:
                ignored.append(path.name)
                continue
            raster = self._check_raster(path, "ignition grid", mask, read=False)
            if raster is not None:
                grids.add((match.group(1), match.group(2)))
                if dem is not None:
                    self._compare_resolution(raster, dem)
        if ignored:
            self.out.warning(
                f"Ignored file(s) not named hexNN_ignGrid_<H|N>_<season>.tif: {_listing(ignored)}.",
                folder,
            )
        if not grids:
            self.out.error("No usable ignition grids (hexNN_ignGrid_<H|N>_<season>.tif).", folder)
        return grids

    # --- tabular -----------------------------------------------------------------------------

    def _check_firezones_table(self, zones: Counter[int] | None) -> dict[str, int] | None:
        path = self.paths.firezones_table(self.hex_id)
        df = _read_csv(path, "fire-zone table", self.out)
        if df is None:
            return None
        name_col = next((c for c in ("Name", "name") if c in df.columns), None)
        id_col = next((c for c in ("ID", "id") if c in df.columns), None)
        if name_col is None or id_col is None:
            self.out.error(f"Fire-zone table needs columns Name and ID. Found: {list(df.columns)}.", path)
            return None
        ids = pd.to_numeric(df[id_col], errors="coerce")
        if ids.isna().any() or (ids != ids.round()).any():
            self.out.error("Fire-zone table ID column must hold integer zone IDs matching the fire-zone raster.", path)
            return None
        names = {str(name).strip(): int(zone_id) for name, zone_id in zip(df[name_col], ids, strict=True)}
        if zones:
            unlisted = sorted(set(zones) - set(names.values()))
            if unlisted:
                self.out.warning(
                    f"Fire zone(s) {_listing(unlisted)} of the fire-zone raster are not listed; their ignition likelihoods are ignored.",
                    path,
                )
        return names

    def _check_ignition_distribution(
        self,
        zones: Counter[int] | None,
        zone_names: dict[str, int] | None,
        grids: set[tuple[str, str]],
    ) -> pd.DataFrame | None:
        path = self.paths.ignition_distribution_table(self.hex_id)
        df = _read_csv(path, "ignition distribution table", self.out)
        if df is None:
            return None
        required = ["Season", "Cause", "FireZone", "RelativeLikelihood"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            self.out.error(f"Ignition distribution table is missing column(s) {missing}. Found: {list(df.columns)}.", path)
            return None
        likelihood = pd.to_numeric(df["RelativeLikelihood"], errors="coerce")
        if (df["RelativeLikelihood"].notna() & likelihood.isna()).any():
            self.out.error("RelativeLikelihood must be numeric.", path)
            return None
        df = df.assign(
            Season=df["Season"].astype(str).str.strip(),
            Cause=df["Cause"].astype(str).str.strip(),
            FireZone=df["FireZone"].astype(str).str.strip(),
            RelativeLikelihood=likelihood.fillna(0.0),
        )
        cause_letters = {label: letter for letter, label in fire_cause_label_mapping.items()}
        unknown_causes = sorted(set(df["Cause"]) - set(cause_letters))
        if unknown_causes:
            self.out.warning(f"Rows with Cause {_listing(unknown_causes)} are ignored (expected {sorted(cause_letters)}).", path)
        if zone_names is not None:
            unknown_zones = sorted(set(df["FireZone"]) - set(zone_names))
            if unknown_zones:
                self.out.warning(
                    f"FireZone name(s) {_listing(unknown_zones)} are not in the fire-zone table; those rows are ignored.", path
                )
            if zones and grids:
                # Mirror load_ignition_grid_weighted: weight per (cause, season) grid present in this hexel.
                total = sum(zones.values())
                weight = sum(
                    zones[zone_names[row.FireZone]] / total * row.RelativeLikelihood
                    for row in df.itertuples()
                    if row.FireZone in zone_names
                    and zone_names[row.FireZone] in zones
                    and (cause_letters.get(row.Cause), row.Season) in grids
                )
                if weight <= 0:
                    self.out.warning(
                        "No ignition likelihood matches this hexel's ignition grids (cause/season) and fire zones; "
                        "the ignition grids will be weighted equally.",
                        path,
                    )
        return df

    def _check_greenup(self, distribution: pd.DataFrame | None) -> None:
        # Needed to blend the leafless/green fuel curves of mixedwood and deciduous fuels.
        path = self.paths.seasons_greenup_table(self.hex_id)
        df = _read_csv(path, "green-up table (Season, GreenUp)", self.out)
        if df is None:
            return
        missing = [column for column in ("Season", "GreenUp") if column not in df.columns]
        if missing:
            self.out.error(f"Green-up table is missing column(s) {missing}. Found: {list(df.columns)}.", path)
            return
        values = df["GreenUp"].astype(str).str.strip().str.lower()
        odd = sorted(set(values) - {"yes", "no"})
        if odd:
            self.out.warning(f"GreenUp value(s) {_listing(odd)} are treated as 'No' (leafless); use Yes or No.", path)
        if distribution is None:
            return
        seasons = set(df["Season"].astype(str).str.strip())
        unmapped = sorted(set(distribution["Season"]) - seasons)
        if unmapped:
            self.out.error(
                f"Season(s) {_listing(unmapped)} of the ignition distribution table have no green-up entry; "
                f"the green-up table lists {_listing(sorted(seasons))}.",
                path,
            )
        elif distribution["RelativeLikelihood"].sum() <= 0:
            self.out.error(
                "The ignition distribution has no positive RelativeLikelihood.", self.paths.ignition_distribution_table(self.hex_id)
            )

    def _check_weather(self, zones: Counter[int] | None) -> None:
        path = self.paths.weather_table(self.hex_id)
        df = _read_csv(path, "daily weather table", self.out)
        if df is None:
            return
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        renamed = df.rename(
            columns={old: new for old, new in weather_column_aliases.items() if old in df.columns and new not in df.columns}
        )
        missing = [column for column in weather_column_names if column not in renamed.columns]
        if missing:
            self.out.error(f"Daily weather table is missing column(s) {missing}.", path)
            return
        value_columns = [column for column in weather_column_names if column not in ("WeatherZone", "Season")]
        numeric = renamed[value_columns].apply(pd.to_numeric, errors="coerce")
        non_numeric = [column for column in value_columns if (renamed[column].notna() & numeric[column].isna()).any()]
        if non_numeric:
            self.out.error(f"Non-numeric values in weather column(s) {non_numeric}.", path)
            return
        empty = {column: int(count) for column, count in numeric.isna().sum().items() if count}
        if empty:
            self.out.warning(f"Empty weather values (ignored when averaging): {empty}.", path)
        for column, (low, high) in WEATHER_VALUE_RANGES.items():
            values = numeric[column].dropna()
            outside = int((values < low).sum() if low is not None else 0) + int((values > high).sum() if high is not None else 0)
            if outside:
                bounds = f"[{low if low is not None else '-inf'}, {high if high is not None else 'inf'}]"
                self.out.warning(f"{column}: {outside} value(s) outside the plausible range {bounds}.", path)

        try:
            parsed = check_weather_list(renamed.copy())
        except (ValueError, TypeError) as exc:
            self.out.error(f"Cannot parse the daily weather table: {exc}", path)
            return
        weather_zones = pd.to_numeric(parsed["WeatherZone"], errors="coerce")
        if weather_zones.isna().any() or (weather_zones != weather_zones.round()).any():
            self.out.error(
                "WeatherZone values must be zone IDs matching the fire-zone raster, either numbers (21) or a prefix and number (fru21).",
                path,
            )
            return
        if not zones:
            return
        weather_zone_ids = {int(zone) for zone in weather_zones}
        missing_zones = sorted(set(zones) - weather_zone_ids)
        if len(missing_zones) == len(zones):
            self.out.error(
                f"None of the fire zones in the fire-zone raster ({_listing(sorted(zones))}) have weather rows "
                f"(weather zones: {_listing(sorted(weather_zone_ids))}). WeatherZone must use the fire-zone IDs.",
                path,
            )
        elif missing_zones:
            self.out.warning(
                f"Fire zone(s) {_listing(missing_zones)} have no weather rows; the model uses this hexel's average weather there.",
                path,
            )


def describe_fire_size_table(bundle: ModelBundle, fire_size_table: Path) -> str:
    """Tell the user which fire-size table the model will use."""
    shipped = bundle.optional_resource_path(RESOURCE_FIRE_SIZE_TABLE)
    reference = bundle.manifest.inputs.fire_size_training_table
    if shipped is not None and Path(fire_size_table).resolve() == shipped.resolve():
        details = f"{reference.num_rows:,} fires" if reference is not None else ""
        if reference is not None and reference.note:
            details += f"; {reference.note}"
        return (
            f"Fire sizes: using the national training table shipped with the model ({shipped.name}"
            f"{', ' + details if details else ''}). Pass --fire_size_table to use your own table instead."
        )
    return f"Fire sizes: using your table {fire_size_table} instead of the model's national training table."


def check_project(
    bundle: ModelBundle,
    project_dir: str | Path,
    hex_ids: list[str] | None = None,
    fire_size_table: str | Path | None = None,
    mask_scope: str | None = None,
    scenario_name: str | None = None,
    inputs: bool = True,
    outputs: bool = False,
) -> CheckReport:
    """Check every requested hexel of ``project_dir`` against ``bundle``; never raises for bad inputs.

    ``inputs`` checks the files the model reads; ``outputs`` checks the BurnP3+ result rasters used by evaluate.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    scope = resolve_mask_scope(mask_scope, bundle)
    report = CheckReport(
        project_dir=str(project_dir),
        bundle=f"{bundle.manifest.name} v{bundle.manifest.version}",
        mask_scope=scope,
        action="evaluate" if outputs else "predict",
    )
    out = _Collector(report, project_dir)
    if not project_dir.is_dir():
        out.error(f"Project folder not found: {project_dir}")
        return report

    requirements = ModelRequirements.from_bundle(bundle)
    available = sorted(find_hex_ids(str(project_dir)))
    if not available:
        out.error("No hexel folders found. Put each area in a folder named hexNN (e.g. hex07) with spatial/ and tabular/ inside.")
        return report
    selected = available
    if hex_ids:
        selected, unknown = resolve_hex_ids(hex_ids, available)
        if unknown:
            out.error(f"Hexel(s) {unknown} not found. Available: {_listing(available)}.")

    fire_size_zones = None
    if inputs and requirements.uses_fire_size:
        table = Path(fire_size_table).expanduser().resolve() if fire_size_table else bundle.optional_resource_path(RESOURCE_FIRE_SIZE_TABLE)
        if table is None:
            out.error("This model needs a fire-size table and the bundle has none: pass --fire_size_table (GRIDCODE, SIZE_HA).")
        else:
            out.note(describe_fire_size_table(bundle, table))
            fire_size_zones = check_fire_size_table(table, out)

    for hex_id in selected:
        report.hexels.append(f"hex{hex_id}")
        _HexelCheck(report, project_dir, hex_id, requirements, scope, scenario_name, fire_size_zones, inputs, outputs).run()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference.check",
        description="Check a project's input files against a model bundle before predicting.",
    )
    parser.add_argument("--bundle", required=True, help="Model bundle folder (contains manifest.yaml).")
    parser.add_argument("--project", required=True, help="Project folder containing hexNN/ input folders.")
    parser.add_argument("--hex_ids", nargs="+", default=None, help="Hexels to check, e.g. 12 or hex12 (default: all).")
    parser.add_argument("--fire_size_table", default=None, help="Fire-size CSV to use instead of the bundle's national table.")
    parser.add_argument(
        "--mask_scope",
        choices=MASK_SCOPES,
        default=None,
        help="Area to check: the hexel mask (actual, default), the buffered mask, or none (whole raster extent).",
    )
    parser.add_argument(
        "--scenario_name",
        default=None,
        help="Check fuel raster hexNN_fbp_<scenario_name>.tif instead of hexNN_fbp.tif (default: the model's training scenario, if any).",
    )
    parser.add_argument(
        "--outputs", action="store_true", help="Also check the BurnP3+ output rasters (results/) that inference.evaluate compares against."
    )
    parser.add_argument("--json", default=None, help="Also write the report as JSON to this file.")
    parser.add_argument("--skip_checksums", action="store_true", help="Skip bundle checksum verification (faster start-up).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit code 0: no errors; 1: input errors found; 2: the check itself could not run."""
    args = build_arg_parser().parse_args(argv)
    try:
        bundle = load_bundle(args.bundle, verify_checksums=not args.skip_checksums)
        report = check_project(
            bundle,
            args.project,
            hex_ids=args.hex_ids,
            fire_size_table=args.fire_size_table,
            mask_scope=args.mask_scope,
            scenario_name=resolve_scenario_name(args.scenario_name, bundle),
            outputs=args.outputs,
        )
    except (BundleError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(report.format())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
