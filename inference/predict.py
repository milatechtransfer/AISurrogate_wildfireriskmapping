"""
Predict burn probability, fire intensity and rate of spread for a BurnP3+ project from a model bundle.

Only the project *inputs* are needed (fuel, DEM, fire zones, ignition grids, weather, ignition
distribution, green-up); BurnP3+ outputs are not read. Every model-side resource (weights,
normalization statistics, fuel curves, channel layout, the national fire-size table) comes from the
bundle, so this runs without access to the training data.

Example:
    python -m inference.predict \\
        --bundle nrcan-surrogate-bp-fi-ros-v1.0 \\
        --project path/to/project \\
        --output predictions/

Outputs (per hexel ``hexNN/``): ``hexNN_bp.tif``, ``hexNN_fi.tif``, ``hexNN_ros.tif`` in physical units and,
when the bundle predicts BP and FI, ``hexNN_hazard_raw.tif``, ``hexNN_hazard_scaled.tif``,
``hexNN_hazard_class.tif``. A ``run_manifest.json`` and ``predict.log`` are written to the output folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation.hexel_loader import load_spatial_features_per_hexel
from data_preparation.paths import Paths
from data_preparation.process_hexels_into_grids import get_split_hexel_window
from data_preparation.process_tabular_data import build_weather_table, process_fire_size_distribution_table
from data_preparation.utils import find_hex_ids
from inference.bundle import (
    FIRE_SIZE_SOURCE,
    MASK_SCOPES,
    NO_MASK_SCOPE,
    RESOURCE_DATASET_NORM_STATS,
    RESOURCE_FIRE_SIZE_NORM_PARAMS,
    RESOURCE_FIRE_SIZE_TABLE,
    RESOURCE_FUEL_CURVES,
    RESOURCE_WEATHER_NORM_PARAMS,
    WEATHER_SOURCE,
    BundleError,
    ModelBundle,
    data_mask_scope,
    load_bundle,
    resolve_device,
    resolve_mask_scope,
    resolve_scenario_name,
    sha256_file,
    utc_timestamp,
)
from inference.check import CheckReport, check_project, describe_fire_size_table, resolve_hex_ids
from inference.fuels import FuelCurveResolution, read_project_fuel_codes, resolve_fuel_curves
from inference.predictor import BurnRiskPredictor
from src.config import SpatializedTabularParams
from src.datasets.dataset import MultiSourceDataset
from src.datasets.fuel_utils import _FEATURE_COLUMN, FUEL_CURVE_ENCODINGS
from src.datasets.postprocessing.hazard import bin_scaled_hazard, compute_raw_hazard, scale_hazard
from src.datasets.postprocessing.utils import get_predicted_hexel, get_prediction_mask_channel_indices
from src.datasets.sources.grids import GridSourceResources
from src.datasets.targets import get_target_spec
from src.datasets.utils import get_data_source_class, get_dataset_dimensions

logger = logging.getLogger("inference.predict")

WORK_DIRNAME = "_work"
RUN_MANIFEST_FILENAME = "run_manifest.json"
LOG_FILENAME = "predict.log"
PREDICT_MASK_SCOPES = MASK_SCOPES
OUTPUT_NODATA = -9999.0
HAZARD_CLASS_NODATA = 0


class PredictError(RuntimeError):
    """A problem with the user's inputs or options, reported without a traceback."""


@dataclass
class HexelResult:
    hex_id: str
    outputs: dict[str, str]
    num_patches: int
    seconds: float
    fuel_curves: dict[str, dict[str, str]] | None = None  # fuel codes whose curves came from the project's fuel tables


@dataclass
class PredictRun:
    output_dir: Path
    hexels: list[HexelResult] = field(default_factory=list)
    input_check: CheckReport | None = None
    fire_size_note: str | None = None


def run_input_checks(
    bundle: ModelBundle,
    project_dir: Path,
    hex_ids: list[str],
    fire_size_table: Path | None,
    mask_scope: str,
    scenario_name: str | None,
) -> CheckReport:
    """Run the input checks and log their warnings."""
    logger.info("Checking the project inputs")
    report = check_project(
        bundle, project_dir, hex_ids, fire_size_table=fire_size_table, mask_scope=mask_scope, scenario_name=scenario_name
    )
    for finding in report.warnings:
        logger.warning("%s", finding.describe())
    return report


def raise_on_input_errors(report: CheckReport) -> None:
    if not report.ok:
        details = "\n".join(f"  - {finding.describe()}" for finding in report.errors)
        raise PredictError(f"The project inputs have {len(report.errors)} error(s); nothing was predicted:\n{details}")


def discover_hex_ids(project_dir: Path, requested: list[str] | None) -> list[str]:
    available = sorted(find_hex_ids(str(project_dir)))
    if not available:
        raise PredictError(f"No hexel folders (hexNN/) found in project {project_dir}.")
    if not requested:
        return available
    found, missing = resolve_hex_ids(requested, available)
    if missing:
        raise PredictError(f"Hexel(s) {missing} not found in {project_dir}. Available: {available}")
    return found


def _spatialized_params(bundle: ModelBundle) -> dict[str, SpatializedTabularParams]:
    params: dict[str, SpatializedTabularParams] = {}
    for source in bundle.manifest.data.input_sources:
        if source.name == "grid":
            continue
        if source.name not in (WEATHER_SOURCE, FIRE_SIZE_SOURCE) or not isinstance(source.params, SpatializedTabularParams):
            raise BundleError(f"Input source {source.name!r} of this bundle is not supported by predict.")
        params[source.name] = source.params
    return params


def prepare_project_tables(bundle: ModelBundle, project_dir: Path, work_dir: Path, fire_size_table: Path | None) -> None:
    """Build the weather and fire-size tables with the bundle's (training) normalization parameters."""
    sources = _spatialized_params(bundle)
    if WEATHER_SOURCE in sources:
        logger.info("Building weather table from %s", project_dir)
        build_weather_table(
            root_dir=project_dir,
            save_path=work_dir / sources[WEATHER_SOURCE].csv_name,
            norm_params_path=bundle.resource_path(RESOURCE_WEATHER_NORM_PARAMS),
        )
    if FIRE_SIZE_SOURCE in sources:
        if fire_size_table is None:
            raise PredictError(
                "This model needs a fire-size table and the bundle has none: pass --fire_size_table (CSV with columns GRIDCODE, SIZE_HA)."
            )
        if not fire_size_table.is_file():
            raise PredictError(f"Fire-size table not found: {fire_size_table}")
        logger.info("Processing fire-size table %s", fire_size_table)
        process_fire_size_distribution_table(
            input_path=fire_size_table,
            output_path=work_dir / sources[FIRE_SIZE_SOURCE].csv_name,
            norm_params_path=bundle.resource_path(RESOURCE_FIRE_SIZE_NORM_PARAMS),
        )


def prepare_hexel_patches(
    bundle: ModelBundle,
    project_dir: Path,
    hex_id: str,
    work_dir: Path,
    mask_scope: str,
    scenario_name: str | None = None,
) -> int:
    """Stack the hexel's input rasters and split them into model-sized patches in ``work_dir``."""
    prep = bundle.manifest.data_prep
    feature_channel_map_path = work_dir / f"feature_channel_map_{prep.modelling_approach}.json"
    stacked_feats, mask, season_cause_mapping = load_spatial_features_per_hexel(
        root_dir=str(project_dir),
        hex_id=hex_id,
        feature_channel_map_path=str(feature_channel_map_path),
        modelling_approach=prep.modelling_approach,
        mask_scope=data_mask_scope(mask_scope),
        ignition_weighting=prep.ignition_weighting,
        fuel_representation=prep.fuel_representation,
        scenario_name=scenario_name,
        load_targets=False,
    )
    if stacked_feats is None or mask is None:
        raise PredictError(f"Could not load the input rasters of hex{hex_id}.")

    with open(feature_channel_map_path) as handle:
        prepared_map = json.load(handle)
    if prepared_map != bundle.manifest.model_io.feature_channel_map:
        raise BundleError(
            f"Prepared channel layout {prepared_map} does not match the layout the model was trained on "
            f"{bundle.manifest.model_io.feature_channel_map}. The bundle and this software version are incompatible."
        )

    get_split_hexel_window(
        season_cause_stacked_feats=stacked_feats,
        season_cause_mask=mask,
        season_cause_mapping=season_cause_mapping,
        out_dir=str(work_dir),
        root_dir=str(project_dir),
        hex_id=hex_id,
        win_h=prep.win_h,
        win_w=prep.win_w,
        overlap_ratio=prep.overlap_ratio,
        mask_scope=data_mask_scope(mask_scope),
    )
    metadata = pd.read_csv(work_dir / f"meta_hex_{hex_id}.csv")
    return int((metadata["valid_ratio"] > bundle.manifest.data.valid_mask_threshold).sum()) if not metadata.empty else 0


def resolve_hexel_fuel_curves(
    bundle: ModelBundle, project_dir: Path, hex_id: str, work_dir: Path, scenario_name: str | None = None
) -> tuple[Path | None, FuelCurveResolution | None]:
    """Fuel curve table for one hexel: the bundle's, plus curves for its other fuel codes from the project's fuel tables.

    Returns the table to use (the bundle's own file when nothing had to be added) and how the codes were resolved.
    """
    if RESOURCE_FUEL_CURVES not in bundle.manifest.resources:
        return None, None
    model_path = bundle.resource_path(RESOURCE_FUEL_CURVES)
    encoding = bundle.manifest.grid_params().fuel_feats_encoding
    if encoding not in FUEL_CURVE_ENCODINGS:
        return model_path, None
    paths = Paths(hex_id=hex_id, root_dir=project_dir)
    with rasterio.open(paths.fuel_grid(hex_id, scenario_name=scenario_name)) as src:
        values = src.read(1, masked=True).compressed()
    codes = {int(code) for code in np.unique(values[np.isfinite(values)]).astype(np.int64)}
    resolution = resolve_fuel_curves(
        pd.read_csv(model_path), read_project_fuel_codes(paths, hex_id), codes=codes, feature_col=_FEATURE_COLUMN[encoding]
    )
    if not resolution.changed:
        return model_path, resolution
    path = work_dir / f"fuel_curves_hex{hex_id}.csv"
    resolution.curves.to_csv(path, index=False)
    return path, resolution


def build_dataset(
    bundle: ModelBundle, project_dir: Path, hex_id: str, work_dir: Path, fuel_curves_path: Path | None = None
) -> MultiSourceDataset:
    """Build the dataset for one prepared hexel, taking every model-side resource from the bundle.

    ``fuel_curves_path`` replaces the bundle's fuel curve table (see ``resolve_hexel_fuel_curves``).
    """
    manifest = bundle.manifest
    if fuel_curves_path is None and RESOURCE_FUEL_CURVES in manifest.resources:
        fuel_curves_path = bundle.resource_path(RESOURCE_FUEL_CURVES)
    sources = {}
    for source in manifest.data.input_sources:
        source_kwargs: dict[str, Any] = {"root_dir": str(work_dir), "params": source.params}
        if source.name == "grid":
            source_kwargs["resources"] = GridSourceResources(
                norm_stats=bundle.read_json_resource(RESOURCE_DATASET_NORM_STATS),
                channel_feature_map=manifest.model_io.feature_channel_map,
                fuel_curves_path=fuel_curves_path,
                season_data_dir=project_dir,
                hex_ids=[hex_id],
            )
        sources[source.name] = get_data_source_class(source.name)(**source_kwargs)
    dataset = MultiSourceDataset(
        csv_name=f"meta_hex_{hex_id}.csv",
        root_dir=str(work_dir),
        sources=sources,
        filename_col=manifest.data.filename_col,
        valid_mask_threshold=manifest.data.valid_mask_threshold,
    )
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(dataset)
    expected = (manifest.model_io.spatial_input_channels, dict(manifest.model_io.auxiliary_input_dims))
    if (spatial_channels, auxiliary_input_dims) != expected:
        raise BundleError(
            f"Prepared inputs have {spatial_channels} channels / {auxiliary_input_dims}, but the model expects "
            f"{expected[0]} / {expected[1]}."
        )
    return dataset


def run_model(predictor: BurnRiskPredictor, dataset: MultiSourceDataset, batch_size: int, num_workers: int) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(predictor.device).type == "cuda",
    )
    predictions = []
    for batch in tqdm(loader, desc="Predicting patches", leave=False):
        predictions.append(predictor(batch["grid"][0], auxiliary_inputs=batch))
    return torch.cat(predictions, dim=0).numpy()


def stitch_predictions(
    bundle: ModelBundle,
    project_dir: Path,
    hex_id: str,
    work_dir: Path,
    dataset: MultiSourceDataset,
    predictions: np.ndarray,
    mask_scope: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Stitch patch predictions back into full hexel grids in physical units."""
    manifest = bundle.manifest
    approach = str(manifest.data_prep.modelling_approach)
    mask_channel_indices = get_prediction_mask_channel_indices(
        data_dir=str(work_dir),
        modelling_approach=approach,
        grid_params=manifest.grid_params(),
        prediction_support_policy=manifest.evaluation.prediction_support_policy,
    )
    if predictions.shape[1] != len(manifest.targets):
        raise BundleError(f"Model returned {predictions.shape[1]} channels for targets {manifest.target_names}.")

    grids: dict[str, np.ndarray] = {}
    profile: dict[str, Any] = {}
    for index, target in enumerate(manifest.targets):
        spec = get_target_spec(target.name)
        grids[target.name], profile = get_predicted_hexel(
            base_dir=str(work_dir),
            raw_data_dir=str(project_dir),
            test_df=dataset.metadata,
            predictions=predictions[:, index],
            min_target_val=target.min_value,
            max_target_val=target.max_value,
            hex_id=hex_id,
            modelling_approach=approach,
            out_norm=target.out_norm,
            target_log_mean=target.log_mean,
            target_log_std=target.log_std,
            target_channel_index=manifest.model_io.feature_channel_map[spec.channel_key][0],
            prediction_mask_channel_indices=mask_channel_indices,
            mask_scope=data_mask_scope(mask_scope),
        )
    return grids, dict(profile)


def compute_hazard_grids(bundle: ModelBundle, grids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not {"bp", "fi"} <= set(grids):
        return {}
    hazard = bundle.manifest.hazard
    hazard_grids = {"hazard_raw": compute_raw_hazard(grids["bp"], grids["fi"], fi_cap=hazard.fi_cap)}
    if hazard.scale_denominator is not None:
        scaled = scale_hazard(hazard_grids["hazard_raw"], denominator=hazard.scale_denominator, scale_to=hazard.scale_to)
        hazard_grids["hazard_scaled"] = scaled
        hazard_grids["hazard_class"] = bin_scaled_hazard(scaled, thresholds=hazard.bin_thresholds, invalid_class=HAZARD_CLASS_NODATA)
    return hazard_grids


def write_geotiff(path: Path, grid: np.ndarray, profile: dict[str, Any], description: str, units: str | None = None) -> None:
    out_profile = {key: value for key, value in profile.items() if key not in {"blockxsize", "blockysize", "tiled"}}
    out_profile.update(driver="GTiff", count=1, compress="deflate")
    if np.issubdtype(grid.dtype, np.integer):
        out_profile.update(dtype="uint8", nodata=HAZARD_CLASS_NODATA)
        data = grid.astype(np.uint8)
    else:
        out_profile.update(dtype="float32", nodata=OUTPUT_NODATA)
        data = np.where(np.isfinite(grid), grid, OUTPUT_NODATA).astype(np.float32)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)
        if units:
            dst.update_tags(1, units=units)


def write_hexel_outputs(
    bundle: ModelBundle,
    hex_id: str,
    grids: dict[str, np.ndarray],
    hazard_grids: dict[str, np.ndarray],
    profile: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    hexel_dir = output_dir / f"hex{hex_id}"
    hexel_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for target in bundle.manifest.targets:
        path = hexel_dir / f"hex{hex_id}_{target.name}.tif"
        write_geotiff(path, grids[target.name], profile, description=target.label, units=target.units)
        outputs[target.name] = str(path.relative_to(output_dir).as_posix())
    descriptions = {
        "hazard_raw": ("Raw hazard (BP x capped FI)", "kW/m"),
        "hazard_scaled": ("Scaled hazard index", f"0-{bundle.manifest.hazard.scale_to:g}"),
        "hazard_class": ("Hazard class", "class (1-based, 0 = nodata)"),
    }
    for name, grid in hazard_grids.items():
        path = hexel_dir / f"hex{hex_id}_{name}.tif"
        write_geotiff(path, grid, profile, description=descriptions[name][0], units=descriptions[name][1])
        outputs[name] = str(path.relative_to(output_dir).as_posix())
    return outputs


def predict_hexel(
    bundle: ModelBundle,
    predictor: BurnRiskPredictor,
    project_dir: Path,
    hex_id: str,
    work_dir: Path,
    output_dir: Path,
    mask_scope: str,
    batch_size: int,
    num_workers: int,
    hazard: bool = True,
    scenario_name: str | None = None,
) -> HexelResult:
    start = time.perf_counter()
    logger.info("hex%s: preparing input patches", hex_id)
    num_patches = prepare_hexel_patches(bundle, project_dir, hex_id, work_dir, mask_scope, scenario_name=scenario_name)
    if num_patches == 0:
        area = "rasters" if mask_scope == NO_MASK_SCOPE else f"{mask_scope!r} mask"
        raise PredictError(f"hex{hex_id} has no valid input pixels inside the {area}; check the rasters and the mask.")
    fuel_curves_path, fuel_resolution = resolve_hexel_fuel_curves(bundle, project_dir, hex_id, work_dir, scenario_name=scenario_name)
    if fuel_resolution is not None and fuel_resolution.derived:
        logger.info("hex%s: fuel curves computed from the project's fuel tables for code(s) %s", hex_id, sorted(fuel_resolution.derived))
    dataset = build_dataset(bundle, project_dir, hex_id, work_dir, fuel_curves_path=fuel_curves_path)
    logger.info("hex%s: running the model on %d patches", hex_id, len(dataset))
    predictions = run_model(predictor, dataset, batch_size=batch_size, num_workers=num_workers)
    grids, profile = stitch_predictions(bundle, project_dir, hex_id, work_dir, dataset, predictions, mask_scope)
    hazard_grids = compute_hazard_grids(bundle, grids) if hazard else {}
    outputs = write_hexel_outputs(bundle, hex_id, grids, hazard_grids, profile, output_dir)
    seconds = time.perf_counter() - start
    logger.info("hex%s: done in %.1f s -> %s", hex_id, seconds, output_dir / f"hex{hex_id}")
    return HexelResult(
        hex_id=hex_id,
        outputs=outputs,
        num_patches=len(dataset),
        seconds=round(seconds, 2),
        fuel_curves=fuel_resolution.summary() if fuel_resolution is not None and fuel_resolution.changed else None,
    )


def output_overlap_error(output_dir: Path, project_dir: Path) -> str | None:
    """Why ``output_dir`` cannot be used for a project's results (it would mix with or overwrite its inputs), else None."""
    if output_dir == project_dir or output_dir in project_dir.parents:
        return f"--output {output_dir} contains the project {project_dir}; choose a folder outside the project."
    if output_dir.is_relative_to(project_dir) and output_dir.relative_to(project_dir).parts[0].startswith("hex"):
        return f"--output {output_dir} is inside the project's input folder; choose a folder outside the hexNN folders."
    return None


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise PredictError(f"Output folder {output_dir} is not empty. Choose another --output or pass --overwrite.")
        for child in output_dir.iterdir():
            if (child.is_dir() and child.name.startswith("hex")) or child.name in {WORK_DIRNAME, RUN_MANIFEST_FILENAME, LOG_FILENAME}:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def log_to_file(path: Path) -> Iterator[logging.FileHandler]:
    """Copy every log message of the run to ``path``, even when the caller did not configure logging."""
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    package_logger = logging.getLogger("inference")
    previous_level = package_logger.level
    if package_logger.getEffectiveLevel() > logging.INFO:
        package_logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        package_logger.setLevel(previous_level)


@contextmanager
def console_logging() -> Iterator[None]:
    """Show INFO messages on the console for the duration of a CLI command."""
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    # Some data_preparation modules call logging.basicConfig at import time; reuse that console handler.
    has_console = any(type(handler) is logging.StreamHandler for handler in root_logger.handlers)
    console_handler: logging.Handler = logging.NullHandler()
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        root_logger.removeHandler(console_handler)
        root_logger.setLevel(previous_level)


def git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_fire_size_table(bundle: ModelBundle, fire_size_table: str | Path | None) -> Path | None:
    """The user's fire-size table if given, otherwise the training table shipped in the bundle (if any)."""
    if fire_size_table is not None:
        return Path(fire_size_table).expanduser().resolve()
    return bundle.optional_resource_path(RESOURCE_FIRE_SIZE_TABLE)


def _write_run_manifest(
    run: PredictRun,
    bundle: ModelBundle,
    project_dir: Path,
    fire_size_table: Path | None,
    options: dict[str, Any],
    started_at: str,
    status: str,
    error: str | None = None,
) -> None:
    fire_size_info = None
    if fire_size_table is not None and fire_size_table.is_file():
        training_table = bundle.manifest.inputs.fire_size_training_table
        sha = sha256_file(fire_size_table)
        fire_size_info = {
            "source": "bundle" if fire_size_table == bundle.optional_resource_path(RESOURCE_FIRE_SIZE_TABLE) else "user",
            "path": str(fire_size_table),
            "sha256": sha,
            "matches_training_table": bool(training_table and training_table.sha256 == sha),
        }
    manifest = {
        "status": status,
        "error": error,
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "bundle": {
            "name": bundle.manifest.name,
            "version": bundle.manifest.version,
            "path": str(bundle.root),
            "weights_sha256": bundle.manifest.weights.sha256,
        },
        "project_dir": str(project_dir),
        "fire_size_table": fire_size_info,
        "options": options,
        "software": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "units": {target.name: target.units for target in bundle.manifest.targets},
        "nodata": {"float_rasters": OUTPUT_NODATA, "hazard_class": HAZARD_CLASS_NODATA},
        "input_check": (
            {"errors": [asdict(f) for f in run.input_check.errors], "warnings": [asdict(f) for f in run.input_check.warnings]}
            if run.input_check is not None
            else None
        ),
        "hexels": [result.__dict__ for result in run.hexels],
    }
    with open(run.output_dir / RUN_MANIFEST_FILENAME, "w") as handle:
        json.dump(manifest, handle, indent=2)


def run_predict(
    bundle_dir: str | Path,
    project_dir: str | Path,
    output_dir: str | Path,
    fire_size_table: str | Path | None = None,
    hex_ids: list[str] | None = None,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    mask_scope: str | None = None,
    hazard: bool = True,
    overwrite: bool = False,
    keep_work_dir: bool = False,
    scenario_name: str | None = None,
    verify_checksums: bool = True,
    check_inputs: bool = True,
) -> PredictRun:
    """Run the model bundle on every requested hexel of a project and write GeoTIFFs to ``output_dir``."""
    started_at = utc_timestamp()
    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise PredictError(f"Project folder not found: {project_dir}")
    overlap = output_overlap_error(output_dir, project_dir)
    if overlap:
        raise PredictError(overlap)

    bundle = load_bundle(bundle_dir, verify_checksums=verify_checksums)
    fire_size_path = resolve_fire_size_table(bundle, fire_size_table)
    try:
        scope = resolve_mask_scope(mask_scope, bundle)
    except (BundleError, ValueError) as exc:
        raise PredictError(str(exc)) from exc
    scenario_name = resolve_scenario_name(scenario_name, bundle)
    selected_hex_ids = discover_hex_ids(project_dir, hex_ids)

    _prepare_output_dir(output_dir, overwrite)
    with log_to_file(output_dir / LOG_FILENAME) as log_handler:
        return _run_predict(
            bundle,
            project_dir,
            output_dir,
            fire_size_path,
            selected_hex_ids,
            scope,
            log_handler,
            started_at,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            hazard=hazard,
            keep_work_dir=keep_work_dir,
            scenario_name=scenario_name,
            check_inputs=check_inputs,
        )


def _run_predict(
    bundle: ModelBundle,
    project_dir: Path,
    output_dir: Path,
    fire_size_path: Path | None,
    selected_hex_ids: list[str],
    scope: str,
    log_handler: logging.FileHandler,
    started_at: str,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    hazard: bool,
    keep_work_dir: bool,
    scenario_name: str | None,
    check_inputs: bool,
) -> PredictRun:
    resolved_device = resolve_device(device)
    options = {
        "hex_ids": selected_hex_ids,
        "device": str(resolved_device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "mask_scope": scope,
        "hazard": hazard,
        "scenario_name": scenario_name,
        "check_inputs": check_inputs,
    }
    run = PredictRun(output_dir=output_dir)
    work_dir = output_dir / WORK_DIRNAME
    try:
        logger.info(
            "Bundle %s v%s | device %s | %d hexel(s)", bundle.manifest.name, bundle.manifest.version, resolved_device, len(selected_hex_ids)
        )
        if FIRE_SIZE_SOURCE in bundle.manifest.input_source_names() and fire_size_path is not None:
            run.fire_size_note = describe_fire_size_table(bundle, fire_size_path)
            logger.info("%s", run.fire_size_note)
        if check_inputs:
            run.input_check = run_input_checks(bundle, project_dir, selected_hex_ids, fire_size_path, scope, scenario_name)
            raise_on_input_errors(run.input_check)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "numpy_files").mkdir(exist_ok=True)
        prepare_project_tables(bundle, project_dir, work_dir, fire_size_path)
        predictor = BurnRiskPredictor.from_bundle(bundle, device=resolved_device)
        for hex_id in selected_hex_ids:
            run.hexels.append(
                predict_hexel(
                    bundle,
                    predictor,
                    project_dir,
                    hex_id,
                    work_dir,
                    output_dir,
                    mask_scope=scope,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    hazard=hazard,
                    scenario_name=scenario_name,
                )
            )
        _write_run_manifest(run, bundle, project_dir, fire_size_path, options, started_at, status="success")
        logger.info("Wrote predictions for %d hexel(s) to %s", len(run.hexels), output_dir)
        return run
    except Exception as exc:
        log_handler.stream.write(traceback.format_exc())
        _write_run_manifest(run, bundle, project_dir, fire_size_path, options, started_at, status="failed", error=str(exc))
        raise
    finally:
        if not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference.predict",
        description="Predict burn probability, fire intensity and rate of spread for a BurnP3+ project from a model bundle.",
    )
    parser.add_argument("--bundle", required=True, help="Model bundle folder (contains manifest.yaml).")
    parser.add_argument("--project", required=True, help="Project folder containing hexNN/ input folders.")
    parser.add_argument("--output", required=True, help="Folder to write predictions to.")
    parser.add_argument(
        "--fire_size_table", default=None, help="Fire-size CSV (GRIDCODE, SIZE_HA) to use instead of the bundle's national table."
    )
    parser.add_argument("--hex_ids", nargs="+", default=None, help="Hexels to predict, e.g. 12 or hex12 (default: all).")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, cuda:1 or mps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Patches per model call (lower it if memory runs out).")
    parser.add_argument("--num_workers", type=int, default=0, help="Data-loading worker processes (0 is safest on Windows/macOS).")
    parser.add_argument(
        "--mask_scope",
        choices=PREDICT_MASK_SCOPES,
        default=None,
        help="Area to predict: actual hexel mask (default), buffer, or none (whole raster extent, e.g. a regional study area).",
    )
    parser.add_argument("--no_hazard", action="store_true", help="Do not write hazard rasters.")
    parser.add_argument(
        "--scenario_name",
        default=None,
        help="Use fuel raster hexNN_fbp_<scenario_name>.tif instead of hexNN_fbp.tif (default: the model's training scenario, if any).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace previous predictions in --output.")
    parser.add_argument("--keep_work_dir", action="store_true", help=f"Keep intermediate patches in <output>/{WORK_DIRNAME}.")
    parser.add_argument("--skip_checksums", action="store_true", help="Skip bundle checksum verification (faster start-up).")
    parser.add_argument("--skip_check", action="store_true", help="Do not check the project inputs first (see inference.check).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        with console_logging():
            run = run_predict(
                bundle_dir=args.bundle,
                project_dir=args.project,
                output_dir=args.output,
                fire_size_table=args.fire_size_table,
                hex_ids=args.hex_ids,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                mask_scope=args.mask_scope,
                hazard=not args.no_hazard,
                overwrite=args.overwrite,
                keep_work_dir=args.keep_work_dir,
                scenario_name=args.scenario_name,
                verify_checksums=not args.skip_checksums,
                check_inputs=not args.skip_check,
            )
    except (PredictError, BundleError, FileNotFoundError, ValueError, KeyError) as exc:
        log_path = Path(args.output) / LOG_FILENAME
        details = f"\n(Full details in {log_path})" if log_path.is_file() else ""
        print(f"\nError: {exc}{details}", file=sys.stderr)
        return 2
    print(f"\nPredicted {len(run.hexels)} hexel(s) into {run.output_dir}")
    if run.fire_size_note:
        print(run.fire_size_note)
    if run.input_check is not None and run.input_check.warnings:
        print(f"{len(run.input_check.warnings)} input warning(s) were logged; see {run.output_dir / LOG_FILENAME}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
