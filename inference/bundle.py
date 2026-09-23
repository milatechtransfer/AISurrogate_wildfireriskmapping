"""
Portable model bundles for operational inference.

A bundle is a self-contained directory holding everything needed to run a trained surrogate
model on a new project without access to the original training filesystem:

    <bundle>/
    ├── manifest.yaml            # model/data/prep config, target normalization, provenance, checksums
    ├── model.pt                 # weights only (state_dict), loadable with torch.load(weights_only=True)
    ├── MODEL_CARD.md
    ├── SHA256SUMS
    ├── norm/                    # normalization statistics fitted on the training split
    └── lookups/                 # static lookup tables (fuel curves, fire sizes, training feature-channel layout)

Bundles are created with ``python -m inference.export_bundle`` and loaded with :func:`load_bundle`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field, ValidationError

from data_preparation.paths import normalize_mask_scope
from src.config import DataPrepConfig, DataSourceConfig, GridParams, ModelConfig, SpatializedTabularParams
from src.datasets.fuel_utils import _FEATURE_COLUMN, FUEL_CURVE_ENCODINGS, read_curves
from src.datasets.targets import get_target_spec
from src.datasets.utils import apply_bp_nodata_zero_range

BUNDLE_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
WEIGHTS_FILENAME = "model.pt"
MODEL_CARD_FILENAME = "MODEL_CARD.md"
CHECKSUMS_FILENAME = "SHA256SUMS"

# Physical units of each target as written to the output GeoTIFFs (after denormalization).
TARGET_UNITS: dict[str, str] = {
    "bp": "probability (0-1)",
    "fi": "kW/m",
    "ros": "m/min",
}

# Resource keys stored in every bundle and their location inside the bundle directory.
RESOURCE_DATASET_NORM_STATS = "dataset_norm_stats"
RESOURCE_WEATHER_NORM_PARAMS = "weather_norm_params"
RESOURCE_FIRE_SIZE_NORM_PARAMS = "fire_size_norm_params"
RESOURCE_FUEL_CURVES = "fuel_curves"

# Area to predict/evaluate: the hexel mask, the buffered mask, or no mask (the whole raster extent,
# e.g. a regional study area without a hexel shapefile).
NO_MASK_SCOPE = "none"
MASK_SCOPES = ("actual", "buffer", NO_MASK_SCOPE)
RESOURCE_FEATURE_CHANNEL_MAP = "feature_channel_map"
RESOURCE_FIRE_SIZE_TABLE = "fire_size_table"

# Names of the zone-level (spatialized tabular) input sources built from project tables.
WEATHER_SOURCE = "spatialized_weather"
FIRE_SIZE_SOURCE = "spatialized_fire_size"


class BundleError(RuntimeError):
    """Raised when a bundle is missing, incomplete, corrupted, or incompatible."""


class ResourceEntry(BaseModel):
    path: str  # relative to the bundle root, POSIX separators
    sha256: str
    description: str = ""


class TargetEntry(BaseModel):
    name: str
    label: str
    units: str
    activation: Literal["sigmoid", "identity"]
    out_norm: str
    min_value: float | None = None
    max_value: float | None = None
    log_mean: float | None = None
    log_std: float | None = None


class HazardEntry(BaseModel):
    fi_cap: float | None
    scale_to: float
    bin_thresholds: list[float]
    scale_denominator: float | None = None
    scale_denominator_source: str | None = None


class ModelIOEntry(BaseModel):
    spatial_input_channels: int
    auxiliary_input_dims: dict[str, int]
    # Channel layout of the prepared patch arrays the model was trained on.
    feature_channel_map: dict[str, list[int]]


class InferenceDataEntry(BaseModel):
    filename_col: str = "filename"
    valid_mask_threshold: float = 0.01
    input_sources: list[DataSourceConfig]


class EvaluationEntry(BaseModel):
    bp_nodata_as_zero: bool = True
    prediction_support_policy: str = "input"


class ReferenceTableEntry(BaseModel):
    """Describes a raw table the model was trained with."""

    filename: str
    sha256: str
    num_rows: int
    columns: list[str]
    note: str = ""


class InputSpecEntry(BaseModel):
    crs: str = "ESRI:102002"
    resolution_m: float = 100.0
    fire_size_training_table: ReferenceTableEntry | None = None


class ProvenanceEntry(BaseModel):
    exported_at: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    epoch: int | None = None
    seed: int | None = None
    best_checkpoint_metrics: dict[str, float] = Field(default_factory=dict)
    selection_note: str = ""
    code_git_commit: str | None = None
    code_git_dirty: bool | None = None
    training_data_root: str | None = None
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)


class BundleManifest(BaseModel):
    bundle_format_version: int
    name: str
    version: str
    description: str = ""
    targets: list[TargetEntry]
    model: ModelConfig
    model_io: ModelIOEntry
    data: InferenceDataEntry
    data_prep: DataPrepConfig
    evaluation: EvaluationEntry
    hazard: HazardEntry
    inputs: InputSpecEntry
    weights: ResourceEntry
    resources: dict[str, ResourceEntry]
    provenance: ProvenanceEntry

    @property
    def target_names(self) -> list[str]:
        return [target.name for target in self.targets]

    def grid_params(self) -> GridParams:
        return get_grid_params(self.data.input_sources)

    def input_source_names(self) -> list[str]:
        return [source.name for source in self.data.input_sources]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def get_grid_params(input_sources: list[DataSourceConfig]) -> GridParams:
    for source in input_sources:
        if source.name == "grid" and isinstance(source.params, GridParams):
            return source.params
    raise BundleError("The model configuration has no 'grid' input source.")


def fuel_curve_length(fuel_curves_csv: str | Path, fuel_feats_encoding: str) -> int:
    """Number of ISI bins in each fuel curve (the model's ``fuel_curve`` auxiliary input size)."""
    curves_by_code = read_curves(ros_csv_path=fuel_curves_csv, feature_col=_FEATURE_COLUMN[fuel_feats_encoding])
    lengths = {len(curve) for season_curves in curves_by_code.values() for curve in season_curves.values()}
    if len(lengths) != 1:
        raise BundleError(f"Fuel curves in {fuel_curves_csv} have inconsistent lengths: {sorted(lengths)}.")
    return lengths.pop()


def compute_model_io_dims(
    input_sources: list[DataSourceConfig],
    feature_channel_map: dict[str, list[int]],
    fuel_curves_csv: str | Path | None,
) -> tuple[int, dict[str, int]]:
    """
    Compute ``(spatial_input_channels, auxiliary_input_dims)`` from the data config alone.

    Mirrors ``src.datasets.utils.get_dataset_dimensions`` without building a dataset (which would
    need prepared patches and the full weather table). The result is verified at export time by a
    strict ``load_state_dict`` against the checkpoint weights.
    """
    spatial_channels = 0
    auxiliary_input_dims: dict[str, int] = {}
    for source in input_sources:
        params = source.params
        if source.name == "grid" and isinstance(params, GridParams):
            missing = [name for name in params.feature_names_list if name not in feature_channel_map]
            if missing:
                raise BundleError(f"Grid features {missing} are missing from the feature channel map {sorted(feature_channel_map)}.")
            channels = sum(len(feature_channel_map[name]) for name in params.feature_names_list)
            if "fuel_grid" in params.feature_names_list:
                if params.fuel_feats_encoding in FUEL_CURVE_ENCODINGS:
                    channels -= len(feature_channel_map["fuel_grid"])
                    if fuel_curves_csv is None:
                        raise BundleError(f"fuel_feats_encoding={params.fuel_feats_encoding!r} requires a fuel curves table.")
                    auxiliary_input_dims["fuel_curve"] = fuel_curve_length(fuel_curves_csv, params.fuel_feats_encoding)
                elif params.fuel_feats_encoding == "one_hot":
                    from data_preparation.spatial.utils import FUEL_GROUP_MAP

                    channels += int(max(FUEL_GROUP_MAP.values()) + 1) - 1
            spatial_channels += channels + len(params.terrain_derivatives)
        elif isinstance(params, SpatializedTabularParams):
            multiplier = len(params.quantiles) if params.quantiles else 1
            spatial_channels += len(params.feature_names_list) * multiplier + int(params.include_missing_firezone_mask)
        else:
            raise BundleError(f"Input source {source.name!r} is not supported by the bundle format yet.")
    return spatial_channels, auxiliary_input_dims


def resolve_targets(
    grid_params: GridParams,
    dataset_norm_stats: dict[str, Any],
    bp_nodata_as_zero: bool,
) -> list[TargetEntry]:
    """Resolve the denormalization constants of every target from the training norm stats."""
    entries = []
    for target_config in grid_params.resolved_targets():
        spec = get_target_spec(target_config.name)
        stats = dataset_norm_stats.get(spec.output_type, {})
        entry = TargetEntry(
            name=spec.name,
            label=spec.label,
            units=TARGET_UNITS.get(spec.name, "unknown"),
            activation="sigmoid" if spec.probability_scale else "identity",
            out_norm=target_config.out_norm,
        )
        if target_config.out_norm == "min_max":
            if stats.get("min") is None or stats.get("max") is None:
                raise BundleError(f"Normalization stats have no min/max for {spec.output_type!r}.")
            max_value, min_value = apply_bp_nodata_zero_range(
                target_name=spec.name,
                max_value=float(stats["max"]),
                min_value=float(stats["min"]),
                bp_nodata_as_zero=bp_nodata_as_zero,
            )
            entry.min_value, entry.max_value = float(min_value), float(max_value)
        elif target_config.out_norm == "log_standard":
            log_mean = target_config.log_mean if target_config.log_mean is not None else stats.get("log_mean")
            log_std = target_config.log_std if target_config.log_std is not None else stats.get("log_std")
            if log_mean is None or log_std is None:
                raise BundleError(f"Normalization stats have no log_mean/log_std for {spec.output_type!r}.")
            entry.log_mean, entry.log_std = float(log_mean), float(log_std)
        elif target_config.out_norm not in {"log", "none"}:
            raise BundleError(f"out_norm={target_config.out_norm!r} for target {spec.name!r} is not supported for inference.")
        entries.append(entry)
    return entries


def build_model_from_manifest(manifest: BundleManifest) -> torch.nn.Module:
    from src.models.factory import build_model

    return build_model(
        model_config=manifest.model,
        spatial_input_channels=manifest.model_io.spatial_input_channels,
        auxiliary_input_dims=dict(manifest.model_io.auxiliary_input_dims),
        target_names=manifest.target_names,
    )


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Map ``"auto"`` to the best available device (CUDA, then Apple MPS, then CPU)."""
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelBundle:
    root: Path
    manifest: BundleManifest

    def resource_path(self, key: str) -> Path:
        if key not in self.manifest.resources:
            raise BundleError(f"Bundle {self.root} has no resource {key!r}. Available: {sorted(self.manifest.resources)}.")
        return self.root / self.manifest.resources[key].path

    def optional_resource_path(self, key: str) -> Path | None:
        return self.resource_path(key) if key in self.manifest.resources else None

    def read_json_resource(self, key: str) -> dict[str, Any]:
        with open(self.resource_path(key)) as handle:
            return json.load(handle)

    def target(self, name: str) -> TargetEntry:
        normalized = get_target_spec(name).name
        for target in self.manifest.targets:
            if target.name == normalized:
                return target
        raise KeyError(f"Target {normalized!r} is not predicted by bundle {self.manifest.name!r}.")

    def load_state_dict(self, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        # weights_only=True: the weights file never needs arbitrary unpickling.
        return torch.load(self.root / self.manifest.weights.path, map_location=map_location, weights_only=True)

    def build_model(self, device: str | torch.device = "auto") -> torch.nn.Module:
        """Instantiate the architecture, load the weights strictly, and return it in eval mode on ``device``."""
        resolved_device = resolve_device(device)
        model = build_model_from_manifest(self.manifest)
        try:
            model.load_state_dict(self.load_state_dict(map_location="cpu"), strict=True)
        except RuntimeError as exc:
            raise BundleError(
                f"The weights in {self.root} do not match the architecture described in its manifest. "
                "The bundle may be corrupted or was exported with an incompatible code version."
            ) from exc
        return model.to(resolved_device).eval()


def _read_manifest(bundle_dir: Path) -> BundleManifest:
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BundleError(f"{bundle_dir} is not a model bundle: {MANIFEST_FILENAME} not found.")
    with open(manifest_path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise BundleError(f"{manifest_path} is not a valid bundle manifest.")
    version = raw.get("bundle_format_version")
    if version != BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"Bundle format version {version!r} in {manifest_path} is not supported by this software "
            f"(expected {BUNDLE_FORMAT_VERSION}). Use a matching software release or re-export the bundle."
        )
    try:
        return BundleManifest(**raw)
    except ValidationError as exc:
        raise BundleError(f"{manifest_path} is not a valid bundle manifest:\n{exc}") from exc


def load_bundle(bundle_dir: str | Path, verify_checksums: bool = True) -> ModelBundle:
    """
    Load and validate a model bundle.

    Checks that the manifest is readable and of a supported format version, that every referenced
    file exists, and (by default) that each file matches its recorded SHA-256 checksum.
    """
    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(f"Model bundle directory not found: {root}")
    manifest = _read_manifest(root)

    entries = {"weights": manifest.weights, **manifest.resources}
    for key, entry in entries.items():
        path = (root / entry.path).resolve()
        if not path.is_relative_to(root):
            raise BundleError(f"Bundle manifest entry {key} points outside the bundle directory: {entry.path}")
        if not path.is_file():
            raise BundleError(f"Bundle {root} is incomplete: {key} file {entry.path} is missing. Re-download the bundle.")
        if verify_checksums and sha256_file(path) != entry.sha256:
            raise BundleError(f"Bundle {root} is corrupted: checksum mismatch for {entry.path}. Re-download the bundle.")
    return ModelBundle(root=root, manifest=manifest)


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def resolve_mask_scope(requested: str | None, bundle: ModelBundle) -> str:
    """The area to use: ``requested``, else the bundle's training patch scope, else the actual hexel mask."""
    scope = requested or bundle.manifest.data_prep.mask_scope or "actual"
    if scope == NO_MASK_SCOPE:
        return scope
    scope = normalize_mask_scope(scope)
    if scope not in MASK_SCOPES:
        raise BundleError(f"mask_scope must be one of {MASK_SCOPES}, got {scope!r}.")
    return scope


def resolve_scenario_name(requested: str | None, bundle: ModelBundle) -> str | None:
    """The scenario to use: ``requested``, else the bundle's training scenario (None = national rasters)."""
    return requested or bundle.manifest.data_prep.scenario_name or None


def data_mask_scope(scope: str) -> str | None:
    """The mask scope understood by data_preparation/src (None = no mask)."""
    return None if scope == NO_MASK_SCOPE else scope
