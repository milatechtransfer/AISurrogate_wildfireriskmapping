# base configurations for experiments
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.datasets.postprocessing.hazard import (
    DEFAULT_FI_CAP,
    DEFAULT_HAZARD_BIN_THRESHOLDS,
    DEFAULT_SCALE_TO,
    validate_bin_thresholds,
)
from src.datasets.targets import TargetName, get_target_spec

TargetNorm = Literal["min_max", "log", "log_standard", "none", "total_iters", "season_cause_iters"]


class LoggerConfig(BaseModel):
    enabled: bool = True
    project_name: str
    workspace: str
    experiment_name: str
    tags: list[str] = []
    log_every_n_step: int = 1


class ModelConfig(BaseModel):
    architecture: str = "auto"
    num_classes: int = 1
    output_head: Literal["shared", "bp_behavior"] = "shared"
    hidden_features: list[int] = [64, 128, 256, 512]

    # Controls if we use MultiSourceUNet or BaselineUNet
    # Use ["spatial"] for base unet
    # Extra tabular features are detected automatically from the dataset config.
    input_branches: list[str] = ["spatial"]

    # encoder/decoder
    use_skip_connections: bool = True
    use_transpose_conv: bool = False
    use_activation_after_upsampling: bool = False
    use_coordconv: bool = False

    # specific to auxiliary model
    auxiliary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] = {"tabular_weather": [32, 64]}
    auxiliary_embed_dims: dict[str, int] = {"tabular_weather": 128}
    auxiliary_feature_encoder_poolings: dict[str, str] = {"tabular_weather": "max"}

    # Non-neural baseline hyperparameters (XGBoost/RandomForest/LinearRegression).
    # Unused by src/models/factory.py; consumed only by src/train_tabular_baseline.py's
    # build_baseline_model dispatch.
    params: dict[str, Any] | None = None


class TargetLossConfig(BaseModel):
    loss: str | list[str]
    loss_weights: dict[str, float] = {}
    task_weight: float = Field(default=1.0, gt=0.0)
    huber_beta: float = Field(default=1.0, gt=0.0)


class OptimizerConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    loss: str | list[str] | None = None
    loss_weights: dict[str, float] = {}
    huber_beta: float = Field(default=1.0, gt=0.0)
    target_losses: dict[TargetName, TargetLossConfig] = {}

    @model_validator(mode="after")
    def validate_loss_configuration(self) -> "OptimizerConfig":
        if self.target_losses:
            if self.loss is not None or self.loss_weights:
                raise ValueError("Use either legacy loss/loss_weights or target_losses, not both.")
            return self
        if self.loss is None:
            raise ValueError("loss is required when target_losses is not configured.")
        return self


class SchedulerConfig(BaseModel):
    name: str | None = None  # "cosine_warmup", "plateau", "onecycle", "multistep" (or null)

    # params specific to each scheduler
    warmup_epochs: int = 5  # for cosine_warmup
    max_lr: float = 1e-3  # for onecycle
    patience: int = 10  # for plateau
    factor: float = 0.1  # for plateau and multistep
    milestones: list[int] = [30, 40]  # for multistep


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class EvaluationConfig(BaseModel):
    best_ckpt_metrics: list[str] = ["spearman"]  # metric to choose best checkpoint
    best_ckpt_metrics_mode: list[str] = ["max"]  # max, or min
    checkpoint_filename: str = "best.pth"
    robust_plot_percentile: float | None = Field(default=None, gt=0.0, le=100.0)
    bp_nodata_as_zero: bool = True
    prediction_support_policy: str = "input"
    hazard_fi_cap: float | None = Field(default=DEFAULT_FI_CAP, gt=0.0)


class TargetConfig(BaseModel):
    name: TargetName
    out_norm: TargetNorm
    log_mean: float | None = None
    log_std: float | None = Field(default=None, gt=0.0)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> TargetName:
        return get_target_spec(str(value)).name

    @model_validator(mode="after")
    def validate_log_stats(self) -> "TargetConfig":
        if (self.log_mean is None) != (self.log_std is None):
            raise ValueError("log_mean and log_std must either both be set or both be omitted.")
        return self


class GridParams(BaseModel):
    """Specific parameters for the GridSource."""

    feature_names_list: list[str]
    target_name: str = "bp"
    # TODO: move out_norm outside of grid source config since it's for GT
    out_norm: TargetNorm = "min_max"
    target_log_mean: float | None = None
    target_log_std: float | None = None
    fuel_feats_encoding: str = "one_hot"
    normalize_fuel_feats_ordinal: bool = True
    transforms_list: list[str] = Field(default_factory=list)
    augmentation_prob: float = 0.0
    terrain_derivatives: list[str] = Field(default_factory=list)
    terrain_cell_size_m: float = Field(default=100.0, gt=0.0)
    bp_nodata_as_zero: bool = True
    targets: list[TargetConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def prevent_mixed_target_configuration(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("targets") is None:
            return data
        conflicting_legacy_fields = {
            field_name
            for field_name, default_value in {
                "target_name": "bp",
                "out_norm": "min_max",
                "target_log_mean": None,
                "target_log_std": None,
            }.items()
            if field_name in data and data[field_name] != default_value
        }
        if conflicting_legacy_fields:
            raise ValueError(f"Use either targets or non-default legacy target fields, not both: {sorted(conflicting_legacy_fields)}")
        return data

    @model_validator(mode="after")
    def validate_targets(self) -> "GridParams":
        if self.targets is None:
            get_target_spec(self.target_name)
            return self
        if not self.targets:
            raise ValueError("targets must contain at least one target.")
        target_names = [target.name for target in self.targets]
        if len(target_names) != len(set(target_names)):
            raise ValueError(f"targets contains duplicate names: {target_names}")
        return self

    def resolved_targets(self) -> list[TargetConfig]:
        if self.targets is not None:
            return list(self.targets)
        return [
            TargetConfig(
                name=get_target_spec(self.target_name).name,
                out_norm=self.out_norm,
                log_mean=self.target_log_mean,
                log_std=self.target_log_std,
            )
        ]

    def target_config(self, target_name: str) -> TargetConfig:
        normalized_name = get_target_spec(target_name).name
        for target in self.resolved_targets():
            if target.name == normalized_name:
                return target
        raise KeyError(f"Target {normalized_name!r} is not configured.")


class TabularParams(BaseModel):
    """Specific parameters for any tabular source relying on mapped weather zone id"""

    csv_name: str = "weather_table_processed.csv"
    feature_names_list: list[str]
    fire_weather_zone_id_col: str = "WeatherZone"
    fire_weather_zone_selection_approach: str = "mode"  # Selection method to determine weather zone to be used for the patch: "mode" (for most common zone) or "weighted" (for frequency-weighted sampling)
    num_samples_per_patch: int = 256
    transforms_list: list[str] = Field(default_factory=list)
    augmentation_prob: float = 0.0
    sampling_bias: str | None = (
        None  # Whether to bias sampling towards high or low values of the feature of interest, or no bias (None, "high_values")
    )
    feature_to_bias: str | None = None  # The feature to bias sampling towards if sampling_bias is not None
    # Column name identifying the hexel each row belongs to (e.g. "hex_id"). When set, the LUT is
    # keyed by (hex_id, zone) instead of zone alone, so a patch's features are only ever drawn from
    # rows belonging to its own hexel — never pooled across hexels that share a fire-weather zone but
    # live in different train/val/test splits. Requires ``patch_info["hex_id"]`` to be present.
    hex_id_col: str | None = None


class SpatializedTabularParams(TabularParams):
    """Parameters for rasterizing zone-level tabular covariates onto patch pixels."""

    zone_channel_key: str = "firezones_grid"
    aggregation: str = "mean"
    shuffle_lut: bool = False
    shuffle_seed: int = 42
    include_missing_firezone_mask: bool = False
    missing_value_strategy: str = "global_mean"
    global_fill_csv_name: str | None = None


class DataSourceConfig(BaseModel):
    name: str
    params: GridParams | TabularParams | SpatializedTabularParams

    @model_validator(mode="before")
    @classmethod
    def parse_params_for_source(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        name = data.get("name")
        params = data.get("params")
        if not isinstance(name, str):
            return data
        if not isinstance(params, dict):
            return data

        param_classes = {
            "grid": GridParams,
            "tabular_weather": TabularParams,
            "tabular_fire_size": TabularParams,
            "spatialized_weather": SpatializedTabularParams,
            "spatialized_fire_size": SpatializedTabularParams,
        }
        param_class = param_classes.get(name)
        if param_class is None:
            return data

        parsed = dict(data)
        parsed["params"] = param_class(**params)
        return parsed


class DataConfig(BaseModel):
    root_dir: str
    raw_data_dir: str
    batch_size: int = 64
    num_workers: int = 0

    train_split: str
    val_split: str
    test_split: str
    filename_col: str = "filename"
    valid_mask_threshold: float = 0.01
    include_patch_metadata: bool = False

    input_sources: list[DataSourceConfig]


class DataPrepConfig(BaseModel):
    modelling_approach: int = 1
    win_h: int = 256
    win_w: int = 256
    overlap_ratio: float = 0.2


class Config(BaseModel):
    save_dir: str = "experiments/default"
    base_dir: str = "../yan_bp3"
    seed: int = 42
    deterministic: bool = True
    modelling_approach: str = "2"
    model: ModelConfig
    optimizer: OptimizerConfig
    lr_scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    training: TrainingConfig
    evaluation: EvaluationConfig
    data: DataConfig
    logger: LoggerConfig
    metrics: list[str] = ["mse", "mae", "spearman", "ssim"]
    data_prep: DataPrepConfig = Field(default_factory=DataPrepConfig)

    @model_validator(mode="after")
    def validate_target_alignment(self) -> "Config":
        grid_params = next(
            (source.params for source in self.data.input_sources if source.name == "grid" and isinstance(source.params, GridParams)),
            None,
        )
        if grid_params is None:
            return self

        target_names = [target.name for target in grid_params.resolved_targets()]
        if self.model.num_classes != len(target_names):
            raise ValueError(f"model.num_classes={self.model.num_classes} must match configured targets {target_names}.")
        configured_loss_targets = set(self.optimizer.target_losses)
        if len(target_names) > 1 and not configured_loss_targets:
            raise ValueError("optimizer.target_losses is required when multiple targets are configured.")
        if configured_loss_targets and configured_loss_targets != set(target_names):
            raise ValueError(
                f"optimizer.target_losses must match configured targets, got {sorted(configured_loss_targets)} "
                f"and expected {sorted(target_names)}."
            )
        if len(target_names) > 1:
            valid_checkpoint_metrics = {"loss"}
            valid_checkpoint_metrics.update(f"{target_name}/{metric_name}" for target_name in target_names for metric_name in self.metrics)
            for target_name, loss_config in self.optimizer.target_losses.items():
                valid_checkpoint_metrics.add(f"loss_{target_name}/total")
                if isinstance(loss_config.loss, list):
                    valid_checkpoint_metrics.update(f"loss_{target_name}/{loss_name}" for loss_name in loss_config.loss)
            if {"bp", "fi"} <= set(target_names):
                valid_checkpoint_metrics.add("hazard/ccc")
            invalid_checkpoint_metrics = set(self.evaluation.best_ckpt_metrics) - valid_checkpoint_metrics
            if invalid_checkpoint_metrics:
                raise ValueError(
                    f"Multi-target best_ckpt_metrics must use namespaced metric keys. Invalid values: {sorted(invalid_checkpoint_metrics)}."
                )
        return self


#: Fixed pool of seeds used to derive a run-specific seed from `run_id`.
SEEDS: list[int] = [42, 1337, 2024]


def apply_run_id_overrides(config: "Config", run_id: int) -> int:
    """
    Look up a run-specific seed for `run_id` (e.g. SLURM_ARRAY_TASK_ID) in `SEEDS`,
    nest `save_dir` under a per-seed subdirectory, and append the seed to the Comet
    experiment name so parallel multi-run jobs don't collide. Mutates `config` in place
    and returns the derived seed.
    """
    if not 0 <= run_id < len(SEEDS):
        raise ValueError(f"run_id must be between 0 and {len(SEEDS) - 1}, but received {run_id}")

    run_seed = SEEDS[run_id]
    config.seed = run_seed
    config.save_dir = str(Path(config.save_dir) / f"seed_{run_seed}")
    if config.logger.experiment_name:
        config.logger.experiment_name = f"{config.logger.experiment_name}_seed{run_seed}"
    return run_seed


DenominatorSource = Literal[
    "reference_file",
    "all_raw_ground_truth",
    "train_ground_truth",
    "eval_ground_truth",
    "prediction",
]


class HazardModelEntry(BaseModel):
    """A single trained multi-output model contributing BP/FI predictions to hazard evaluation."""

    config_path: str
    checkpoint_filename: str = "best.pth"
    save_predictions: bool = False


class HazardEvalConfig(BaseModel):
    """Config for hazard evaluation using a single multi-output BP/FI model checkpoint."""

    save_dir: str = "experiments/hazard_eval"

    root_dir: str
    raw_data_dir: str
    test_split: str = "test_indices.csv"
    valid_mask_threshold: float = 0.01
    mask_scope: Literal["actual", "buffer", "buffer_only"] = "actual"
    stitch_mode: Literal["mean", "max"] = "mean"

    model: HazardModelEntry

    fi_cap: float | None = Field(default=DEFAULT_FI_CAP, gt=0.0)
    scale_to: float = Field(default=DEFAULT_SCALE_TO, gt=0.0)
    bin_thresholds: list[float] = Field(default_factory=lambda: list(DEFAULT_HAZARD_BIN_THRESHOLDS))
    scale_denominator: float | None = Field(default=None, gt=0.0)
    scale_denominator_source: DenominatorSource = "all_raw_ground_truth"
    reference_denominator_path: str | None = None
    self_normalized_prediction: bool = False

    save_hazard_map: bool = True

    @field_validator("bin_thresholds")
    @classmethod
    def _validate_bin_thresholds(cls, thresholds: list[float]) -> list[float]:
        return validate_bin_thresholds(thresholds, name="bin_thresholds").tolist()

    @model_validator(mode="after")
    def _require_reference_denominator_path(self) -> "HazardEvalConfig":
        if self.scale_denominator_source == "reference_file" and self.scale_denominator is None and self.reference_denominator_path is None:
            raise ValueError(
                "reference_denominator_path is required when scale_denominator_source='reference_file' "
                "and no explicit scale_denominator is set"
            )
        return self
