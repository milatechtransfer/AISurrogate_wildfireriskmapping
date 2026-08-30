import json
import logging
import os
import random
import time
from collections.abc import Mapping
from typing import Any, Literal, cast, overload

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation.spatial.utils import NORM_STATS_JSON, get_output_log_stats_cached, get_range_output_cached, read_split_hex_ids
from src.config import Config, GridParams, SpatializedTabularParams
from src.datasets.context_crop import centered_crop_slices, validate_context_crop_metadata
from src.datasets.fuel_utils import FUEL_CURVE_ENCODINGS
from src.datasets.targets import TargetName, activate_target_predictions, get_target_specs
from src.datasets.utils import apply_bp_nodata_zero_range, get_dataset_spatial_feature_names, get_fuel_curve_normalization_stats
from src.logger import CometLogger
from src.losses import MultiTaskLoss, WeightedLoss
from src.models.factory import build_model, resolve_model_architecture
from src.models.utils import get_nbr_model_parameters
from src.schedulers import build_lr_scheduler
from src.utils import AVAILABLE_METRICS, build_single_loss, set_device

logger = logging.getLogger(__name__)


def validate_mechanistic_normalization_params(config: Config, resolved_architecture: str) -> None:
    """Ensure mechanistic denormalization constants match the dataset artifacts."""
    travel_time_architectures = {
        "mechanistic_travel_time_v4",
        "travel_time_propagation_v4",
        "mechanistic_propagation_v4",
    }
    interpretable_architectures = {"interpretable_mechanistic", "physical_mechanistic"}
    interpretable_v2_architectures = {"interpretable_mechanistic_v2", "physical_mechanistic_v2"}
    interpretable_v3_architectures = {
        "interpretable_mechanistic_v3",
        "physical_mechanistic_v3",
        "gray_box_mechanistic",
    }
    hybrid_v22_architectures = {"mechanistic_hybrid_v22", "cnn_physics_hybrid_v22"}
    hybrid_v23_architectures = {"mechanistic_hybrid_v23", "cnn_physics_hybrid_v23"}
    hybrid_v24_architectures = {"mechanistic_hybrid_v24", "cnn_physics_hybrid_v24"}
    direct_log_hybrid_architectures = hybrid_v23_architectures | hybrid_v24_architectures
    hybrid_architectures = hybrid_v22_architectures | direct_log_hybrid_architectures
    count_aware_interpretable_architectures = interpretable_v2_architectures | interpretable_v3_architectures
    count_aware_architectures = count_aware_interpretable_architectures | hybrid_architectures
    all_interpretable_architectures = interpretable_architectures | count_aware_interpretable_architectures
    physical_behavior_architectures = all_interpretable_architectures | hybrid_architectures
    if resolved_architecture not in travel_time_architectures | physical_behavior_architectures:
        return

    if resolved_architecture in hybrid_architectures:
        if not config.data_prep.preserve_native_grid:
            raise ValueError("Native mechanistic hybrids require data_prep.preserve_native_grid=true.")
        if config.model.propagation_downsample_factor != 8:
            raise ValueError("Native mechanistic hybrids require model.propagation_downsample_factor=8.")
        if not np.isclose(config.model.propagation_cell_size_m, 100.0):
            raise ValueError("Native mechanistic hybrids require native 100 m pixels.")
        crop_h, crop_w = config.data_prep.resolved_target_crop()
        context_pixels = min((config.data_prep.win_h - crop_h) // 2, (config.data_prep.win_w - crop_w) // 2)
        max_context_steps = context_pixels // config.model.propagation_downsample_factor
        if config.model.propagation_steps > max_context_steps:
            raise ValueError(
                f"model.propagation_steps={config.model.propagation_steps} exceeds the centered context margin "
                f"of {max_context_steps} coarse cells."
            )

    checks: list[tuple[str, dict[str, tuple[str, float]]]] = []
    if (
        resolved_architecture in travel_time_architectures and config.model.propagation_ignition_mode == "probability_mass"
    ) or resolved_architecture in count_aware_architectures:
        count_checks = {
            "log1p_mean_minimum": (
                "model.propagation_count_log_mean_min",
                config.model.propagation_count_log_mean_min,
            ),
            "log1p_mean_maximum": (
                "model.propagation_count_log_mean_max",
                config.model.propagation_count_log_mean_max,
            ),
        }
        if resolved_architecture in count_aware_architectures:
            count_checks.update(
                {
                    "cv_minimum": (
                        "model.propagation_count_cv_min",
                        config.model.propagation_count_cv_min,
                    ),
                    "cv_maximum": (
                        "model.propagation_count_cv_max",
                        config.model.propagation_count_cv_max,
                    ),
                }
            )
        checks.append(
            (
                "ignition_count_norm_params.json",
                count_checks,
            )
        )
    if config.model.propagation_scenario_mode == "fire_size" and resolved_architecture not in direct_log_hybrid_architectures:
        checks.append(
            (
                "fire_size_norm_params.json",
                {
                    "log_size_min": (
                        "model.propagation_fire_size_log_min",
                        config.model.propagation_fire_size_log_min,
                    ),
                    "log_size_max": (
                        "model.propagation_fire_size_log_max",
                        config.model.propagation_fire_size_log_max,
                    ),
                },
            )
        )

    if resolved_architecture in direct_log_hybrid_architectures:
        fire_size_params = next(
            (
                source.params
                for source in config.data.input_sources
                if source.name == "spatialized_fire_size" and isinstance(source.params, SpatializedTabularParams)
            ),
            None,
        )
        if fire_size_params is None:
            raise ValueError("Mechanistic hybrid v2.3 requires a spatialized_fire_size source.")
        if fire_size_params.feature_names_list != ["LOG_SIZE_HA"]:
            raise ValueError("Mechanistic hybrid v2.3 requires spatialized_fire_size.feature_names_list=['LOG_SIZE_HA'].")
        if fire_size_params.quantiles != [0.1, 0.5, 0.9]:
            raise ValueError("Mechanistic hybrid v2.3 requires fire-size quantiles [0.1, 0.5, 0.9].")
        if not fire_size_params.include_missing_firezone_mask:
            raise ValueError("Mechanistic hybrid v2.3 requires include_missing_firezone_mask=true.")
        if fire_size_params.missing_value_strategy != "global_mean" or fire_size_params.global_fill_csv_name is None:
            raise ValueError("Mechanistic hybrid v2.3 requires a frozen global-mean fire-size fill CSV.")
        global_fill_path = os.path.join(config.data.root_dir, fire_size_params.global_fill_csv_name)
        if not os.path.isfile(global_fill_path):
            raise FileNotFoundError(f"Mechanistic fire-size fallback table does not exist: {global_fill_path}")

        fire_size_stats_path = os.path.join(config.data.root_dir, "fire_size_log_stats.json")
        if not os.path.isfile(fire_size_stats_path):
            raise FileNotFoundError(f"Mechanistic fire-size statistics do not exist: {fire_size_stats_path}")
        with open(fire_size_stats_path) as handle:
            fire_size_stats = json.load(handle)
        if fire_size_stats.get("contract") != "log10_1p_hectares":
            raise ValueError(f"Mechanistic fire-size statistics {fire_size_stats_path} do not use the log10_1p_hectares contract.")
        if fire_size_stats.get("feature_name") != "LOG_SIZE_HA":
            raise ValueError(f"Mechanistic fire-size statistics {fire_size_stats_path} do not describe LOG_SIZE_HA.")
        if fire_size_stats.get("quantiles") != fire_size_params.quantiles:
            raise ValueError(
                f"Mechanistic fire-size statistics {fire_size_stats_path} quantiles={fire_size_stats.get('quantiles')} "
                f"do not match configured quantiles={fire_size_params.quantiles}."
            )
        if 36 not in fire_size_stats.get("excluded_gridcodes", []):
            raise ValueError(f"Mechanistic fire-size statistics {fire_size_stats_path} must exclude synthetic GRIDCODE 36.")
        fire_size_checks = {
            "neural_mean": (
                "model.propagation_fire_size_neural_mean",
                config.model.propagation_fire_size_neural_mean,
            ),
            "neural_std": (
                "model.propagation_fire_size_neural_std",
                config.model.propagation_fire_size_neural_std,
            ),
        }
        for stat_name, (config_name, configured_value) in fire_size_checks.items():
            try:
                artifact_value = float(fire_size_stats[stat_name])
            except KeyError as exc:
                raise ValueError(f"Mechanistic fire-size statistics {fire_size_stats_path} are missing {stat_name!r}.") from exc
            if not np.isclose(configured_value, artifact_value, rtol=1e-9, atol=1e-12):
                raise ValueError(f"{config_name}={configured_value} does not match {fire_size_stats_path}:{stat_name}={artifact_value}.")

    if resolved_architecture in physical_behavior_architectures:
        grid_params = next(
            (source.params for source in config.data.input_sources if source.name == "grid" and isinstance(source.params, GridParams)),
            None,
        )
        if grid_params is None:
            raise ValueError("Physical behavior mechanism requires a configured grid data source.")
        if resolved_architecture in hybrid_architectures and grid_params.fuel_feats_encoding != "iROS_HFI":
            raise ValueError("Native mechanistic hybrids require grid.fuel_feats_encoding='iROS_HFI'.")
        targets_by_name = {target.name: target for target in grid_params.resolved_targets()}
        expected_norms: dict[TargetName, str] = {"bp": "none", "fi": "log_standard", "ros": "log_standard"}
        for target_name, expected_norm in expected_norms.items():
            target = targets_by_name.get(target_name)
            if target is None:
                raise ValueError(f"Physical behavior mechanism requires a {target_name!r} grid target.")
            if target.out_norm != expected_norm:
                raise ValueError(f"Physical behavior mechanism requires {target_name}.out_norm={expected_norm!r}, got {target.out_norm!r}.")

        path = os.path.join(config.data.root_dir, NORM_STATS_JSON)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mechanistic normalization metadata does not exist: {path}")
        with open(path) as handle:
            payload = json.load(handle)
        target_checks = {
            ("fire_intensity", "log_mean"): (
                "model.interpretable_fi_log_mean",
                config.model.interpretable_fi_log_mean,
            ),
            ("fire_intensity", "log_std"): (
                "model.interpretable_fi_log_std",
                config.model.interpretable_fi_log_std,
            ),
            ("fire_ros", "log_mean"): (
                "model.interpretable_ros_log_mean",
                config.model.interpretable_ros_log_mean,
            ),
            ("fire_ros", "log_std"): (
                "model.interpretable_ros_log_std",
                config.model.interpretable_ros_log_std,
            ),
        }
        if resolved_architecture in hybrid_architectures:
            target_checks.update(
                {
                    ("elevation", "min"): (
                        "model.propagation_elevation_min_m",
                        config.model.propagation_elevation_min_m,
                    ),
                    ("elevation", "max"): (
                        "model.propagation_elevation_max_m",
                        config.model.propagation_elevation_max_m,
                    ),
                }
            )
        for (artifact_target_name, stat_name), (config_name, configured_value) in target_checks.items():
            try:
                artifact_value = float(payload[artifact_target_name][stat_name])
            except KeyError as exc:
                raise ValueError(f"Mechanistic normalization metadata {path} is missing {artifact_target_name}.{stat_name}.") from exc
            if not np.isclose(configured_value, artifact_value, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"{config_name}={configured_value} does not match "
                    f"{path}:{artifact_target_name}.{stat_name}={artifact_value}. "
                    "Update the config or regenerate the matching normalization artifact."
                )

        if resolved_architecture in hybrid_architectures:
            for curve_name in ("fuel_curve_iROS", "fuel_curve_HFI"):
                entry = payload.get(curve_name)
                if not isinstance(entry, dict) or "log_mean" not in entry or "log_std" not in entry:
                    raise ValueError(f"Mechanistic normalization metadata {path} is missing {curve_name} log statistics.")
            weather_path = os.path.join(config.data.root_dir, "weather_norm_params.json")
            if not os.path.isfile(weather_path):
                raise FileNotFoundError(f"Mechanistic normalization metadata does not exist: {weather_path}")
            with open(weather_path) as handle:
                weather_payload = json.load(handle)
            z_score = weather_payload.get("z_score", {})
            columns = list(z_score.get("cols", []))
            means = list(z_score.get("mean", []))
            stds = list(z_score.get("std", []))
            if not (len(columns) == len(means) == len(stds)):
                raise ValueError(f"Mechanistic normalization metadata {weather_path} has inconsistent z-score arrays.")
            weather_stats = {name: (float(mean), float(std)) for name, mean, std in zip(columns, means, stds, strict=True)}
            weather_checks = {
                "InitialSpreadIndex": (
                    ("model.propagation_isi_mean", config.model.propagation_isi_mean),
                    ("model.propagation_isi_std", config.model.propagation_isi_std),
                ),
                "wind_x": (
                    ("model.propagation_wind_x_mean", config.model.propagation_wind_x_mean),
                    ("model.propagation_wind_x_std", config.model.propagation_wind_x_std),
                ),
                "wind_y": (
                    ("model.propagation_wind_y_mean", config.model.propagation_wind_y_mean),
                    ("model.propagation_wind_y_std", config.model.propagation_wind_y_std),
                ),
            }
            for feature_name, ((mean_name, configured_mean), (std_name, configured_std)) in weather_checks.items():
                if feature_name not in weather_stats:
                    raise ValueError(f"Mechanistic normalization metadata {weather_path} is missing {feature_name!r}.")
                artifact_mean, artifact_std = weather_stats[feature_name]
                if not np.isclose(configured_mean, artifact_mean, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"{mean_name}={configured_mean} does not match {weather_path}:{feature_name}.mean={artifact_mean}.")
                if not np.isclose(configured_std, artifact_std, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"{std_name}={configured_std} does not match {weather_path}:{feature_name}.std={artifact_std}.")
        configured_target_stats: dict[TargetName, tuple[float, float]] = {
            "fi": (
                config.model.interpretable_fi_log_mean,
                config.model.interpretable_fi_log_std,
            ),
            "ros": (
                config.model.interpretable_ros_log_mean,
                config.model.interpretable_ros_log_std,
            ),
        }
        for target_name, (model_mean, model_std) in configured_target_stats.items():
            target = targets_by_name[target_name]
            if target.log_mean is None or target.log_std is None:
                continue
            if not np.isclose(model_mean, target.log_mean, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"model.interpretable_{target_name}_log_mean={model_mean} does not match the grid target log_mean={target.log_mean}."
                )
            if not np.isclose(model_std, target.log_std, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"model.interpretable_{target_name}_log_std={model_std} does not match the grid target log_std={target.log_std}."
                )

    for filename, expected_values in checks:
        path = os.path.join(config.data.root_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mechanistic normalization metadata does not exist: {path}")
        with open(path) as handle:
            payload = json.load(handle)
        for json_key, (config_name, configured_value) in expected_values.items():
            if json_key not in payload:
                raise ValueError(f"Mechanistic normalization metadata {path} is missing {json_key!r}.")
            artifact_value = float(payload[json_key])
            if not np.isclose(configured_value, artifact_value, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"{config_name}={configured_value} does not match {path}:{json_key}={artifact_value}. "
                    "Update the config or regenerate the matching normalization artifact."
                )


class Trainer:
    def __init__(
        self,
        config: Config,
        spatial_input_channels: int | None = None,
        auxiliary_input_dims: dict[str, int] | None = None,
        train_dataset=None,
        spatial_input_names: list[str] | None = None,
    ):
        self.config = config
        self.spatial_input_channels = spatial_input_channels
        self.auxiliary_input_dims = auxiliary_input_dims if auxiliary_input_dims is not None else {}
        self.train_dataset = train_dataset
        self.spatial_input_names = spatial_input_names
        if self.spatial_input_names is None and train_dataset is not None:
            self.spatial_input_names = get_dataset_spatial_feature_names(train_dataset)
        if self.spatial_input_names:
            if spatial_input_channels is not None and len(self.spatial_input_names) != spatial_input_channels:
                raise ValueError(f"Detected {spatial_input_channels} spatial channels but resolved names {self.spatial_input_names}.")
            self.config.model.spatial_input_names = list(self.spatial_input_names)
        if train_dataset is not None and hasattr(train_dataset, "metadata"):
            validate_context_crop_metadata(
                train_dataset.metadata,
                config.data_prep.resolved_target_crop() if config.data_prep.context_crop_enabled else None,
            )

        # Set device
        self.device = set_device()
        print(f"\n[Device] Using: {self.device}")

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Comet Logger
        self.logger = None
        # Only initialize logger if not in test-only mode
        if self.config.logger.enabled:
            previous_experiment_key = self._load_previous_experiment_key()
            self.logger = CometLogger(
                project_name=self.config.logger.project_name,
                workspace=self.config.logger.workspace,
                experiment_name=self.config.logger.experiment_name,
                experiment_tags=self.config.logger.tags,
                previous_experiment_key=previous_experiment_key,
            )
            if previous_experiment_key:
                print(f"[Comet] Resuming experiment {previous_experiment_key} instead of creating a new one.")
            self.log_every_n_step = self.config.logger.log_every_n_step
            # log all the params.
            self.logger.log_params(self.config.model_dump())

        self.setup()

        # metrics for best checkpoint saving
        self.best_ckpt_metrics = list(self.config.evaluation.best_ckpt_metrics)
        self.best_ckpt_modes = list(self.config.evaluation.best_ckpt_metrics_mode)
        if len(self.best_ckpt_metrics) != len(self.best_ckpt_modes):
            raise ValueError("Number of best_ckpt_metric and best_ckpt_metric_mode must match!")
        self._stitched_best_ckpt_metrics = self._validate_stitched_best_ckpt_metrics()
        self._best_metric_list: list[float] = []

    def _load_previous_experiment_key(self) -> str | None:
        """
        If a checkpoint from a previous (e.g. preempted/requeued) run of this same
        save_dir exists, return its stored Comet experiment key so the logger can
        continue logging into that same experiment instead of starting a new one.
        """
        last_path = os.path.join(self.save_dir, "last.pth")
        if not os.path.exists(last_path):
            return None
        try:
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - best-effort lookup, never block startup
            print(f"[Comet] Could not read previous checkpoint at {last_path} to resume experiment: {exc}")
            return None
        return checkpoint.get("comet_experiment_key")

    def setup(self):
        """
        Define model, loss function and optimizer.
        """

        self._grid_params = self._get_grid_params()
        target_names = [target.name for target in self._grid_params.resolved_targets()] if self._grid_params is not None else ["bp"]
        self._target_specs = get_target_specs(target_names)
        self._target_names = [target.name for target in self._target_specs]
        if self.config.model.num_classes != len(self._target_specs):
            raise ValueError(
                f"model.num_classes={self.config.model.num_classes} must match number of configured targets "
                f"({len(self._target_specs)}: {self._target_names})."
            )

        # Flag to indicate we are including auxiliary features
        self.auxiliary = "auxiliary" in self.config.model.input_branches

        resolved_architecture = resolve_model_architecture(self.config.model)
        validate_mechanistic_normalization_params(self.config, resolved_architecture)
        print(f"[Trainer] Model architecture: {self.config.model.architecture} -> {resolved_architecture}")
        print(f"[Trainer] Spatial Channels: {self.spatial_input_channels}, Auxiliary Dim: {self.auxiliary_input_dims}")

        self.model = build_model(
            model_config=self.config.model,
            spatial_input_channels=self.spatial_input_channels,
            auxiliary_input_dims=self.auxiliary_input_dims,
            **self._get_fuel_curve_stats(),
            target_names=self._target_names,
        )

        self._load_initial_checkpoint()
        self.model.to(self.device)

        # Get and log number of model params.
        total_params, trainable_params = get_nbr_model_parameters(self.model)
        print(f"Model Params: Total={total_params:,} | Trainable={trainable_params:,}")
        if self.logger:
            self.logger.log_params({"model_total_params": total_params, "model_trainable_params": trainable_params})

        self.loss_fn = self._build_loss()
        self._coarse_bp_supervision_weight = self.config.model.propagation_coarse_bp_supervision_weight
        self._bp_hazard_residual_weight = self.config.model.propagation_bp_hazard_residual_weight
        self._coarse_bp_loss: WeightedLoss | None = None
        if self._coarse_bp_supervision_weight > 0.0:
            if "bp" not in self._target_names:
                raise ValueError("Coarse BP supervision requires a configured BP target.")
            if not callable(getattr(self.model, "pop_coarse_bp_probability", None)):
                raise ValueError("model.propagation_coarse_bp_supervision_weight requires a model exposing pop_coarse_bp_probability().")
            self._coarse_bp_loss = WeightedLoss(
                losses={
                    "kl": build_single_loss("kl"),
                    "ccc": build_single_loss("ccc"),
                },
                weights={"kl": 0.5, "ccc": 0.5},
            )
        if self._bp_hazard_residual_weight > 0.0:
            if "bp" not in self._target_names:
                raise ValueError("BP hazard residual regularization requires a configured BP target.")
            if not callable(getattr(self.model, "pop_bp_hazard_residual", None)):
                raise ValueError("model.propagation_bp_hazard_residual_weight requires a model exposing pop_bp_hazard_residual().")

        # Setup optimizer
        opt_name = self.config.optimizer.name
        # TODO: add other parameters
        opt_params = {"lr": self.config.optimizer.lr}

        OptimizerClass = getattr(optim, opt_name)
        self.optimizer = OptimizerClass(self._optimizer_parameter_groups(), **opt_params)

        self.global_step = 0
        self._pretrained_frozen: bool | None = None

        # MPS-only memory diagnostic populated by validate()/test(); not a loss/metric,
        # so it is kept out of the results dict and exposed separately to avoid leaking
        # into metrics CSVs/Comet logging.
        self.last_peak_mps_driver_allocated_gb: float | None = None

        # Validate and load metrics from config.
        self._validate_and_load_metrics()

        self._configure_metric_target_transform()

    def _load_initial_checkpoint(self) -> None:
        configured_path = self.config.training.initial_checkpoint
        if configured_path is None:
            return
        checkpoint_path = os.path.expanduser(os.path.expandvars(configured_path))
        if "$" in checkpoint_path:
            raise ValueError(f"training.initial_checkpoint contains an unresolved environment variable: {configured_path}")
        last_path = os.path.join(self.save_dir, "last.pth")
        if os.path.exists(last_path):
            print(f"[Warm Start] Existing {last_path} will be resumed; skipping initial checkpoint.")
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"training.initial_checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state")
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"Initial checkpoint {checkpoint_path} does not contain a model_state mapping.")
        custom_loader = getattr(self.model, "load_pretrained_unet_state_dict", None)
        if custom_loader is None:
            self.model.load_state_dict(state_dict)
            report = {"loaded": len(state_dict), "new": 0}
        else:
            report = custom_loader(state_dict, checkpoint.get("config"))
        print(f"[Warm Start] Loaded {report['loaded']} U-Net tensors from {checkpoint_path}; initialized {report['new']} new v4 tensors.")

    def _configure_pretrained_freeze(self, epoch: int) -> None:
        freeze_method = getattr(self.model, "set_pretrained_frozen", None)
        if freeze_method is None:
            if self.config.training.freeze_pretrained_epochs:
                raise ValueError("freeze_pretrained_epochs requires a model that supports pretrained-module freezing.")
            return
        frozen = epoch <= self.config.training.freeze_pretrained_epochs
        freeze_method(frozen)
        if frozen != self._pretrained_frozen:
            state = "frozen" if frozen else "trainable"
            print(f"[Warm Start] Pretrained U-Net modules are {state} for epoch {epoch}.")
            self._pretrained_frozen = frozen

    def _optimizer_parameter_groups(self):
        lr_scales = self.config.optimizer.parameter_lr_scales
        if not lr_scales:
            return self.model.parameters()

        named_parameters = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        assigned_names: set[str] = set()
        scaled_groups: list[dict] = []
        base_lr = self.config.optimizer.lr

        for prefix, scale in lr_scales.items():
            matched = [(name, parameter) for name, parameter in named_parameters if name == prefix or name.startswith(f"{prefix}.")]
            if not matched:
                raise ValueError(f"optimizer.parameter_lr_scales prefix {prefix!r} matched no trainable parameters.")
            duplicate_names = sorted(name for name, _ in matched if name in assigned_names)
            if duplicate_names:
                raise ValueError(f"optimizer.parameter_lr_scales prefixes overlap for parameters: {duplicate_names[:20]}")
            assigned_names.update(name for name, _ in matched)
            scaled_groups.append({"params": [parameter for _, parameter in matched], "lr": base_lr * scale})

        default_parameters = [parameter for name, parameter in named_parameters if name not in assigned_names]
        groups: list[dict] = []
        if default_parameters:
            groups.append({"params": default_parameters, "lr": base_lr})
        groups.extend(scaled_groups)
        return groups

    def _get_fuel_curve_stats(self) -> dict[str, torch.Tensor | None]:
        """Extract fuel curve normalization stats for model initialisation.

        Priority:
        1. GridSource on the training dataset (training mode).
        2. ``dataset_norm_stats.json`` in ``root_dir`` (eval-only mode, no train dataset).
        3. None → model falls back to zeros/ones (shape mismatch risk if checkpoint differs).
        """
        grid_params = self._get_grid_params()
        if grid_params is None or grid_params.fuel_feats_encoding not in FUEL_CURVE_ENCODINGS:
            return {"fuel_curve_mean": None, "fuel_curve_std": None}

        resolved_architecture = resolve_model_architecture(self.config.model)
        if resolved_architecture in {
            "mechanistic_hybrid_v22",
            "cnn_physics_hybrid_v22",
            "mechanistic_hybrid_v23",
            "cnn_physics_hybrid_v23",
            "mechanistic_hybrid_v24",
            "cnn_physics_hybrid_v24",
        }:
            cache_path = os.path.join(self.config.data.root_dir, NORM_STATS_JSON)
            with open(cache_path) as handle:
                cached = json.load(handle)
            ros_entry = cached["fuel_curve_iROS"]
            hfi_entry = cached["fuel_curve_HFI"]
            return {
                "fuel_curve_mean": torch.tensor(
                    [float(ros_entry["log_mean"]), float(hfi_entry["log_mean"])],
                    dtype=torch.float32,
                ),
                "fuel_curve_std": torch.tensor(
                    [float(ros_entry["log_std"]), float(hfi_entry["log_std"])],
                    dtype=torch.float32,
                ),
            }

        # Training mode: read from GridSource which already computed/cached the stats.
        if self.train_dataset is not None:
            fuel_curve_mean, fuel_curve_std = get_fuel_curve_normalization_stats(self.train_dataset)
            return {"fuel_curve_mean": fuel_curve_mean, "fuel_curve_std": fuel_curve_std}

        # Eval-only mode: try dataset_norm_stats.json.
        cache_path = os.path.join(self.config.data.root_dir, NORM_STATS_JSON)
        cache_key = f"fuel_curve_{grid_params.fuel_feats_encoding}"
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            entry = cached.get(cache_key, {})
            mean = entry.get("log_mean")
            std = entry.get("log_std")
            if mean is not None and std is not None:
                logger.info("Fuel curve stats for model init loaded from %s: mean=%.4f, std=%.4f", cache_path, mean, std)
                return {
                    "fuel_curve_mean": torch.tensor([float(mean)], dtype=torch.float32),
                    "fuel_curve_std": torch.tensor([float(std)], dtype=torch.float32),
                }
        logger.warning(
            "Fuel curve stats not available (no train dataset and %r not found or missing key %r). "
            "Model will use default zeros/ones — shape may mismatch checkpoint.",
            cache_path,
            cache_key,
        )
        return {"fuel_curve_mean": None, "fuel_curve_std": None}

    def _build_loss(self) -> torch.nn.Module:
        if self.config.optimizer.target_losses:
            configured_targets = set(self.config.optimizer.target_losses)
            expected_targets = set(self._target_names)
            if configured_targets != expected_targets:
                raise ValueError(
                    f"optimizer.target_losses must match configured targets, got {sorted(configured_targets)} "
                    f"and expected {sorted(expected_targets)}."
                )
            task_losses = {}
            task_weights = {}
            for target_name in self._target_names:
                target_config = self.config.optimizer.target_losses[target_name]
                task_losses[target_name] = self._build_loss_group(
                    target_config.loss,
                    target_config.loss_weights,
                    target_config.huber_beta,
                )
                task_weights[target_name] = target_config.task_weight
            return MultiTaskLoss(target_names=self._target_names, losses=task_losses, task_weights=task_weights)

        if self.config.optimizer.loss is None:
            raise RuntimeError("Legacy loss configuration is missing.")
        return self._build_loss_group(
            self.config.optimizer.loss,
            self.config.optimizer.loss_weights,
            self.config.optimizer.huber_beta,
        )

    @staticmethod
    def _build_loss_group(
        loss_config: str | list[str],
        weights: dict[str, float],
        huber_beta: float,
    ) -> torch.nn.Module:
        loss_kwargs = {"huber_beta": huber_beta}
        if isinstance(loss_config, str):
            return build_single_loss(loss_config, **loss_kwargs)
        loss_names = loss_config
        losses = {n: build_single_loss(n, **loss_kwargs) for n in loss_names}
        return WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

    def _get_grid_params(self) -> GridParams | None:
        for source in self.config.data.input_sources:
            if source.name == "grid" and isinstance(source.params, GridParams):
                return source.params
        return None

    def _target_out_norm(self, target_name: str) -> str:
        if self._grid_params is None:
            return "none"
        return self._grid_params.target_config(target_name).out_norm

    def _target_log_stats(self, target_name: str) -> tuple[float | None, float | None]:
        if self._grid_params is None:
            return None, None
        target_config = self._grid_params.target_config(target_name)
        return target_config.log_mean, target_config.log_std

    def _configure_metric_target_transform(self) -> None:
        self._metric_out_norms: list[str] = []
        self._metric_target_mins: list[float] = []
        self._metric_target_maxs: list[float] = []
        self._metric_target_log_means: list[float | None] = []
        self._metric_target_log_stds: list[float | None] = []

        if self._grid_params is None:
            self._metric_out_norms = ["none"] * len(self._target_specs)
            self._metric_target_mins = [0.0] * len(self._target_specs)
            self._metric_target_maxs = [1.0] * len(self._target_specs)
            self._metric_target_log_means = [None] * len(self._target_specs)
            self._metric_target_log_stds = [None] * len(self._target_specs)
            return

        # Normalization stats are train-only: derive the allowed hexes from the train split so
        # held-out hexes never leak into target normalization constants.
        train_hex_ids: set[int] | None = None
        if self.config.data.root_dir and self.config.data.train_split:
            split_path = os.path.join(self.config.data.root_dir, self.config.data.train_split)
            if os.path.isfile(split_path):
                train_hex_ids = read_split_hex_ids(split_path)
            else:
                logger.warning(
                    "Train split file %r not found — normalization stats will use all hexels.",
                    split_path,
                )

        for target in self._target_specs:
            out_norm = self._target_out_norm(target.name)
            target_min = 0.0
            target_max = 1.0
            target_log_mean, target_log_std = self._target_log_stats(target.name)

            if out_norm == "min_max":
                root_dir = self.config.data.root_dir or self.config.data.raw_data_dir
                if root_dir:
                    target_max, target_min = get_range_output_cached(
                        root_dir=root_dir,
                        output_type=target.output_type,
                        allowed_hex_ids=train_hex_ids,
                        raw_data_dir=self.config.data.raw_data_dir,
                    )
                    target_max, target_min = apply_bp_nodata_zero_range(
                        target_name=target.name,
                        max_value=target_max,
                        min_value=target_min,
                        bp_nodata_as_zero=self._grid_params.bp_nodata_as_zero,
                    )
            elif out_norm == "log_standard":
                if (target_log_mean is None or target_log_std is None) and self.config.data.root_dir:
                    target_log_mean, target_log_std = get_output_log_stats_cached(
                        root_dir=self.config.data.root_dir,
                        output_type=target.output_type,
                        allowed_hex_ids=train_hex_ids,
                        raw_data_dir=self.config.data.raw_data_dir,
                    )
                if target_log_mean is None or target_log_std is None:
                    raise ValueError(f"target_log_mean/std are required for target={target.name!r} with out_norm='log_standard'.")
                if target_log_std <= 0.0:
                    raise ValueError(f"target_log_std must be positive for target={target.name!r}, got {target_log_std}.")
            elif out_norm not in {"log", "none", "total_iters", "season_cause_iters"}:
                raise ValueError(f"Unsupported output normalization for target={target.name!r}: {out_norm!r}")

            self._metric_out_norms.append(out_norm)
            self._metric_target_mins.append(target_min)
            self._metric_target_maxs.append(target_max)
            self._metric_target_log_means.append(target_log_mean)
            self._metric_target_log_stds.append(target_log_std)

    def _inverse_model_target_for_metrics(self, data: torch.Tensor) -> torch.Tensor:
        data = data.float()
        transformed_channels = []
        for idx, out_norm in enumerate(self._metric_out_norms):
            channel = data[:, idx : idx + 1]
            if out_norm == "min_max":
                target_min = self._metric_target_mins[idx]
                target_max = self._metric_target_maxs[idx]
                transformed_channels.append(channel * (target_max - target_min) + target_min)
            elif out_norm == "log":
                transformed_channels.append(torch.expm1(channel * float(np.log1p(1000.0))) / 1000.0)
            elif out_norm == "log_standard":
                target_log_mean = self._metric_target_log_means[idx]
                target_log_std = self._metric_target_log_stds[idx]
                if target_log_mean is None or target_log_std is None:
                    raise RuntimeError("log_standard metric transform was not configured.")
                transformed_channels.append(torch.expm1(channel * target_log_std + target_log_mean).clamp_min(0.0))
            else:
                transformed_channels.append(channel)
        return torch.cat(transformed_channels, dim=1)

    def _prepare_metric_tensors(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inverse_model_target_for_metrics(predictions), self._inverse_model_target_for_metrics(targets)

    def _metric_result_keys(self) -> list[str]:
        if len(self._target_specs) == 1:
            return list(self.metric_functions)
        keys = [f"{target.name}/{metric_name}" for target in self._target_specs for metric_name in self.metric_functions]
        if {"bp", "fi"} <= set(self._target_names):
            keys.append("hazard/ccc")
        return keys

    def _compute_metric_values(self, predictions: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        if predictions.device.type == "mps":
            predictions = predictions.cpu()
            targets = targets.cpu()
            masks = masks.cpu()

        if len(self._target_specs) == 1:
            return {name: metric_fn(predictions, targets, masks) for name, metric_fn in self.metric_functions.items()}

        metric_values = {}
        for channel_idx, target in enumerate(self._target_specs):
            channel = slice(channel_idx, channel_idx + 1)
            for metric_name, metric_fn in self.metric_functions.items():
                metric_values[f"{target.name}/{metric_name}"] = metric_fn(
                    predictions[:, channel],
                    targets[:, channel],
                    masks[:, channel],
                )

        if {"bp", "fi"} <= set(self._target_names):
            bp_idx = self._target_names.index("bp")
            fi_idx = self._target_names.index("fi")
            bp_predictions = predictions[:, bp_idx : bp_idx + 1]
            bp_targets = targets[:, bp_idx : bp_idx + 1]
            fi_predictions = predictions[:, fi_idx : fi_idx + 1]
            fi_targets = targets[:, fi_idx : fi_idx + 1]
            bp_mask = masks[:, bp_idx : bp_idx + 1].bool()
            fi_mask = masks[:, fi_idx : fi_idx + 1].bool()
            if self.config.evaluation.hazard_fi_cap is not None:
                fi_cap = self.config.evaluation.hazard_fi_cap
                fi_predictions = fi_predictions.clamp_max(fi_cap)
                fi_targets = fi_targets.clamp_max(fi_cap)
            no_burn = bp_targets <= 0.0
            hazard_mask = bp_mask & (fi_mask | no_burn)
            hazard_targets = bp_targets * torch.where(fi_mask, fi_targets, torch.zeros_like(fi_targets))
            metric_values["hazard/ccc"] = AVAILABLE_METRICS["ccc"](
                bp_predictions * fi_predictions,
                hazard_targets,
                hazard_mask,
            )

        return metric_values

    def _validate_and_load_metrics(self) -> None:
        """Helper to validate and load metrics to be computed."""
        if not set(self.config.metrics).issubset(AVAILABLE_METRICS):
            raise ValueError(f"Invalid metrics found.Available options: {list(AVAILABLE_METRICS)}")
        self.metric_functions = {k: AVAILABLE_METRICS[k] for k in self.config.metrics}

    def _compute_coarse_bp_supervision(
        self,
        coarse_probability: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        full_spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._coarse_bp_loss is None:
            raise RuntimeError("Coarse BP supervision loss was not configured.")
        if coarse_probability.ndim != 4 or coarse_probability.shape[1] != 1:
            raise ValueError(f"Expected coarse BP probability shape (B, 1, H, W), got {tuple(coarse_probability.shape)}.")
        if coarse_probability.shape[0] != targets.shape[0]:
            raise ValueError(f"Coarse BP batch size {coarse_probability.shape[0]} does not match target batch size {targets.shape[0]}.")

        full_h, full_w = full_spatial_shape
        coarse_h, coarse_w = coarse_probability.shape[-2:]
        if full_h % coarse_h != 0 or full_w % coarse_w != 0:
            raise ValueError(f"Full spatial shape {full_spatial_shape} must be divisible by coarse BP shape {(coarse_h, coarse_w)}.")
        scale_h = full_h // coarse_h
        scale_w = full_w // coarse_w
        target_h, target_w = targets.shape[-2:]
        if target_h % scale_h != 0 or target_w % scale_w != 0:
            raise ValueError(f"Target shape {(target_h, target_w)} must be divisible by coarse BP scale {(scale_h, scale_w)}.")
        supervised_h = target_h // scale_h
        supervised_w = target_w // scale_w
        row_slice, col_slice = centered_crop_slices(coarse_h, coarse_w, supervised_h, supervised_w)
        coarse_probability = coarse_probability[..., row_slice, col_slice]

        bp_idx = self._target_names.index("bp")
        bp_target = targets[:, bp_idx : bp_idx + 1]
        bp_mask = masks[:, bp_idx : bp_idx + 1].to(dtype=bp_target.dtype)
        valid_fraction = F.adaptive_avg_pool2d(bp_mask, (supervised_h, supervised_w))
        pooled_target = F.adaptive_avg_pool2d(bp_target * bp_mask, (supervised_h, supervised_w))
        pooled_target = pooled_target / valid_fraction.clamp_min(1e-8)
        pooled_mask = valid_fraction > 0.0

        eps = self.config.model.propagation_logit_eps
        coarse_logits = torch.logit(coarse_probability.clamp(eps, 1.0 - eps))
        coarse_loss, coarse_parts = self._coarse_bp_loss(coarse_logits, pooled_target, pooled_mask)
        weighted_loss = self._coarse_bp_supervision_weight * coarse_loss
        loss_parts = {
            "model/coarse_bp_kl": coarse_parts["kl"],
            "model/coarse_bp_ccc": coarse_parts["ccc"],
            "model/coarse_bp_total": coarse_loss,
            "model/coarse_bp_weighted": weighted_loss,
        }
        return weighted_loss, loss_parts

    def _compute_bp_hazard_residual_regularization(
        self,
        residual_fraction: torch.Tensor,
        burnability: torch.Tensor,
        masks: torch.Tensor,
        full_spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if residual_fraction.shape != burnability.shape:
            raise ValueError(
                f"BP hazard residual shape {tuple(residual_fraction.shape)} does not match burnability {tuple(burnability.shape)}."
            )
        if residual_fraction.shape[-2:] != full_spatial_shape:
            raise ValueError(
                f"BP hazard residual shape {tuple(residual_fraction.shape[-2:])} does not match model output {full_spatial_shape}."
            )

        target_h, target_w = masks.shape[-2:]
        if (target_h, target_w) != full_spatial_shape:
            row_slice, col_slice = centered_crop_slices(
                full_spatial_shape[0],
                full_spatial_shape[1],
                target_h,
                target_w,
            )
            residual_fraction = residual_fraction[..., row_slice, col_slice]
            burnability = burnability[..., row_slice, col_slice]

        bp_idx = self._target_names.index("bp")
        valid = masks[:, bp_idx : bp_idx + 1].to(dtype=residual_fraction.dtype) * burnability
        valid_pixels = valid.sum().clamp_min(1.0)
        residual_l2 = (residual_fraction.square() * valid).sum() / valid_pixels
        weighted_loss = self._bp_hazard_residual_weight * residual_l2
        return weighted_loss, {
            "model/bp_hazard_residual_l2": residual_l2,
            "model/bp_hazard_residual_weighted": weighted_loss,
        }

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """
        Default step. Expects batch -> {'grid': (inputs, targets, masks), 'tabular_weather': ...}.
        Returns (predictions, loss, loss_parts, targets_on_device, masks_on_device).
        """
        # Get the spatial grid inputs, targets and masks.
        if "grid" not in batch:
            raise ValueError("Batch is missing required 'grid' data.")
        inputs, targets, masks = [t.to(self.device) for t in batch["grid"]]

        patch_metadata = batch.get("patch_metadata")

        # Unpack all potential auxiliary data
        auxiliary_data = {}
        for key, value in batch.items():
            if key in {"grid", "patch_metadata"}:
                continue
            auxiliary_data[key] = value.to(self.device)

        predictions = self.model(inputs, auxiliary_data)
        full_spatial_shape = predictions.shape[-2:]
        pop_coarse_bp_probability = getattr(self.model, "pop_coarse_bp_probability", None)
        coarse_bp_probability = pop_coarse_bp_probability() if callable(pop_coarse_bp_probability) else None
        pop_bp_hazard_residual = getattr(self.model, "pop_bp_hazard_residual", None)
        bp_hazard_residual = pop_bp_hazard_residual() if callable(pop_bp_hazard_residual) else None
        if predictions.shape[-2:] != targets.shape[-2:] or targets.shape[-2:] != masks.shape[-2:]:
            raise ValueError(
                "Predictions, targets, and masks must share spatial dimensions before context cropping, got "
                f"{predictions.shape[-2:]}, {targets.shape[-2:]}, and {masks.shape[-2:]}."
            )
        if self.config.data_prep.context_crop_enabled:
            crop_h, crop_w = self.config.data_prep.resolved_target_crop()
            row_slice, col_slice = centered_crop_slices(
                predictions.shape[-2],
                predictions.shape[-1],
                crop_h,
                crop_w,
            )
            predictions = predictions[..., row_slice, col_slice]
            targets = targets[..., row_slice, col_slice]
            masks = masks[..., row_slice, col_slice]

        if getattr(self.loss_fn, "requires_patch_metadata", False):
            if patch_metadata is None:
                raise ValueError("Configured loss requires patch metadata, but batch does not include 'patch_metadata'.")
            patch_metadata = {key: value.to(self.device) for key, value in patch_metadata.items()}
            loss_out = self.loss_fn(predictions, targets, masks, patch_metadata=patch_metadata)
        else:
            loss_out = self.loss_fn(predictions, targets, masks)

        # Support if it is a single loss or weighted loss
        if isinstance(loss_out, tuple):
            total_loss, loss_parts = loss_out
        else:
            total_loss = cast(torch.Tensor, loss_out)
            loss_parts = None

        pop_regularization_loss = getattr(self.model, "pop_regularization_loss", None)
        if pop_regularization_loss is not None:
            regularization_loss = pop_regularization_loss()
            if regularization_loss is not None:
                total_loss = total_loss + regularization_loss
                loss_parts = dict(loss_parts or {})
                loss_parts["model/field_regularization"] = regularization_loss

        if self._coarse_bp_loss is not None:
            if coarse_bp_probability is None:
                raise RuntimeError("Model did not provide the coarse BP probability required for supervision.")
            coarse_bp_loss, coarse_bp_parts = self._compute_coarse_bp_supervision(
                coarse_bp_probability,
                targets,
                masks,
                full_spatial_shape,
            )
            total_loss = total_loss + coarse_bp_loss
            loss_parts = dict(loss_parts or {})
            loss_parts.update(coarse_bp_parts)

        if self._bp_hazard_residual_weight > 0.0:
            if bp_hazard_residual is None:
                raise RuntimeError("Model did not provide the BP hazard residual required for regularization.")
            residual_fraction, residual_burnability = bp_hazard_residual
            residual_loss, residual_parts = self._compute_bp_hazard_residual_regularization(
                residual_fraction=residual_fraction,
                burnability=residual_burnability,
                masks=masks,
                full_spatial_shape=full_spatial_shape,
            )
            total_loss = total_loss + residual_loss
            loss_parts = dict(loss_parts or {})
            loss_parts.update(residual_parts)

        metric_predictions = activate_target_predictions(predictions, self._target_specs)
        return metric_predictions, total_loss, loss_parts, targets, masks

    @staticmethod
    def _are_metrics_better(curr: list[float], best: list[float], modes: list[str]):
        if not all(np.isfinite(value) for value in curr):
            return False
        if not best:
            return True

        improved = False
        for c, b, mode in zip(curr, best, modes, strict=False):
            if mode == "min":
                if c > b:
                    return False  # a metric got worse!
                elif c < b:
                    improved = True
            elif mode == "max":
                if c < b:
                    return False  # a metric got worse!
                elif c > b:
                    improved = True
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return improved  # Only True if at least one metric improved, none worse

    @staticmethod
    def _accumulate_metrics(
        running_metrics: dict[str, float],
        metric_counts: dict[str, int],
        metric_values: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        for name, value in metric_values.items():
            scalar = value.item()
            if not np.isfinite(scalar):
                continue
            running_metrics[name] += scalar * batch_size
            metric_counts[name] += batch_size

    @staticmethod
    def _average_metrics(running_metrics: dict[str, float], metric_counts: dict[str, int]) -> dict[str, float]:
        return {
            name: total_value / metric_counts[name] if metric_counts[name] > 0 else float("nan")
            for name, total_value in running_metrics.items()
        }

    def _add_derived_metrics(self, results: dict[str, float]) -> None:
        ccc_keys = [f"{target_name}/ccc" for target_name in self._target_names]
        if len(ccc_keys) > 1 and all(key in results and np.isfinite(results[key]) for key in ccc_keys):
            results["mean/ccc"] = float(np.mean([results[key] for key in ccc_keys]))

    def _validate_stitched_best_ckpt_metrics(self) -> list[str]:
        stitched_metrics = [name for name in self.best_ckpt_metrics if name.startswith("hex/")]
        for name in stitched_metrics:
            parts = name.split("/")
            if len(parts) != 3:
                raise ValueError(
                    f"Stitched checkpoint metric {name!r} must use 'hex/<target-or-mean>/<metric>', for example 'hex/mean/ccc'."
                )
            _, target_name, metric_name = parts
            if target_name != "mean" and target_name not in self._target_names:
                raise ValueError(
                    f"Stitched checkpoint metric {name!r} references unknown target {target_name!r}; "
                    f"configured targets are {self._target_names}."
                )
            if metric_name not in AVAILABLE_METRICS:
                raise ValueError(
                    f"Stitched checkpoint metric {name!r} references unknown metric {metric_name!r}; "
                    f"available metrics are {sorted(AVAILABLE_METRICS)}."
                )
        return stitched_metrics

    def _compute_stitched_validation_metrics(
        self,
        predictions: np.ndarray,
        loader: DataLoader,
    ) -> dict[str, float]:
        """Stitch predictions already produced by validation and compute requested hex metrics."""
        if not self._stitched_best_ckpt_metrics:
            return {}
        dataset = loader.dataset
        metadata = getattr(dataset, "metadata", None)
        if metadata is None:
            raise ValueError("Stitched validation checkpoint selection requires dataset patch metadata.")

        requested_metric_names = sorted({name.rsplit("/", 1)[1] for name in self._stitched_best_ckpt_metrics})
        metric_functions = {name: AVAILABLE_METRICS[name] for name in requested_metric_names}
        out_norm = self._grid_params.resolved_targets()[0].out_norm if self._grid_params is not None else "none"

        from src.datasets.postprocessing.utils import evaluate_and_visualize_hexels

        raw_metrics = evaluate_and_visualize_hexels(
            test_predictions=predictions,
            config=self.config,
            out_norm=out_norm,
            device="cpu",
            metric_functions=metric_functions,
            save_artifacts=False,
            save_plots=False,
            split_csv=self.config.data.val_split,
            test_metadata=metadata,
        )

        results: dict[str, float] = {}
        multi_target = len(self._target_names) > 1
        for metric_name in requested_metric_names:
            target_values = []
            for target_name in self._target_names:
                raw_key = f"all/{target_name}_{metric_name}" if multi_target else f"all/{metric_name}"
                value = float(raw_metrics.get(raw_key, float("nan")))
                results[f"hex/{target_name}/{metric_name}"] = value
                target_values.append(value)
            results[f"hex/mean/{metric_name}"] = (
                float(np.mean(target_values)) if target_values and all(np.isfinite(value) for value in target_values) else float("nan")
            )
        return results

    # Supports any LRScheduler object and metric-based ReduceLROnPlateau schedulers
    def train_epoch(
        self, loader: DataLoader, lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None, lr_scheduler_type: str | None = None
    ) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self._metric_result_keys()}
        metric_counts = {name: 0 for name in running_metrics}
        running_loss_parts: dict[str, float] = {}

        training_loop = tqdm(loader, desc="Training", leave=True)
        accumulation_steps = self.config.training.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(training_loop):
            predictions, loss, loss_parts, targets, masks = self._step(batch)
            (loss / accumulation_steps).backward()
            optimizer_step = (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader)
            if optimizer_step:
                if self.config.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.training.gradient_clip_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # use scheduler if its type is batch-level
            if optimizer_step and lr_scheduler is not None and lr_scheduler_type == "batch":
                lr_scheduler.step()

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            if self.logger and self.global_step % self.log_every_n_step == 0:
                self.logger.log_metrics({"train_step_loss": loss.item()}, step=self.global_step)

                # log LR since it can change with scheduler
                current_lr = max(group["lr"] for group in self.optimizer.param_groups)
                self.logger.log_metrics({"learning_rate": current_lr}, step=self.global_step)

            training_loop.set_description(f"Loss: {running_loss / running_batch_count:.4f}")

            # compute the metrics
            with torch.no_grad():
                metric_predictions, metric_targets = self._prepare_metric_tensors(predictions.detach(), targets)
                metric_values = self._compute_metric_values(metric_predictions, metric_targets, masks)
                self._accumulate_metrics(running_metrics, metric_counts, metric_values, batch_size)
                for name, value in metric_values.items():
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics({f"train_step_{name}": value.item()}, step=self.global_step)

                if loss_parts is not None:
                    # log each loss part (raw/unweighted)
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics(
                            {f"train_step_loss_{k}": v.item() for k, v in loss_parts.items()},
                            step=self.global_step,
                        )
                    # accumulate epoch averages
                    for k, v in loss_parts.items():
                        running_loss_parts[k] = running_loss_parts.get(k, 0.0) + v.item() * batch_size

            self.global_step += 1

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        if running_loss_parts:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

        # add averaged metrics to results
        results.update(self._average_metrics(running_metrics, metric_counts))
        self._add_derived_metrics(results)

        return results

    @overload
    def validate(self, loader: DataLoader, return_predictions: Literal[False] = False) -> dict[str, float]: ...

    @overload
    def validate(self, loader: DataLoader, return_predictions: Literal[True]) -> tuple[dict[str, float], np.ndarray]: ...

    @overload
    def validate(self, loader: DataLoader, return_predictions: bool) -> dict[str, float] | tuple[dict[str, float], np.ndarray]: ...

    @torch.no_grad()
    def validate(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self._metric_result_keys()}
        metric_counts = {name: 0 for name in running_metrics}
        running_loss_parts: dict[str, float] = {}

        peak_mps_driver_allocated_gb = 0.0
        self.last_peak_mps_driver_allocated_gb = None

        preds_list = []
        validation_loop = tqdm(loader, desc="Evaluating", leave=True)

        for batch_idx, batch in enumerate(validation_loop):
            predictions, loss, loss_parts, targets, masks = self._step(batch)

            if return_predictions:
                preds_list.append(predictions.detach().cpu().numpy())

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            # compute the metrics
            with torch.no_grad():
                metric_predictions, metric_targets = self._prepare_metric_tensors(predictions.detach(), targets)
                metric_values = self._compute_metric_values(metric_predictions, metric_targets, masks)
                self._accumulate_metrics(running_metrics, metric_counts, metric_values, batch_size)

                if loss_parts is not None:
                    for k, v in loss_parts.items():
                        running_loss_parts[k] = running_loss_parts.get(k, 0.0) + v.item() * batch_size

            # NOTE: MPS-only diagnostic -- captured BEFORE empty_cache() flushes it, so
            # this reflects the real per-batch peak, not the post-flush snapshot. The
            # earlier investigation confirmed individual batches genuinely peak near
            # 30GB even though empty_cache() brings end-of-run allocation back down to
            # a couple GB -- this line is what makes that visible.
            if return_predictions and predictions.device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
                batch_peak_gb = torch.mps.driver_allocated_memory() / 1024**3
                peak_mps_driver_allocated_gb = max(peak_mps_driver_allocated_gb, batch_peak_gb)
                if os.getenv("MPS_DIAG") == "1":
                    print(f"[diag] batch {batch_idx + 1} peak driver_allocated: {batch_peak_gb:.3f} GB")

            # NOTE: MPS-only -- the caching allocator retains freed batch memory
            # without releasing it to the OS, so per-batch driver_allocated_memory
            # climbs across the whole loop instead of resetting. empty_cache()
            # after each batch prevents that accumulation. No-op cost on CPU/CUDA,
            # so this is gated rather than applied unconditionally.
            if predictions.device.type == "mps":
                torch.mps.empty_cache()

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        if running_loss_parts:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

        # add averaged metrics to results
        results.update(self._average_metrics(running_metrics, metric_counts))
        self._add_derived_metrics(results)

        if return_predictions and peak_mps_driver_allocated_gb > 0.0:
            self.last_peak_mps_driver_allocated_gb = peak_mps_driver_allocated_gb

        if return_predictions:
            return results, np.concatenate(preds_list, axis=0)

        return results

    @overload
    def test(self, loader: DataLoader, return_predictions: Literal[False] = False) -> dict[str, float]: ...

    @overload
    def test(self, loader: DataLoader, return_predictions: Literal[True]) -> tuple[dict[str, float], np.ndarray]: ...

    @overload
    def test(self, loader: DataLoader, return_predictions: bool) -> dict[str, float] | tuple[dict[str, float], np.ndarray]: ...

    @torch.no_grad()
    def test(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
        return self.validate(loader, return_predictions=return_predictions)

    def _maybe_resume(self, lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None) -> int:
        """
        Resume training from ``last.pth`` when it exists so a preempted/requeued SLURM job
        continues instead of restarting from scratch. Returns the next epoch to run
        (``1`` when there is no compatible checkpoint to resume from).
        """
        if not self.save_dir:
            return 1
        last_path = os.path.join(self.save_dir, "last.pth")
        if not os.path.exists(last_path):
            return 1

        # weights_only=False: these checkpoints are produced by this same Trainer and
        # include optimizer/scheduler/RNG state (numpy arrays, etc.) that torch's
        # default `weights_only=True` restricted unpickler will not load.
        checkpoint = torch.load(last_path, map_location=self.device, weights_only=False)
        saved_state = checkpoint.get("model_state", {})
        current_state = self.model.state_dict()
        compatible = saved_state.keys() == current_state.keys() and all(
            saved_state[key].shape == current_state[key].shape for key in current_state
        )
        if not compatible:
            print(f"[Resume] {last_path} does not match the current model; starting from scratch.")
            return 1

        self.model.load_state_dict(saved_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if lr_scheduler is not None and checkpoint.get("scheduler_state") is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("best_metric_list"):
            self._best_metric_list = list(checkpoint["best_metric_list"])
        self.global_step = int(checkpoint.get("global_step", self.global_step))

        # Restore RNG states so a resumed run (e.g. after preemption) reproduces the
        # same data order/augmentations as an uninterrupted one, per Mila's
        # checkpointing guidelines.
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python_random_state"])
            np.random.set_state(rng_state["numpy_random_state"])
            # The CPU RNG state tensor must stay on CPU even though `map_location`
            # above may have moved other checkpoint tensors to an accelerator device.
            torch.random.set_rng_state(rng_state["torch_random_state"].cpu())
            if torch.cuda.is_available() and rng_state.get("torch_cuda_random_state") is not None:
                # Like the CPU RNG state above, each per-device state tensor must stay on
                # CPU (as a plain torch.ByteTensor) even though `map_location` may have
                # moved it onto an accelerator device; `set_rng_state_all` rejects
                # anything that isn't a CPU ByteTensor.
                cuda_rng_states = [state.cpu() for state in rng_state["torch_cuda_random_state"]]
                torch.cuda.set_rng_state_all(cuda_rng_states)

        last_epoch = int(checkpoint.get("epoch", 0))
        print(f"[Resume] Resuming from {last_path}: completed epoch {last_epoch}, continuing at epoch {last_epoch + 1}.")
        return last_epoch + 1

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        num_epochs = self.config.training.max_epochs
        log_every_n_epoch = self.config.training.log_every_n_epoch

        # get scheduler and its type
        lr_scheduler, lr_scheduler_type = build_lr_scheduler(self.config, self.optimizer, train_loader)

        start_epoch = self._maybe_resume(lr_scheduler)

        for epoch in range(start_epoch, num_epochs + 1):
            self._configure_pretrained_freeze(epoch)
            start = time.time()
            train_res = self.train_epoch(train_loader, lr_scheduler=lr_scheduler, lr_scheduler_type=lr_scheduler_type)
            elapsed = time.time() - start

            val_result = None
            if val_loader is not None:
                if self._stitched_best_ckpt_metrics:
                    val_result_with_predictions = self.validate(val_loader, return_predictions=True)
                    if not isinstance(val_result_with_predictions, tuple):
                        raise RuntimeError("Validation predictions were requested but not returned.")
                    val_result, val_predictions = val_result_with_predictions
                    val_result.update(self._compute_stitched_validation_metrics(val_predictions, val_loader))
                    del val_predictions
                else:
                    val_result = self.validate(val_loader)
            if isinstance(val_result, tuple):
                val_result = val_result[0]

            # for epoch level schedulers
            if lr_scheduler is not None:
                if lr_scheduler_type == "epoch":
                    lr_scheduler.step()
                elif lr_scheduler_type == "epoch_metric":
                    # Plateau needs a metric to watch. Default to val_loss, fallback to train_loss
                    watch_metric = val_result["loss"] if val_result else train_res["loss"]
                    lr_scheduler.step(watch_metric)

            # log metrics and loss
            if epoch % log_every_n_epoch == 0:
                current_lr = max(group["lr"] for group in self.optimizer.param_groups)
                msg = f"Epoch {epoch}/{num_epochs} - train_loss: {train_res['loss']:.4f}"
                if val_result is not None:
                    msg += f", val_loss: {val_result['loss']:.4f}"
                msg += f", lr: {current_lr:.2e}, time: {elapsed:.1f}s"
                print(msg)

                metrics_to_log = {f"train_{k}": v for k, v in train_res.items()}
                metrics_to_log["epoch_duration"] = elapsed

                if val_result:
                    metrics_to_log.update({f"val_{k}": v for k, v in val_result.items()})
                diagnostic_metrics = getattr(self.model, "diagnostic_metrics", None)
                if diagnostic_metrics is not None:
                    metrics_to_log.update({f"model/{name}": value for name, value in diagnostic_metrics().items()})

                if self.logger:
                    self.logger.log_metrics(metrics_to_log, epoch=epoch)

            # Save best checkpoint based on multi-metrics
            if val_result:
                curr_metric_list = [val_result[m] for m in self.best_ckpt_metrics]
                if self._are_metrics_better(curr_metric_list, self._best_metric_list, self.best_ckpt_modes):
                    self._best_metric_list = curr_metric_list
                    if self.save_dir:
                        best_path = self.save_model(
                            epoch=epoch,
                            metric_value={m: v for m, v in zip(self.best_ckpt_metrics, curr_metric_list, strict=False)},
                            filename="best.pth",
                        )
                        if self.logger:
                            self.logger.experiment.log_model(name="best", file_or_folder=best_path, overwrite=True)

                # save most recent checkpoint
                self.save_model(
                    epoch=epoch,
                    metric_value={m: v for m, v in zip(self.best_ckpt_metrics, curr_metric_list, strict=False)},
                    lr_scheduler=lr_scheduler,
                )

    def save_model(
        self,
        epoch: int,
        metric_value: float | dict,
        filename: str = "last.pth",
        lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    ):
        if not self.save_dir:
            raise ValueError("save_dir not set")

        path = os.path.join(self.save_dir, filename)

        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric_value": metric_value,
            "config": self.config.model_dump(),
            "global_step": self.global_step,
            "best_metric_list": self._best_metric_list,
            "scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            # RNG states so a resumed run (e.g. after SLURM preemption) can reproduce
            # the same data order/augmentations as an uninterrupted one.
            "rng_state": {
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.random.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            # So a resumed run continues logging into the same Comet experiment
            # instead of starting a new one on each SLURM requeue.
            "comet_experiment_key": self.logger.experiment_key if self.logger else None,
        }
        # Write atomically so a preemption mid-save cannot leave a corrupt checkpoint.
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_model(self, path: str | None = None, filename: str = "last.pth", map_location: str | None = None):
        if path is None:
            path = os.path.join(self.save_dir, filename)

        map_location = map_location or self.device
        # weights_only=False: trusted, self-produced checkpoints that also carry
        # optimizer/scheduler/RNG state beyond plain tensors.
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint
