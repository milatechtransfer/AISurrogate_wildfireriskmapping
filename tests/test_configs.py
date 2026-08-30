import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    SEEDS,
    Config,
    GridParams,
    HazardEvalConfig,
    HazardModelEntry,
    OptimizerConfig,
    SpatializedTabularParams,
    TargetConfig,
    TargetLossConfig,
    apply_run_id_overrides,
)
from src.config_io import load_config as load_resolved_config
from src.datasets.postprocessing.hazard import DEFAULT_FI_CAP, DEFAULT_SCALE_TO
from src.utils import AVAILABLE_METRICS, build_single_loss

BP_CONFIG = Path("configs/bp_spatial_weather.yaml")
FI_CONFIG = Path("configs/fi_spatial_weather.yaml")
ROS_CONFIG = Path("configs/ros_spatial_weather.yaml")
MULTI_OUTPUT_CONFIG = Path("configs/multi_output_spatial_weather.yaml")
HAZARD_EVAL_CONFIG = Path("configs/hazard_eval_spatial_weather.yaml")
HAZARD_MODEL_CONFIG = Path("configs/multi_output_spatial_weather.yaml")
SPREAD_OPPORTUNITY_CONFIG = Path("configs/context_models/unet_512_crop_256_spread_opportunity_q3.yaml")
MECHANISTIC_V21_CONFIG = Path("configs/mechanistic/mechanistic_propagation_v21_512_crop_256_firesize_q3.yaml")
MECHANISTIC_V21_PILOT_CONFIG = Path("configs/mechanistic/mechanistic_propagation_v21_512_crop_256_firesize_q3_pilot.yaml")
MECHANISTIC_V3_CONFIG = Path("configs/mechanistic/mechanistic_propagation_v3_512_crop_256_spread_opportunity_q3.yaml")
MECHANISTIC_V3_PILOT_CONFIG = Path("configs/mechanistic/mechanistic_propagation_v3_512_crop_256_spread_opportunity_q3_pilot.yaml")
MECHANISTIC_V4_CONFIG = Path("configs/mechanistic/mechanistic_travel_time_v4_512_crop_256_spread_opportunity_q3.yaml")
GRAY_BOX_PHYSICS_V3_CONFIG = Path("configs/mechanistic/gray_box_physics_v3_512_crop_256_firesize_q3.yaml")
MECHANISTIC_HYBRID_V22_CONFIG = Path("configs/mechanistic/mechanistic_hybrid_v22_native_512_crop_256_firesize_q3.yaml")
MECHANISTIC_HYBRID_V23_CONFIG = Path("configs/mechanistic/mechanistic_hybrid_v23_native_512_crop_256_firesize_q3.yaml")
MECHANISTIC_HYBRID_V24_CONFIG = Path("configs/mechanistic/mechanistic_hybrid_v24_native_512_crop_256_firesize_q3.yaml")
COUNT_UNET_CONFIG = Path("configs/context_models/unet_512_crop_256_firesize_q3_ignition_count.yaml")
FIRE_SIZE_UNET_256_CONFIG = Path("configs/context_models/unet_256_firesize_q3.yaml")
COUNT_UNET_256_CONFIG = Path("configs/context_models/unet_256_firesize_q3_ignition_count.yaml")
COMPACT_FIRE_SIZE_UNET_256_CONFIG = Path("configs/context_models/unet_compact_256_firesize_q3.yaml")
COMPACT_FIRE_SIZE_UNET_512_CONFIG = Path("configs/context_models/unet_compact_512_crop_256_firesize_q3.yaml")
COMPACT_COUNT_UNET_CONFIG = Path("configs/context_models/unet_compact_512_crop_256_firesize_q3_ignition_count.yaml")
COMPACT_COUNT_MECHANISTIC_CONFIG = Path("configs/mechanistic/mechanistic_travel_time_compact_512_crop_256_firesize_q3_ignition_count.yaml")
COMPACT_FIRE_SIZE_MECHANISTIC_CONFIG = Path("configs/mechanistic/mechanistic_travel_time_compact_512_crop_256_firesize_q3.yaml")
COMMON_INPUT_PIPELINE_CONFIGS = [BP_CONFIG, FI_CONFIG, ROS_CONFIG]

WEATHER_FEATURES = {
    "Temperature",
    "RelativeHumidity",
    "Precipitation",
    "FineFuelMoistureCode",
    "DuffMoistureCode",
    "DroughtCode",
    "InitialSpreadIndex",
    "BuildupIndex",
    "FireWeatherIndex",
    "wind_x",
    "wind_y",
}


def _load_config(path: Path) -> Config:
    with path.open() as f:
        return Config(**yaml.safe_load(f))


def _load_hazard_eval_config(path: Path) -> HazardEvalConfig:
    with path.open() as f:
        return HazardEvalConfig(**yaml.safe_load(f))


def test_common_input_pipeline_configs_share_unified_input_pipeline():
    for path in COMMON_INPUT_PIPELINE_CONFIGS:
        config = _load_config(path)
        sources = {source.name: source.params for source in config.data.input_sources}

        # Spatial-only model
        assert config.model.input_branches == ["spatial"]

        # Dataset is the leakage-fixed, aggregated-ignition data_samples_v4.
        assert config.data.root_dir.endswith("data_samples_v4")

        # Grid: aggregated 2-channel ignition + terrain derivatives.
        assert isinstance(sources["grid"], GridParams)
        assert sources["grid"].feature_names_list[:2] == ["ignition_grid_human", "ignition_grid_lightning"]
        assert sources["grid"].terrain_derivatives == ["slope", "aspect_sin", "aspect_cos"]

        # Spatialized weather. The weather LUT covers every firezone, so its
        # missing-firezone mask is dropped; the fire-size table lacks some zones, so it is kept.
        assert isinstance(sources["spatialized_weather"], SpatializedTabularParams)
        assert sources["spatialized_weather"].include_missing_firezone_mask is False

        # Wind is expressed as Cartesian components only; WindSpeed is dropped.
        weather_features = set(sources["spatialized_weather"].feature_names_list)
        assert {"wind_x", "wind_y"} <= weather_features
        assert "WindSpeed" not in weather_features
        assert weather_features == WEATHER_FEATURES


def test_common_input_pipeline_configs_reference_supported_losses_and_metrics():
    for path in COMMON_INPUT_PIPELINE_CONFIGS:
        config = _load_config(path)
        loss_names = config.optimizer.loss if isinstance(config.optimizer.loss, list) else [config.optimizer.loss]

        assert config.logger.enabled is True
        assert config.logger.log_every_n_step == 10
        for loss_name in loss_names:
            build_single_loss(loss_name, huber_beta=config.optimizer.huber_beta)
        for metric_name in config.metrics:
            assert metric_name in AVAILABLE_METRICS


def test_spread_opportunity_context_unet_is_conventional_minmax_ablation():
    config = load_resolved_config(SPREAD_OPPORTUNITY_CONFIG)
    sources = {source.name: source.params for source in config.data.input_sources}
    grid = sources["grid"]
    spread = sources["spatialized_spread_opportunity"]

    assert config.model.architecture == "auto"
    assert config.data.root_dir.endswith("data_samples_v4_context_512_crop_256")
    assert config.data.batch_size == 8
    assert config.training.gradient_accumulation_steps == 8
    assert isinstance(grid, GridParams)
    assert grid.target_config("bp").out_norm == "min_max"
    assert isinstance(spread, SpatializedTabularParams)
    assert spread.feature_names_list == [
        "NORM_TOTAL_BURN_HOURS_Q10",
        "NORM_TOTAL_BURN_HOURS_Q50",
        "NORM_TOTAL_BURN_HOURS_Q90",
    ]
    assert spread.hex_id_col == "hex_id"
    assert spread.quantiles is None
    assert spread.missing_value_strategy == "raise"


def test_count_conditioned_configs_resolve_shared_feature_contract():
    full = load_resolved_config(COUNT_UNET_CONFIG)
    full_256 = load_resolved_config(COUNT_UNET_256_CONFIG)
    compact = load_resolved_config(COMPACT_COUNT_UNET_CONFIG)
    mechanistic = load_resolved_config(COMPACT_COUNT_MECHANISTIC_CONFIG)

    for config in (full, full_256, compact, mechanistic):
        count = next(source.params for source in config.data.input_sources if source.name == "spatialized_ignition_count")
        assert isinstance(count, SpatializedTabularParams)
        assert count.feature_names_list == [
            "NORM_LOG1P_IGNITION_COUNT_MEAN",
            "NORM_IGNITION_COUNT_CV",
        ]
        assert count.hex_id_col == "hex_id"
        assert count.missing_value_strategy == "raise"
        assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]

    assert full.model.hidden_features == [64, 128, 256, 512]
    assert full_256.model.hidden_features == [64, 128, 256, 512]
    assert full_256.data.batch_size == 64
    assert full_256.training.gradient_accumulation_steps == 1
    assert full_256.data_prep.resolved_target_crop() == (256, 256)
    assert not full_256.data_prep.context_crop_enabled
    assert compact.model.hidden_features == [32, 64, 128, 256]
    assert compact.data_prep.ignition_weighting == "probability_mass"
    assert mechanistic.model.architecture == "mechanistic_travel_time_v4"
    assert mechanistic.model.propagation_ignition_mode == "probability_mass"
    assert mechanistic.model.propagation_scenario_mode == "fire_size"
    assert mechanistic.training.freeze_pretrained_epochs == mechanistic.training.max_epochs


def test_256_fire_size_count_ablation_changes_only_count_source():
    baseline = load_resolved_config(FIRE_SIZE_UNET_256_CONFIG)
    count_conditioned = load_resolved_config(COUNT_UNET_256_CONFIG)

    baseline_sources = [source.name for source in baseline.data.input_sources]
    count_sources = [source.name for source in count_conditioned.data.input_sources]

    assert baseline_sources == ["grid", "spatialized_weather", "spatialized_fire_size"]
    assert count_sources == [*baseline_sources, "spatialized_ignition_count"]
    assert baseline.model == count_conditioned.model
    assert baseline.optimizer == count_conditioned.optimizer
    assert baseline.training == count_conditioned.training
    assert baseline.evaluation == count_conditioned.evaluation
    assert baseline.data.root_dir == count_conditioned.data.root_dir
    assert baseline.data_prep == count_conditioned.data_prep


def test_compact_fire_size_configs_drop_count_and_preserve_geometry():
    compact_256 = load_resolved_config(COMPACT_FIRE_SIZE_UNET_256_CONFIG)
    compact_512 = load_resolved_config(COMPACT_FIRE_SIZE_UNET_512_CONFIG)
    mechanistic = load_resolved_config(COMPACT_FIRE_SIZE_MECHANISTIC_CONFIG)

    for config in (compact_256, compact_512, mechanistic):
        assert config.model.hidden_features == [32, 64, 128, 256]
        assert "spatialized_ignition_count" not in [source.name for source in config.data.input_sources]
        fire_size = next(source.params for source in config.data.input_sources if source.name == "spatialized_fire_size")
        assert isinstance(fire_size, SpatializedTabularParams)
        assert fire_size.quantiles == [0.1, 0.5, 0.9]

    assert compact_256.data_prep.resolved_target_crop() == (256, 256)
    assert not compact_256.data_prep.context_crop_enabled
    assert compact_256.data.batch_size * compact_256.training.gradient_accumulation_steps == 64
    assert compact_512.data_prep.resolved_target_crop() == (256, 256)
    assert compact_512.data_prep.context_crop_enabled
    assert compact_512.data.batch_size * compact_512.training.gradient_accumulation_steps == 64
    assert mechanistic.model.propagation_ignition_mode == "legacy_intensity"
    assert mechanistic.model.propagation_scenario_mode == "fire_size"
    assert mechanistic.training.freeze_pretrained_epochs == mechanistic.training.max_epochs


@pytest.mark.parametrize(
    ("config_path", "architecture"),
    [
        (MECHANISTIC_V21_CONFIG, "mechanistic_propagation_v21"),
        (MECHANISTIC_V3_CONFIG, "mechanistic_propagation_v3"),
    ],
)
def test_stabilized_mechanistic_configs_resolve(config_path, architecture):
    config = load_resolved_config(config_path)

    assert config.model.architecture == architecture
    assert config.optimizer.lr == pytest.approx(3.0e-4)
    assert config.optimizer.parameter_lr_scales["raw_ignition_scale"] == pytest.approx(0.1)
    assert config.optimizer.parameter_lr_scales["raw_bp_scale"] == pytest.approx(0.1)
    assert config.optimizer.parameter_lr_scales["bp_local_calibration"] == pytest.approx(0.25)
    assert config.training.gradient_clip_norm == pytest.approx(1.0)
    assert config.evaluation.best_ckpt_metrics == ["mean/ccc"]
    assert config.evaluation.best_ckpt_metrics_mode == ["max"]


def test_mechanistic_hybrid_v22_config_is_native_scratch_q3() -> None:
    config = load_resolved_config(MECHANISTIC_HYBRID_V22_CONFIG)

    assert config.model.architecture == "mechanistic_hybrid_v22"
    assert config.seed == 42
    assert config.training.initial_checkpoint is None
    assert config.training.freeze_pretrained_epochs == 0
    assert config.data_prep.preserve_native_grid
    assert config.data_prep.resolved_target_crop() == (256, 256)
    assert config.model.propagation_downsample_factor == 8
    assert config.model.propagation_steps == 16
    assert config.model.propagation_speed_correction_log_limit == pytest.approx(math.log(2.0))
    assert config.model.propagation_max_local_log_calibration == pytest.approx(math.log(1.5))
    assert config.data.batch_size * config.training.gradient_accumulation_steps == 64
    grid = next(source.params for source in config.data.input_sources if source.name == "grid")
    assert isinstance(grid, GridParams)
    assert grid.fuel_feats_encoding == "iROS_HFI"
    fire_size = next(source.params for source in config.data.input_sources if source.name == "spatialized_fire_size")
    assert isinstance(fire_size, SpatializedTabularParams)
    assert fire_size.quantiles == [0.1, 0.5, 0.9]
    assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]
    assert config.evaluation.best_ckpt_metrics_mode == ["max"]


def test_mechanistic_hybrid_v23_config_uses_direct_log_fire_size_and_additive_hazard() -> None:
    config = load_resolved_config(MECHANISTIC_HYBRID_V23_CONFIG)

    assert config.model.architecture == "mechanistic_hybrid_v23"
    assert config.seed == 42
    assert config.training.initial_checkpoint is None
    assert config.data_prep.preserve_native_grid
    assert config.data_prep.resolved_target_crop() == (256, 256)
    assert config.model.propagation_fire_size_neural_mean == pytest.approx(2.689850106599859)
    assert config.model.propagation_fire_size_neural_std == pytest.approx(0.7868705075427128)
    assert config.model.propagation_max_additive_bp_hazard == pytest.approx(-math.log(0.8))
    assert config.model.propagation_additive_bp_hazard_weight == pytest.approx(0.1)
    assert config.model.propagation_coarse_bp_supervision_weight == pytest.approx(0.05)
    fire_size = next(source.params for source in config.data.input_sources if source.name == "spatialized_fire_size")
    assert isinstance(fire_size, SpatializedTabularParams)
    assert fire_size.feature_names_list == ["LOG_SIZE_HA"]
    assert fire_size.quantiles == [0.1, 0.5, 0.9]
    assert fire_size.include_missing_firezone_mask
    assert fire_size.global_fill_csv_name == "fire_size_global_fill.csv"
    assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]


def test_mechanistic_hybrid_v24_config_decouples_tasks_and_uses_signed_hazard_residual() -> None:
    config = load_resolved_config(MECHANISTIC_HYBRID_V24_CONFIG)

    assert config.model.architecture == "mechanistic_hybrid_v24"
    assert config.seed == 42
    assert config.training.initial_checkpoint is None
    assert config.data_prep.preserve_native_grid
    assert config.data_prep.resolved_target_crop() == (256, 256)
    assert config.model.propagation_additive_bp_hazard_weight == 0.0
    assert config.model.propagation_initial_bp_hazard_residual == pytest.approx(1e-4)
    assert config.model.propagation_max_bp_hazard_residual == pytest.approx(2e-3)
    assert config.model.propagation_max_bp_hazard_attenuation == pytest.approx(0.9)
    assert config.model.propagation_bp_hazard_residual_weight == pytest.approx(1e-2)
    assert config.model.propagation_coarse_bp_supervision_weight == pytest.approx(0.05)
    assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]


def test_mechanistic_v3_resolves_scenario_budget_contract():
    config = load_resolved_config(MECHANISTIC_V3_CONFIG)
    spread = next(source.params for source in config.data.input_sources if source.name == "spatialized_spread_opportunity")

    assert isinstance(spread, SpatializedTabularParams)
    assert spread.feature_names_list == [
        "NORM_TOTAL_BURN_HOURS_Q10",
        "NORM_TOTAL_BURN_HOURS_Q50",
        "NORM_TOTAL_BURN_HOURS_Q90",
        "SCENARIO_FALLBACK",
    ]
    assert config.model.propagation_budget_min_hours == pytest.approx(1.0)
    assert config.model.propagation_budget_max_hours == pytest.approx(140.0)
    assert config.model.propagation_budget_hours_per_step == pytest.approx(2.0)
    assert config.model.propagation_budget_temperature_hours == pytest.approx(2.0)


def test_mechanistic_v4_resolves_warm_started_travel_time_contract():
    config = load_resolved_config(MECHANISTIC_V4_CONFIG)
    spread = next(source.params for source in config.data.input_sources if source.name == "spatialized_spread_opportunity")

    assert config.model.architecture == "mechanistic_travel_time_v4"
    assert config.model.propagation_downsample_factor == 16
    assert config.model.propagation_steps == 32
    assert config.model.propagation_budget_min_hours == pytest.approx(1.0)
    assert config.model.propagation_budget_max_hours == pytest.approx(140.0)
    assert config.training.max_epochs == 10
    assert config.training.freeze_pretrained_epochs == 1
    assert config.training.initial_checkpoint is not None
    assert config.optimizer.parameter_lr_scales["encoder"] == pytest.approx(0.1)
    assert config.evaluation.best_ckpt_metrics == ["mean/ccc"]
    assert isinstance(spread, SpatializedTabularParams)
    assert spread.feature_names_list == [
        "NORM_TOTAL_BURN_HOURS_Q10",
        "NORM_TOTAL_BURN_HOURS_Q50",
        "NORM_TOTAL_BURN_HOURS_Q90",
    ]


def test_gray_box_physics_v3_resolves_bounded_field_contract():
    config = load_resolved_config(GRAY_BOX_PHYSICS_V3_CONFIG)

    assert config.model.architecture == "interpretable_mechanistic_v3"
    assert config.model.interpretable_field_hidden_channels == 48
    assert config.model.interpretable_source_grid_size == 4
    assert config.model.interpretable_ignition_field_log_limit == pytest.approx(1.3862943611198906)
    assert config.model.interpretable_ros_field_log_limit == pytest.approx(0.6931471805599453)
    assert config.model.interpretable_fire_size_field_log_limit == pytest.approx(1.3862943611198906)
    assert config.model.interpretable_reach_field_log_limit == pytest.approx(1.3862943611198906)
    assert config.model.interpretable_consumption_field_log_limit == pytest.approx(0.6931471805599453)
    assert config.model.interpretable_field_l2_weight == pytest.approx(5.0e-3)
    assert config.model.interpretable_field_tv_weight == pytest.approx(5.0e-3)
    assert config.optimizer.name == "AdamW"
    assert config.optimizer.lr == pytest.approx(2.0e-3)
    assert config.optimizer.parameter_lr_scales["raw_per_fire_reach_scale"] == pytest.approx(0.25)
    assert config.optimizer.parameter_lr_scales["travel_time_propagation.raw_area_multiplier"] == pytest.approx(0.25)
    assert config.training.gradient_clip_norm == pytest.approx(1.0)
    assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]
    assert config.data_prep.resolved_target_crop() == (256, 256)
    assert "spatialized_ignition_count" in [source.name for source in config.data.input_sources]


@pytest.mark.parametrize("config_path", [MECHANISTIC_V21_PILOT_CONFIG, MECHANISTIC_V3_PILOT_CONFIG])
def test_mechanistic_pilot_configs_inherit_five_epoch_recipes(config_path):
    config = load_resolved_config(config_path)

    assert config.training.max_epochs == 5
    assert config.training.gradient_clip_norm == pytest.approx(1.0)
    assert config.evaluation.best_ckpt_metrics == ["mean/ccc"]


def test_bp_common_input_pipeline_keeps_its_training_recipe():
    config = _load_config(BP_CONFIG)
    grid = {source.name: source.params for source in config.data.input_sources}["grid"]

    assert config.optimizer.loss == ["kl", "ccc", "hex_mean_pairwise_rank", "hex_top10_pairwise_rank"]
    assert config.optimizer.loss_weights == {
        "kl": 0.45,
        "ccc": 0.45,
        "hex_mean_pairwise_rank": 0.05,
        "hex_top10_pairwise_rank": 0.05,
    }
    assert grid.out_norm == "min_max"
    assert config.evaluation.bp_nodata_as_zero is True
    assert config.evaluation.prediction_support_policy == "input"


def test_fi_ros_common_input_pipeline_use_log_standard_regression_recipe():
    for path in (FI_CONFIG, ROS_CONFIG):
        config = _load_config(path)
        grid = {source.name: source.params for source in config.data.input_sources}["grid"]

        assert grid.out_norm == "log_standard"
        assert config.optimizer.loss == ["huber", "raw_pearson"]
        assert config.evaluation.robust_plot_percentile == 99.0


def test_legacy_target_config_resolves_to_one_target():
    grid = GridParams(
        feature_names_list=["ignition_grid"],
        target_name="fire_intensity",
        out_norm="log_standard",
        target_log_mean=2.0,
        target_log_std=0.5,
    )

    assert grid.resolved_targets() == [
        TargetConfig(name="fi", out_norm="log_standard", log_mean=2.0, log_std=0.5),
    ]
    assert grid.target_config("fi").name == "fi"


def test_multi_target_config_keeps_order_and_per_target_normalization():
    grid = GridParams(
        feature_names_list=["ignition_grid"],
        targets=[
            TargetConfig(name="bp", out_norm="min_max"),
            TargetConfig(name="fi", out_norm="log_standard"),
            TargetConfig(name="ros", out_norm="log_standard"),
        ],
    )

    assert [target.name for target in grid.resolved_targets()] == ["bp", "fi", "ros"]
    assert grid.target_config("bp").out_norm == "min_max"
    assert grid.target_config("fi").out_norm == "log_standard"


def test_grid_params_rejects_mixed_or_duplicate_target_config():
    with pytest.raises(ValidationError, match="non-default legacy target fields"):
        GridParams(
            feature_names_list=["ignition_grid"],
            target_name="fi",
            targets=[TargetConfig(name="bp", out_norm="min_max")],
        )

    with pytest.raises(ValidationError, match="duplicate"):
        GridParams(
            feature_names_list=["ignition_grid"],
            targets=[
                TargetConfig(name="bp", out_norm="min_max"),
                TargetConfig(name="burn_probability", out_norm="min_max"),
            ],
        )


def test_optimizer_supports_legacy_or_per_target_losses():
    legacy = OptimizerConfig(loss="mse")
    assert legacy.loss == "mse"
    assert legacy.target_losses == {}

    multi_target = OptimizerConfig(
        target_losses={
            "bp": TargetLossConfig(loss=["kl", "ccc"], loss_weights={"kl": 0.5, "ccc": 0.5}, task_weight=0.5),
            "fi": TargetLossConfig(loss=["huber", "raw_pearson"], task_weight=0.25),
            "ros": TargetLossConfig(loss=["huber", "raw_pearson"], task_weight=0.25),
        }
    )
    assert multi_target.loss is None
    assert multi_target.target_losses["bp"].task_weight == 0.5

    with pytest.raises(ValidationError, match="either legacy loss"):
        OptimizerConfig(loss="mse", target_losses={"bp": TargetLossConfig(loss="kl")})


def test_multi_output_common_input_pipeline_config():
    config = _load_config(MULTI_OUTPUT_CONFIG)
    grid = {source.name: source.params for source in config.data.input_sources}["grid"]

    assert config.model.num_classes == 3
    assert config.model.output_head == "bp_behavior"
    assert [target.name for target in grid.resolved_targets()] == ["bp", "fi", "ros"]
    assert [target.out_norm for target in grid.resolved_targets()] == ["min_max", "log_standard", "log_standard"]
    assert set(config.optimizer.target_losses) == {"bp", "fi", "ros"}
    assert sum(target.task_weight for target in config.optimizer.target_losses.values()) == pytest.approx(1.0)
    assert config.data.include_patch_metadata is True


def test_multi_output_config_round_trips_through_checkpoint_dump():
    config = _load_config(MULTI_OUTPUT_CONFIG)
    dumped_config = config.model_dump()
    dumped_grid_params = next(source["params"] for source in dumped_config["data"]["input_sources"] if source["name"] == "grid")

    restored_grid = GridParams(**dumped_grid_params)

    assert [target.name for target in restored_grid.resolved_targets()] == ["bp", "fi", "ros"]


@pytest.mark.parametrize("config_path", [BP_CONFIG, FI_CONFIG, ROS_CONFIG])
def test_legacy_config_round_trips_through_checkpoint_dump(config_path):
    config = _load_config(config_path)

    restored_config = Config(**config.model_dump())

    assert restored_config == config


def test_multi_output_config_requires_target_specific_losses():
    with MULTI_OUTPUT_CONFIG.open() as f:
        raw_config = yaml.safe_load(f)
    raw_config["optimizer"] = {"name": "AdamW", "lr": 7.0e-4, "loss": "kl"}

    with pytest.raises(ValidationError, match="target_losses is required"):
        Config(**raw_config)


def test_multi_output_config_rejects_unnamespaced_checkpoint_metric():
    with MULTI_OUTPUT_CONFIG.open() as f:
        raw_config = yaml.safe_load(f)
    raw_config["evaluation"]["best_ckpt_metrics"] = ["spearman"]

    with pytest.raises(ValidationError, match="namespaced metric keys"):
        Config(**raw_config)


def test_multi_output_config_accepts_stitched_hex_checkpoint_metric():
    with MULTI_OUTPUT_CONFIG.open() as f:
        raw_config = yaml.safe_load(f)
    raw_config["evaluation"]["best_ckpt_metrics"] = ["hex/mean/ccc"]

    config = Config(**raw_config)

    assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]


def test_multi_output_config_rejects_mean_ccc_when_ccc_is_not_computed():
    with MULTI_OUTPUT_CONFIG.open() as f:
        raw_config = yaml.safe_load(f)
    raw_config["metrics"] = [metric for metric in raw_config["metrics"] if metric != "ccc"]
    raw_config["evaluation"]["best_ckpt_metrics"] = ["mean/ccc"]

    with pytest.raises(ValidationError, match="namespaced metric keys"):
        Config(**raw_config)


def test_multi_output_config_rejects_scalar_loss_component_checkpoint_metric():
    with MULTI_OUTPUT_CONFIG.open() as f:
        raw_config = yaml.safe_load(f)
    raw_config["optimizer"]["target_losses"]["bp"] = {
        "loss": "kl",
        "task_weight": 0.5,
    }
    raw_config["evaluation"]["best_ckpt_metrics"] = ["loss_bp/kl"]

    with pytest.raises(ValidationError, match="namespaced metric keys"):
        Config(**raw_config)


def test_hazard_eval_config_parses_and_references_model_config():
    config = _load_hazard_eval_config(HAZARD_EVAL_CONFIG)

    assert config.model.config_path == str(HAZARD_MODEL_CONFIG)
    assert config.root_dir.endswith("data_samples_v4")
    assert config.test_split == "test_indices.csv"
    assert config.mask_scope == "actual"
    assert config.stitch_mode == "mean"


def test_hazard_eval_config_defaults():
    config = HazardEvalConfig(
        root_dir="root",
        raw_data_dir="raw",
        model=HazardModelEntry(config_path="model.yaml"),
    )

    assert len(config.bin_thresholds) == 12
    assert config.scale_denominator_source == "all_raw_ground_truth"
    assert config.self_normalized_prediction is False
    assert config.fi_cap == DEFAULT_FI_CAP
    assert config.scale_to == DEFAULT_SCALE_TO


def test_hazard_eval_config_rejects_non_positive_fi_cap():
    with pytest.raises(ValidationError):
        HazardEvalConfig(
            root_dir="root",
            raw_data_dir="raw",
            model={"config_path": "model.yaml"},
            fi_cap=0.0,
        )


def test_hazard_eval_config_rejects_non_positive_scale_denominator():
    with pytest.raises(ValidationError):
        HazardEvalConfig(
            root_dir="root",
            raw_data_dir="raw",
            model={"config_path": "model.yaml"},
            scale_denominator=-1.0,
        )


def test_hazard_eval_config_rejects_non_increasing_bin_thresholds():
    with pytest.raises(ValidationError):
        HazardEvalConfig(
            root_dir="root",
            raw_data_dir="raw",
            model={"config_path": "model.yaml"},
            bin_thresholds=[0.1, 0.1, 0.2],
        )


def test_hazard_eval_config_rejects_non_finite_bin_thresholds():
    with pytest.raises(ValidationError):
        HazardEvalConfig(
            root_dir="root",
            raw_data_dir="raw",
            model={"config_path": "model.yaml"},
            bin_thresholds=[0.1, float("nan"), 0.2],
        )


def test_hazard_eval_config_reference_file_requires_denominator_or_path():
    kwargs = {
        "root_dir": "root",
        "raw_data_dir": "raw",
        "model": {"config_path": "model.yaml"},
        "scale_denominator_source": "reference_file",
    }

    with pytest.raises(ValidationError):
        HazardEvalConfig(**kwargs)

    config = HazardEvalConfig(**kwargs, scale_denominator=42.0)
    assert config.scale_denominator == 42.0


def test_hazard_eval_config_parses_uncapped_fi_cap():
    config = HazardEvalConfig(
        root_dir="root",
        raw_data_dir="raw",
        model={"config_path": "model.yaml"},
        fi_cap=None,
    )

    assert config.fi_cap is None


def test_hazard_eval_config_reference_file_with_path_parses():
    config = HazardEvalConfig(
        root_dir="root",
        raw_data_dir="raw",
        model={"config_path": "model.yaml"},
        scale_denominator_source="reference_file",
        reference_denominator_path="denominator.tif",
    )

    assert config.scale_denominator_source == "reference_file"
    assert config.reference_denominator_path == "denominator.tif"


def test_apply_run_id_overrides_derives_seed_save_dir_and_experiment_name():
    config = _load_config(BP_CONFIG)
    config.save_dir = "experiments/my_run"
    config.logger.experiment_name = "my_experiment"

    run_seed = apply_run_id_overrides(config, run_id=2)

    assert run_seed == SEEDS[2]
    assert config.seed == SEEDS[2]
    assert config.save_dir == f"experiments/my_run/seed_{SEEDS[2]}"
    assert config.logger.experiment_name == f"my_experiment_seed{SEEDS[2]}"


def test_apply_run_id_overrides_skips_empty_experiment_name():
    config = _load_config(BP_CONFIG)
    config.logger.experiment_name = ""

    apply_run_id_overrides(config, run_id=0)

    assert config.logger.experiment_name == ""


@pytest.mark.parametrize("run_id", [-1, len(SEEDS)])
def test_apply_run_id_overrides_rejects_out_of_range_run_id(run_id):
    config = _load_config(BP_CONFIG)

    with pytest.raises(ValueError, match="run_id must be between"):
        apply_run_id_overrides(config, run_id=run_id)
