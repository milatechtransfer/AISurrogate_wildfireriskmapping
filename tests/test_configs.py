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
from src.datasets.postprocessing.hazard import DEFAULT_FI_CAP, DEFAULT_SCALE_TO
from src.utils import AVAILABLE_METRICS, build_single_loss

BP_CONFIG = Path("configs/bp_spatial_weather.yaml")
FI_CONFIG = Path("configs/fi_spatial_weather.yaml")
ROS_CONFIG = Path("configs/ros_spatial_weather.yaml")
MULTI_OUTPUT_CONFIG = Path("configs/multi_output_spatial_weather.yaml")
HAZARD_EVAL_CONFIG = Path("configs/hazard_eval_spatial_weather.yaml")
HAZARD_MODEL_CONFIG = Path("configs/multi_output_spatial_weather.yaml")
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

        # Dataset is the leakage-fixed, aggregated-ignition data samples.
        assert config.data.root_dir.endswith("data_samples")

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
    assert config.root_dir.endswith("data_samples")
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
