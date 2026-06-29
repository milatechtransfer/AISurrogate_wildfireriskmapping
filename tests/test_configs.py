from pathlib import Path

import yaml

from src.config import Config, GridParams, SpatializedTabularParams
from src.utils import AVAILABLE_METRICS, build_single_loss

BP_CONFIG = Path("configs/bp_common_input_pipeline.yaml")
FI_CONFIG = Path("configs/fi_common_input_pipeline.yaml")
ROS_CONFIG = Path("configs/ros_common_input_pipeline.yaml")
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


def test_common_input_pipeline_configs_share_unified_input_pipeline():
    for path in COMMON_INPUT_PIPELINE_CONFIGS:
        config = _load_config(path)
        sources = {source.name: source.params for source in config.data.input_sources}

        # Spatial-only model with patch-local coordconv.
        assert config.model.input_branches == ["spatial"]
        assert config.model.use_coordconv is True

        # Dataset is the leakage-fixed, aggregated-ignition data_samples_v2.
        assert config.data.root_dir.endswith("data_samples_v2")

        # Grid: aggregated 2-channel ignition + terrain derivatives.
        assert isinstance(sources["grid"], GridParams)
        assert sources["grid"].feature_names_list[:2] == ["ignition_grid_human", "ignition_grid_lightning"]
        assert sources["grid"].terrain_derivatives == ["slope", "aspect_sin", "aspect_cos"]

        # Spatialized weather + fire-size. The weather LUT covers every firezone, so its
        # missing-firezone mask is dropped; the fire-size table lacks some zones, so it is kept.
        assert isinstance(sources["spatialized_weather"], SpatializedTabularParams)
        assert isinstance(sources["spatialized_fire_size"], SpatializedTabularParams)
        assert sources["spatialized_weather"].include_missing_firezone_mask is False
        assert sources["spatialized_fire_size"].include_missing_firezone_mask is True

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
