import torch

from src.config import ModelConfig
from src.models.factory import build_model
from src.models.interpretable_mechanistic import (
    InterpretableMechanisticModel,
    SourceCohortTravelTimePropagation,
    count_distribution_burn_probability,
)
from src.models.utils import get_nbr_model_parameters

SPATIAL_NAMES = [
    "grid/ignition_grid_human",
    "grid/ignition_grid_lightning",
    "grid/elevation_grid",
    "spatialized_weather/InitialSpreadIndex",
    "spatialized_weather/wind_x",
    "spatialized_weather/wind_y",
    "spatialized_fire_size/NORM_LOG_SIZE_HA_q10",
    "spatialized_fire_size/NORM_LOG_SIZE_HA_q50",
    "spatialized_fire_size/NORM_LOG_SIZE_HA_q90",
]
SPATIAL_NAMES_V2 = [
    *SPATIAL_NAMES,
    "spatialized_ignition_count/NORM_LOG1P_IGNITION_COUNT_MEAN",
    "spatialized_ignition_count/NORM_IGNITION_COUNT_CV",
]


def _config(*, ignition_rate: float = 1.0) -> ModelConfig:
    return ModelConfig(
        architecture="interpretable_mechanistic",
        num_classes=3,
        output_head="bp_behavior",
        spatial_input_names=SPATIAL_NAMES,
        propagation_downsample_factor=4,
        propagation_steps=3,
        propagation_cell_size_m=100.0,
        propagation_budget_temperature_hours=1.0,
        propagation_scenario_mode="fire_size",
        propagation_fire_size_log_min=0.0,
        propagation_fire_size_log_max=4.0,
        propagation_isi_bins=[0.0, 10.0, 20.0],
        propagation_isi_mean=0.0,
        propagation_isi_std=1.0,
        propagation_wind_x_mean=0.0,
        propagation_wind_x_std=1.0,
        propagation_wind_y_mean=0.0,
        propagation_wind_y_std=1.0,
        propagation_elevation_min_m=0.0,
        propagation_elevation_max_m=1_000.0,
        propagation_min_ros_m_per_min=0.01,
        propagation_min_area_multiplier=0.5,
        propagation_max_area_multiplier=2.0,
        interpretable_initial_ignition_rate=ignition_rate,
        interpretable_min_ignition_rate=0.01,
        interpretable_max_ignition_rate=10.0,
        interpretable_initial_bp_hazard_scale=1.0,
        interpretable_min_bp_hazard_scale=0.01,
        interpretable_max_bp_hazard_scale=10.0,
        interpretable_fi_log_mean=0.0,
        interpretable_fi_log_std=1.0,
        interpretable_ros_log_mean=0.0,
        interpretable_ros_log_std=1.0,
    )


def _inputs(*, fire_size: float = 0.5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.zeros(1, len(SPATIAL_NAMES), 16, 16)
    x[:, 0, 8, 8] = 1_000_000.0
    x[:, 3] = 10.0
    x[:, 6:] = fire_size

    ros_curve = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1).expand(1, 3, 16, 16)
    hfi_curve = torch.tensor([10.0, 20.0, 30.0]).view(1, 3, 1, 1).expand(1, 3, 16, 16)
    return x, {"fuel_curve": torch.cat([ros_curve, hfi_curve], dim=1)}


def _build(*, ignition_rate: float = 1.0) -> InterpretableMechanisticModel:
    model = build_model(
        model_config=_config(ignition_rate=ignition_rate),
        spatial_input_channels=len(SPATIAL_NAMES),
        auxiliary_input_dims={"fuel_curve": 6},
        target_names=["bp", "fi", "ros"],
    )
    assert isinstance(model, InterpretableMechanisticModel)
    return model


def test_interpretable_model_uses_direct_physical_behavior_curves() -> None:
    model = _build()
    x, auxiliary = _inputs()

    with torch.no_grad():
        output = model(x, auxiliary)

    assert output.shape == (1, 3, 16, 16)
    assert torch.isfinite(output).all()
    assert torch.allclose(torch.expm1(output[:, 1]), torch.full_like(output[:, 1], 20.0), atol=1e-5)
    assert torch.allclose(torch.expm1(output[:, 2]), torch.full_like(output[:, 2], 2.0), atol=1e-5)
    assert get_nbr_model_parameters(model) == (5, 5)


def test_larger_fire_size_and_ignition_rate_cannot_reduce_bp() -> None:
    low_rate_model = _build(ignition_rate=0.2)
    high_rate_model = _build(ignition_rate=5.0)
    small_x, auxiliary = _inputs(fire_size=0.05)
    large_x, _ = _inputs(fire_size=0.95)

    with torch.no_grad():
        small_bp = torch.sigmoid(low_rate_model(small_x, auxiliary)[:, 0])
        large_bp = torch.sigmoid(low_rate_model(large_x, auxiliary)[:, 0])
        high_rate_bp = torch.sigmoid(high_rate_model(small_x, auxiliary)[:, 0])

    assert torch.all(large_bp >= small_bp - 1e-7)
    assert large_bp.sum() > small_bp.sum()
    assert torch.all(high_rate_bp >= small_bp - 1e-7)
    assert high_rate_bp.sum() > small_bp.sum()


def test_nonburnable_cells_cannot_receive_bp_and_larger_curves_raise_behavior() -> None:
    model = _build()
    x, auxiliary = _inputs()
    higher_curves = {"fuel_curve": auxiliary["fuel_curve"] * 2.0}
    blocked_curves = auxiliary["fuel_curve"].clone()
    blocked_curves[:, :, :4, :4] = 0.0

    with torch.no_grad():
        baseline = model(x, auxiliary)
        higher = model(x, higher_curves)
        blocked = model(x, {"fuel_curve": blocked_curves})

    assert torch.all(higher[:, 1] > baseline[:, 1])
    assert torch.all(higher[:, 2] > baseline[:, 2])
    assert torch.allclose(torch.sigmoid(blocked[:, 0, :4, :4]), torch.full((1, 4, 4), 1e-6), atol=1e-8)


def _v2_config() -> ModelConfig:
    config = _config()
    config.architecture = "interpretable_mechanistic_v2"
    config.spatial_input_names = SPATIAL_NAMES_V2
    config.propagation_count_log_mean_min = 0.0
    config.propagation_count_log_mean_max = torch.log1p(torch.tensor(20.0)).item()
    config.propagation_count_cv_min = 0.5
    config.propagation_count_cv_max = 1.5
    return config


def _build_v2() -> InterpretableMechanisticModel:
    model = build_model(
        model_config=_v2_config(),
        spatial_input_channels=len(SPATIAL_NAMES_V2),
        auxiliary_input_dims={"fuel_curve": 6},
        target_names=["bp", "fi", "ros"],
    )
    assert isinstance(model, InterpretableMechanisticModel)
    return model


def _v2_inputs(*, normalized_log_mean: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base, auxiliary = _inputs()
    count_channels = torch.empty(1, 2, 16, 16)
    count_channels[:, 0] = normalized_log_mean
    count_channels[:, 1] = 0.5
    return torch.cat([base, count_channels], dim=1), auxiliary


def test_count_distribution_aggregation_uses_mean_and_dispersion() -> None:
    reach = torch.full((1, 1, 2, 2), 0.05)
    zero_count = count_distribution_burn_probability(
        per_fire_reach=reach,
        mean_count=torch.zeros_like(reach),
        coefficient_of_variation=torch.ones_like(reach),
        per_fire_reach_scale=torch.tensor(1.0),
    )
    low_count = count_distribution_burn_probability(
        per_fire_reach=reach,
        mean_count=torch.full_like(reach, 2.0),
        coefficient_of_variation=torch.full_like(reach, 1.0),
        per_fire_reach_scale=torch.tensor(1.0),
    )
    high_count = count_distribution_burn_probability(
        per_fire_reach=reach,
        mean_count=torch.full_like(reach, 20.0),
        coefficient_of_variation=torch.full_like(reach, 1.0),
        per_fire_reach_scale=torch.tensor(1.0),
    )
    overdispersed = count_distribution_burn_probability(
        per_fire_reach=reach,
        mean_count=torch.full_like(reach, 20.0),
        coefficient_of_variation=torch.full_like(reach, 1.5),
        per_fire_reach_scale=torch.tensor(1.0),
    )

    assert torch.equal(zero_count, torch.zeros_like(zero_count))
    assert torch.all(high_count > low_count)
    assert torch.all(overdispersed < high_count)
    assert torch.isfinite(overdispersed).all()


def test_probabilistic_travel_time_sums_distinct_ignition_sources() -> None:
    propagation = SourceCohortTravelTimePropagation(
        source_grid_size=5,
        steps=1,
        coarse_cell_size_m=100.0,
        budget_temperature_hours=0.1,
        quantile_levels=[0.5],
        min_ros_m_per_min=0.01,
        scenario_mode="fire_size",
        min_area_multiplier=0.5,
        max_area_multiplier=2.0,
    )
    one_source = torch.zeros(1, 1, 5, 5)
    one_source[..., 2, 1] = 0.1
    two_sources = one_source.clone()
    two_sources[..., 2, 3] = 0.1
    speed = torch.full((1, 8, 5, 5), 100.0)
    fire_size = torch.full((1, 1, 5, 5), 4.0)
    burnability = torch.ones(1, 1, 5, 5)

    one_reach, _ = propagation(one_source, speed, fire_size, burnability)
    two_reach, _ = propagation(two_sources, speed, fire_size, burnability)

    assert two_reach[..., 2, 2] > one_reach[..., 2, 2]
    assert torch.allclose(two_reach[..., 2, 2], 2.0 * one_reach[..., 2, 2], rtol=1e-5, atol=1e-7)


def test_source_cohorts_preserve_context_mass_across_spatial_strata() -> None:
    propagation = SourceCohortTravelTimePropagation(
        source_grid_size=2,
        steps=1,
        coarse_cell_size_m=100.0,
        budget_temperature_hours=0.1,
        quantile_levels=[0.5],
        min_ros_m_per_min=0.01,
        scenario_mode="fire_size",
        min_area_multiplier=0.5,
        max_area_multiplier=2.0,
    )
    location = torch.full((1, 1, 4, 4), 1.0 / 16.0)

    indices, weights, total = propagation._source_cohorts(location)

    assert torch.allclose(total, torch.ones_like(total))
    assert torch.allclose(weights, torch.full_like(weights, 0.25))
    assert indices.shape == (1, 4)
    assert len(set(indices[0].tolist())) == 4


def test_v2_uses_hex_count_and_has_six_interpretable_parameters() -> None:
    model = _build_v2()
    low_count_x, auxiliary = _v2_inputs(normalized_log_mean=0.1)
    high_count_x, _ = _v2_inputs(normalized_log_mean=0.9)

    with torch.no_grad():
        low_count_bp = torch.sigmoid(model(low_count_x, auxiliary)[:, 0])
        high_count_bp = torch.sigmoid(model(high_count_x, auxiliary)[:, 0])

    assert torch.all(high_count_bp >= low_count_bp - 1e-7)
    assert high_count_bp.sum() > low_count_bp.sum()
    assert get_nbr_model_parameters(model) == (6, 6)


def test_v2_fire_size_barriers_behavior_and_gradients_preserve_physics() -> None:
    model = _build_v2()
    small_x, auxiliary = _v2_inputs(normalized_log_mean=0.5)
    large_x, _ = _v2_inputs(normalized_log_mean=0.5)
    small_x[:, 6:9] = 0.05
    large_x[:, 6:9] = 0.95
    blocked_curves = auxiliary["fuel_curve"].clone()
    blocked_curves[:, :, :4, :4] = 0.0

    with torch.no_grad():
        small_bp = torch.sigmoid(model(small_x, auxiliary)[:, 0])
        large_bp = torch.sigmoid(model(large_x, auxiliary)[:, 0])
        blocked = model(large_x, {"fuel_curve": blocked_curves})

    assert torch.all(large_bp >= small_bp - 1e-7)
    assert large_bp.sum() > small_bp.sum()
    assert torch.allclose(torch.sigmoid(blocked[:, 0, :4, :4]), torch.full((1, 4, 4), 1e-6), atol=1e-8)
    assert torch.equal(blocked[:, 1:, :4, :4], torch.zeros_like(blocked[:, 1:, :4, :4]))

    output = model(large_x, auxiliary)
    output.mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
