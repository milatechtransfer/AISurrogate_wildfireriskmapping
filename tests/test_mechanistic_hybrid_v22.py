import math

import pytest
import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.mechanistic_hybrid_v22 import MechanisticHybridV22

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
    "spatialized_ignition_count/NORM_LOG1P_IGNITION_COUNT_MEAN",
    "spatialized_ignition_count/NORM_IGNITION_COUNT_CV",
]


def _config() -> ModelConfig:
    return ModelConfig(
        architecture="mechanistic_hybrid_v22",
        num_classes=3,
        output_head="bp_behavior",
        spatial_input_names=SPATIAL_NAMES,
        auxiliary_embed_dims={"fuel_curve": 4},
        propagation_base_channels=8,
        propagation_downsample_factor=8,
        propagation_steps=2,
        propagation_cell_size_m=100.0,
        propagation_budget_temperature_hours=0.25,
        propagation_isi_bins=[0.0, 10.0, 20.0],
        propagation_isi_mean=0.0,
        propagation_isi_std=1.0,
        propagation_wind_x_mean=0.0,
        propagation_wind_x_std=1.0,
        propagation_wind_y_mean=0.0,
        propagation_wind_y_std=1.0,
        propagation_wind_anisotropy=math.log(4.0),
        propagation_wind_anisotropy_min=0.0,
        propagation_wind_anisotropy_max=math.log(8.0),
        propagation_elevation_min_m=0.0,
        propagation_elevation_max_m=1_000.0,
        propagation_min_ros_m_per_min=0.01,
        propagation_speed_correction_log_limit=math.log(2.0),
        propagation_max_local_log_calibration=math.log(1.5),
        propagation_fire_size_log_min=0.0,
        propagation_fire_size_log_max=4.0,
        propagation_count_log_mean_min=0.0,
        propagation_count_log_mean_max=math.log1p(20.0),
        propagation_count_cv_min=0.5,
        propagation_count_cv_max=1.5,
        interpretable_initial_per_fire_reach_scale=1.0,
        interpretable_min_per_fire_reach_scale=0.5,
        interpretable_max_per_fire_reach_scale=2.0,
        interpretable_source_grid_size=2,
        interpretable_fi_log_mean=0.0,
        interpretable_fi_log_std=1.0,
        interpretable_ros_log_mean=0.0,
        interpretable_ros_log_std=1.0,
    )


def _build() -> MechanisticHybridV22:
    model = build_model(
        model_config=_config(),
        spatial_input_channels=len(SPATIAL_NAMES),
        auxiliary_input_dims={"fuel_curve": 6},
        fuel_curve_mean=torch.tensor([1.0, 3.0]),
        fuel_curve_std=torch.tensor([0.5, 2.0]),
        target_names=["bp", "fi", "ros"],
    )
    assert isinstance(model, MechanisticHybridV22)
    return model


def _inputs(*, count: float = 0.5, fire_size: float = 0.5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.zeros(1, len(SPATIAL_NAMES), 32, 32)
    x[:, 0, 12, 12] = 1_000_000.0
    x[:, 3] = 10.0
    x[:, 6:9] = fire_size
    x[:, 9] = count
    x[:, 10] = 0.5
    ros_curve = torch.tensor([4.0, 8.0, 12.0]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    hfi_curve = torch.tensor([100.0, 200.0, 300.0]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    return x, {"fuel_curve": torch.cat([ros_curve, hfi_curve], dim=1)}


def test_factory_builds_scratch_hybrid_with_q3_prior() -> None:
    model = _build()

    assert resolve_model_architecture(_config()) == "mechanistic_hybrid_v22"
    assert torch.allclose(model.quantile_weights, torch.tensor([0.3, 0.4, 0.3]))
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any("ignition_scale" in name or "bp_scale" in name for name in parameter_names)


def test_forward_backward_is_finite_and_regularization_is_consumed_once() -> None:
    model = _build()
    x, auxiliary = _inputs()

    output = model(x, auxiliary)
    regularization = model.pop_regularization_loss()
    assert regularization is not None
    loss = output.square().mean() + regularization
    loss.backward()

    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert model.pop_regularization_loss() is None
    named_parameters = dict(model.named_parameters())
    for name in (
        "raw_quantile_logits",
        "raw_per_fire_reach_scale",
        "raw_wind_anisotropy",
        "speed_correction.3.weight",
        "bp_local_calibration.weight",
        "behavior_head.weight",
    ):
        parameter = named_parameters[name]
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_larger_count_and_fire_size_raise_initial_physical_bp() -> None:
    model = _build().eval()
    low_count_x, auxiliary = _inputs(count=0.1, fire_size=0.2)
    high_count_x, _ = _inputs(count=0.9, fire_size=0.2)
    large_fire_x, _ = _inputs(count=0.1, fire_size=0.9)

    with torch.no_grad():
        low = torch.sigmoid(model(low_count_x, auxiliary)[:, 0])
        high_count = torch.sigmoid(model(high_count_x, auxiliary)[:, 0])
        large_fire = torch.sigmoid(model(large_fire_x, auxiliary)[:, 0])

    assert torch.all(high_count >= low - 1e-7)
    assert high_count.sum() > low.sum()
    assert torch.all(large_fire >= low - 1e-7)
    assert large_fire.sum() > low.sum()


def test_source_cohorts_preserve_probability_mass() -> None:
    model = _build()
    location = torch.full((1, 1, 4, 4), 1.0 / 16.0)

    indices, weights = model._source_cohorts(location)

    assert indices.shape == (1, 4)
    assert len(set(indices[0].tolist())) == 4
    assert torch.allclose(weights, torch.full_like(weights, 0.25))
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))


def test_small_fire_area_correction_only_scales_down() -> None:
    model = _build()
    reach = torch.full((1, 2, 3, 4, 4), 0.75)
    target_area = torch.full((1, 2, 3), model.coarse_cell_area_ha)

    corrected = model._apply_area_bias_correction(reach, target_area)
    corrected_area = corrected.sum(dim=(-2, -1)) * model.coarse_cell_area_ha

    assert torch.all(corrected <= reach)
    assert torch.all(corrected_area <= target_area + 1e-5)


def test_wind_and_edge_grade_change_incoming_travel_time_directionally() -> None:
    model = _build()
    elevation = torch.zeros(1, 1, 5, 5)
    elevation[..., 2, 3] = 100.0
    base_speed = torch.full((1, 1, 5, 5), 10.0)
    wind_factor = torch.ones(1, 8, 5, 5)
    wind_factor[:, 2] = 2.0
    wind_factor[:, 6] = 0.5
    correction = torch.zeros(1, 8, 5, 5)
    burnability = torch.ones(1, 1, 5, 5)

    hours = model._incoming_edge_hours(
        elevation_m=elevation,
        base_speed=base_speed,
        wind_factor=wind_factor,
        log_speed_correction=correction,
        burnability=burnability,
    )

    assert hours[0, 2, 2, 3] < hours[0, 6, 2, 1]
    flat_elevation = torch.zeros_like(elevation)
    flat_hours = model._incoming_edge_hours(
        elevation_m=flat_elevation,
        base_speed=base_speed,
        wind_factor=wind_factor,
        log_speed_correction=correction,
        burnability=burnability,
    )
    assert hours[0, 2, 2, 3] < flat_hours[0, 2, 2, 3]


def test_zero_physical_probability_remains_zero_after_hazard_correction() -> None:
    physical = torch.tensor([[[[0.0, 0.2]]]])
    delta = torch.tensor([[[[5.0, 0.0]]]])

    corrected = MechanisticHybridV22._bp_hazard_correction(physical, delta)

    assert corrected[..., 0].item() == 0.0
    assert corrected[..., 1].item() == pytest.approx(0.2)


def test_bilinear_bp_upsampling_does_not_leak_onto_nonburnable_fine_pixels() -> None:
    model = _build().eval()
    x, auxiliary = _inputs()
    auxiliary["fuel_curve"][:, :, 12, 13] = 0.0

    with torch.no_grad():
        bp = torch.sigmoid(model(x, auxiliary)[:, 0])

    assert bp[0, 12, 13].item() <= 2.0 * model.bp_logit_eps
    assert bp[0, 12, 12].item() > bp[0, 12, 13].item()
