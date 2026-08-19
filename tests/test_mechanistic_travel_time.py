import pytest
import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.mechanistic_travel_time import (
    DifferentiableTravelTimePropagation,
    MechanisticTravelTimeUNet,
)

SPATIAL_NAMES = [
    "grid/ignition_grid_human",
    "grid/ignition_grid_lightning",
    "grid/elevation_grid",
    "spatialized_weather/InitialSpreadIndex",
    "spatialized_weather/wind_x",
    "spatialized_weather/wind_y",
    "spatialized_spread_opportunity/NORM_TOTAL_BURN_HOURS_Q10",
    "spatialized_spread_opportunity/NORM_TOTAL_BURN_HOURS_Q50",
    "spatialized_spread_opportunity/NORM_TOTAL_BURN_HOURS_Q90",
]


def _model_config(architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        num_classes=3,
        output_head="bp_behavior",
        hidden_features=[8, 16],
        spatial_input_names=SPATIAL_NAMES,
        auxiliary_embed_dims={"fuel_curve": 2},
        propagation_base_channels=8,
        propagation_downsample_factor=16,
        propagation_steps=2,
        propagation_budget_min_hours=1.0,
        propagation_budget_max_hours=20.0,
        propagation_isi_bins=[1e-6, 5.0, 10.0, 15.0],
        propagation_isi_mean=5.0,
        propagation_isi_std=2.0,
        propagation_elevation_min_m=0.0,
        propagation_elevation_max_m=1000.0,
    )


def _build(architecture: str):
    return build_model(
        model_config=_model_config(architecture),
        spatial_input_channels=len(SPATIAL_NAMES),
        auxiliary_input_dims={"fuel_curve": 4},
        fuel_curve_mean=torch.zeros(1),
        fuel_curve_std=torch.ones(1),
        target_names=["bp", "fi", "ros"],
    )


def test_travel_time_uses_physical_edge_hours():
    propagation = DifferentiableTravelTimePropagation(
        steps=1,
        coarse_cell_size_m=1200.0,
        budget_temperature_hours=1.0,
        quantile_levels=[0.5],
        min_ros_m_per_min=0.01,
    )
    seed = torch.zeros(1, 1, 5, 5)
    seed[..., 2, 2] = 0.5
    speed = torch.full((1, 8, 5, 5), 10.0)
    budget = torch.full((1, 1, 5, 5), 3.0)
    burnability = torch.ones_like(seed)

    _, quantile_reach = propagation(seed, speed, budget, burnability)

    expected_cardinal = torch.sigmoid(torch.tensor(1.0))
    expected_diagonal = torch.sigmoid(torch.tensor(3.0 - 2.0 * 2.0**0.5))
    assert quantile_reach[0, 0, 2, 3] == pytest.approx(expected_cardinal.item())
    assert quantile_reach[0, 0, 1, 3] == pytest.approx(expected_diagonal.item())
    assert propagation.diagnostic_metrics()["cap_hit_rate"] > 0.0


def test_longer_source_budget_reaches_farther():
    propagation = DifferentiableTravelTimePropagation(
        steps=2,
        coarse_cell_size_m=600.0,
        budget_temperature_hours=1.0,
        quantile_levels=[0.5],
        min_ros_m_per_min=0.01,
    )
    seed = torch.zeros(1, 1, 7, 7)
    seed[..., 3, 3] = 0.1
    speed = torch.full((1, 8, 7, 7), 10.0)
    burnability = torch.ones_like(seed)

    short, _ = propagation(seed, speed, torch.ones(1, 1, 7, 7), burnability)
    long, _ = propagation(seed, speed, torch.full((1, 1, 7, 7), 5.0), burnability)

    assert long.sum() > short.sum()


def test_zero_initialized_v4_exactly_matches_pretrained_unet():
    baseline = _build("unet")
    v4 = _build("mechanistic_travel_time_v4")
    assert isinstance(v4, MechanisticTravelTimeUNet)
    report = v4.load_pretrained_unet_state_dict(
        baseline.state_dict(),
        {"model": {"spatial_input_names": SPATIAL_NAMES}},
    )
    baseline.eval()
    v4.eval()
    spatial = torch.randn(1, len(SPATIAL_NAMES), 32, 32)
    spatial[:, 0:2] = spatial[:, 0:2].sigmoid()
    spatial[:, 2] = spatial[:, 2].sigmoid()
    spatial[:, 6:] = spatial[:, 6:].sigmoid()
    fuel_curve = torch.rand(1, 4, 32, 32) * 10.0

    with torch.no_grad():
        baseline_output = baseline(spatial, {"fuel_curve": fuel_curve})
        v4_output = v4(spatial, {"fuel_curve": fuel_curve})

    assert report["loaded"] == len(baseline.state_dict())
    assert torch.equal(v4_output, baseline_output)
    assert resolve_model_architecture(_model_config("mechanistic_travel_time_v4")) == "mechanistic_travel_time_v4"


def test_v4_residual_changes_only_bp():
    baseline = _build("unet")
    v4 = _build("mechanistic_travel_time_v4")
    assert isinstance(v4, MechanisticTravelTimeUNet)
    v4.load_pretrained_unet_state_dict(baseline.state_dict())
    with torch.no_grad():
        v4.bp_residual_head.bias.fill_(0.5)
    baseline.eval()
    v4.eval()
    spatial = torch.zeros(1, len(SPATIAL_NAMES), 32, 32)
    spatial[:, 0:2] = 0.2
    spatial[:, 2] = 0.5
    spatial[:, 6:] = 0.25
    fuel_curve = torch.ones(1, 4, 32, 32)

    with torch.no_grad():
        baseline_output = baseline(spatial, {"fuel_curve": fuel_curve})
        v4_output = v4(spatial, {"fuel_curve": fuel_curve})

    assert not torch.equal(v4_output[:, 0], baseline_output[:, 0])
    assert torch.equal(v4_output[:, 1:], baseline_output[:, 1:])


def test_meteorological_west_wind_produces_eastward_fastest_spread():
    v4 = _build("mechanistic_travel_time_v4")
    assert isinstance(v4, MechanisticTravelTimeUNet)
    spatial = torch.zeros(1, len(SPATIAL_NAMES), 32, 32)
    spatial[:, v4.wind_x_index] = -10.0
    fuel_curve = torch.full((1, 4, 32, 32), 10.0)
    decoded = torch.zeros(1, 8, 32, 32)
    normalized_budget = torch.zeros(1, 3, 2, 2)
    coarse_burnability = torch.ones(1, 1, 2, 2)
    coarse_ignition = torch.ones(1, 1, 2, 2)

    speed, base_speed, _, _, _ = v4._directional_speed(
        x=spatial,
        fuel_curve=fuel_curve,
        decoded_features=decoded,
        normalized_budget=normalized_budget,
        coarse_burnability=coarse_burnability,
        coarse_ignition=coarse_ignition,
    )

    east_index = 2
    west_index = 6
    assert torch.allclose(speed[:, east_index], base_speed[:, 0])
    assert torch.all(speed[:, east_index] > speed[:, west_index])


def test_pretrained_freeze_does_not_freeze_new_branch():
    v4 = _build("mechanistic_travel_time_v4")
    assert isinstance(v4, MechanisticTravelTimeUNet)

    v4.set_pretrained_frozen(True)

    assert not any(parameter.requires_grad for parameter in v4.encoder.parameters())
    assert all(parameter.requires_grad for parameter in v4.bp_residual_head.parameters())

    v4.set_pretrained_frozen(False)
    assert all(parameter.requires_grad for parameter in v4.encoder.parameters())
