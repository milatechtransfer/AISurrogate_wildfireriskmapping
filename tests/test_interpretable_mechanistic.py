import torch

from src.config import ModelConfig
from src.models.factory import build_model
from src.models.interpretable_mechanistic import InterpretableMechanisticModel
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
