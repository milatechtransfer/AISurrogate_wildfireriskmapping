import math

import pytest
import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.mechanistic_hybrid_v23 import MechanisticHybridV23

SPATIAL_NAMES = [
    "grid/ignition_grid_human",
    "grid/ignition_grid_lightning",
    "grid/elevation_grid",
    "spatialized_weather/InitialSpreadIndex",
    "spatialized_weather/wind_x",
    "spatialized_weather/wind_y",
    "spatialized_fire_size/LOG_SIZE_HA_q10",
    "spatialized_fire_size/LOG_SIZE_HA_q50",
    "spatialized_fire_size/LOG_SIZE_HA_q90",
    "spatialized_fire_size/missing_firezone_mask",
    "spatialized_ignition_count/NORM_LOG1P_IGNITION_COUNT_MEAN",
    "spatialized_ignition_count/NORM_IGNITION_COUNT_CV",
]


def _config() -> ModelConfig:
    return ModelConfig(
        architecture="mechanistic_hybrid_v23",
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
        propagation_fire_size_neural_mean=2.0,
        propagation_fire_size_neural_std=0.5,
        propagation_initial_additive_bp_hazard=1e-4,
        propagation_max_additive_bp_hazard=-math.log(0.8),
        propagation_additive_bp_hazard_weight=0.1,
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


def _build() -> MechanisticHybridV23:
    model = build_model(
        model_config=_config(),
        spatial_input_channels=len(SPATIAL_NAMES),
        auxiliary_input_dims={"fuel_curve": 6},
        fuel_curve_mean=torch.tensor([1.0, 3.0]),
        fuel_curve_std=torch.tensor([0.5, 2.0]),
        target_names=["bp", "fi", "ros"],
    )
    assert isinstance(model, MechanisticHybridV23)
    return model


def _inputs(*, fire_size_log10_ha: float = 2.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.zeros(1, len(SPATIAL_NAMES), 32, 32)
    x[:, 0, 12, 12] = 1_000_000.0
    x[:, 3] = 10.0
    x[:, 6:9] = fire_size_log10_ha
    x[:, 10] = 0.5
    x[:, 11] = 0.5
    ros_curve = torch.tensor([4.0, 8.0, 12.0]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    hfi_curve = torch.tensor([100.0, 200.0, 300.0]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    return x, {"fuel_curve": torch.cat([ros_curve, hfi_curve], dim=1)}


def test_factory_builds_v23_with_direct_log_fire_size_and_missing_mask() -> None:
    model = _build()

    assert resolve_model_architecture(_config()) == "mechanistic_hybrid_v23"
    assert model.scenario_indices == [6, 7, 8]
    assert model.fire_size_missing_index == 9
    assert torch.allclose(model.quantile_weights, torch.tensor([0.3, 0.4, 0.3]))


def test_direct_log_fire_size_is_only_standardized_for_neural_encoder() -> None:
    model = _build()
    x, _ = _inputs(fire_size_log10_ha=2.5)
    x[:, model.fire_size_missing_index] = 1.0

    neural_x = model._neural_encoder_input(x)
    physical = model._physical_fire_size_log10_ha(x[:, model.scenario_indices])

    assert torch.allclose(neural_x[:, model.scenario_indices], torch.ones_like(neural_x[:, model.scenario_indices]))
    assert torch.allclose(physical, torch.full_like(physical, 2.5))
    assert torch.all(neural_x[:, model.fire_size_missing_index] == 1.0)


def test_additive_hazard_can_repair_zero_physical_bp_but_not_nonburnable_pixels() -> None:
    model = _build()
    decoded = torch.zeros(1, 8, 2, 2)
    physical_bp = torch.zeros(1, 1, 2, 2)
    burnability = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    probability, regularization, diagnostics = model._calibrate_bp(
        physical_bp=physical_bp,
        decoded=decoded,
        fine_burnability=burnability,
    )

    assert torch.all(probability[burnability.bool()] > 0.0)
    assert torch.all(probability[~burnability.bool()] == 0.0)
    assert regularization.item() > 0.0
    assert diagnostics["max_additive_bp_hazard"].item() <= model.max_additive_bp_hazard


def test_additive_hazard_is_bounded_and_regularized_with_gradients() -> None:
    model = _build()
    assert model.bp_additive_hazard.bias is not None
    with torch.no_grad():
        model.bp_additive_hazard.bias.fill_(100.0)
    decoded = torch.randn(1, 8, 4, 4, requires_grad=True)
    burnability = torch.ones(1, 1, 4, 4)

    additive_hazard = model._additive_hazard(decoded, burnability)
    assert torch.all(additive_hazard >= 0.0)
    assert torch.all(additive_hazard <= model.max_additive_bp_hazard)

    with torch.no_grad():
        model.bp_additive_hazard.bias.zero_()
    _, regularization, _ = model._calibrate_bp(
        physical_bp=torch.zeros_like(burnability),
        decoded=decoded,
        fine_burnability=burnability,
    )
    regularization.backward()

    assert model.bp_additive_hazard.weight.grad is not None
    assert torch.isfinite(model.bp_additive_hazard.weight.grad).all()
    assert model.bp_additive_hazard.weight.grad.abs().sum() > 0.0


def test_forward_backward_is_finite_and_additive_head_trains() -> None:
    model = _build()
    x, auxiliary = _inputs()

    output = model(x, auxiliary)
    regularization = model.pop_regularization_loss()
    coarse_bp = model.pop_coarse_bp_probability()
    assert regularization is not None
    assert coarse_bp is not None
    loss = output.square().mean() + regularization
    loss.backward()

    assert output.shape == (1, 3, 32, 32)
    assert coarse_bp.shape == (1, 1, 4, 4)
    assert torch.all((coarse_bp >= 0.0) & (coarse_bp <= 1.0))
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert model.pop_coarse_bp_probability() is None
    assert model.bp_additive_hazard.weight.grad is not None
    assert torch.isfinite(model.bp_additive_hazard.weight.grad).all()
    assert model.diagnostic_metrics()["mean_additive_bp_hazard"] == pytest.approx(1e-4, rel=1e-4)


def test_missing_fire_size_mask_is_required() -> None:
    config = _config()
    config.spatial_input_names = [name for name in SPATIAL_NAMES if not name.endswith("missing_firezone_mask")]

    with pytest.raises(ValueError, match="missing_firezone_mask"):
        build_model(
            model_config=config,
            spatial_input_channels=len(config.spatial_input_names),
            auxiliary_input_dims={"fuel_curve": 6},
            target_names=["bp", "fi", "ros"],
        )
