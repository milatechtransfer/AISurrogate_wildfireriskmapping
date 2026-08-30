import math

import pytest
import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.mechanistic_hybrid_v24 import MechanisticHybridV24

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
        architecture="mechanistic_hybrid_v24",
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
        propagation_initial_bp_hazard_residual=1e-4,
        propagation_max_bp_hazard_residual=2e-3,
        propagation_max_bp_hazard_attenuation=0.9,
        propagation_bp_hazard_residual_weight=1e-2,
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


def _build() -> MechanisticHybridV24:
    model = build_model(
        model_config=_config(),
        spatial_input_channels=len(SPATIAL_NAMES),
        auxiliary_input_dims={"fuel_curve": 6},
        fuel_curve_mean=torch.tensor([1.0, 3.0]),
        fuel_curve_std=torch.tensor([0.5, 2.0]),
        target_names=["bp", "fi", "ros"],
    )
    assert isinstance(model, MechanisticHybridV24)
    return model


def _inputs() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.zeros(1, len(SPATIAL_NAMES), 32, 32)
    x[:, 0, 12, 12] = 1_000_000.0
    x[:, 3] = 10.0
    x[:, 6:9] = 2.0
    x[:, 10] = 0.5
    x[:, 11] = 0.5
    ros_curve = torch.tensor([4.0, 8.0, 12.0]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    hfi_curve = torch.tensor([100.0, 200.0, 300.0]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    return x, {"fuel_curve": torch.cat([ros_curve, hfi_curve], dim=1)}


def _has_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())


def test_factory_builds_v24_with_independent_task_decoders() -> None:
    model = _build()

    assert resolve_model_architecture(_config()) == "mechanistic_hybrid_v24"
    assert model.behavior_mechanistic_fusion is not model.mechanistic_fusion
    assert model.behavior_decoder_blocks is not model.decoder_blocks
    assert not hasattr(model, "bp_additive_hazard")
    assert torch.allclose(model.quantile_weights, torch.tensor([0.3, 0.4, 0.3]))


def test_fi_ros_gradients_do_not_update_bp_decoder_or_physics() -> None:
    model = _build()
    x, auxiliary = _inputs()

    output = model(x, auxiliary)
    output[:, 1:].square().mean().backward()

    assert _has_nonzero_gradient(model.behavior_mechanistic_fusion)
    assert _has_nonzero_gradient(model.behavior_decoder_blocks)
    assert _has_nonzero_gradient(model.full_encoder)
    assert not _has_nonzero_gradient(model.mechanistic_fusion)
    assert not _has_nonzero_gradient(model.decoder_blocks)
    assert not _has_nonzero_gradient(model.speed_correction)
    assert model.raw_per_fire_reach_scale.grad is None or model.raw_per_fire_reach_scale.grad == 0.0
    assert model.raw_wind_anisotropy.grad is None or model.raw_wind_anisotropy.grad == 0.0
    assert model.raw_quantile_logits.grad is None or torch.equal(
        model.raw_quantile_logits.grad, torch.zeros_like(model.raw_quantile_logits)
    )


def test_bp_gradients_do_not_update_behavior_decoder() -> None:
    model = _build()
    with torch.no_grad():
        model.bp_local_calibration.weight.fill_(0.01)
        model.bp_hazard_residual.weight.fill_(0.01)
    x, auxiliary = _inputs()

    output = model(x, auxiliary)
    output[:, :1].square().mean().backward()

    assert _has_nonzero_gradient(model.mechanistic_fusion)
    assert _has_nonzero_gradient(model.decoder_blocks)
    assert _has_nonzero_gradient(model.bp_hazard_residual)
    assert not _has_nonzero_gradient(model.behavior_mechanistic_fusion)
    assert not _has_nonzero_gradient(model.behavior_decoder_blocks)
    assert not _has_nonzero_gradient(model.behavior_head)


def test_signed_hazard_residual_adds_and_attenuates_without_negative_hazard() -> None:
    model = _build()
    local_calibration_bias = model.bp_local_calibration.bias
    residual_bias = model.bp_hazard_residual.bias
    assert local_calibration_bias is not None
    assert residual_bias is not None
    decoded = torch.zeros(1, 8, 2, 2)
    burnability = torch.ones(1, 1, 2, 2)

    with torch.no_grad():
        model.bp_local_calibration.weight.zero_()
        local_calibration_bias.zero_()
        residual_bias.fill_(2.0)
    added, _, _ = model._calibrate_bp(
        physical_bp=torch.zeros_like(burnability),
        decoded=decoded,
        fine_burnability=burnability,
    )

    with torch.no_grad():
        residual_bias.fill_(-2.0)
    physical_bp = torch.full_like(burnability, 0.01)
    attenuated, _, _ = model._calibrate_bp(
        physical_bp=physical_bp,
        decoded=decoded,
        fine_burnability=burnability,
    )

    assert torch.all(added > 0.0)
    assert torch.all((attenuated >= 0.0) & (attenuated < physical_bp))


def test_initial_residual_repairs_zero_physical_bp_with_nonzero_gradient() -> None:
    model = _build()
    decoded = torch.zeros(1, 8, 2, 2, requires_grad=True)
    burnability = torch.ones(1, 1, 2, 2)

    probability, _, diagnostics = model._calibrate_bp(
        physical_bp=torch.zeros_like(burnability),
        decoded=decoded,
        fine_burnability=burnability,
    )
    probability.mean().backward()

    assert probability.mean().item() == pytest.approx(1.0 - math.exp(-1e-4), rel=1e-4)
    residual_weight_grad = model.bp_hazard_residual.weight.grad
    residual_bias = model.bp_hazard_residual.bias
    assert residual_weight_grad is not None
    assert residual_bias is not None
    assert residual_bias.grad is not None
    assert residual_weight_grad.abs().sum() == 0.0
    assert residual_bias.grad.abs().sum() > 0.0
    assert diagnostics["mean_positive_bp_hazard_residual"].item() == pytest.approx(1e-4, rel=1e-4)


def test_forward_backward_is_finite_and_residual_is_consumed_once() -> None:
    model = _build()
    x, auxiliary = _inputs()

    output = model(x, auxiliary)
    regularization = model.pop_regularization_loss()
    coarse_bp = model.pop_coarse_bp_probability()
    residual = model.pop_bp_hazard_residual()
    assert regularization is not None
    assert coarse_bp is not None
    assert residual is not None
    loss = output.square().mean() + regularization
    loss.backward()

    assert output.shape == (1, 3, 32, 32)
    assert coarse_bp.shape == (1, 1, 4, 4)
    assert residual[0].shape == residual[1].shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert model.pop_bp_hazard_residual() is None
    assert model.bp_hazard_residual.weight.grad is not None
    assert torch.isfinite(model.bp_hazard_residual.weight.grad).all()
