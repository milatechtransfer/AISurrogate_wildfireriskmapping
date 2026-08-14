import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.mechanistic_propagation import DifferentiableFirePropagation, MechanisticFirePropagationUNet


def _mechanistic_model(steps: int = 2) -> MechanisticFirePropagationUNet:
    spatial_input_names = [
        "grid/ignition_grid_human",
        "grid/ignition_grid_lightning",
        "grid/elevation_grid",
        "spatialized_fire_size/LOG_SIZE_HA_q10",
        "spatialized_fire_size/LOG_SIZE_HA_q50",
        "spatialized_fire_size/LOG_SIZE_HA_q90",
    ]
    config = ModelConfig(
        architecture="mechanistic_propagation",
        num_classes=3,
        output_head="bp_behavior",
        spatial_input_names=spatial_input_names,
        propagation_base_channels=8,
        propagation_steps=steps,
    )
    model = build_model(
        model_config=config,
        spatial_input_channels=len(spatial_input_names),
        auxiliary_input_dims={"fuel_curve": 4},
        target_names=["bp", "fi", "ros"],
    )
    assert isinstance(model, MechanisticFirePropagationUNet)
    return model


def test_directional_propagation_moves_frontier_east():
    propagation = DifferentiableFirePropagation(
        steps=1,
        coarse_cell_size_m=100.0,
        survival_sharpness=6.0,
        min_area_multiplier=0.05,
        max_area_multiplier=20.0,
    )
    seed = torch.zeros(1, 1, 7, 7)
    seed[..., 3, 3] = 1.0
    transmission = torch.zeros(1, 8, 7, 7)
    transmission[:, 2] = 1.0
    quantiles = torch.full((1, 3, 7, 7), 8.0)
    reach = propagation(seed, transmission, quantiles, torch.ones_like(seed))

    assert reach[0, 0, 3, 4] > 0.99
    assert reach[0, 0, 3, 2] == 0.0
    assert reach[0, 0, 2, 3] == 0.0


def test_larger_fire_size_distribution_allows_more_spread():
    propagation = DifferentiableFirePropagation(
        steps=3,
        coarse_cell_size_m=800.0,
        survival_sharpness=6.0,
        min_area_multiplier=0.05,
        max_area_multiplier=20.0,
    )
    seed = torch.zeros(1, 1, 9, 9)
    seed[..., 4, 4] = 0.5
    transmission = torch.full((1, 8, 9, 9), 0.8)
    burnability = torch.ones_like(seed)

    small_fire_reach = propagation(seed, transmission, torch.zeros(1, 5, 9, 9), burnability)
    large_fire_reach = propagation(seed, transmission, torch.full((1, 5, 9, 9), 6.0), burnability)

    assert large_fire_reach.sum() > small_fire_reach.sum()


def test_zero_ignition_produces_finite_near_zero_bp():
    model = _mechanistic_model()
    spatial = torch.zeros(2, 6, 64, 64)
    spatial[:, 3:] = 4.0
    fuel_curve = torch.ones(2, 4, 64, 64)

    predictions = model(spatial, {"fuel_curve": fuel_curve})
    bp_probability = torch.sigmoid(predictions[:, 0:1])

    assert predictions.shape == (2, 3, 64, 64)
    assert torch.isfinite(predictions).all()
    assert bp_probability.max() <= 1.1e-6


def test_mechanistic_model_has_finite_gradients():
    model = _mechanistic_model()
    spatial = torch.zeros(1, 6, 64, 64)
    spatial[:, 0, 32, 32] = 0.2
    spatial[:, 3:] = 4.0
    fuel_curve = torch.ones(1, 4, 64, 64)

    predictions = model(spatial, {"fuel_curve": fuel_curve})
    loss = torch.sigmoid(predictions[:, 0:1]).mean() + predictions[:, 1:].square().mean()
    loss.backward()

    assert resolve_model_architecture(model_config=ModelConfig(architecture="mechanistic_propagation")) == "mechanistic_propagation"
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
