"""Native-grid hybrid v2.3 with direct log-hectare scenarios and additive BP hazard repair."""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.models.mechanistic_hybrid_v22 import MechanisticHybridV22, _inverse_bounded_sigmoid


class MechanisticHybridV23(MechanisticHybridV22):
    """Hybrid v2.2 physics with a bounded fine-resolution additive BP hazard residual."""

    MODEL_NAME = "MechanisticHybridV23"
    FIRE_SIZE_CHANNEL_PREFIX = "spatialized_fire_size/LOG_SIZE_HA_q"

    def __init__(
        self,
        input_channels: int,
        spatial_input_names: list[str],
        model_config: ModelConfig,
        fuel_curve_input_dim: int,
        fuel_curve_embed_dim: int,
        fuel_curve_mean: torch.Tensor | None,
        fuel_curve_std: torch.Tensor | None,
        target_names: list[str],
    ):
        super().__init__(
            input_channels=input_channels,
            spatial_input_names=spatial_input_names,
            model_config=model_config,
            fuel_curve_input_dim=fuel_curve_input_dim,
            fuel_curve_embed_dim=fuel_curve_embed_dim,
            fuel_curve_mean=fuel_curve_mean,
            fuel_curve_std=fuel_curve_std,
            target_names=target_names,
        )

        missing_name = "spatialized_fire_size/missing_firezone_mask"
        missing_indices = [index for index, name in enumerate(self.spatial_input_names) if name == missing_name]
        if len(missing_indices) != 1:
            raise ValueError(f"{self.MODEL_NAME} requires exactly one {missing_name!r} channel, resolved {missing_indices}.")
        self.fire_size_missing_index = missing_indices[0]
        self.fire_size_neural_mean = model_config.propagation_fire_size_neural_mean
        self.fire_size_neural_std = model_config.propagation_fire_size_neural_std
        self.max_additive_bp_hazard = model_config.propagation_max_additive_bp_hazard
        self.additive_bp_hazard_weight = model_config.propagation_additive_bp_hazard_weight

        self.bp_additive_hazard = nn.Conv2d(model_config.propagation_base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.bp_additive_hazard.weight)
        initial_bias = _inverse_bounded_sigmoid(
            model_config.propagation_initial_additive_bp_hazard,
            0.0,
            self.max_additive_bp_hazard,
        )
        if self.bp_additive_hazard.bias is None:
            raise RuntimeError("The additive BP hazard head requires a bias term.")
        nn.init.constant_(self.bp_additive_hazard.bias, float(initial_bias))
        self._last_fire_size_missing_fraction = torch.tensor(0.0)

    def _neural_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        neural_x = x.clone()
        neural_x[:, self.scenario_indices] = (neural_x[:, self.scenario_indices] - self.fire_size_neural_mean) / self.fire_size_neural_std
        self._last_fire_size_missing_fraction = x[:, self.fire_size_missing_index].detach().mean()
        return neural_x

    def _physical_fire_size_log10_ha(self, scenario_values: torch.Tensor) -> torch.Tensor:
        return scenario_values

    def _coarse_scenario_values(
        self,
        x: torch.Tensor,
        fine_burnability: torch.Tensor,
        coarse_burnability: torch.Tensor,
    ) -> torch.Tensor:
        scenario_values = self._weighted_pool(x[:, self.scenario_indices], fine_burnability, coarse_burnability)
        if not torch.isfinite(scenario_values).all():
            raise ValueError("MechanisticHybridV23 received non-finite direct log-hectare fire sizes.")
        if torch.any(scenario_values < 0.0):
            raise ValueError("MechanisticHybridV23 received a negative log10(1 + hectares) fire size.")
        return scenario_values

    def _additive_hazard(self, decoded: torch.Tensor, fine_burnability: torch.Tensor) -> torch.Tensor:
        hazard = self.max_additive_bp_hazard * torch.sigmoid(self.bp_additive_hazard(decoded))
        return hazard * fine_burnability

    def _calibrate_bp(
        self,
        *,
        physical_bp: torch.Tensor,
        decoded: torch.Tensor,
        fine_burnability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        physical_hazard = -torch.log1p(-physical_bp.clamp(0.0, 1.0 - 1e-7))
        delta = self.max_local_log_calibration * torch.tanh(self.bp_local_calibration(decoded))
        additive_hazard = self._additive_hazard(decoded, fine_burnability)
        final_hazard = physical_hazard * torch.exp(delta) + additive_hazard
        probability = -torch.expm1(-final_hazard)

        burnable_pixels = fine_burnability.sum().clamp_min(1.0)
        mean_additive_hazard = additive_hazard.sum() / burnable_pixels
        regularization = self.additive_bp_hazard_weight * mean_additive_hazard
        diagnostics = {
            "mean_additive_bp_hazard": mean_additive_hazard.detach(),
            "max_additive_bp_hazard": additive_hazard.detach().amax(),
            "mean_abs_bp_log_hazard_correction": delta.detach().abs().mean(),
            "fire_size_missing_fraction": self._last_fire_size_missing_fraction,
        }
        return probability, regularization, diagnostics
