"""Native-grid hybrid v2.4 with task-specific decoders and signed BP hazard repair."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from src.config import ModelConfig
from src.models.mechanistic_hybrid_v23 import MechanisticHybridV23


class MechanisticHybridV24(MechanisticHybridV23):
    """Hybrid v2.3 physics with decoupled BP/behavior decoding and signed hazard repair."""

    MODEL_NAME = "MechanisticHybridV24"

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

        del self.bp_additive_hazard
        self.behavior_mechanistic_fusion = copy.deepcopy(self.mechanistic_fusion)
        self.behavior_decoder_blocks = copy.deepcopy(self.decoder_blocks)

        self.initial_bp_hazard_residual = model_config.propagation_initial_bp_hazard_residual
        self.max_bp_hazard_residual = model_config.propagation_max_bp_hazard_residual
        self.max_bp_hazard_attenuation = model_config.propagation_max_bp_hazard_attenuation
        self.bp_hazard_residual = nn.Conv2d(model_config.propagation_base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.bp_hazard_residual.weight)
        if self.bp_hazard_residual.bias is None:
            raise RuntimeError("The BP hazard residual head requires a bias term.")
        initial_fraction = self.initial_bp_hazard_residual / self.max_bp_hazard_residual
        nn.init.constant_(self.bp_hazard_residual.bias, math.atanh(initial_fraction))

        self._pending_bp_hazard_residual: tuple[torch.Tensor, torch.Tensor] | None = None

    def _decode_task_features(
        self,
        *,
        coarse_features: torch.Tensor,
        per_fire_reach: torch.Tensor,
        coarse_bp: torch.Tensor,
        mean_log_speed_correction_coarse: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bp_fused = self.mechanistic_fusion(torch.cat([coarse_features, per_fire_reach, coarse_bp, mean_log_speed_correction_coarse], dim=1))
        behavior_fused = self.behavior_mechanistic_fusion(
            torch.cat(
                [
                    coarse_features,
                    per_fire_reach.detach(),
                    coarse_bp.detach(),
                    mean_log_speed_correction_coarse.detach(),
                ],
                dim=1,
            )
        )
        bp_decoded = self._decode_features(bp_fused, skips, self.decoder_blocks)
        behavior_decoded = self._decode_features(behavior_fused, skips, self.behavior_decoder_blocks)
        return bp_decoded, behavior_decoded

    def _behavior_reference_fields(
        self,
        *,
        log1p_hfi_full: torch.Tensor,
        log1p_ros_full: torch.Tensor,
        mean_log_speed_correction_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return log1p_hfi_full.detach(), log1p_ros_full.detach(), mean_log_speed_correction_full.detach()

    def _calibrate_bp(
        self,
        *,
        physical_bp: torch.Tensor,
        decoded: torch.Tensor,
        fine_burnability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        physical_hazard = -torch.log1p(-physical_bp.clamp(0.0, 1.0 - 1e-7))
        calibration_fraction = torch.tanh(self.bp_local_calibration(decoded))
        calibrated_hazard = physical_hazard * torch.exp(self.max_local_log_calibration * calibration_fraction)

        residual_fraction = torch.tanh(self.bp_hazard_residual(decoded)) * fine_burnability
        positive_hazard = self.max_bp_hazard_residual * residual_fraction.clamp_min(0.0)
        attenuation = 1.0 - self.max_bp_hazard_attenuation * (-residual_fraction).clamp_min(0.0)
        final_hazard = calibrated_hazard * attenuation + positive_hazard
        probability = -torch.expm1(-final_hazard)

        self._pending_bp_hazard_residual = (residual_fraction, fine_burnability)
        burnable_pixels = fine_burnability.sum().clamp_min(1.0)
        diagnostics = {
            "mean_signed_bp_hazard_residual_fraction": residual_fraction.sum().detach() / burnable_pixels,
            "mean_abs_bp_hazard_residual_fraction": residual_fraction.abs().sum().detach() / burnable_pixels,
            "bp_hazard_residual_saturation_fraction": (
                ((residual_fraction.abs() > 0.9) * fine_burnability).sum().detach() / burnable_pixels
            ),
            "bp_log_hazard_correction_saturation_fraction": (
                ((calibration_fraction.abs() > 0.9) * fine_burnability).sum().detach() / burnable_pixels
            ),
            "mean_positive_bp_hazard_residual": positive_hazard.sum().detach() / burnable_pixels,
            "mean_bp_hazard_attenuation": (attenuation * fine_burnability).sum().detach() / burnable_pixels,
            "fire_size_missing_fraction": self._last_fire_size_missing_fraction,
        }
        return probability, physical_bp.new_zeros(()), diagnostics

    def pop_bp_hazard_residual(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        residual = self._pending_bp_hazard_residual
        self._pending_bp_hazard_residual = None
        return residual

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        self._pending_bp_hazard_residual = None
        return super().forward(x, x_auxiliary)
