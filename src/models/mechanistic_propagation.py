"""Differentiable ignition-to-spread model for wildfire dense prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.datasets.targets import get_target_spec
from src.models.encoders import FuelCurveEncoder


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(),
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = _conv_block(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class DifferentiableFirePropagation(nn.Module):
    """Recurrently spread an ignition frontier through directional transmission fields."""

    DIRECTIONS = (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    )

    def __init__(
        self,
        *,
        steps: int,
        coarse_cell_size_m: float,
        survival_sharpness: float,
        min_area_multiplier: float,
        max_area_multiplier: float,
    ):
        super().__init__()
        self.steps = steps
        self.coarse_cell_size_m = coarse_cell_size_m
        self.survival_sharpness = survival_sharpness
        self.min_area_multiplier = min_area_multiplier
        self.max_area_multiplier = max_area_multiplier
        initial_fraction = (1.0 - min_area_multiplier) / (max_area_multiplier - min_area_multiplier)
        initial_fraction = min(max(initial_fraction, 1e-6), 1.0 - 1e-6)
        self.raw_area_multiplier = nn.Parameter(torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction))))

    @property
    def area_multiplier(self) -> torch.Tensor:
        fraction = torch.sigmoid(self.raw_area_multiplier)
        return self.min_area_multiplier + (self.max_area_multiplier - self.min_area_multiplier) * fraction

    @staticmethod
    def _shift(values: torch.Tensor, row_offset: int, col_offset: int) -> torch.Tensor:
        shifted = torch.roll(values, shifts=(row_offset, col_offset), dims=(-2, -1))
        mask = torch.ones_like(shifted)
        if row_offset > 0:
            mask[..., :row_offset, :] = 0.0
        elif row_offset < 0:
            mask[..., row_offset:, :] = 0.0
        if col_offset > 0:
            mask[..., :, :col_offset] = 0.0
        elif col_offset < 0:
            mask[..., :, col_offset:] = 0.0
        return shifted * mask

    def _survival_probability(self, fire_size_log10_ha_quantiles: torch.Tensor, step: int) -> torch.Tensor:
        radius_m = step * self.coarse_cell_size_m
        required_area_ha = self.area_multiplier * math.pi * radius_m**2 / 10_000.0
        required_log10_ha = torch.log10(required_area_ha + 1.0)
        exceedance = torch.sigmoid(self.survival_sharpness * (fire_size_log10_ha_quantiles - required_log10_ha))
        return exceedance.mean(dim=1, keepdim=True)

    def forward(
        self,
        seed_probability: torch.Tensor,
        directional_transmission: torch.Tensor,
        fire_size_log10_ha_quantiles: torch.Tensor,
        burnability: torch.Tensor,
    ) -> torch.Tensor:
        if directional_transmission.shape[1] != len(self.DIRECTIONS):
            raise ValueError(f"Expected 8 directional transmission channels, got {directional_transmission.shape}.")
        if fire_size_log10_ha_quantiles.shape[1] < 3:
            raise ValueError("At least three fire-size quantiles are required for propagation.")

        reach = seed_probability.clamp(0.0, 1.0) * burnability
        frontier = reach
        previous_survival = torch.ones_like(reach)

        for step in range(1, self.steps + 1):
            messages = [
                self._shift(
                    frontier * directional_transmission[:, direction_index : direction_index + 1],
                    row_offset,
                    col_offset,
                )
                for direction_index, (row_offset, col_offset) in enumerate(self.DIRECTIONS)
            ]
            arrival = 1.0 - torch.prod(1.0 - torch.stack(messages, dim=1), dim=1)
            survival = self._survival_probability(fire_size_log10_ha_quantiles, step)
            continuation = (survival / previous_survival.clamp_min(1e-6)).clamp(0.0, 1.0)
            frontier = (1.0 - reach) * arrival * continuation * burnability
            reach = reach + frontier
            previous_survival = survival

        return reach.clamp(0.0, 1.0)


class MechanisticFirePropagationUNet(nn.Module):
    """Local fire-behavior encoder coupled to differentiable directional spread."""

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
        super().__init__()
        required_targets = {"bp", "fi", "ros"}
        normalized_targets = [get_target_spec(name).name for name in target_names]
        if len(normalized_targets) != 3 or set(normalized_targets) != required_targets:
            raise ValueError(f"Mechanistic propagation requires exactly {sorted(required_targets)}, got {normalized_targets}.")
        if fuel_curve_input_dim <= 0:
            raise ValueError("Mechanistic propagation requires an iROS fuel-curve input.")
        if len(spatial_input_names) != input_channels:
            raise ValueError(
                f"Expected {input_channels} semantic spatial input names, got {len(spatial_input_names)}: {spatial_input_names}."
            )

        self.target_names = normalized_targets
        self.ignition_indices = self._indices_with_suffix(
            spatial_input_names,
            ("ignition_grid_human", "ignition_grid_lightning"),
        )
        if len(self.ignition_indices) != 2:
            raise ValueError(
                "Mechanistic propagation requires human and lightning ignition channels; "
                f"resolved indices={self.ignition_indices} from {spatial_input_names}."
            )
        self.fire_size_indices = [
            index for index, name in enumerate(spatial_input_names) if name.startswith("spatialized_fire_size/LOG_SIZE_HA_q")
        ]
        if len(self.fire_size_indices) < 3:
            raise ValueError(
                "Mechanistic propagation requires spatialized raw LOG_SIZE_HA quantile channels; "
                f"resolved indices={self.fire_size_indices} from {spatial_input_names}."
            )

        curve_mean = fuel_curve_mean if fuel_curve_mean is not None else torch.zeros(1)
        curve_std = fuel_curve_std if fuel_curve_std is not None else torch.ones(1)
        self.fuel_curve_encoder = FuelCurveEncoder(
            curve_mean=curve_mean,
            curve_std=curve_std,
            in_channels=fuel_curve_input_dim,
            embed_dim=fuel_curve_embed_dim,
        )

        base_channels = model_config.propagation_base_channels
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]
        self.full_encoder = _conv_block(input_channels + fuel_curve_embed_dim, channels[0])
        self.half_encoder = _conv_block(channels[0], channels[1], stride=2)
        self.quarter_encoder = _conv_block(channels[1], channels[2], stride=2)
        self.eighth_encoder = _conv_block(channels[2], channels[3], stride=2)

        self.downsample_factor = model_config.propagation_downsample_factor
        propagation_channels = channels[2] if self.downsample_factor == 4 else channels[3]
        self.directional_transmission = nn.Conv2d(propagation_channels, 8, kernel_size=3, padding=1)
        if self.directional_transmission.bias is not None:
            nn.init.constant_(self.directional_transmission.bias, math.log(0.15 / 0.85))
        self.raw_ignition_scale = nn.Parameter(torch.tensor(0.0))
        self.propagation = DifferentiableFirePropagation(
            steps=model_config.propagation_steps,
            coarse_cell_size_m=model_config.propagation_cell_size_m * self.downsample_factor,
            survival_sharpness=model_config.propagation_survival_sharpness,
            min_area_multiplier=model_config.propagation_min_area_multiplier,
            max_area_multiplier=model_config.propagation_max_area_multiplier,
        )

        self.mechanistic_fusion = _conv_block(propagation_channels + 2, propagation_channels)
        if self.downsample_factor == 8:
            self.decoder_blocks = nn.ModuleList(
                [
                    DecoderBlock(propagation_channels, channels[2], channels[2]),
                    DecoderBlock(channels[2], channels[1], channels[1]),
                    DecoderBlock(channels[1], channels[0], channels[0]),
                ]
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    DecoderBlock(propagation_channels, channels[1], channels[1]),
                    DecoderBlock(channels[1], channels[0], channels[0]),
                ]
            )

        self.bp_calibration = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.behavior_head = nn.Conv2d(channels[0], 2, kernel_size=1)
        self.bp_logit_eps = model_config.propagation_logit_eps

    @staticmethod
    def _indices_with_suffix(names: list[str], suffixes: tuple[str, ...]) -> list[int]:
        return [index for suffix in suffixes for index, name in enumerate(names) if name.endswith(f"/{suffix}")]

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if x_auxiliary is None or "fuel_curve" not in x_auxiliary:
            raise ValueError("Mechanistic propagation requires a 'fuel_curve' tensor.")
        fuel_curve = x_auxiliary["fuel_curve"]
        fuel_embedding = self.fuel_curve_encoder(fuel_curve)

        full = self.full_encoder(torch.cat([x, fuel_embedding], dim=1))
        half = self.half_encoder(full)
        quarter = self.quarter_encoder(half)
        eighth = self.eighth_encoder(quarter)
        coarse_features = quarter if self.downsample_factor == 4 else eighth
        coarse_size = coarse_features.shape[-2:]

        ignition = x[:, self.ignition_indices].clamp_min(0.0).sum(dim=1, keepdim=True)
        pooled_ignition = F.adaptive_avg_pool2d(ignition, coarse_size)
        pooling_area = (x.shape[-2] / coarse_size[0]) * (x.shape[-1] / coarse_size[1])
        pooled_ignition = pooled_ignition * pooling_area

        burnability = (fuel_curve.clamp_min(0.0).sum(dim=1, keepdim=True) > 0.0).to(x.dtype)
        coarse_burnability = F.adaptive_avg_pool2d(burnability, coarse_size)
        ignition_scale = F.softplus(self.raw_ignition_scale) + 1e-4
        seed_probability = (1.0 - torch.exp(-ignition_scale * pooled_ignition)) * coarse_burnability

        fire_size_quantiles = F.adaptive_avg_pool2d(x[:, self.fire_size_indices], coarse_size)
        transmission = torch.sigmoid(self.directional_transmission(coarse_features)) * coarse_burnability
        reach = self.propagation(
            seed_probability=seed_probability,
            directional_transmission=transmission,
            fire_size_log10_ha_quantiles=fire_size_quantiles,
            burnability=coarse_burnability,
        )

        decoded = self.mechanistic_fusion(torch.cat([coarse_features, reach, transmission.mean(dim=1, keepdim=True)], dim=1))
        skips = [quarter, half, full] if self.downsample_factor == 8 else [half, full]
        for decoder_block, skip in zip(self.decoder_blocks, skips, strict=True):
            decoded = decoder_block(decoded, skip)

        reach_full = F.interpolate(reach, size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        bp_probability = reach_full * torch.sigmoid(self.bp_calibration(decoded))
        bp_logits = torch.logit(bp_probability, eps=self.bp_logit_eps)
        behavior = self.behavior_head(decoded)
        outputs = {
            "bp": bp_logits,
            "fi": behavior[:, 0:1],
            "ros": behavior[:, 1:2],
        }
        return torch.cat([outputs[name] for name in self.target_names], dim=1)
