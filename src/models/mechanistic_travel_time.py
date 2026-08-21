"""Warm-started U-Net with a physics-guided travel-time BP residual."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.datasets.targets import get_target_spec
from src.models.mechanistic_propagation import _group_count, _quantile_mass_weights
from src.models.unet import BaselineUNet


def _shift(values: torch.Tensor, row_offset: int, col_offset: int, fill_value: float) -> torch.Tensor:
    shifted = torch.roll(values, shifts=(row_offset, col_offset), dims=(-2, -1))
    valid = torch.ones_like(shifted, dtype=torch.bool)
    if row_offset > 0:
        valid[..., :row_offset, :] = False
    elif row_offset < 0:
        valid[..., row_offset:, :] = False
    if col_offset > 0:
        valid[..., :, :col_offset] = False
    elif col_offset < 0:
        valid[..., :, col_offset:] = False
    return torch.where(valid, shifted, shifted.new_full((), fill_value))


def probability_mass_seed_probability(
    *,
    scaled_location_mass: torch.Tensor,
    normalized_log_mean_count: torch.Tensor,
    downsample_factor: int,
    probability_mass_scale: float,
    log_mean_minimum: float,
    log_mean_maximum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert absolute location mass and mean ignition count into coarse Poisson seed probability."""
    if scaled_location_mass.shape[1] != 1 or normalized_log_mean_count.shape[1] != 1:
        raise ValueError("Location mass and normalized count mean must each contain one channel.")
    if scaled_location_mass.shape[-2:] != normalized_log_mean_count.shape[-2:]:
        raise ValueError("Location mass and normalized count mean must have the same spatial shape.")
    if probability_mass_scale <= 0.0:
        raise ValueError(f"probability_mass_scale must be positive, got {probability_mass_scale}.")

    pooling_area = float(downsample_factor**2)
    coarse_location_mass = (
        F.avg_pool2d(
            scaled_location_mass.clamp_min(0.0) / probability_mass_scale,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )
        * pooling_area
    )
    log_mean = log_mean_minimum + normalized_log_mean_count * (log_mean_maximum - log_mean_minimum)
    mean_count = torch.expm1(log_mean).clamp_min(0.0)
    expected_ignitions = (
        F.avg_pool2d(
            scaled_location_mass.clamp_min(0.0) / probability_mass_scale * mean_count,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )
        * pooling_area
    )
    return -torch.expm1(-expected_ignitions), coarse_location_mass


class DifferentiableTravelTimePropagation(nn.Module):
    """Max-plus propagation of ignition log-odds and source-cell time budgets."""

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
        budget_temperature_hours: float,
        quantile_levels: list[float],
        min_ros_m_per_min: float,
        scenario_mode: Literal["burn_hours", "fire_size"] = "burn_hours",
        min_area_multiplier: float = 0.05,
        max_area_multiplier: float = 20.0,
    ):
        super().__init__()
        self.steps = steps
        self.coarse_cell_size_m = coarse_cell_size_m
        self.budget_temperature_hours = budget_temperature_hours
        self.min_ros_m_per_min = min_ros_m_per_min
        self.scenario_mode = scenario_mode
        self.min_area_multiplier = min_area_multiplier
        self.max_area_multiplier = max_area_multiplier
        if scenario_mode == "fire_size":
            initial_fraction = (1.0 - min_area_multiplier) / (max_area_multiplier - min_area_multiplier)
            initial_fraction = min(max(initial_fraction, 1e-6), 1.0 - 1e-6)
            self.raw_area_multiplier = nn.Parameter(torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction))))
        else:
            self.register_parameter("raw_area_multiplier", None)
        self.cohort_weights: torch.Tensor
        self.register_buffer("cohort_weights", _quantile_mass_weights(quantile_levels), persistent=False)
        edge_lengths = [
            coarse_cell_size_m * (math.sqrt(2.0) if row_offset and col_offset else 1.0) for row_offset, col_offset in self.DIRECTIONS
        ]
        self.edge_lengths_m: torch.Tensor
        self.register_buffer("edge_lengths_m", torch.tensor(edge_lengths).view(1, 8, 1, 1), persistent=False)
        self._last_cap_hit_rate = torch.tensor(0.0)

    @property
    def area_multiplier(self) -> torch.Tensor | None:
        if self.raw_area_multiplier is None:
            return None
        fraction = torch.sigmoid(self.raw_area_multiplier)
        return self.min_area_multiplier + (self.max_area_multiplier - self.min_area_multiplier) * fraction

    def _fire_size_budget_hours(
        self,
        fire_size_log10_ha_quantiles: torch.Tensor,
        directional_speed_m_per_min: torch.Tensor,
    ) -> torch.Tensor:
        """Convert desired area to a source-carried equivalent radial travel budget."""
        area_multiplier = self.area_multiplier
        if area_multiplier is None:
            raise RuntimeError("Fire-size budget requested without an area multiplier.")
        fire_size_ha = (torch.pow(10.0, fire_size_log10_ha_quantiles) - 1.0).clamp_min(0.0)
        radius_squared_m2 = fire_size_ha * 10_000.0 / (math.pi * area_multiplier)
        equivalent_radius_m = torch.sqrt(radius_squared_m2 + 1e-6) - 1e-3
        reference_speed = torch.exp(torch.log(directional_speed_m_per_min.clamp_min(self.min_ros_m_per_min)).mean(dim=1, keepdim=True))
        return equivalent_radius_m / (60.0 * reference_speed)

    def forward(
        self,
        seed_probability: torch.Tensor,
        directional_speed_m_per_min: torch.Tensor,
        scenario_quantiles: torch.Tensor,
        burnability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if directional_speed_m_per_min.shape[1] != len(self.DIRECTIONS):
            raise ValueError(f"Expected 8 directional speed channels, got {directional_speed_m_per_min.shape}.")
        if scenario_quantiles.shape[1] != self.cohort_weights.shape[1]:
            raise ValueError(
                "Scenario channel count does not match configured percentile levels: "
                f"{scenario_quantiles.shape[1]} versus {self.cohort_weights.shape[1]}."
            )
        if seed_probability.shape[1] != 1 or burnability.shape[1] != 1:
            raise ValueError("seed_probability and burnability must each contain one channel.")

        temperature = self.budget_temperature_hours
        seed = seed_probability.clamp(0.0, 1.0)
        seed_log_odds = torch.logit(seed.clamp(1e-6, 1.0 - 1e-6))
        if self.scenario_mode == "burn_hours":
            budget_hours_quantiles = scenario_quantiles
        else:
            budget_hours_quantiles = self._fire_size_budget_hours(scenario_quantiles, directional_speed_m_per_min)
        initial_scores = seed_log_odds + budget_hours_quantiles / temperature
        valid_seed = (seed > 0.0) & (burnability > 0.0)
        negative_sentinel = -1.0e4
        scores = torch.where(valid_seed, initial_scores, initial_scores.new_full((), negative_sentinel))

        speed = directional_speed_m_per_min.clamp_min(self.min_ros_m_per_min)
        destination_burnable = burnability > 0.0
        cap_hit_rate = scores.new_zeros(())

        for step in range(self.steps):
            candidates = []
            for direction_index, (row_offset, col_offset) in enumerate(self.DIRECTIONS):
                source_scores = _shift(scores, row_offset, col_offset, negative_sentinel)
                source_speed = _shift(
                    speed[:, direction_index : direction_index + 1],
                    row_offset,
                    col_offset,
                    self.min_ros_m_per_min,
                )
                destination_speed = speed[:, direction_index : direction_index + 1]
                edge_hours = self.edge_lengths_m[:, direction_index : direction_index + 1].to(speed.dtype) / 120.0
                edge_hours = edge_hours * (source_speed.reciprocal() + destination_speed.reciprocal())
                candidate = source_scores - edge_hours / temperature
                source_burnable = _shift(burnability, row_offset, col_offset, 0.0) > 0.0
                candidate = torch.where(
                    source_burnable & destination_burnable,
                    candidate,
                    candidate.new_full((), negative_sentinel),
                )
                candidates.append(candidate)

            best_candidate = torch.stack(candidates, dim=2).amax(dim=2)
            if step == self.steps - 1:
                advancing = (best_candidate > scores + 1e-4) & (torch.sigmoid(best_candidate) > 0.01) & destination_burnable
                denominator = destination_burnable.expand_as(advancing).sum().clamp_min(1)
                cap_hit_rate = advancing.sum().to(scores.dtype) / denominator
            scores = torch.maximum(scores, best_candidate)

        quantile_reach = torch.sigmoid(scores) * burnability
        weights = self.cohort_weights.to(dtype=quantile_reach.dtype)
        mixture_reach = (quantile_reach * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        self._last_cap_hit_rate = cap_hit_rate.detach()
        return mixture_reach, quantile_reach

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = {"cap_hit_rate": float(self._last_cap_hit_rate.cpu())}
        area_multiplier = self.area_multiplier
        if area_multiplier is not None:
            metrics["area_multiplier"] = float(area_multiplier.detach().cpu())
        return metrics


class MechanisticTravelTimeUNet(BaselineUNet):
    """Scenario U-Net whose BP logit receives a bounded mechanistic residual."""

    _PRETRAINED_MODULE_NAMES = ("fuel_curve_encoder", "encoder", "bottleneck", "decoder", "multi_output_head", "out_conv")
    _NEW_STATE_PREFIXES = (
        "ros_isi_bins",
        "direction_xy",
        "travel_time_propagation.",
        "speed_correction.",
        "bp_residual_features.",
        "bp_residual_head.",
        "bp_gate_head.",
    )

    def __init__(
        self,
        *,
        input_channels: int,
        spatial_input_names: list[str],
        model_config: ModelConfig,
        fuel_curve_input_dim: int,
        fuel_curve_embed_dim: int,
        fuel_curve_mean: torch.Tensor | None,
        fuel_curve_std: torch.Tensor | None,
        target_names: list[str],
    ):
        if len(spatial_input_names) != input_channels:
            raise ValueError(
                f"Expected {input_channels} semantic spatial input names, got {len(spatial_input_names)}: {spatial_input_names}."
            )
        if fuel_curve_input_dim <= 0:
            raise ValueError("Mechanistic travel-time propagation requires an iROS fuel-curve input.")
        normalized_targets = [get_target_spec(name).name for name in target_names]
        if len(normalized_targets) != 3 or set(normalized_targets) != {"bp", "fi", "ros"}:
            raise ValueError(f"Mechanistic travel-time propagation requires bp, fi, and ros targets, got {normalized_targets}.")
        if model_config.output_head != "bp_behavior":
            raise ValueError("Mechanistic travel-time propagation requires output_head='bp_behavior'.")
        if len(model_config.propagation_isi_bins) != fuel_curve_input_dim:
            raise ValueError(
                "Configured physical ISI bins must match the iROS curve length: "
                f"{len(model_config.propagation_isi_bins)} versus {fuel_curve_input_dim}."
            )

        super().__init__(
            input_channels=input_channels,
            num_classes=model_config.num_classes,
            hidden_features=model_config.hidden_features,
            input_branches=model_config.input_branches,
            use_skip_connections=model_config.use_skip_connections,
            use_transpose_conv=model_config.use_transpose_conv,
            use_activation_after_upsampling=model_config.use_activation_after_upsampling,
            use_coordconv=model_config.use_coordconv,
            fuel_curve_input_dim=fuel_curve_input_dim,
            fuel_curve_embed_dim=fuel_curve_embed_dim,
            fuel_curve_mean=fuel_curve_mean,
            fuel_curve_std=fuel_curve_std,
            output_head=model_config.output_head,
            target_names=target_names,
        )

        self.spatial_input_names = list(spatial_input_names)
        self.target_names = normalized_targets
        self.bp_target_index = self.target_names.index("bp")
        self.ignition_indices = self._indices_with_suffix(("ignition_grid_human", "ignition_grid_lightning"))
        self.ignition_mode = model_config.propagation_ignition_mode
        self.count_log_mean_index = (
            self._single_index("NORM_LOG1P_IGNITION_COUNT_MEAN") if self.ignition_mode == "probability_mass" else None
        )
        self.elevation_index = self._single_index("elevation_grid")
        self.isi_index = self._single_index("InitialSpreadIndex")
        self.wind_x_index = self._single_index("wind_x")
        self.wind_y_index = self._single_index("wind_y")
        self.scenario_mode = model_config.propagation_scenario_mode
        scenario_prefix = (
            "spatialized_spread_opportunity/NORM_TOTAL_BURN_HOURS_Q"
            if self.scenario_mode == "burn_hours"
            else "spatialized_fire_size/NORM_LOG_SIZE_HA_q"
        )
        scenario_channels = []
        for index, name in enumerate(self.spatial_input_names):
            if name.startswith(scenario_prefix):
                label = name.removeprefix(scenario_prefix)
                if not label.isdigit():
                    raise ValueError(f"Could not parse scenario percentile from channel {name!r}.")
                scenario_channels.append((float(label) / 100.0, index))
        scenario_channels.sort()
        if not scenario_channels:
            raise ValueError(
                f"Mechanistic travel-time propagation in {self.scenario_mode!r} mode requires channels prefixed by {scenario_prefix!r}."
            )
        self.scenario_quantile_levels = [level for level, _ in scenario_channels]
        self.scenario_indices = [index for _, index in scenario_channels]

        self.downsample_factor = model_config.propagation_downsample_factor
        self.coarse_cell_size_m = model_config.propagation_cell_size_m * self.downsample_factor
        self.budget_min_hours = model_config.propagation_budget_min_hours
        self.budget_max_hours = model_config.propagation_budget_max_hours
        self.fire_size_log_min = model_config.propagation_fire_size_log_min
        self.fire_size_log_max = model_config.propagation_fire_size_log_max
        self.ignition_scale = model_config.propagation_initial_ignition_scale
        self.ignition_probability_mass_scale = model_config.propagation_ignition_probability_mass_scale
        self.count_log_mean_minimum = model_config.propagation_count_log_mean_min
        self.count_log_mean_maximum = model_config.propagation_count_log_mean_max
        self.isi_mean = model_config.propagation_isi_mean
        self.isi_std = model_config.propagation_isi_std
        self.wind_x_mean = model_config.propagation_wind_x_mean
        self.wind_x_std = model_config.propagation_wind_x_std
        self.wind_y_mean = model_config.propagation_wind_y_mean
        self.wind_y_std = model_config.propagation_wind_y_std
        self.wind_direction_is_from = model_config.propagation_wind_direction_is_from
        self.wind_anisotropy = model_config.propagation_wind_anisotropy
        self.wind_half_saturation_kmh = model_config.propagation_wind_half_saturation_kmh
        self.elevation_min_m = model_config.propagation_elevation_min_m
        self.elevation_max_m = model_config.propagation_elevation_max_m
        self.slope_coefficient = model_config.propagation_slope_coefficient
        self.max_abs_grade = model_config.propagation_max_abs_grade
        self.min_ros_m_per_min = model_config.propagation_min_ros_m_per_min
        self.speed_correction_log_limit = model_config.propagation_speed_correction_log_limit
        self.max_bp_logit_correction = model_config.propagation_max_bp_logit_correction
        self.ros_isi_bins: torch.Tensor
        self.register_buffer("ros_isi_bins", torch.tensor(model_config.propagation_isi_bins, dtype=torch.float32))
        direction_xy = []
        for row_offset, col_offset in DifferentiableTravelTimePropagation.DIRECTIONS:
            norm = math.sqrt(float(row_offset * row_offset + col_offset * col_offset))
            direction_xy.append((col_offset / norm, -row_offset / norm))
        self.direction_xy: torch.Tensor
        self.register_buffer("direction_xy", torch.tensor(direction_xy, dtype=torch.float32).view(1, 8, 2, 1, 1))

        self.travel_time_propagation = DifferentiableTravelTimePropagation(
            steps=model_config.propagation_steps,
            coarse_cell_size_m=self.coarse_cell_size_m,
            budget_temperature_hours=model_config.propagation_budget_temperature_hours,
            quantile_levels=self.scenario_quantile_levels,
            min_ros_m_per_min=self.min_ros_m_per_min,
            scenario_mode=self.scenario_mode,
            min_area_multiplier=model_config.propagation_min_area_multiplier,
            max_area_multiplier=model_config.propagation_max_area_multiplier,
        )

        decoder_channels = model_config.hidden_features[0]
        branch_channels = model_config.propagation_base_channels
        physics_channels = 6 + len(self.scenario_indices)
        speed_correction_output = nn.Conv2d(branch_channels, 8, kernel_size=1)
        self.speed_correction = nn.Sequential(
            nn.Conv2d(decoder_channels + physics_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.SiLU(),
            speed_correction_output,
        )
        nn.init.zeros_(speed_correction_output.weight)
        if speed_correction_output.bias is not None:
            nn.init.zeros_(speed_correction_output.bias)

        residual_channels = max(8, branch_channels // 2)
        mechanistic_channels = 1 + len(self.scenario_indices)
        self.bp_residual_features = nn.Sequential(
            nn.Conv2d(decoder_channels + mechanistic_channels, residual_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.bp_residual_head = nn.Conv2d(residual_channels, 1, kernel_size=1)
        self.bp_gate_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        nn.init.zeros_(self.bp_residual_head.weight)
        if self.bp_residual_head.bias is not None:
            nn.init.zeros_(self.bp_residual_head.bias)
        nn.init.zeros_(self.bp_gate_head.weight)
        if self.bp_gate_head.bias is not None:
            nn.init.constant_(self.bp_gate_head.bias, -2.0)

        self._pretrained_frozen = False
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    def _indices_with_suffix(self, suffixes: tuple[str, ...]) -> list[int]:
        indices = [index for suffix in suffixes for index, name in enumerate(self.spatial_input_names) if name.endswith(f"/{suffix}")]
        if len(indices) != len(suffixes):
            raise ValueError(f"Could not resolve required channels {suffixes} from {self.spatial_input_names}.")
        return indices

    def _single_index(self, suffix: str) -> int:
        indices = [index for index, name in enumerate(self.spatial_input_names) if name.endswith(f"/{suffix}")]
        if len(indices) != 1:
            raise ValueError(f"Expected one channel ending in /{suffix}, resolved {indices} from {self.spatial_input_names}.")
        return indices[0]

    @staticmethod
    def _neighbor(values: torch.Tensor, row_offset: int, col_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor = _shift(values, -row_offset, -col_offset, 0.0)
        valid = _shift(torch.ones_like(values), -row_offset, -col_offset, 0.0)
        return neighbor, valid

    def _pool(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2] % self.downsample_factor or values.shape[-1] % self.downsample_factor:
            raise ValueError(
                f"Input shape {values.shape[-2:]} must be divisible by propagation_downsample_factor={self.downsample_factor}."
            )
        return F.avg_pool2d(values, kernel_size=self.downsample_factor, stride=self.downsample_factor)

    def _weighted_pool(
        self,
        values: torch.Tensor,
        fine_weights: torch.Tensor,
        coarse_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._pool(values * fine_weights) / coarse_weights.clamp_min(1e-6)

    def _interpolate_iros(self, fuel_curve: torch.Tensor, physical_isi: torch.Tensor) -> torch.Tensor:
        bins = self.ros_isi_bins.to(device=physical_isi.device, dtype=physical_isi.dtype)
        isi = physical_isi.clamp(float(bins[0]), float(bins[-1])).contiguous()
        upper = torch.searchsorted(bins, isi).clamp(1, bins.numel() - 1)
        lower = upper - 1
        lower_bin = bins[lower]
        upper_bin = bins[upper]
        fraction = (isi - lower_bin) / (upper_bin - lower_bin).clamp_min(1e-6)
        lower_ros = fuel_curve.gather(1, lower)
        upper_ros = fuel_curve.gather(1, upper)
        return lower_ros + fraction * (upper_ros - lower_ros)

    def _directional_speed(
        self,
        *,
        x: torch.Tensor,
        fuel_curve: torch.Tensor,
        decoded_features: torch.Tensor,
        normalized_scenario: torch.Tensor,
        coarse_burnability: torch.Tensor,
        coarse_ignition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        physical_isi = x[:, self.isi_index : self.isi_index + 1] * self.isi_std + self.isi_mean
        fine_base_ros = self._interpolate_iros(fuel_curve, physical_isi)
        fine_burnability = (fuel_curve.amax(dim=1, keepdim=True) > 0.0).to(x.dtype)
        safe_ros = fine_base_ros.clamp_min(self.min_ros_m_per_min)
        pooling_area = float(self.downsample_factor**2)
        burnable_count = self._pool(fine_burnability) * pooling_area
        inverse_speed_sum = self._pool(fine_burnability / safe_ros) * pooling_area
        harmonic_ros = burnable_count / inverse_speed_sum.clamp_min(1e-6)
        base_speed = harmonic_ros * coarse_burnability

        wind_x = x[:, self.wind_x_index : self.wind_x_index + 1] * self.wind_x_std + self.wind_x_mean
        wind_y = x[:, self.wind_y_index : self.wind_y_index + 1] * self.wind_y_std + self.wind_y_mean
        if self.wind_direction_is_from:
            wind_x = -wind_x
            wind_y = -wind_y
        coarse_wind_x = self._weighted_pool(wind_x, fine_burnability, coarse_burnability)
        coarse_wind_y = self._weighted_pool(wind_y, fine_burnability, coarse_burnability)
        wind_speed = torch.sqrt(coarse_wind_x.square() + coarse_wind_y.square())
        wind_unit_x = coarse_wind_x / wind_speed.clamp_min(1e-6)
        wind_unit_y = coarse_wind_y / wind_speed.clamp_min(1e-6)
        direction_xy = self.direction_xy.to(dtype=x.dtype)
        alignment = wind_unit_x * direction_xy[:, :, 0] + wind_unit_y * direction_xy[:, :, 1]
        wind_strength = wind_speed / (wind_speed + self.wind_half_saturation_kmh)
        wind_factor = torch.exp(-self.wind_anisotropy * wind_strength * (1.0 - alignment))

        elevation_norm = x[:, self.elevation_index : self.elevation_index + 1]
        elevation_m = elevation_norm * (self.elevation_max_m - self.elevation_min_m) + self.elevation_min_m
        coarse_elevation_m = self._weighted_pool(elevation_m, fine_burnability, coarse_burnability)
        directional_grades = []
        for direction_index, (row_offset, col_offset) in enumerate(DifferentiableTravelTimePropagation.DIRECTIONS):
            neighbor_elevation, neighbor_valid = self._neighbor(coarse_elevation_m, row_offset, col_offset)
            edge_length = self.travel_time_propagation.edge_lengths_m[:, direction_index : direction_index + 1].to(x.dtype)
            grade = ((neighbor_elevation - coarse_elevation_m) / edge_length) * neighbor_valid
            directional_grades.append(grade)
        directional_grade = torch.cat(directional_grades, dim=1).clamp(-self.max_abs_grade, self.max_abs_grade)
        slope_factor = torch.exp(self.slope_coefficient * directional_grade)

        coarse_decoder = self._pool(decoded_features)
        physics_features = torch.cat(
            [
                torch.log1p(base_speed) / math.log(201.0),
                coarse_wind_x / 20.0,
                coarse_wind_y / 20.0,
                self._weighted_pool(elevation_norm, fine_burnability, coarse_burnability),
                normalized_scenario,
                coarse_burnability,
                coarse_ignition,
            ],
            dim=1,
        )
        raw_speed_correction = self.speed_correction(torch.cat([coarse_decoder, physics_features], dim=1))
        log_speed_correction = self.speed_correction_log_limit * torch.tanh(raw_speed_correction)
        learned_factor = torch.exp(log_speed_correction)
        directional_speed = base_speed * wind_factor * slope_factor * learned_factor
        return directional_speed, base_speed, wind_speed, coarse_elevation_m, log_speed_correction

    def load_pretrained_unet_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        checkpoint_config: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
        if checkpoint_config is not None:
            checkpoint_names = checkpoint_config.get("model", {}).get("spatial_input_names")
            if checkpoint_names and list(checkpoint_names) != self.spatial_input_names:
                raise ValueError(
                    "Initial U-Net checkpoint spatial inputs do not match the v4 backbone: "
                    f"{checkpoint_names} versus {self.spatial_input_names}."
                )
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not any(name == prefix or name.startswith(prefix) for prefix in self._NEW_STATE_PREFIXES)
        ]
        if unexpected or invalid_missing:
            raise ValueError(
                f"Initial checkpoint is not a compatible scenario U-Net: unexpected={unexpected}, invalid_missing={invalid_missing}."
            )
        return {
            "loaded": len(state_dict),
            "new": len(incompatible.missing_keys),
        }

    def _pretrained_modules(self) -> list[nn.Module]:
        modules = []
        for name in self._PRETRAINED_MODULE_NAMES:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
        return modules

    def set_pretrained_frozen(self, frozen: bool) -> None:
        self._pretrained_frozen = frozen
        for module in self._pretrained_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)
            if frozen:
                module.eval()

    def train(self, mode: bool = True) -> MechanisticTravelTimeUNet:
        super().train(mode)
        if mode and self._pretrained_frozen:
            for module in self._pretrained_modules():
                module.eval()
        return self

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = self.travel_time_propagation.diagnostic_metrics()
        metrics.update({name: float(value.cpu()) for name, value in self._last_diagnostics.items()})
        return metrics

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if x_auxiliary is None or "fuel_curve" not in x_auxiliary:
            raise ValueError("Mechanistic travel-time propagation requires a 'fuel_curve' tensor.")
        fuel_curve = x_auxiliary["fuel_curve"]
        decoded_features = self.forward_features(x, x_auxiliary)
        baseline_outputs = self._project_output(decoded_features)

        fine_burnability = (fuel_curve.amax(dim=1, keepdim=True) > 0.0).to(x.dtype)
        coarse_burnability = self._pool(fine_burnability)
        ignition = x[:, self.ignition_indices].clamp_min(0.0).sum(dim=1, keepdim=True)
        if self.ignition_mode == "probability_mass":
            if self.count_log_mean_index is None:
                raise RuntimeError("Probability-mass ignition mode requires a resolved normalized mean-count channel.")
            seed_probability, coarse_ignition = probability_mass_seed_probability(
                scaled_location_mass=ignition,
                normalized_log_mean_count=x[:, self.count_log_mean_index : self.count_log_mean_index + 1],
                downsample_factor=self.downsample_factor,
                probability_mass_scale=self.ignition_probability_mass_scale,
                log_mean_minimum=self.count_log_mean_minimum,
                log_mean_maximum=self.count_log_mean_maximum,
            )
        else:
            coarse_ignition = self._pool(ignition)
            seed_probability = 1.0 - torch.exp(-self.ignition_scale * coarse_ignition)
        normalized_scenario = self._weighted_pool(
            x[:, self.scenario_indices],
            fine_burnability,
            coarse_burnability,
        ).clamp(0.0, 1.0)
        if self.scenario_mode == "burn_hours":
            scenario_quantiles = self.budget_min_hours + normalized_scenario * (self.budget_max_hours - self.budget_min_hours)
        else:
            scenario_quantiles = self.fire_size_log_min + normalized_scenario * (self.fire_size_log_max - self.fire_size_log_min)

        directional_speed, base_speed, wind_speed, _, log_speed_correction = self._directional_speed(
            x=x,
            fuel_curve=fuel_curve,
            decoded_features=decoded_features,
            normalized_scenario=normalized_scenario,
            coarse_burnability=coarse_burnability,
            coarse_ignition=coarse_ignition,
        )
        mixture_reach, quantile_reach = self.travel_time_propagation(
            seed_probability=seed_probability,
            directional_speed_m_per_min=directional_speed,
            scenario_quantiles=scenario_quantiles,
            burnability=coarse_burnability,
        )
        mechanistic_features = torch.cat([mixture_reach, quantile_reach], dim=1)
        mechanistic_full = F.interpolate(
            mechanistic_features,
            size=decoded_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual_features = self.bp_residual_features(torch.cat([decoded_features, mechanistic_full], dim=1))
        raw_delta = self.bp_residual_head(residual_features)
        bp_delta = self.max_bp_logit_correction * torch.tanh(raw_delta)
        bp_gate = torch.sigmoid(self.bp_gate_head(decoded_features))

        outputs = baseline_outputs.clone()
        bp_slice = slice(self.bp_target_index, self.bp_target_index + 1)
        outputs[:, bp_slice] = outputs[:, bp_slice] + bp_gate * bp_delta
        self._last_diagnostics = {
            "mean_base_ros_m_per_min": base_speed.detach().mean(),
            "mean_wind_speed_kmh": wind_speed.detach().mean(),
            "mean_mechanistic_reach": mixture_reach.detach().mean(),
            "std_mechanistic_reach": mixture_reach.detach().std(unbiased=False),
            "mean_abs_log_speed_correction": log_speed_correction.detach().abs().mean(),
            "mean_bp_gate": bp_gate.detach().mean(),
            "mean_abs_bp_logit_delta": bp_delta.detach().abs().mean(),
        }
        return outputs
