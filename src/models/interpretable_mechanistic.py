import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.config import ModelConfig
from src.datasets.targets import get_target_spec
from src.models.mechanistic_travel_time import DifferentiableTravelTimePropagation, _shift


def _inverse_bounded_sigmoid(value: float, minimum: float, maximum: float) -> torch.Tensor:
    if not minimum < value < maximum:
        raise ValueError(f"Expected {minimum} < {value} < {maximum}.")
    fraction = (value - minimum) / (maximum - minimum)
    return torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float32)


def _bounded_value(raw_value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return minimum + (maximum - minimum) * torch.sigmoid(raw_value)


def count_distribution_burn_probability(
    *,
    per_fire_reach: torch.Tensor,
    mean_count: torch.Tensor,
    coefficient_of_variation: torch.Tensor,
    per_fire_reach_scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Aggregate per-fire reach through a moment-matched ignition-count distribution."""
    reach = -torch.expm1(per_fire_reach_scale * torch.log1p(-per_fire_reach.clamp(0.0, 1.0 - eps)))
    mean = mean_count.clamp_min(0.0)
    variance = (coefficient_of_variation.clamp_min(0.0) * mean).square()

    poisson_log_no_burn = -mean * reach

    overdispersion = (variance - mean).clamp_min(eps)
    dispersion = mean.square() / overdispersion
    negative_binomial_log_no_burn = -dispersion * torch.log1p(mean * reach / dispersion.clamp_min(eps))

    underdispersion = (mean - variance).clamp_min(eps)
    trials = mean.square() / underdispersion
    ignition_probability = (mean / trials.clamp_min(eps)).clamp(0.0, 1.0 - eps)
    binomial_log_no_burn = trials * torch.log1p(-ignition_probability * reach)

    tolerance = 1e-4 * mean.clamp_min(1.0)
    log_no_burn = torch.where(
        variance > mean + tolerance,
        negative_binomial_log_no_burn,
        torch.where(variance < mean - tolerance, binomial_log_no_burn, poisson_log_no_burn),
    )
    return (-torch.expm1(log_no_burn)).clamp(0.0, 1.0)


class SourceCohortTravelTimePropagation(DifferentiableTravelTimePropagation):
    """Approximate per-fire reach by propagating stratified ignition-source cohorts."""

    def __init__(
        self,
        *,
        source_grid_size: int,
        steps: int,
        coarse_cell_size_m: float,
        budget_temperature_hours: float,
        quantile_levels: list[float],
        min_ros_m_per_min: float,
        scenario_mode: Literal["burn_hours", "fire_size"] = "fire_size",
        min_area_multiplier: float = 0.05,
        max_area_multiplier: float = 20.0,
    ):
        super().__init__(
            steps=steps,
            coarse_cell_size_m=coarse_cell_size_m,
            budget_temperature_hours=budget_temperature_hours,
            quantile_levels=quantile_levels,
            min_ros_m_per_min=min_ros_m_per_min,
            scenario_mode=scenario_mode,
            min_area_multiplier=min_area_multiplier,
            max_area_multiplier=max_area_multiplier,
        )
        self.source_grid_size = source_grid_size
        self.source_samples = source_grid_size**2
        self._last_context_location_mass = torch.tensor(0.0)

    def _source_cohorts(self, location: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, height, width = location.shape
        if self.source_grid_size > min(height, width):
            raise ValueError(f"source_grid_size={self.source_grid_size} exceeds the propagation grid shape {(height, width)}.")

        row_edges = [index * height // self.source_grid_size for index in range(self.source_grid_size + 1)]
        col_edges = [index * width // self.source_grid_size for index in range(self.source_grid_size + 1)]
        source_indices = []
        source_weights = []
        for row_start, row_end in zip(row_edges, row_edges[1:], strict=False):
            for col_start, col_end in zip(col_edges, col_edges[1:], strict=False):
                tile = location[:, 0, row_start:row_end, col_start:col_end]
                flat_tile = tile.flatten(1)
                tile_mass = flat_tile.sum(dim=1)
                rows = torch.arange(row_start, row_end, device=location.device, dtype=location.dtype)
                cols = torch.arange(col_start, col_end, device=location.device, dtype=location.dtype)
                row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
                flat_rows = row_grid.flatten().unsqueeze(0)
                flat_cols = col_grid.flatten().unsqueeze(0)
                centroid_row = (flat_tile * flat_rows).sum(dim=1) / tile_mass.clamp_min(1e-12)
                centroid_col = (flat_tile * flat_cols).sum(dim=1) / tile_mass.clamp_min(1e-12)
                squared_distance = (flat_rows - centroid_row.unsqueeze(1)).square()
                squared_distance = squared_distance + (flat_cols - centroid_col.unsqueeze(1)).square()
                squared_distance = torch.where(
                    flat_tile > 0.0,
                    squared_distance,
                    squared_distance.new_full((), float("inf")),
                )
                local_index = squared_distance.argmin(dim=1)
                local_width = col_end - col_start
                source_row = row_start + torch.div(local_index, local_width, rounding_mode="floor")
                source_col = col_start + torch.remainder(local_index, local_width)
                source_indices.append(source_row * width + source_col)
                source_weights.append(tile_mass)

        indices = torch.stack(source_indices, dim=1)
        weights = torch.stack(source_weights, dim=1)
        raw_context_mass = weights.sum(dim=1, keepdim=True)
        weights = weights * raw_context_mass.clamp(0.0, 1.0) / raw_context_mass.clamp_min(1e-12)
        return indices, weights, raw_context_mass

    def forward(
        self,
        location_mass: torch.Tensor,
        directional_speed_m_per_min: torch.Tensor,
        fire_size_log10_ha_quantiles: torch.Tensor,
        burnability: torch.Tensor,
        fire_size_multiplier: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if directional_speed_m_per_min.shape[1] != len(self.DIRECTIONS):
            raise ValueError(f"Expected 8 directional speed channels, got {directional_speed_m_per_min.shape}.")
        if fire_size_log10_ha_quantiles.shape[1] != self.cohort_weights.shape[1]:
            raise ValueError(
                "Fire-size channel count does not match configured percentile levels: "
                f"{fire_size_log10_ha_quantiles.shape[1]} versus {self.cohort_weights.shape[1]}."
            )
        if location_mass.shape[1] != 1 or burnability.shape[1] != 1:
            raise ValueError("location_mass and burnability must each contain one channel.")
        if fire_size_multiplier is not None and (
            fire_size_multiplier.shape[1] != 1 or fire_size_multiplier.shape[-2:] != location_mass.shape[-2:]
        ):
            raise ValueError(
                "fire_size_multiplier must contain one channel and match the location-mass spatial shape, got "
                f"{fire_size_multiplier.shape} versus {location_mass.shape}."
            )

        location = location_mass.clamp(0.0, 1.0) * burnability
        batch_size, _, height, width = location.shape
        quantile_count = fire_size_log10_ha_quantiles.shape[1]
        source_indices, source_weights, raw_context_mass = self._source_cohorts(location)

        effective_fire_size = fire_size_log10_ha_quantiles
        if fire_size_multiplier is not None:
            fire_size_ha = torch.expm1(math.log(10.0) * fire_size_log10_ha_quantiles).clamp_min(0.0)
            effective_fire_size = torch.log1p(fire_size_ha * fire_size_multiplier.clamp_min(0.0)) / math.log(10.0)
        budget_hours = self._fire_size_budget_hours(effective_fire_size, directional_speed_m_per_min)
        source_budgets = budget_hours.flatten(2).gather(
            dim=2,
            index=source_indices.unsqueeze(1).expand(-1, quantile_count, -1),
        )
        source_budgets = source_budgets.permute(0, 2, 1)
        negative_sentinel = -1.0e4
        source_budgets = torch.where(
            source_weights.unsqueeze(-1) > 0.0,
            source_budgets,
            source_budgets.new_full((), negative_sentinel),
        )
        remaining_hours = budget_hours.new_full(
            (batch_size, self.source_samples, quantile_count, height * width),
            negative_sentinel,
        ).scatter(
            dim=-1,
            index=source_indices[:, :, None, None].expand(-1, -1, quantile_count, 1),
            src=source_budgets.unsqueeze(-1),
        )
        remaining_hours = remaining_hours.view(batch_size, self.source_samples, quantile_count, height, width)

        speed = directional_speed_m_per_min.clamp_min(self.min_ros_m_per_min)
        destination_burnable = burnability > 0.0
        cap_hit_rate = location.new_zeros(())

        for step in range(self.steps):
            candidates = []
            for direction_index, (row_offset, col_offset) in enumerate(self.DIRECTIONS):
                source_remaining = _shift(remaining_hours, row_offset, col_offset, negative_sentinel)
                source_speed = _shift(
                    speed[:, direction_index : direction_index + 1],
                    row_offset,
                    col_offset,
                    self.min_ros_m_per_min,
                ).unsqueeze(1)
                destination_speed = speed[:, direction_index : direction_index + 1].unsqueeze(1)
                edge_hours = self.edge_lengths_m[:, direction_index : direction_index + 1].to(speed.dtype).unsqueeze(1) / 120.0
                edge_hours = edge_hours * (source_speed.reciprocal() + destination_speed.reciprocal())
                remaining_after_edge = source_remaining - edge_hours
                source_burnable = (_shift(burnability, row_offset, col_offset, 0.0) > 0.0).unsqueeze(1)
                valid_edge = source_burnable & destination_burnable.unsqueeze(1)
                candidates.append(torch.where(valid_edge, remaining_after_edge, remaining_after_edge.new_full((), negative_sentinel)))

            best_candidate = torch.stack(candidates, dim=3).amax(dim=3)

            if step == self.steps - 1:
                advancing = (
                    (best_candidate > remaining_hours + 1e-4)
                    & (torch.sigmoid(best_candidate / self.budget_temperature_hours) > 0.01)
                    & destination_burnable.unsqueeze(1)
                )
                denominator = destination_burnable.unsqueeze(1).expand_as(advancing).sum().clamp_min(1)
                cap_hit_rate = advancing.sum().to(location.dtype) / denominator
            remaining_hours = torch.maximum(remaining_hours, best_candidate)

        source_reach = torch.sigmoid(remaining_hours / self.budget_temperature_hours)
        source_reach = source_reach * destination_burnable.unsqueeze(1)
        quantile_reach = (source_reach * source_weights[:, :, None, None, None]).sum(dim=1).clamp(0.0, 1.0)
        weights = self.cohort_weights.to(dtype=quantile_reach.dtype)
        mixture_reach = (quantile_reach * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        self._last_cap_hit_rate = cap_hit_rate.detach()
        self._last_context_location_mass = raw_context_mass.detach().mean()
        return mixture_reach, quantile_reach

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = super().diagnostic_metrics()
        metrics["source_samples"] = float(self.source_samples)
        metrics["source_grid_size"] = float(self.source_grid_size)
        metrics["mean_context_location_mass"] = float(self._last_context_location_mass.cpu())
        return metrics


class PhysicalParameterFieldNetwork(nn.Module):
    """Predict bounded local corrections to named physical quantities."""

    FIELD_NAMES = ("ignition", "ros", "fire_size", "reach", "consumption")

    def __init__(self, *, input_channels: int, hidden_channels: int):
        super().__init__()
        normalization_groups = math.gcd(hidden_channels, 8)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(normalization_groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(normalization_groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(normalization_groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=8, dilation=8),
            nn.GroupNorm(normalization_groups, hidden_channels),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden_channels, len(self.FIELD_NAMES), kernel_size=1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is None:
            raise RuntimeError("Physical parameter-field head must include a bias.")
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = self.head(self.features(features))
        return {name: fields[:, index : index + 1] for index, name in enumerate(self.FIELD_NAMES)}


class InterpretableMechanisticModel(nn.Module):
    """Mechanism-dominant BP, FI, and ROS model with named scalar parameters."""

    FIELD_INPUT_CHANNELS = 15

    def __init__(
        self,
        *,
        input_channels: int,
        spatial_input_names: list[str],
        model_config: ModelConfig,
        fuel_curve_input_dim: int,
        target_names: list[str],
        variant: Literal["v1", "v2", "v3"] = "v1",
    ):
        super().__init__()
        if len(spatial_input_names) != input_channels:
            raise ValueError(
                f"Expected {input_channels} semantic spatial input names, got {len(spatial_input_names)}: {spatial_input_names}."
            )
        normalized_targets = [get_target_spec(name).name for name in target_names]
        if len(normalized_targets) != 3 or set(normalized_targets) != {"bp", "fi", "ros"}:
            raise ValueError(f"Interpretable mechanism requires bp, fi, and ros targets, got {normalized_targets}.")
        curve_length = len(model_config.propagation_isi_bins)
        if fuel_curve_input_dim != 2 * curve_length:
            raise ValueError(
                "Interpretable mechanism requires concatenated iROS and HFI curves: "
                f"expected {2 * curve_length} channels, got {fuel_curve_input_dim}."
            )
        if model_config.propagation_scenario_mode != "fire_size":
            raise ValueError("Interpretable mechanism requires propagation_scenario_mode='fire_size'.")

        self.spatial_input_names = list(spatial_input_names)
        self.target_names = normalized_targets
        self.variant = variant
        self.curve_length = curve_length
        self.downsample_factor = model_config.propagation_downsample_factor
        self.fire_size_log_min = model_config.propagation_fire_size_log_min
        self.fire_size_log_max = model_config.propagation_fire_size_log_max
        self.ignition_probability_mass_scale = model_config.propagation_ignition_probability_mass_scale
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
        self.fi_log_mean = model_config.interpretable_fi_log_mean
        self.fi_log_std = model_config.interpretable_fi_log_std
        self.ros_log_mean = model_config.interpretable_ros_log_mean
        self.ros_log_std = model_config.interpretable_ros_log_std
        self.behavior_log_scale_limit = model_config.interpretable_behavior_log_scale_limit
        self.min_ignition_rate = model_config.interpretable_min_ignition_rate
        self.max_ignition_rate = model_config.interpretable_max_ignition_rate
        self.min_bp_hazard_scale = model_config.interpretable_min_bp_hazard_scale
        self.max_bp_hazard_scale = model_config.interpretable_max_bp_hazard_scale
        self.count_log_mean_minimum = model_config.propagation_count_log_mean_min
        self.count_log_mean_maximum = model_config.propagation_count_log_mean_max
        self.count_cv_minimum = model_config.propagation_count_cv_min
        self.count_cv_maximum = model_config.propagation_count_cv_max
        self.min_per_fire_reach_scale = model_config.interpretable_min_per_fire_reach_scale
        self.max_per_fire_reach_scale = model_config.interpretable_max_per_fire_reach_scale
        self.behavior_log_slope_min = model_config.interpretable_behavior_log_slope_min
        self.behavior_log_slope_max = model_config.interpretable_behavior_log_slope_max
        self.ignition_field_log_limit = model_config.interpretable_ignition_field_log_limit
        self.ros_field_log_limit = model_config.interpretable_ros_field_log_limit
        self.fire_size_field_log_limit = model_config.interpretable_fire_size_field_log_limit
        self.reach_field_log_limit = model_config.interpretable_reach_field_log_limit
        self.consumption_field_log_limit = model_config.interpretable_consumption_field_log_limit
        self.field_l2_weight = model_config.interpretable_field_l2_weight
        self.field_tv_weight = model_config.interpretable_field_tv_weight

        self.ignition_indices = self._indices_with_suffix(("ignition_grid_human", "ignition_grid_lightning"))
        self.count_log_mean_index = self._single_index("NORM_LOG1P_IGNITION_COUNT_MEAN") if self.variant != "v1" else None
        self.count_cv_index = self._single_index("NORM_IGNITION_COUNT_CV") if self.variant != "v1" else None
        self.elevation_index = self._single_index("elevation_grid")
        self.isi_index = self._single_index("InitialSpreadIndex")
        self.wind_x_index = self._single_index("wind_x")
        self.wind_y_index = self._single_index("wind_y")
        scenario_channels = []
        scenario_prefix = "spatialized_fire_size/NORM_LOG_SIZE_HA_q"
        for index, name in enumerate(self.spatial_input_names):
            if name.startswith(scenario_prefix):
                quantile_label = name.removeprefix(scenario_prefix)
                if not quantile_label.isdigit():
                    raise ValueError(f"Could not parse fire-size percentile from channel {name!r}.")
                scenario_channels.append((float(quantile_label) / 100.0, index))
        scenario_channels.sort()
        if not scenario_channels:
            raise ValueError(f"Interpretable mechanism requires channels prefixed by {scenario_prefix!r}.")
        self.scenario_quantile_levels = [quantile for quantile, _ in scenario_channels]
        self.scenario_indices = [index for _, index in scenario_channels]

        self.ros_isi_bins: torch.Tensor
        self.register_buffer("ros_isi_bins", torch.tensor(model_config.propagation_isi_bins, dtype=torch.float32))
        direction_xy = []
        for row_offset, col_offset in DifferentiableTravelTimePropagation.DIRECTIONS:
            norm = math.sqrt(float(row_offset * row_offset + col_offset * col_offset))
            direction_xy.append((col_offset / norm, -row_offset / norm))
        self.direction_xy: torch.Tensor
        self.register_buffer("direction_xy", torch.tensor(direction_xy, dtype=torch.float32).view(1, 8, 2, 1, 1))

        coarse_cell_size_m = model_config.propagation_cell_size_m * self.downsample_factor
        self.travel_time_propagation: DifferentiableTravelTimePropagation
        if self.variant != "v1":
            self.travel_time_propagation = SourceCohortTravelTimePropagation(
                source_grid_size=model_config.interpretable_source_grid_size,
                steps=model_config.propagation_steps,
                coarse_cell_size_m=coarse_cell_size_m,
                budget_temperature_hours=model_config.propagation_budget_temperature_hours,
                quantile_levels=self.scenario_quantile_levels,
                min_ros_m_per_min=self.min_ros_m_per_min,
                scenario_mode="fire_size",
                min_area_multiplier=model_config.propagation_min_area_multiplier,
                max_area_multiplier=model_config.propagation_max_area_multiplier,
            )
        else:
            self.travel_time_propagation = DifferentiableTravelTimePropagation(
                steps=model_config.propagation_steps,
                coarse_cell_size_m=coarse_cell_size_m,
                budget_temperature_hours=model_config.propagation_budget_temperature_hours,
                quantile_levels=self.scenario_quantile_levels,
                min_ros_m_per_min=self.min_ros_m_per_min,
                scenario_mode="fire_size",
                min_area_multiplier=model_config.propagation_min_area_multiplier,
                max_area_multiplier=model_config.propagation_max_area_multiplier,
            )

        if self.variant == "v1":
            self.raw_effective_ignition_rate = nn.Parameter(
                _inverse_bounded_sigmoid(
                    model_config.interpretable_initial_ignition_rate,
                    self.min_ignition_rate,
                    self.max_ignition_rate,
                )
            )
            self.raw_bp_hazard_scale = nn.Parameter(
                _inverse_bounded_sigmoid(
                    model_config.interpretable_initial_bp_hazard_scale,
                    self.min_bp_hazard_scale,
                    self.max_bp_hazard_scale,
                )
            )
            self.raw_fi_log_scale = nn.Parameter(torch.zeros(()))
            self.raw_ros_log_scale = nn.Parameter(torch.zeros(()))
            self.register_parameter("raw_per_fire_reach_scale", None)
            self.register_parameter("raw_fi_log_slope", None)
            self.register_parameter("raw_ros_log_slope", None)
        else:
            self.register_parameter("raw_effective_ignition_rate", None)
            self.register_parameter("raw_bp_hazard_scale", None)
            self.raw_per_fire_reach_scale = nn.Parameter(
                _inverse_bounded_sigmoid(
                    model_config.interpretable_initial_per_fire_reach_scale,
                    self.min_per_fire_reach_scale,
                    self.max_per_fire_reach_scale,
                )
            )
            self.raw_fi_log_scale = nn.Parameter(torch.zeros(()))
            self.raw_ros_log_scale = nn.Parameter(torch.zeros(()))
            initial_slope = _inverse_bounded_sigmoid(
                1.0,
                self.behavior_log_slope_min,
                self.behavior_log_slope_max,
            )
            self.raw_fi_log_slope = nn.Parameter(initial_slope.clone())
            self.raw_ros_log_slope = nn.Parameter(initial_slope.clone())
        self.parameter_field_network = (
            PhysicalParameterFieldNetwork(
                input_channels=self.FIELD_INPUT_CHANNELS,
                hidden_channels=model_config.interpretable_field_hidden_channels,
            )
            if self.variant == "v3"
            else None
        )
        self._pending_regularization_loss: torch.Tensor | None = None
        self._last_field_diagnostics: dict[str, torch.Tensor] = {}
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    def _indices_with_suffix(self, suffixes: tuple[str, ...]) -> list[int]:
        resolved = []
        for suffix in suffixes:
            indices = [index for index, name in enumerate(self.spatial_input_names) if name == suffix or name.endswith(f"/{suffix}")]
            if len(indices) != 1:
                raise ValueError(f"Expected one channel ending in /{suffix}, resolved {indices} from {self.spatial_input_names}.")
            resolved.append(indices[0])
        return resolved

    def _single_index(self, suffix: str) -> int:
        return self._indices_with_suffix((suffix,))[0]

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

    def _parameter_field_features(
        self,
        *,
        x: torch.Tensor,
        ros_curve: torch.Tensor,
        hfi_curve: torch.Tensor,
        fine_base_hfi: torch.Tensor,
        fine_burnability: torch.Tensor,
        coarse_burnability: torch.Tensor,
        coarse_support: torch.Tensor,
        coarse_location_mass: torch.Tensor,
        normalized_fire_size: torch.Tensor,
        normalized_log_mean: torch.Tensor,
        normalized_cv: torch.Tensor,
        base_speed: torch.Tensor,
    ) -> torch.Tensor:
        context_mean_mass = coarse_location_mass.mean(dim=(-2, -1), keepdim=True)
        relative_ignition_density = coarse_location_mass / context_mean_mass.clamp_min(1e-12)
        relative_ignition_density = torch.log1p(relative_ignition_density).clamp_max(4.0) / 4.0
        coarse_isi = self._weighted_pool(
            x[:, self.isi_index : self.isi_index + 1],
            fine_burnability,
            coarse_burnability,
        )
        coarse_wind_x = self._weighted_pool(
            x[:, self.wind_x_index : self.wind_x_index + 1],
            fine_burnability,
            coarse_burnability,
        )
        coarse_wind_y = self._weighted_pool(
            x[:, self.wind_y_index : self.wind_y_index + 1],
            fine_burnability,
            coarse_burnability,
        )
        coarse_elevation = self._weighted_pool(
            x[:, self.elevation_index : self.elevation_index + 1],
            fine_burnability,
            coarse_burnability,
        )
        coarse_base_hfi = self._weighted_pool(fine_base_hfi, fine_burnability, coarse_burnability)
        coarse_max_ros = self._weighted_pool(ros_curve.amax(dim=1, keepdim=True), fine_burnability, coarse_burnability)
        coarse_max_hfi = self._weighted_pool(hfi_curve.amax(dim=1, keepdim=True), fine_burnability, coarse_burnability)
        coarse_ros_feature = (torch.log1p(base_speed.clamp_min(0.0)) - self.ros_log_mean) / self.ros_log_std
        coarse_hfi_feature = (torch.log1p(coarse_base_hfi.clamp_min(0.0)) - self.fi_log_mean) / self.fi_log_std
        coarse_max_ros_feature = (torch.log1p(coarse_max_ros.clamp_min(0.0)) - self.ros_log_mean) / self.ros_log_std
        coarse_max_hfi_feature = (torch.log1p(coarse_max_hfi.clamp_min(0.0)) - self.fi_log_mean) / self.fi_log_std
        features = torch.cat(
            [
                relative_ignition_density,
                normalized_log_mean,
                normalized_cv,
                normalized_fire_size,
                coarse_isi,
                coarse_wind_x,
                coarse_wind_y,
                coarse_elevation,
                coarse_ros_feature,
                coarse_hfi_feature,
                coarse_max_ros_feature,
                coarse_max_hfi_feature,
                coarse_burnability,
            ],
            dim=1,
        )
        if features.shape[1] != self.FIELD_INPUT_CHANNELS:
            raise RuntimeError(f"Expected {self.FIELD_INPUT_CHANNELS} parameter-field features, got {features.shape}.")
        return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0) * coarse_support

    def _bounded_parameter_fields(
        self,
        features: torch.Tensor,
        coarse_support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.parameter_field_network is None:
            raise RuntimeError("Local physical parameter fields are only defined for interpretable mechanism v3.")
        raw_fields = self.parameter_field_network(features)
        limits = {
            "ignition": self.ignition_field_log_limit,
            "ros": self.ros_field_log_limit,
            "fire_size": self.fire_size_field_log_limit,
            "reach": self.reach_field_log_limit,
            "consumption": self.consumption_field_log_limit,
        }
        return {name: limit * torch.tanh(raw_fields[name]) * coarse_support for name, limit in limits.items()}

    @staticmethod
    def _redistribute_location_mass(
        location_mass: torch.Tensor,
        ignition_log_correction: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        original_total = location_mass.sum(dim=(-2, -1), keepdim=True)
        corrected = location_mass * torch.exp(ignition_log_correction) * support
        corrected_total = corrected.sum(dim=(-2, -1), keepdim=True)
        return corrected * original_total / corrected_total.clamp_min(1e-12)

    def _set_parameter_field_regularization(
        self,
        fields: dict[str, torch.Tensor],
        coarse_support: torch.Tensor,
    ) -> None:
        stacked = torch.cat([fields[name] for name in PhysicalParameterFieldNetwork.FIELD_NAMES], dim=1)
        support_count = coarse_support.sum().clamp_min(1.0)
        l2 = stacked.square().sum() / (support_count * stacked.shape[1])
        horizontal_mask = coarse_support[..., :, 1:] * coarse_support[..., :, :-1]
        vertical_mask = coarse_support[..., 1:, :] * coarse_support[..., :-1, :]
        horizontal_tv = (stacked[..., :, 1:] - stacked[..., :, :-1]).abs() * horizontal_mask
        vertical_tv = (stacked[..., 1:, :] - stacked[..., :-1, :]).abs() * vertical_mask
        tv_denominator = ((horizontal_mask.sum() + vertical_mask.sum()) * stacked.shape[1]).clamp_min(1.0)
        total_variation = (horizontal_tv.sum() + vertical_tv.sum()) / tv_denominator
        self._pending_regularization_loss = self.field_l2_weight * l2 + self.field_tv_weight * total_variation
        self._last_field_diagnostics = {
            "field_l2": l2.detach(),
            "field_total_variation": total_variation.detach(),
            **{f"{name}_field_abs_mean": value.detach().abs().mean() for name, value in fields.items()},
            **{f"{name}_field_abs_max": value.detach().abs().amax() for name, value in fields.items()},
        }

    def pop_regularization_loss(self) -> torch.Tensor | None:
        regularization = self._pending_regularization_loss
        self._pending_regularization_loss = None
        return regularization

    @staticmethod
    def _neighbor(values: torch.Tensor, row_offset: int, col_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor = _shift(values, -row_offset, -col_offset, 0.0)
        valid = _shift(torch.ones_like(values), -row_offset, -col_offset, 0.0)
        return neighbor, valid

    def _interpolate_curve(self, curve: torch.Tensor, physical_isi: torch.Tensor) -> torch.Tensor:
        bins = self.ros_isi_bins.to(device=physical_isi.device, dtype=physical_isi.dtype)
        isi = physical_isi.clamp(float(bins[0]), float(bins[-1])).contiguous()
        upper = torch.searchsorted(bins, isi).clamp(1, bins.numel() - 1)
        lower = upper - 1
        lower_bin = bins[lower]
        upper_bin = bins[upper]
        fraction = (isi - lower_bin) / (upper_bin - lower_bin).clamp_min(1e-6)
        lower_value = curve.gather(1, lower)
        upper_value = curve.gather(1, upper)
        return lower_value + fraction * (upper_value - lower_value)

    @property
    def effective_ignition_rate(self) -> torch.Tensor:
        if self.raw_effective_ignition_rate is None:
            raise RuntimeError("A global ignition rate is only defined for interpretable mechanism v1.")
        return _bounded_value(self.raw_effective_ignition_rate, self.min_ignition_rate, self.max_ignition_rate)

    @property
    def bp_hazard_scale(self) -> torch.Tensor:
        if self.raw_bp_hazard_scale is None:
            raise RuntimeError("A global BP hazard scale is only defined for interpretable mechanism v1.")
        return _bounded_value(self.raw_bp_hazard_scale, self.min_bp_hazard_scale, self.max_bp_hazard_scale)

    @property
    def fi_scale(self) -> torch.Tensor:
        return torch.exp(self.behavior_log_scale_limit * torch.tanh(self.raw_fi_log_scale))

    @property
    def ros_scale(self) -> torch.Tensor:
        return torch.exp(self.behavior_log_scale_limit * torch.tanh(self.raw_ros_log_scale))

    @property
    def per_fire_reach_scale(self) -> torch.Tensor:
        if self.raw_per_fire_reach_scale is None:
            raise RuntimeError("A per-fire reach scale is only defined for interpretable mechanism v2.")
        return _bounded_value(
            self.raw_per_fire_reach_scale,
            self.min_per_fire_reach_scale,
            self.max_per_fire_reach_scale,
        )

    @property
    def fi_log_slope(self) -> torch.Tensor:
        if self.raw_fi_log_slope is None:
            raise RuntimeError("A behavior log slope is only defined for interpretable mechanism v2.")
        return _bounded_value(
            self.raw_fi_log_slope,
            self.behavior_log_slope_min,
            self.behavior_log_slope_max,
        )

    @property
    def ros_log_slope(self) -> torch.Tensor:
        if self.raw_ros_log_slope is None:
            raise RuntimeError("A behavior log slope is only defined for interpretable mechanism v2.")
        return _bounded_value(
            self.raw_ros_log_slope,
            self.behavior_log_slope_min,
            self.behavior_log_slope_max,
        )

    def _directional_speed(
        self,
        *,
        x: torch.Tensor,
        fine_base_ros: torch.Tensor,
        fine_burnability: torch.Tensor,
        coarse_burnability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooling_area = float(self.downsample_factor**2)
        safe_ros = fine_base_ros.clamp_min(self.min_ros_m_per_min)
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
        return base_speed * wind_factor * slope_factor, base_speed, wind_speed

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = self.travel_time_propagation.diagnostic_metrics()
        if self.variant == "v1":
            metrics.update(
                {
                    "effective_ignition_rate": float(self.effective_ignition_rate.detach().cpu()),
                    "bp_hazard_scale": float(self.bp_hazard_scale.detach().cpu()),
                    "fi_scale": float(self.fi_scale.detach().cpu()),
                    "ros_scale": float(self.ros_scale.detach().cpu()),
                }
            )
        else:
            metrics.update(
                {
                    "per_fire_reach_scale": float(self.per_fire_reach_scale.detach().cpu()),
                    "fi_log_scale": float(self.fi_scale.detach().cpu()),
                    "fi_log_slope": float(self.fi_log_slope.detach().cpu()),
                    "ros_log_scale": float(self.ros_scale.detach().cpu()),
                    "ros_log_slope": float(self.ros_log_slope.detach().cpu()),
                }
            )
        metrics.update({name: float(value.cpu()) for name, value in self._last_diagnostics.items()})
        metrics.update({name: float(value.cpu()) for name, value in self._last_field_diagnostics.items()})
        return metrics

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if x_auxiliary is None or "fuel_curve" not in x_auxiliary:
            raise ValueError("Interpretable mechanism requires a 'fuel_curve' tensor.")
        self._pending_regularization_loss = None
        fuel_curve = x_auxiliary["fuel_curve"]
        if fuel_curve.shape[1] != 2 * self.curve_length:
            raise ValueError(f"Expected {2 * self.curve_length} concatenated iROS/HFI channels, got {fuel_curve.shape[1]}.")
        ros_curve = fuel_curve[:, : self.curve_length]
        hfi_curve = fuel_curve[:, self.curve_length :]
        physical_isi = x[:, self.isi_index : self.isi_index + 1] * self.isi_std + self.isi_mean
        fine_base_ros = self._interpolate_curve(ros_curve, physical_isi)
        fine_base_hfi = self._interpolate_curve(hfi_curve, physical_isi)
        fine_burnability = (ros_curve.amax(dim=1, keepdim=True) > 0.0).to(x.dtype)
        coarse_burnability = self._pool(fine_burnability)
        coarse_support = (coarse_burnability > 0.0).to(x.dtype)

        directional_speed, base_speed, wind_speed = self._directional_speed(
            x=x,
            fine_base_ros=fine_base_ros,
            fine_burnability=fine_burnability,
            coarse_burnability=coarse_burnability,
        )
        normalized_fire_size = self._weighted_pool(
            x[:, self.scenario_indices],
            fine_burnability,
            coarse_burnability,
        )
        normalized_fire_size = normalized_fire_size.clamp(0.0, 1.0) if self.variant == "v1" else normalized_fire_size.clamp_min(0.0)
        fire_size_log10_quantiles = self.fire_size_log_min + normalized_fire_size * (self.fire_size_log_max - self.fire_size_log_min)

        ignition_mass = x[:, self.ignition_indices].clamp_min(0.0).sum(dim=1, keepdim=True)
        pooling_area = float(self.downsample_factor**2)
        coarse_location_mass = self._pool(ignition_mass * fine_burnability / self.ignition_probability_mass_scale) * pooling_area
        mean_count = None
        coefficient_of_variation = None
        normalized_log_mean = None
        normalized_cv = None
        if self.variant != "v1":
            if self.count_log_mean_index is None or self.count_cv_index is None:
                raise RuntimeError("Count-aware interpretable mechanisms require resolved ignition-count mean and CV channels.")
            normalized_log_mean = self._weighted_pool(
                x[:, self.count_log_mean_index : self.count_log_mean_index + 1],
                fine_burnability,
                coarse_burnability,
            )
            normalized_cv = self._weighted_pool(
                x[:, self.count_cv_index : self.count_cv_index + 1],
                fine_burnability,
                coarse_burnability,
            )
            log_mean_count = self.count_log_mean_minimum + normalized_log_mean * (self.count_log_mean_maximum - self.count_log_mean_minimum)
            mean_count = torch.expm1(log_mean_count).clamp_min(0.0)
            coefficient_of_variation = (self.count_cv_minimum + normalized_cv * (self.count_cv_maximum - self.count_cv_minimum)).clamp_min(
                0.0
            )

        corrected_location_mass = coarse_location_mass
        fire_size_multiplier = None
        reach_scale: torch.Tensor | None = None
        fine_ros_log_correction = None
        fine_consumption_log_correction = None
        if self.variant == "v3":
            if normalized_log_mean is None or normalized_cv is None:
                raise RuntimeError("Interpretable mechanism v3 requires ignition-count features.")
            parameter_features = self._parameter_field_features(
                x=x,
                ros_curve=ros_curve,
                hfi_curve=hfi_curve,
                fine_base_hfi=fine_base_hfi,
                fine_burnability=fine_burnability,
                coarse_burnability=coarse_burnability,
                coarse_support=coarse_support,
                coarse_location_mass=coarse_location_mass,
                normalized_fire_size=normalized_fire_size,
                normalized_log_mean=normalized_log_mean,
                normalized_cv=normalized_cv,
                base_speed=base_speed,
            )
            parameter_fields = self._bounded_parameter_fields(parameter_features, coarse_support)
            self._set_parameter_field_regularization(parameter_fields, coarse_support)
            corrected_location_mass = self._redistribute_location_mass(
                coarse_location_mass,
                parameter_fields["ignition"],
                coarse_support,
            )
            directional_speed = directional_speed * torch.exp(parameter_fields["ros"])
            fire_size_multiplier = torch.exp(parameter_fields["fire_size"])
            reach_scale = self.per_fire_reach_scale * torch.exp(parameter_fields["reach"])
            fine_ros_log_correction = F.interpolate(
                parameter_fields["ros"],
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fine_consumption_log_correction = F.interpolate(
                parameter_fields["consumption"],
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.variant == "v1":
            seed_probability = -torch.expm1(-self.effective_ignition_rate * coarse_location_mass)
            mixture_reach, _ = self.travel_time_propagation(
                seed_probability=seed_probability,
                directional_speed_m_per_min=directional_speed,
                scenario_quantiles=fire_size_log10_quantiles,
                burnability=coarse_burnability,
            )
            coarse_bp = -torch.expm1(-self.bp_hazard_scale * mixture_reach)
        else:
            if not isinstance(self.travel_time_propagation, SourceCohortTravelTimePropagation):
                raise RuntimeError("Count-aware interpretable mechanism has the wrong propagation module.")
            if mean_count is None or coefficient_of_variation is None:
                raise RuntimeError("Count-aware interpretable mechanism did not reconstruct the ignition-count distribution.")
            per_fire_reach, _ = self.travel_time_propagation(
                location_mass=corrected_location_mass,
                directional_speed_m_per_min=directional_speed,
                fire_size_log10_ha_quantiles=fire_size_log10_quantiles,
                burnability=coarse_support,
                fire_size_multiplier=fire_size_multiplier,
            )
            if reach_scale is None:
                reach_scale = self.per_fire_reach_scale
            coarse_bp = count_distribution_burn_probability(
                per_fire_reach=per_fire_reach,
                mean_count=mean_count,
                coefficient_of_variation=coefficient_of_variation,
                per_fire_reach_scale=reach_scale,
            )
            seed_probability = corrected_location_mass
            mixture_reach = per_fire_reach
        fine_bp = (F.interpolate(coarse_bp, size=x.shape[-2:], mode="bilinear", align_corners=False) * fine_burnability).clamp(
            1e-6, 1.0 - 1e-6
        )
        bp_logits = torch.logit(fine_bp)

        if self.variant == "v1":
            fi_log = torch.log1p(fine_base_hfi.clamp_min(0.0) * self.fi_scale)
            ros_log = torch.log1p(fine_base_ros.clamp_min(0.0) * self.ros_scale)
        else:
            corrected_fine_ros = fine_base_ros
            corrected_fine_hfi = fine_base_hfi
            if fine_ros_log_correction is not None and fine_consumption_log_correction is not None:
                corrected_fine_ros = corrected_fine_ros * torch.exp(fine_ros_log_correction)
                corrected_fine_hfi = corrected_fine_hfi * torch.exp(fine_ros_log_correction + fine_consumption_log_correction)
            physical_fi_log = torch.log1p(corrected_fine_hfi.clamp_min(0.0))
            physical_ros_log = torch.log1p(corrected_fine_ros.clamp_min(0.0))
            fi_log = self.fi_scale * physical_fi_log.clamp_min(1e-6).pow(self.fi_log_slope)
            ros_log = self.ros_scale * physical_ros_log.clamp_min(1e-6).pow(self.ros_log_slope)
            fi_log = torch.where(physical_fi_log > 0.0, fi_log, torch.zeros_like(fi_log))
            ros_log = torch.where(physical_ros_log > 0.0, ros_log, torch.zeros_like(ros_log))
        fi_output = (fi_log - self.fi_log_mean) / self.fi_log_std
        ros_output = (ros_log - self.ros_log_mean) / self.ros_log_std
        outputs_by_target = {
            "bp": bp_logits,
            "fi": fi_output,
            "ros": ros_output,
        }
        self._last_diagnostics = {
            "mean_seed_probability": seed_probability.detach().mean(),
            "mean_base_ros": base_speed.detach().mean(),
            "mean_wind_speed_kmh": wind_speed.detach().mean(),
            "mean_per_fire_reach": mixture_reach.detach().mean(),
            "mean_bp": fine_bp.detach().mean(),
        }
        if mean_count is not None:
            self._last_diagnostics["mean_ignition_count"] = mean_count.detach().mean()
        return torch.cat([outputs_by_target[target] for target in self.target_names], dim=1)
