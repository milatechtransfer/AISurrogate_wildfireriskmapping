import math

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


class InterpretableMechanisticModel(nn.Module):
    """Mechanism-dominant BP, FI, and ROS model with five named scalar parameters."""

    def __init__(
        self,
        *,
        input_channels: int,
        spatial_input_names: list[str],
        model_config: ModelConfig,
        fuel_curve_input_dim: int,
        target_names: list[str],
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

        self.ignition_indices = self._indices_with_suffix(("ignition_grid_human", "ignition_grid_lightning"))
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
        return _bounded_value(self.raw_effective_ignition_rate, self.min_ignition_rate, self.max_ignition_rate)

    @property
    def bp_hazard_scale(self) -> torch.Tensor:
        return _bounded_value(self.raw_bp_hazard_scale, self.min_bp_hazard_scale, self.max_bp_hazard_scale)

    @property
    def fi_scale(self) -> torch.Tensor:
        return torch.exp(self.behavior_log_scale_limit * torch.tanh(self.raw_fi_log_scale))

    @property
    def ros_scale(self) -> torch.Tensor:
        return torch.exp(self.behavior_log_scale_limit * torch.tanh(self.raw_ros_log_scale))

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
        metrics.update(
            {
                "effective_ignition_rate": float(self.effective_ignition_rate.detach().cpu()),
                "bp_hazard_scale": float(self.bp_hazard_scale.detach().cpu()),
                "fi_scale": float(self.fi_scale.detach().cpu()),
                "ros_scale": float(self.ros_scale.detach().cpu()),
            }
        )
        metrics.update({name: float(value.cpu()) for name, value in self._last_diagnostics.items()})
        return metrics

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if x_auxiliary is None or "fuel_curve" not in x_auxiliary:
            raise ValueError("Interpretable mechanism requires a 'fuel_curve' tensor.")
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
        ).clamp(0.0, 1.0)
        fire_size_log10_quantiles = self.fire_size_log_min + normalized_fire_size * (self.fire_size_log_max - self.fire_size_log_min)

        ignition_mass = x[:, self.ignition_indices].clamp_min(0.0).sum(dim=1, keepdim=True)
        pooling_area = float(self.downsample_factor**2)
        coarse_location_mass = self._pool(ignition_mass * fine_burnability / self.ignition_probability_mass_scale) * pooling_area
        seed_probability = -torch.expm1(-self.effective_ignition_rate * coarse_location_mass)
        mixture_reach, _ = self.travel_time_propagation(
            seed_probability=seed_probability,
            directional_speed_m_per_min=directional_speed,
            scenario_quantiles=fire_size_log10_quantiles,
            burnability=coarse_burnability,
        )
        coarse_bp = -torch.expm1(-self.bp_hazard_scale * mixture_reach)
        fine_bp = (F.interpolate(coarse_bp, size=x.shape[-2:], mode="bilinear", align_corners=False) * fine_burnability).clamp(
            1e-6, 1.0 - 1e-6
        )
        bp_logits = torch.logit(fine_bp)

        physical_fi = fine_base_hfi.clamp_min(0.0) * self.fi_scale
        physical_ros = fine_base_ros.clamp_min(0.0) * self.ros_scale
        fi_output = (torch.log1p(physical_fi) - self.fi_log_mean) / self.fi_log_std
        ros_output = (torch.log1p(physical_ros) - self.ros_log_mean) / self.ros_log_std
        outputs_by_target = {
            "bp": bp_logits,
            "fi": fi_output,
            "ros": ros_output,
        }
        self._last_diagnostics = {
            "mean_seed_probability": seed_probability.detach().mean(),
            "mean_base_ros": base_speed.detach().mean(),
            "mean_wind_speed_kmh": wind_speed.detach().mean(),
            "mean_bp": fine_bp.detach().mean(),
        }
        return torch.cat([outputs_by_target[target] for target in self.target_names], dim=1)
