"""Scratch-trained CNN/physics hybrid: 1/8-resolution source-cohort travel time fused with a U-Net decoder.

This variant differs from the other mechanistic families in three ways: (1) ignition mass is
stratified into a small grid of representative source cohorts and propagated as a factorized
``[B, S, H, W]`` min-travel-time state (no quantile axis), (2) terrain grade is computed once per
edge and shared between the source and destination halves of that edge's travel time, and (3) the
CNN only supplies a bounded local BP calibration and a learned FI/ROS head on top of physical
reference fields, with no learned global ignition or BP scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.config import ModelConfig
from src.datasets.targets import get_target_spec
from src.models.encoders import FuelCurveEncoder
from src.models.interpretable_mechanistic import count_distribution_burn_probability
from src.models.mechanistic_propagation import DecoderBlock, _conv_block, _group_count, _quantile_mass_weights
from src.models.mechanistic_travel_time import _shift

# Sentinel used for "unreached" travel-time states and invalid (off-grid or non-burnable) edges.
_LARGE_TRAVEL_HOURS = 1.0e4


def _inverse_bounded_sigmoid(value: float, minimum: float, maximum: float) -> torch.Tensor:
    if not minimum < value < maximum:
        raise ValueError(f"Expected {minimum} < {value} < {maximum}.")
    fraction = (value - minimum) / (maximum - minimum)
    return torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float32)


def _bounded_value(raw_value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return minimum + (maximum - minimum) * torch.sigmoid(raw_value)


def _split_curve_stat(stat: torch.Tensor | None, curve_length: int, default_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    if stat is None:
        filled = torch.full((1,), default_value)
        return filled, filled
    if stat.numel() == 1:
        return stat, stat
    if stat.numel() == 2:
        return stat[:1], stat[1:]
    if stat.numel() != 2 * curve_length:
        raise ValueError(f"Expected fuel-curve statistics of length 1 or {2 * curve_length}, got {stat.numel()}.")
    return stat[:curve_length], stat[curve_length:]


class MechanisticHybridV22(nn.Module):
    """1/8-resolution CNN encoder-decoder fused with a source-cohort physical travel-time model."""

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
        normalized_targets = [get_target_spec(name).name for name in target_names]
        if len(normalized_targets) != 3 or set(normalized_targets) != {"bp", "fi", "ros"}:
            raise ValueError(f"MechanisticHybridV22 requires bp, fi, and ros targets, got {normalized_targets}.")
        if model_config.output_head != "bp_behavior":
            raise ValueError("MechanisticHybridV22 requires output_head='bp_behavior'.")
        if len(spatial_input_names) != input_channels:
            raise ValueError(
                f"Expected {input_channels} semantic spatial input names, got {len(spatial_input_names)}: {spatial_input_names}."
            )
        curve_length = len(model_config.propagation_isi_bins)
        if fuel_curve_input_dim != 2 * curve_length:
            raise ValueError(
                "MechanisticHybridV22 requires concatenated iROS and HFI curves: "
                f"expected {2 * curve_length} channels, got {fuel_curve_input_dim}."
            )
        if fuel_curve_embed_dim < 2:
            raise ValueError("MechanisticHybridV22 requires fuel_curve_embed_dim >= 2 to embed iROS and HFI separately.")

        self.spatial_input_names = list(spatial_input_names)
        self.target_names = normalized_targets
        self.curve_length = curve_length
        if model_config.propagation_downsample_factor != 8:
            raise ValueError("MechanisticHybridV22 requires propagation_downsample_factor=8.")
        self.downsample_factor = model_config.propagation_downsample_factor

        self.ignition_indices = self._indices_with_suffix(("ignition_grid_human", "ignition_grid_lightning"))
        self.count_log_mean_index = self._single_index("NORM_LOG1P_IGNITION_COUNT_MEAN")
        self.count_cv_index = self._single_index("NORM_IGNITION_COUNT_CV")
        self.elevation_index = self._single_index("elevation_grid")
        self.isi_index = self._single_index("InitialSpreadIndex")
        self.wind_x_index = self._single_index("wind_x")
        self.wind_y_index = self._single_index("wind_y")

        scenario_prefix = "spatialized_fire_size/NORM_LOG_SIZE_HA_q"
        scenario_channels: list[tuple[float, int]] = []
        for index, name in enumerate(self.spatial_input_names):
            if name.startswith(scenario_prefix):
                label = name.removeprefix(scenario_prefix)
                if not label.isdigit():
                    raise ValueError(f"Could not parse fire-size percentile from channel {name!r}.")
                scenario_channels.append((float(label) / 100.0, index))
        scenario_channels.sort()
        if not scenario_channels:
            raise ValueError(f"MechanisticHybridV22 requires channels prefixed by {scenario_prefix!r}.")
        self.scenario_quantile_levels = [level for level, _ in scenario_channels]
        self.scenario_indices = [index for _, index in scenario_channels]

        # Physical / normalization configuration (all fields below already exist on ModelConfig).
        self.fire_size_log_min = model_config.propagation_fire_size_log_min
        self.fire_size_log_max = model_config.propagation_fire_size_log_max
        self.ignition_probability_mass_scale = model_config.propagation_ignition_probability_mass_scale
        self.count_log_mean_minimum = model_config.propagation_count_log_mean_min
        self.count_log_mean_maximum = model_config.propagation_count_log_mean_max
        self.count_cv_minimum = model_config.propagation_count_cv_min
        self.count_cv_maximum = model_config.propagation_count_cv_max
        self.isi_mean = model_config.propagation_isi_mean
        self.isi_std = model_config.propagation_isi_std
        self.wind_x_mean = model_config.propagation_wind_x_mean
        self.wind_x_std = model_config.propagation_wind_x_std
        self.wind_y_mean = model_config.propagation_wind_y_mean
        self.wind_y_std = model_config.propagation_wind_y_std
        self.wind_direction_is_from = model_config.propagation_wind_direction_is_from
        self.wind_half_saturation_kmh = model_config.propagation_wind_half_saturation_kmh
        self.elevation_min_m = model_config.propagation_elevation_min_m
        self.elevation_max_m = model_config.propagation_elevation_max_m
        self.slope_coefficient = model_config.propagation_slope_coefficient
        self.max_abs_grade = model_config.propagation_max_abs_grade
        self.min_ros_m_per_min = model_config.propagation_min_ros_m_per_min
        self.speed_correction_log_limit = model_config.propagation_speed_correction_log_limit
        self.max_local_log_calibration = model_config.propagation_max_local_log_calibration
        self.bp_logit_eps = model_config.propagation_logit_eps
        self.budget_temperature_hours = model_config.propagation_budget_temperature_hours
        self.propagation_steps = model_config.propagation_steps
        self.source_grid_size = model_config.interpretable_source_grid_size
        self.fi_log_mean = model_config.interpretable_fi_log_mean
        self.fi_log_std = model_config.interpretable_fi_log_std
        self.ros_log_mean = model_config.interpretable_ros_log_mean
        self.ros_log_std = model_config.interpretable_ros_log_std

        self.wind_anisotropy_min = model_config.propagation_wind_anisotropy_min
        self.wind_anisotropy_max = model_config.propagation_wind_anisotropy_max
        self.quantile_kl_weight = model_config.propagation_quantile_kl_weight
        self.behavior_consistency_weight = model_config.propagation_behavior_consistency_weight

        ros_mean, hfi_mean = _split_curve_stat(fuel_curve_mean, curve_length, 0.0)
        ros_std, hfi_std = _split_curve_stat(fuel_curve_std, curve_length, 1.0)
        embed_dim_ros = fuel_curve_embed_dim - fuel_curve_embed_dim // 2
        embed_dim_hfi = fuel_curve_embed_dim // 2
        self.fuel_curve_encoder_ros = FuelCurveEncoder(
            curve_mean=ros_mean, curve_std=ros_std, in_channels=curve_length, embed_dim=embed_dim_ros
        )
        self.fuel_curve_encoder_hfi = FuelCurveEncoder(
            curve_mean=hfi_mean, curve_std=hfi_std, in_channels=curve_length, embed_dim=embed_dim_hfi
        )

        base_channels = model_config.propagation_base_channels
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]
        self.full_encoder = _conv_block(input_channels + fuel_curve_embed_dim, channels[0])
        self.half_encoder = _conv_block(channels[0], channels[1], stride=2)
        self.quarter_encoder = _conv_block(channels[1], channels[2], stride=2)
        self.eighth_encoder = _conv_block(channels[2], channels[3], stride=2)
        propagation_channels = channels[3]

        physics_context_channels = 6
        speed_hidden = max(8, propagation_channels // 2)
        speed_correction_output = nn.Conv2d(speed_hidden, 8, kernel_size=1)
        self.speed_correction = nn.Sequential(
            nn.Conv2d(propagation_channels + physics_context_channels, speed_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(speed_hidden), speed_hidden),
            nn.SiLU(),
            speed_correction_output,
        )
        nn.init.zeros_(speed_correction_output.weight)
        if speed_correction_output.bias is not None:
            nn.init.zeros_(speed_correction_output.bias)

        self.mechanistic_fusion = _conv_block(propagation_channels + 3, propagation_channels)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(propagation_channels, channels[2], channels[2]),
                DecoderBlock(channels[2], channels[1], channels[1]),
                DecoderBlock(channels[1], channels[0], channels[0]),
            ]
        )

        self.bp_local_calibration = nn.Conv2d(channels[0], 1, kernel_size=1)
        nn.init.zeros_(self.bp_local_calibration.weight)
        if self.bp_local_calibration.bias is not None:
            nn.init.zeros_(self.bp_local_calibration.bias)
        self.behavior_head = nn.Conv2d(channels[0] + 3, 2, kernel_size=1)

        self.raw_per_fire_reach_scale = nn.Parameter(
            _inverse_bounded_sigmoid(
                model_config.interpretable_initial_per_fire_reach_scale,
                model_config.interpretable_min_per_fire_reach_scale,
                model_config.interpretable_max_per_fire_reach_scale,
            )
        )
        self.per_fire_reach_scale_min = model_config.interpretable_min_per_fire_reach_scale
        self.per_fire_reach_scale_max = model_config.interpretable_max_per_fire_reach_scale
        self.raw_wind_anisotropy = nn.Parameter(
            _inverse_bounded_sigmoid(model_config.propagation_wind_anisotropy, self.wind_anisotropy_min, self.wind_anisotropy_max)
        )

        quantile_prior = _quantile_mass_weights(self.scenario_quantile_levels).view(-1)
        self.quantile_prior: torch.Tensor
        self.register_buffer("quantile_prior", quantile_prior, persistent=False)
        self.raw_quantile_logits = nn.Parameter(torch.log(quantile_prior.clamp_min(1e-8)))

        self.ros_isi_bins: torch.Tensor
        self.register_buffer("ros_isi_bins", torch.tensor(model_config.propagation_isi_bins, dtype=torch.float32))
        direction_xy = []
        for row_offset, col_offset in self.DIRECTIONS:
            norm = math.sqrt(float(row_offset * row_offset + col_offset * col_offset))
            direction_xy.append((col_offset / norm, -row_offset / norm))
        self.direction_xy: torch.Tensor
        self.register_buffer("direction_xy", torch.tensor(direction_xy, dtype=torch.float32).view(1, 8, 2, 1, 1))
        self.coarse_cell_size_m = model_config.propagation_cell_size_m * self.downsample_factor
        edge_lengths = [
            self.coarse_cell_size_m * (math.sqrt(2.0) if row_offset and col_offset else 1.0) for row_offset, col_offset in self.DIRECTIONS
        ]
        self.edge_lengths_m: torch.Tensor
        self.register_buffer("edge_lengths_m", torch.tensor(edge_lengths, dtype=torch.float32).view(1, 8, 1, 1), persistent=False)
        self.coarse_cell_area_ha = (self.coarse_cell_size_m**2) / 10_000.0
        self._quantile_channel_labels = [f"q{round(level * 100)}" for level in self.scenario_quantile_levels]

        self._pending_regularization_loss: torch.Tensor | None = None
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_cap_hit_rate = torch.tensor(0.0)

    # -- channel resolution -------------------------------------------------------------------
    def _indices_with_suffix(self, suffixes: tuple[str, ...]) -> list[int]:
        resolved = []
        for suffix in suffixes:
            indices = [index for index, name in enumerate(self.spatial_input_names) if name.endswith(f"/{suffix}")]
            if len(indices) != 1:
                raise ValueError(f"Expected one channel ending in /{suffix}, resolved {indices} from {self.spatial_input_names}.")
            resolved.append(indices[0])
        return resolved

    def _single_index(self, suffix: str) -> int:
        return self._indices_with_suffix((suffix,))[0]

    # -- bounded / learned scalars -------------------------------------------------------------
    @property
    def wind_anisotropy(self) -> torch.Tensor:
        return _bounded_value(self.raw_wind_anisotropy, self.wind_anisotropy_min, self.wind_anisotropy_max)

    @property
    def per_fire_reach_scale(self) -> torch.Tensor:
        return _bounded_value(self.raw_per_fire_reach_scale, self.per_fire_reach_scale_min, self.per_fire_reach_scale_max)

    @property
    def quantile_weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_quantile_logits, dim=0)

    def _quantile_kl_divergence(self) -> torch.Tensor:
        weights = self.quantile_weights
        prior = self.quantile_prior.to(dtype=weights.dtype)
        return (weights * (torch.log(weights.clamp_min(1e-8)) - torch.log(prior.clamp_min(1e-8)))).sum()

    # -- pooling helpers ------------------------------------------------------------------------
    def _pool(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2] % self.downsample_factor or values.shape[-1] % self.downsample_factor:
            raise ValueError(f"Input shape {values.shape[-2:]} must be divisible by {self.downsample_factor}.")
        return F.avg_pool2d(values, kernel_size=self.downsample_factor, stride=self.downsample_factor)

    def _weighted_pool(self, values: torch.Tensor, fine_weights: torch.Tensor, coarse_weights: torch.Tensor) -> torch.Tensor:
        return self._pool(values * fine_weights) / coarse_weights.clamp_min(1e-6)

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

    # -- source cohorts ---------------------------------------------------------------------------
    def _source_cohorts(self, location: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stratify coarse ignition mass into source_grid_size**2 tiles.

        Each tile's representative source is the nearest positive cell to that tile's mass
        centroid (a discrete index is acceptable since the input mass is fixed), and its weight is
        the tile's exact mass. Weights are only ever scaled down, and only as a numerical guard if
        the total mass exceeds one.
        """
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
        raw_total_mass = weights.sum(dim=1, keepdim=True)
        weights = weights * raw_total_mass.clamp(0.0, 1.0) / raw_total_mass.clamp_min(1e-12)
        return indices, weights

    # -- edge-consistent terrain and directional speed ---------------------------------------------
    def _incoming_edge_hours(
        self,
        *,
        elevation_m: torch.Tensor,
        base_speed: torch.Tensor,
        wind_factor: torch.Tensor,
        log_speed_correction: torch.Tensor,
        burnability: torch.Tensor,
    ) -> torch.Tensor:
        """Directly compute the travel hours for each of the 8 edges arriving at every cell.

        For a given edge, the terrain grade is computed exactly once (from the two endpoint
        elevations) and reused identically for both the source-side and destination-side speed
        terms, so the two halves of the same physical edge never see inconsistent slopes.
        """
        destination_burnable = burnability > 0.0
        edge_hours = []
        for direction_index, (row_offset, col_offset) in enumerate(self.DIRECTIONS):
            edge_length = self.edge_lengths_m[:, direction_index : direction_index + 1].to(elevation_m.dtype)

            source_elevation = _shift(elevation_m, row_offset, col_offset, 0.0)
            source_valid = _shift(torch.ones_like(elevation_m), row_offset, col_offset, 0.0) > 0.0
            grade = ((elevation_m - source_elevation) / edge_length).clamp(-self.max_abs_grade, self.max_abs_grade)
            slope_factor = torch.exp(self.slope_coefficient * grade)

            destination_wind_factor = wind_factor[:, direction_index : direction_index + 1]
            source_wind_factor = _shift(destination_wind_factor, row_offset, col_offset, 1.0)
            destination_correction = log_speed_correction[:, direction_index : direction_index + 1]
            source_correction = _shift(destination_correction, row_offset, col_offset, 0.0)
            source_base_speed = _shift(base_speed, row_offset, col_offset, self.min_ros_m_per_min)

            source_speed = (source_base_speed * source_wind_factor * slope_factor * torch.exp(source_correction)).clamp_min(
                self.min_ros_m_per_min
            )
            destination_speed = (base_speed * destination_wind_factor * slope_factor * torch.exp(destination_correction)).clamp_min(
                self.min_ros_m_per_min
            )
            hours = edge_length / 120.0 * (source_speed.reciprocal() + destination_speed.reciprocal())

            source_burnable = _shift(burnability, row_offset, col_offset, 0.0) > 0.0
            valid = source_valid & source_burnable & destination_burnable
            edge_hours.append(torch.where(valid, hours, hours.new_full((), _LARGE_TRAVEL_HOURS)))
        return torch.cat(edge_hours, dim=1)

    def _min_travel_time(
        self,
        *,
        source_indices: torch.Tensor,
        incoming_edge_hours: torch.Tensor,
        burnability: torch.Tensor,
    ) -> torch.Tensor:
        """Iterate min-plus relaxation of a factorized [B, S, H, W] travel-time state."""
        batch_size, source_count = source_indices.shape
        _, _, height, width = burnability.shape
        travel_time = incoming_edge_hours.new_full((batch_size, source_count, height * width), _LARGE_TRAVEL_HOURS)
        travel_time = travel_time.scatter(
            dim=2,
            index=source_indices.unsqueeze(-1),
            src=torch.zeros_like(source_indices, dtype=travel_time.dtype).unsqueeze(-1),
        )
        travel_time = travel_time.view(batch_size, source_count, height, width)
        destination_burnable = burnability > 0.0
        cap_hit_rate = travel_time.new_zeros(())

        for step in range(self.propagation_steps):
            candidates = []
            for direction_index, (row_offset, col_offset) in enumerate(self.DIRECTIONS):
                source_time = _shift(travel_time, row_offset, col_offset, _LARGE_TRAVEL_HOURS)
                candidates.append(source_time + incoming_edge_hours[:, direction_index : direction_index + 1])
            best_candidate = torch.stack(candidates, dim=2).amin(dim=2)
            if step == self.propagation_steps - 1:
                improving = (best_candidate < travel_time - 1e-4) & destination_burnable
                denominator = destination_burnable.expand_as(improving).sum().clamp_min(1)
                cap_hit_rate = improving.sum().to(travel_time.dtype) / denominator
            travel_time = torch.minimum(travel_time, best_candidate)
        self._last_cap_hit_rate = cap_hit_rate.detach()
        return travel_time

    # -- small-fire coarse-cell bias correction -----------------------------------------------------
    def _apply_area_bias_correction(self, reach: torch.Tensor, target_area_ha: torch.Tensor) -> torch.Tensor:
        """Scale reach down (never up) so its implied burnable area never exceeds the target fire area."""
        raw_area_ha = reach.sum(dim=(-2, -1)) * self.coarse_cell_area_ha
        scale = torch.where(
            raw_area_ha > 1e-9,
            (target_area_ha / raw_area_ha.clamp_min(1e-9)).clamp(max=1.0),
            torch.ones_like(raw_area_ha),
        )
        return reach * scale.unsqueeze(-1).unsqueeze(-1)

    # -- BP hazard calibration ------------------------------------------------------------------------
    @staticmethod
    def _bp_hazard_correction(physical_bp: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Multiplicative hazard correction; a zero physical BP field stays exactly zero."""
        hazard = -torch.log1p(-physical_bp.clamp(0.0, 1.0 - 1e-7))
        return -torch.expm1(-hazard * torch.exp(delta))

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = {
            "wind_anisotropy": float(self.wind_anisotropy.detach().cpu()),
            "per_fire_reach_scale": float(self.per_fire_reach_scale.detach().cpu()),
            "cap_hit_rate": float(self._last_cap_hit_rate.detach().cpu()),
        }
        for label, weight in zip(self._quantile_channel_labels, self.quantile_weights.detach().cpu().tolist(), strict=True):
            metrics[f"quantile_weight_{label}"] = float(weight)
        metrics.update({name: float(value.detach().cpu()) for name, value in self._last_diagnostics.items()})
        return metrics

    def pop_regularization_loss(self) -> torch.Tensor | None:
        regularization = self._pending_regularization_loss
        self._pending_regularization_loss = None
        return regularization

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if x_auxiliary is None or "fuel_curve" not in x_auxiliary:
            raise ValueError("MechanisticHybridV22 requires a 'fuel_curve' tensor.")
        self._pending_regularization_loss = None
        fuel_curve = x_auxiliary["fuel_curve"]
        if fuel_curve.shape[1] != 2 * self.curve_length:
            raise ValueError(f"Expected {2 * self.curve_length} concatenated iROS/HFI channels, got {fuel_curve.shape[1]}.")
        ros_curve = fuel_curve[:, : self.curve_length]
        hfi_curve = fuel_curve[:, self.curve_length :]

        fuel_embedding = torch.cat([self.fuel_curve_encoder_ros(ros_curve), self.fuel_curve_encoder_hfi(hfi_curve)], dim=1)
        full = self.full_encoder(torch.cat([x, fuel_embedding], dim=1))
        half = self.half_encoder(full)
        quarter = self.quarter_encoder(half)
        eighth = self.eighth_encoder(quarter)
        coarse_features = eighth
        coarse_size = coarse_features.shape[-2:]

        physical_isi = x[:, self.isi_index : self.isi_index + 1] * self.isi_std + self.isi_mean
        fine_base_ros = self._interpolate_curve(ros_curve, physical_isi)
        fine_base_hfi = self._interpolate_curve(hfi_curve, physical_isi)
        fine_burnability = (ros_curve.amax(dim=1, keepdim=True) > 0.0).to(x.dtype)
        coarse_burnability = self._pool(fine_burnability)
        coarse_support = (coarse_burnability > 0.0).to(x.dtype)
        if coarse_size != coarse_burnability.shape[-2:]:
            raise RuntimeError("Encoder and physical pooling resolutions disagree; check input shape divisibility.")

        safe_fine_ros = fine_base_ros.clamp_min(self.min_ros_m_per_min)
        pooling_area = float(self.downsample_factor**2)
        burnable_count = self._pool(fine_burnability) * pooling_area
        inverse_speed_sum = self._pool(fine_burnability / safe_fine_ros) * pooling_area
        coarse_base_speed = (burnable_count / inverse_speed_sum.clamp_min(1e-6)) * coarse_burnability
        coarse_base_hfi = self._weighted_pool(fine_base_hfi, fine_burnability, coarse_burnability)

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

        ignition_mass = x[:, self.ignition_indices].clamp_min(0.0).sum(dim=1, keepdim=True)
        coarse_location_mass = self._pool(ignition_mass * fine_burnability / self.ignition_probability_mass_scale) * pooling_area
        context_mean_mass = coarse_location_mass.mean(dim=(-2, -1), keepdim=True)
        relative_ignition_density = torch.log1p(coarse_location_mass / context_mean_mass.clamp_min(1e-12)).clamp_max(4.0) / 4.0

        physics_context = torch.cat(
            [
                torch.log1p(coarse_base_speed.clamp_min(0.0)) / math.log(201.0),
                coarse_wind_x / 20.0,
                coarse_wind_y / 20.0,
                self._weighted_pool(elevation_norm, fine_burnability, coarse_burnability),
                coarse_burnability,
                relative_ignition_density,
            ],
            dim=1,
        )
        raw_speed_correction = self.speed_correction(torch.cat([coarse_features, physics_context], dim=1))
        log_speed_correction = self.speed_correction_log_limit * torch.tanh(raw_speed_correction)

        incoming_edge_hours = self._incoming_edge_hours(
            elevation_m=coarse_elevation_m,
            base_speed=coarse_base_speed,
            wind_factor=wind_factor,
            log_speed_correction=log_speed_correction,
            burnability=coarse_support,
        )

        location = coarse_location_mass.clamp(0.0, 1.0) * coarse_support
        source_indices, source_weights = self._source_cohorts(location)
        travel_time = self._min_travel_time(
            source_indices=source_indices,
            incoming_edge_hours=incoming_edge_hours,
            burnability=coarse_support,
        )

        height, width = coarse_size
        flat_base_speed = coarse_base_speed.detach().reshape(x.shape[0], 1, height * width)
        reference_speed = flat_base_speed.gather(2, source_indices.unsqueeze(1)).squeeze(1)  # (B, S)

        normalized_fire_size = self._weighted_pool(x[:, self.scenario_indices], fine_burnability, coarse_burnability).clamp_min(0.0)
        flat_fire_size = normalized_fire_size.reshape(x.shape[0], -1, height * width)
        gathered_fire_size = flat_fire_size.gather(2, source_indices.unsqueeze(1).expand(-1, flat_fire_size.shape[1], -1))
        gathered_fire_size = gathered_fire_size.permute(0, 2, 1)  # (B, S, Q)

        fire_size_log10_ha = self.fire_size_log_min + gathered_fire_size * (self.fire_size_log_max - self.fire_size_log_min)
        fire_size_ha = (torch.pow(10.0, fire_size_log10_ha) - 1.0).clamp_min(0.0)
        equivalent_radius_m = torch.sqrt(fire_size_ha * 10_000.0 / math.pi + 1e-6) - 1e-3
        budget_hours = equivalent_radius_m / (60.0 * reference_speed.unsqueeze(-1).clamp_min(self.min_ros_m_per_min))

        reach = torch.sigmoid((budget_hours.unsqueeze(-1).unsqueeze(-1) - travel_time.unsqueeze(2)) / self.budget_temperature_hours)
        reach = reach * coarse_support.unsqueeze(2)
        reach = self._apply_area_bias_correction(reach, fire_size_ha)

        quantile_weights = self.quantile_weights.to(dtype=reach.dtype)
        source_weighted_reach = (reach * source_weights.view(x.shape[0], -1, 1, 1, 1)).sum(dim=1)
        per_fire_reach = (source_weighted_reach * quantile_weights.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

        normalized_log_mean = self._weighted_pool(
            x[:, self.count_log_mean_index : self.count_log_mean_index + 1], fine_burnability, coarse_burnability
        )
        normalized_cv = self._weighted_pool(x[:, self.count_cv_index : self.count_cv_index + 1], fine_burnability, coarse_burnability)
        log_mean_count = self.count_log_mean_minimum + normalized_log_mean * (self.count_log_mean_maximum - self.count_log_mean_minimum)
        mean_count = torch.expm1(log_mean_count).clamp_min(0.0)
        coefficient_of_variation = (self.count_cv_minimum + normalized_cv * (self.count_cv_maximum - self.count_cv_minimum)).clamp_min(0.0)

        coarse_bp = count_distribution_burn_probability(
            per_fire_reach=per_fire_reach,
            mean_count=mean_count,
            coefficient_of_variation=coefficient_of_variation,
            per_fire_reach_scale=self.per_fire_reach_scale,
        )

        mean_log_speed_correction_coarse = log_speed_correction.mean(dim=1, keepdim=True)
        decoded = self.mechanistic_fusion(torch.cat([coarse_features, per_fire_reach, coarse_bp, mean_log_speed_correction_coarse], dim=1))
        skips = [quarter, half, full]
        for decoder_block, skip in zip(self.decoder_blocks, skips, strict=True):
            decoded = decoder_block(decoded, skip)

        full_size = x.shape[-2:]
        physical_bp_full = F.interpolate(coarse_bp, size=full_size, mode="bilinear", align_corners=False)
        physical_bp_full = (physical_bp_full * fine_burnability).clamp(0.0, 1.0 - 1e-7)
        delta = self.max_local_log_calibration * torch.tanh(self.bp_local_calibration(decoded))
        bp_probability = self._bp_hazard_correction(physical_bp_full, delta)
        bp_logits = torch.logit(bp_probability, eps=self.bp_logit_eps)

        physical_hfi_full = F.interpolate(coarse_base_hfi, size=full_size, mode="bilinear", align_corners=False)
        physical_ros_full = F.interpolate(coarse_base_speed, size=full_size, mode="bilinear", align_corners=False)
        mean_log_speed_correction_full = F.interpolate(
            mean_log_speed_correction_coarse, size=full_size, mode="bilinear", align_corners=False
        )
        log1p_hfi_full = torch.log1p(physical_hfi_full.clamp_min(0.0))
        log1p_ros_full = torch.log1p(physical_ros_full.clamp_min(0.0))
        behavior = self.behavior_head(torch.cat([decoded, log1p_hfi_full, log1p_ros_full, mean_log_speed_correction_full], dim=1))
        fi_output = behavior[:, 0:1]
        ros_output = behavior[:, 1:2]

        physical_fi_log_target = (log1p_hfi_full - self.fi_log_mean) / self.fi_log_std
        physical_ros_log_target = (log1p_ros_full - self.ros_log_mean) / self.ros_log_std
        consistency_fi = F.mse_loss(fi_output, physical_fi_log_target)
        # The ROS "residual" is destandardized back into raw log-ratio units before comparing it to
        # the CNN's own mean log speed correction, since the two are not on the same scale.
        ros_residual = (ros_output - physical_ros_log_target) * self.ros_log_std
        consistency_ros = F.mse_loss(ros_residual, mean_log_speed_correction_full)
        kl_divergence = self._quantile_kl_divergence()
        self._pending_regularization_loss = self.behavior_consistency_weight * (consistency_fi + consistency_ros) + (
            self.quantile_kl_weight * kl_divergence
        )

        outputs_by_target = {"bp": bp_logits, "fi": fi_output, "ros": ros_output}
        self._last_diagnostics = {
            "mean_physical_reach": reach.detach().mean(),
            "mean_per_fire_reach": per_fire_reach.detach().mean(),
            "mean_bp": coarse_bp.detach().mean(),
            "mean_abs_log_speed_correction": log_speed_correction.detach().abs().mean(),
            "quantile_kl": kl_divergence.detach(),
        }
        return torch.cat([outputs_by_target[name] for name in self.target_names], dim=1)
