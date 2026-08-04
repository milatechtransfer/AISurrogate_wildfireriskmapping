from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

TargetName = Literal["bp", "fi", "ros"]


@dataclass(frozen=True)
class TargetSpec:
    name: TargetName
    channel_key: str
    output_type: str
    path_method: str
    label: str
    probability_scale: bool = False


TARGET_SPECS: dict[TargetName, TargetSpec] = {
    "bp": TargetSpec(
        name="bp",
        channel_key="bp_out_grid",
        output_type="fire_burn_probability",
        path_method="output_burn_prob",
        label="Burn Probability",
        probability_scale=True,
    ),
    "fi": TargetSpec(
        name="fi",
        channel_key="fi_out_grid",
        output_type="fire_intensity",
        path_method="output_fire_intensity",
        label="Fire Intensity",
    ),
    "ros": TargetSpec(
        name="ros",
        channel_key="ros_out_grid",
        output_type="fire_ros",
        path_method="output_ros",
        label="Rate of Spread",
    ),
}

TARGET_ALIASES = {
    "burn_probability": "bp",
    "fire_burn_probability": "bp",
    "bp_out_grid": "bp",
    "fire_intensity": "fi",
    "fi_out_grid": "fi",
    "rate_of_spread": "ros",
    "fire_ros": "ros",
    "ros_out_grid": "ros",
}


def get_target_spec(target_name: str) -> TargetSpec:
    normalized = target_name.strip().lower()
    normalized = TARGET_ALIASES.get(normalized, normalized)
    if normalized not in TARGET_SPECS:
        supported = sorted(set(TARGET_SPECS) | set(TARGET_ALIASES))
        raise ValueError(f"Unsupported target_name={target_name!r}. Supported values: {supported}")
    return TARGET_SPECS[cast(TargetName, normalized)]


def get_target_specs(target_names: str | list[str]) -> list[TargetSpec]:
    if isinstance(target_names, str):
        target_names = [target_names]
    if not target_names:
        raise ValueError("At least one target name is required.")
    return [get_target_spec(target_name) for target_name in target_names]


def activate_target_predictions(predictions: torch.Tensor, target_specs: Sequence[TargetSpec]) -> torch.Tensor:
    if predictions.ndim < 2 or predictions.shape[1] != len(target_specs):
        raise ValueError(f"Expected predictions with {len(target_specs)} target channels, got shape {tuple(predictions.shape)}.")
    return torch.cat(
        [
            torch.sigmoid(predictions[:, idx : idx + 1]) if target.probability_scale else predictions[:, idx : idx + 1]
            for idx, target in enumerate(target_specs)
        ],
        dim=1,
    )


def split_target_predictions(
    predictions: torch.Tensor,
    target_specs: Sequence[TargetSpec],
) -> dict[TargetName, torch.Tensor]:
    if predictions.ndim < 2 or predictions.shape[1] != len(target_specs):
        raise ValueError(f"Expected predictions with {len(target_specs)} target channels, got shape {tuple(predictions.shape)}.")
    return {target.name: predictions[:, idx : idx + 1] for idx, target in enumerate(target_specs)}
