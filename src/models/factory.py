import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.interpretable_mechanistic import InterpretableMechanisticModel
from src.models.mechanistic_propagation import MechanisticFirePropagationUNet
from src.models.mechanistic_travel_time import MechanisticTravelTimeUNet
from src.models.unet import BaselineUNet, MultiSourceUNet

BASELINE_UNET_NAMES = {"baseline_unet", "baseline", "unet"}
MULTI_SOURCE_UNET_NAMES = {"multi_source_unet", "multisource_unet", "multi_source", "multisource"}
MECHANISTIC_PROPAGATION_NAMES = {"mechanistic_propagation", "fire_propagation", "propagation_unet"}
MECHANISTIC_PROPAGATION_V2_NAMES = {
    "mechanistic_propagation_v2",
    "fire_propagation_v2",
    "propagation_unet_v2",
}
MECHANISTIC_PROPAGATION_V21_NAMES = {
    "mechanistic_propagation_v21",
    "mechanistic_propagation_stable",
    "fire_propagation_v21",
}
MECHANISTIC_PROPAGATION_V3_NAMES = {
    "mechanistic_propagation_v3",
    "fire_propagation_v3",
    "scenario_propagation_v3",
}
MECHANISTIC_TRAVEL_TIME_V4_NAMES = {
    "mechanistic_travel_time_v4",
    "travel_time_propagation_v4",
    "mechanistic_propagation_v4",
}
INTERPRETABLE_MECHANISTIC_NAMES = {
    "interpretable_mechanistic",
    "physical_mechanistic",
}
INTERPRETABLE_MECHANISTIC_V2_NAMES = {
    "interpretable_mechanistic_v2",
    "physical_mechanistic_v2",
}
INTERPRETABLE_MECHANISTIC_V3_NAMES = {
    "interpretable_mechanistic_v3",
    "physical_mechanistic_v3",
    "gray_box_mechanistic",
}


def _normalize_architecture_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def resolve_model_architecture(model_config: ModelConfig) -> str:
    architecture = _normalize_architecture_name(model_config.architecture)
    if architecture == "auto":
        return "multi_source_unet" if "auxiliary" in model_config.input_branches else "baseline_unet"
    return architecture


def build_model(
    *,
    model_config: ModelConfig,
    spatial_input_channels: int | None,
    auxiliary_input_dims: dict[str, int] | None = None,
    fuel_curve_mean: torch.Tensor | None = None,
    fuel_curve_std: torch.Tensor | None = None,
    target_names: list[str] | None = None,
) -> nn.Module:
    if spatial_input_channels is None:
        raise ValueError("spatial_input_channels must be detected before building the model.")

    auxiliary_input_dims = auxiliary_input_dims or {}
    architecture = resolve_model_architecture(model_config)
    auxiliary_requested = "auxiliary" in model_config.input_branches

    # iROS early-fusion params (used by both BaselineUNet and MultiSourceUNet).
    fuel_curve_input_dim = auxiliary_input_dims.get("fuel_curve", 0)
    fuel_curve_embed_dim = model_config.auxiliary_embed_dims.get("fuel_curve", 4) if fuel_curve_input_dim > 0 else 0

    # If iROS is the only auxiliary dim and no explicit multi-source architecture is requested,
    # BaselineUNet can handle it via its built-in iROS encoder.
    if architecture in BASELINE_UNET_NAMES:
        if auxiliary_requested:
            raise ValueError("BaselineUNet cannot consume requested auxiliary features. Use architecture='auto' or 'multi_source_unet'.")
        return BaselineUNet(
            input_channels=spatial_input_channels,
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

    if architecture in MULTI_SOURCE_UNET_NAMES:
        if auxiliary_requested and not auxiliary_input_dims:
            raise ValueError("Config requests auxiliary features, but no auxiliary dimensions were detected.")
        return MultiSourceUNet(
            input_channels=spatial_input_channels,
            num_classes=model_config.num_classes,
            hidden_features=model_config.hidden_features,
            input_branches=model_config.input_branches,
            use_skip_connections=model_config.use_skip_connections,
            use_transpose_conv=model_config.use_transpose_conv,
            use_activation_after_upsampling=model_config.use_activation_after_upsampling,
            auxiliary_input_dims=auxiliary_input_dims,
            auxiliary_hidden_dims=model_config.auxiliary_hidden_dims,
            auxiliary_embed_dims=model_config.auxiliary_embed_dims,
            auxiliary_feature_encoder_poolings=model_config.auxiliary_feature_encoder_poolings,
            use_coordconv=model_config.use_coordconv,
            fuel_curve_mean=fuel_curve_mean,
            fuel_curve_std=fuel_curve_std,
            output_head=model_config.output_head,
            target_names=target_names,
        )

    if architecture in MECHANISTIC_TRAVEL_TIME_V4_NAMES:
        if auxiliary_requested:
            raise ValueError("Mechanistic travel-time propagation supports spatial inputs and early-fused iROS only.")
        if target_names is None:
            raise ValueError("target_names are required for mechanistic travel-time propagation.")
        return MechanisticTravelTimeUNet(
            input_channels=spatial_input_channels,
            spatial_input_names=model_config.spatial_input_names,
            model_config=model_config,
            fuel_curve_input_dim=fuel_curve_input_dim,
            fuel_curve_embed_dim=fuel_curve_embed_dim,
            fuel_curve_mean=fuel_curve_mean,
            fuel_curve_std=fuel_curve_std,
            target_names=target_names,
        )

    if architecture in INTERPRETABLE_MECHANISTIC_NAMES | INTERPRETABLE_MECHANISTIC_V2_NAMES | INTERPRETABLE_MECHANISTIC_V3_NAMES:
        if auxiliary_requested:
            raise ValueError("Interpretable mechanism supports spatial inputs and raw fuel curves only.")
        if target_names is None:
            raise ValueError("target_names are required for the interpretable mechanism.")
        return InterpretableMechanisticModel(
            input_channels=spatial_input_channels,
            spatial_input_names=model_config.spatial_input_names,
            model_config=model_config,
            fuel_curve_input_dim=fuel_curve_input_dim,
            target_names=target_names,
            variant=(
                "v3"
                if architecture in INTERPRETABLE_MECHANISTIC_V3_NAMES
                else "v2"
                if architecture in INTERPRETABLE_MECHANISTIC_V2_NAMES
                else "v1"
            ),
        )

    mechanistic_names = (
        MECHANISTIC_PROPAGATION_NAMES
        | MECHANISTIC_PROPAGATION_V2_NAMES
        | MECHANISTIC_PROPAGATION_V21_NAMES
        | MECHANISTIC_PROPAGATION_V3_NAMES
    )
    if architecture in mechanistic_names:
        if auxiliary_requested:
            raise ValueError("Mechanistic propagation supports spatial inputs and early-fused iROS only.")
        if target_names is None:
            raise ValueError("target_names are required for mechanistic propagation.")
        return MechanisticFirePropagationUNet(
            input_channels=spatial_input_channels,
            spatial_input_names=model_config.spatial_input_names,
            model_config=model_config,
            fuel_curve_input_dim=fuel_curve_input_dim,
            fuel_curve_embed_dim=fuel_curve_embed_dim,
            fuel_curve_mean=fuel_curve_mean,
            fuel_curve_std=fuel_curve_std,
            target_names=target_names,
            variant=(
                "v3"
                if architecture in MECHANISTIC_PROPAGATION_V3_NAMES
                else "v21"
                if architecture in MECHANISTIC_PROPAGATION_V21_NAMES
                else "v2"
                if architecture in MECHANISTIC_PROPAGATION_V2_NAMES
                else "v1"
            ),
        )

    supported = sorted(
        BASELINE_UNET_NAMES
        | MULTI_SOURCE_UNET_NAMES
        | MECHANISTIC_PROPAGATION_NAMES
        | MECHANISTIC_PROPAGATION_V2_NAMES
        | MECHANISTIC_PROPAGATION_V21_NAMES
        | MECHANISTIC_PROPAGATION_V3_NAMES
        | MECHANISTIC_TRAVEL_TIME_V4_NAMES
        | INTERPRETABLE_MECHANISTIC_NAMES
        | INTERPRETABLE_MECHANISTIC_V2_NAMES
        | INTERPRETABLE_MECHANISTIC_V3_NAMES
        | {"auto"}
    )
    raise ValueError(f"Unknown model architecture '{model_config.architecture}'. Supported values: {supported}")
