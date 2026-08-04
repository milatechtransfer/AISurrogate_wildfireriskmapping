import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.unet import BaselineUNet, MultiSourceUNet

BASELINE_UNET_NAMES = {"baseline_unet", "baseline", "unet"}
MULTI_SOURCE_UNET_NAMES = {"multi_source_unet", "multisource_unet", "multi_source", "multisource"}


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

    supported = sorted(BASELINE_UNET_NAMES | MULTI_SOURCE_UNET_NAMES | {"auto"})
    raise ValueError(f"Unknown model architecture '{model_config.architecture}'. Supported values: {supported}")
