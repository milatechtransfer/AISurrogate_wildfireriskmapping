from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.bottlenecks import MultiSourceBottleneck
from src.models.decoders import BaselineDecoder
from src.models.encoders import (
    BaselineEncoder,
    FuelCurveEncoder,
    TabularFeatureEncoder,
    WindFeatureEncoderMixer,
    WindFeatureEncoderSpatial,
    append_coord_channels,
)
from src.models.heads import BurnProbabilityBehaviorHead
from src.models.utils import double_conv_block


class UNetBase(nn.Module, ABC):
    """Abstract base class for UNet models."""

    def __init__(self):
        super().__init__()

        self.input_channels: int
        self.num_classes: int
        self.hidden_features: list[int] | None
        self.input_branches: list | None
        self.use_skip_connections: bool
        self.use_transpose_conv: bool
        self.use_activation_after_upsampling: bool
        self.use_coordconv: bool

        self.encoder: nn.Module
        self.bottleneck: nn.Module
        self.decoder: nn.Module
        self.out_conv: nn.Conv2d | None
        self.multi_output_head: nn.Module | None

    @abstractmethod
    def build_encoder(self) -> nn.Module | nn.ModuleDict:
        """Return an EncoderBase-derived module (or any module whose forward returns (bottleneck, skips))."""
        raise NotImplementedError

    @abstractmethod
    def build_bottleneck(self) -> nn.Module:
        """Return the bottleneck module (nn.Module) applied to the deepest feature map."""
        raise NotImplementedError

    @abstractmethod
    def build_decoder(self) -> nn.Module:
        """Return a DecoderBase-derived module (or any module that accepts (bottleneck, skips) -> features)."""
        raise NotImplementedError

    def _build_components(self) -> None:
        """
        Top-level hook that calls the build methods.
        Subclasses may override build_* methods or override _build_components itself.
        """
        self.encoder = self.build_encoder()
        self.bottleneck = self.build_bottleneck()
        self.decoder = self.build_decoder()

    def _build_output_layers(self, feature_channels: int, output_head: str, target_names: list[str] | None) -> None:
        if output_head == "shared":
            self.out_conv = nn.Conv2d(feature_channels, self.num_classes, kernel_size=1)
            self.multi_output_head = None
            return
        if output_head == "bp_behavior":
            if target_names is None:
                raise ValueError("target_names are required for output_head='bp_behavior'.")
            if self.num_classes != len(target_names):
                raise ValueError(f"num_classes={self.num_classes} must match configured targets {target_names}.")
            self.out_conv = None
            self.multi_output_head = BurnProbabilityBehaviorHead(feature_channels, target_names)
            return
        raise ValueError(f"Unsupported output_head={output_head!r}.")

    def _project_output(self, features: torch.Tensor) -> torch.Tensor:
        if self.multi_output_head is not None:
            return self.multi_output_head(features)
        if self.out_conv is None:
            raise RuntimeError("UNet output layers are not configured.")
        return self.out_conv(features)

    @abstractmethod
    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        raise NotImplementedError


class BaselineUNet(UNetBase):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list[int] | None = None,
        input_branches: list | None = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
        use_coordconv: bool = False,
        fuel_curve_input_dim: int = 0,
        fuel_curve_embed_dim: int = 4,
        fuel_curve_mean: torch.Tensor | None = None,
        fuel_curve_std: torch.Tensor | None = None,
        output_head: str = "shared",
        target_names: list[str] | None = None,
    ):
        super().__init__()
        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
        if input_branches is None:
            input_branches = ["spatial"]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.use_coordconv = use_coordconv
        self.input_branches = input_branches
        self.fuel_curve_input_dim = fuel_curve_input_dim
        self.fuel_curve_embed_dim = fuel_curve_embed_dim if fuel_curve_input_dim > 0 else 0
        # iROS embedding is concatenated with spatial input before the encoder.
        self._effective_spatial_in = self.input_channels + self.fuel_curve_embed_dim
        self.fuel_curve_encoder: FuelCurveEncoder | None
        if self.fuel_curve_input_dim > 0:
            # Real stats (computed in GridSource) are always a single global scalar
            # (shape (1,)), regardless of fuel_curve_input_dim (number of ISI bins).
            # Default to a scalar too so eval-time construction (no train_dataset,
            # e.g. evaluate_hexels.py) matches the checkpoint's buffer shape before
            # load_state_dict overwrites it with the real values.
            _mean = fuel_curve_mean if fuel_curve_mean is not None else torch.zeros(1)
            _std = fuel_curve_std if fuel_curve_std is not None else torch.ones(1)
            self.fuel_curve_encoder = FuelCurveEncoder(
                curve_mean=_mean, curve_std=_std, in_channels=self.fuel_curve_input_dim, embed_dim=self.fuel_curve_embed_dim
            )
        else:
            self.fuel_curve_encoder = None
        self._build_components()
        self._build_output_layers(self.hidden_features[0], output_head, target_names)

    def build_encoder(self) -> nn.Module:
        encoder = BaselineEncoder(
            in_channels=self._effective_spatial_in, hidden_features=self.hidden_features, use_coordconv=self.use_coordconv
        )
        return encoder

    def build_bottleneck(self) -> nn.Module:
        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        in_channels = self.hidden_features[-1] + 2 if self.use_coordconv else self.hidden_features[-1]
        return double_conv_block(in_channels, self.hidden_features[-1] * 2)

    def build_decoder(self) -> nn.Module:
        decoder = BaselineDecoder(
            hidden_features=self.hidden_features,
            use_skip_connections=self.use_skip_connections,
            use_transpose_conv=self.use_transpose_conv,
            use_activation_after_upsampling=self.use_activation_after_upsampling,
            use_coordconv=self.use_coordconv,
        )
        return decoder

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if self.fuel_curve_encoder is not None and x_auxiliary is not None and "fuel_curve" in x_auxiliary:
            fuel_curve_emb = self.fuel_curve_encoder(x_auxiliary["fuel_curve"])  # (B, fuel_curve_embed_dim, H, W)
            x = torch.cat([x, fuel_curve_emb], dim=1)  # (B, C + fuel_curve_embed_dim, H, W)
        x, skip_connections = self.encoder(x)
        if self.use_coordconv:
            x = append_coord_channels(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self._project_output(x)


class MultiSourceUNet(UNetBase):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list[int] | None = None,
        input_branches: list | None = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
        auxiliary_input_dims: dict[str, int] | None = None,
        auxiliary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] | None = None,
        auxiliary_embed_dims: dict[str, int] | None = None,
        auxiliary_feature_encoder_poolings: dict[str, str] | None = None,
        use_coordconv: bool = False,
        fuel_curve_mean: torch.Tensor | None = None,
        fuel_curve_std: torch.Tensor | None = None,
        output_head: str = "shared",
        target_names: list[str] | None = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_features = hidden_features if hidden_features else [64, 128, 256, 512]
        self.input_branches = input_branches
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.use_coordconv = use_coordconv
        self.auxiliary_input_dims: dict[str, int] = auxiliary_input_dims or {}
        self.auxiliary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] = auxiliary_hidden_dims or {}
        self.auxiliary_embed_dims: dict[str, int] = auxiliary_embed_dims or {}
        self.auxiliary_feature_encoder_poolings: dict[str, str] = auxiliary_feature_encoder_poolings or {}

        # iROS early-fusion: encoded to fuel_curve_embed_dim channels then concatenated with spatial input.
        fuel_curve_in = self.auxiliary_input_dims.get("fuel_curve", 0)
        self.fuel_curve_embed_dim = self.auxiliary_embed_dims.get("fuel_curve", 4) if fuel_curve_in > 0 else 0
        self.fuel_curve_encoder: FuelCurveEncoder | None
        if fuel_curve_in > 0:
            # Real stats (computed in GridSource) are always a single global scalar
            # (shape (1,)), regardless of fuel_curve_in (number of ISI bins). Default
            # to a scalar too so eval-time construction (no train_dataset, e.g.
            # evaluate_hexels.py) matches the checkpoint's buffer shape before
            # load_state_dict overwrites it with the real values.
            _mean = fuel_curve_mean if fuel_curve_mean is not None else torch.zeros(1)
            _std = fuel_curve_std if fuel_curve_std is not None else torch.ones(1)
            self.fuel_curve_encoder = FuelCurveEncoder(
                curve_mean=_mean, curve_std=_std, in_channels=fuel_curve_in, embed_dim=self.fuel_curve_embed_dim
            )
        else:
            self.fuel_curve_encoder = None
        self._effective_spatial_in = self.input_channels + self.fuel_curve_embed_dim

        self._build_components()
        self._build_output_layers(self.hidden_features[0], output_head, target_names)

    def build_encoder(self) -> nn.Module:
        encoders = nn.ModuleDict()

        features = self.input_branches if self.input_branches is not None else []

        # Build base spatial grids encoder; input channels include iROS embedding if present.
        if "spatial" in features:
            encoders["spatial"] = BaselineEncoder(
                in_channels=self._effective_spatial_in,
                hidden_features=self.hidden_features,
                use_coordconv=self.use_coordconv,
            )

        # Build encoders for each extra auxiliary feature type (iROS is handled separately).
        if self.auxiliary_input_dims:
            for name, input_dim in self.auxiliary_input_dims.items():
                if name == "fuel_curve":
                    continue  # early-fused via self.fuel_curve_encoder before the spatial encoder
                if name == "wind_grid_mixer":
                    hidden_dims = self.auxiliary_hidden_dims.get(name, {"mixer": [16], "local": [32, 64, 16], "global": [16]})
                    if isinstance(hidden_dims, dict):
                        # WindFeatureEncoderSpatial, WindFeatureEncoderMixer
                        encoders[name] = WindFeatureEncoderMixer(
                            in_channels=input_dim, hidden_dims=hidden_dims, embed_dim=self.auxiliary_embed_dims.get(name, 16)
                        )
                    else:
                        raise ValueError(
                            """For the spatial wind encoder the hidden_dims should be a dict, eg {"mixer": [16], "local": [32, 64, 16], "global": [16]}"""
                        )
                    continue
                if name == "wind_grid_spatial":
                    hidden_dims = self.auxiliary_hidden_dims.get(name, [16, 32, 64])
                    if isinstance(hidden_dims, list):
                        # WindFeatureEncoderSpatial, WindFeatureEncoderMixer
                        encoders[name] = WindFeatureEncoderSpatial(
                            in_channels=input_dim, hidden_dims=hidden_dims, embed_dim=self.auxiliary_embed_dims.get(name, 16)
                        )
                    else:
                        raise ValueError("For the spatial wind encoder the hidden_dims should be a list, eg: [16, 32, 64]")
                    continue
                # get the architectural values for each different auxillary encoder
                hidden_dims = self.auxiliary_hidden_dims.get(name, [32, 64])
                embed_dim = self.auxiliary_embed_dims.get(name, 64)
                pool = self.auxiliary_feature_encoder_poolings.get(name, "max")

                if isinstance(hidden_dims, list):
                    encoders[name] = TabularFeatureEncoder(
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        embed_dim=embed_dim,
                        pooling_type=pool,
                    )

        return encoders

    def build_bottleneck(self) -> nn.Module:
        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None.")

        # iROS is early-fused and must not appear in the bottleneck aux dims.
        auxillary_dims: dict[str, int] = {}
        if self.auxiliary_input_dims:
            for name in self.auxiliary_input_dims:
                if name == "fuel_curve":
                    continue
                auxillary_dims[name] = self.auxiliary_embed_dims.get(name, 64)

        return MultiSourceBottleneck(
            in_channels=self.hidden_features[-1],
            out_channels=self.hidden_features[-1] * 2,
            aux_dims=auxillary_dims,
            use_coordconv=self.use_coordconv,
        )

    def build_decoder(self) -> nn.Module:
        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None.")

        decoder = BaselineDecoder(
            hidden_features=self.hidden_features,
            use_skip_connections=self.use_skip_connections,
            use_transpose_conv=self.use_transpose_conv,
            use_activation_after_upsampling=self.use_activation_after_upsampling,
            use_coordconv=self.use_coordconv,
        )
        return decoder

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Args:
            x: Spatial input (B, C, H, W) (torch.Tensor)
            x_auxiliary: Dict. of auxiliary inputs {'tabular_weather': (B, N, D), ...} (dict[str, torch.Tensor] | None)
        """

        skip_connections = []
        tabular_embeddings = []

        # iROS early fusion: encode and concatenate with spatial input before the UNet encoder.
        if self.fuel_curve_encoder is not None and x_auxiliary is not None and "fuel_curve" in x_auxiliary:
            fuel_curve_emb = self.fuel_curve_encoder(x_auxiliary["fuel_curve"])  # (B, fuel_curve_embed_dim, H, W)
            x = torch.cat([x, fuel_curve_emb], dim=1)  # (B, C + fuel_curve_embed_dim, H, W)

        # Spatial encoder path.
        if "spatial" in self.encoder:  # type: ignore
            x, skip_connections = self.encoder["spatial"](x)  # type: ignore

        # Extra auxiliary encoders path (iROS is already handled above).
        x_wind = None
        if self.auxiliary_input_dims and x_auxiliary is not None:
            for name in self.auxiliary_input_dims:
                if name == "fuel_curve" or name not in x_auxiliary:
                    continue
                encoder_aux = self.encoder[name]  # type: ignore
                encoder_emb = encoder_aux(x_auxiliary[name])
                if name == "wind_grid_mixer" or name == "wind_grid_spatial":
                    x_wind = encoder_emb
                    continue
                tabular_embeddings.append(encoder_emb)

        # Bottleneck path: concat. all tabular embeds.
        x_fused_tabular = None
        if len(tabular_embeddings) > 0:
            x_fused_tabular = torch.cat(tabular_embeddings, dim=1)
        x = self.bottleneck(x, x_fused_tabular, x_wind)

        # Decoder and head.
        x = self.decoder(x, skip_connections)
        return self._project_output(x)
