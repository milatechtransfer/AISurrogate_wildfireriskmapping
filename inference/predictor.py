"""
Pure inference class for wildfire burn risk prediction.
Wraps the PyTorch model and handles tensor-in, tensor-out operations.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from src.config import ModelConfig
from src.datasets.targets import (
    TargetName,
    TargetSpec,
    activate_target_predictions,
    get_target_specs,
    split_target_predictions,
)
from src.models.factory import build_model

logger = logging.getLogger(__name__)


class BurnRiskPredictor:
    """
    Pure inference class for wildfire burn risk prediction.
    Wraps the PyTorch model and handles tensor-in, tensor-out operations.

    This class focuses only on model operations.

    Example:
        # Assuming you have a trained model checkpoint and prepared input tensors:
        >>> predictor = BurnRiskPredictor.from_checkpoint(checkpoint_path="models/best.pth", spatial_channels=10, auxiliary_input_dims={"tabular_weather": 7, "tabular_fire_size": 5})
        >>> predictions = predictor(spatial_batch, auxiliary_batch)

        # Assuming you loaded model and config separately (not most common usage):
        >>> loaded_model = ... # load model state dict and build model architecture
        >>> loaded_config = ... # load config dict from checkpoint
        >>> predictor = BurnRiskPredictor(model=loaded_model, device="cuda", config=loaded_config)
        >>> predictions = predictor(spatial_batch, auxiliary_batch)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize with an already-built model.

        For typical usage, prefer the `from_checkpoint` classmethod.

        Args:
            model (torch.nn.Module): A PyTorch dense prediction model.
            device (str | torch.device): Device the model is on.
            config (dict[str, Any] | None): Optional config dict for reference.
        """
        self.model = model
        self.device = device
        self.config = config
        self.target_specs = self._get_target_specs()
        self.model.eval()

    def _get_target_specs(self) -> list[TargetSpec]:
        if self.config is None:
            return get_target_specs("bp")
        for source in self.config.get("data", {}).get("input_sources", []):
            if source.get("name") == "grid":
                params = source.get("params", {})
                if params.get("targets"):
                    return get_target_specs([target["name"] for target in params["targets"]])
                return get_target_specs(params.get("target_name", "bp"))
        return get_target_specs("bp")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        spatial_channels: int,
        auxiliary_input_dims: dict[str, int] | None = None,
        device: str | torch.device | None = None,
    ) -> "BurnRiskPredictor":
        """
        Create a predictor from a saved checkpoint file.

        Args:
            checkpoint_path (str | Path): Path to the trained model checkpoint (.pth file). Must contain 'model_state' and 'config'.
            spatial_channels (int): Number of input channels for spatial data.
            auxiliary_input_dims (dict[str, int] | None): Dict mapping auxiliary source names to their dimensions.
            device (str | torch.device | None): Device to run inference on. If None, auto-detects GPU/CPU.

        Returns:
            BurnRiskPredictor instance ready for inference.
        """
        checkpoint_path = Path(checkpoint_path)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        logger.info(f"Loading checkpoint from {checkpoint_path} on {device}")

        config = checkpoint["config"]
        model_config = config["model"]
        target_specs = cls._target_specs_from_config(config)

        # Build model architecture
        model = cls._build_model(
            model_config=model_config,
            spatial_channels=spatial_channels,
            auxiliary_input_dims=auxiliary_input_dims or {},
            target_names=[target.name for target in target_specs],
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully on {device}")
        return cls(model=model, device=device, config=config)

    @staticmethod
    def _target_specs_from_config(config: dict[str, Any]) -> list[TargetSpec]:
        for source in config.get("data", {}).get("input_sources", []):
            if source.get("name") == "grid":
                params = source.get("params", {})
                if params.get("targets"):
                    return get_target_specs([target["name"] for target in params["targets"]])
                return get_target_specs(params.get("target_name", "bp"))
        return get_target_specs("bp")

    @staticmethod
    def _build_model(
        model_config: dict,
        spatial_channels: int,
        auxiliary_input_dims: dict[str, int],
        target_names: list[str] | None = None,
    ) -> torch.nn.Module:
        """Instantiate the model architecture based on config."""
        return build_model(
            model_config=ModelConfig(**model_config),
            spatial_input_channels=spatial_channels,
            auxiliary_input_dims=auxiliary_input_dims,
            target_names=target_names,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        spatial_inputs: torch.Tensor,
        auxiliary_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Run a single batch through the model.

        Args:
            spatial_inputs (torch.Tensor): Spatial grid tensor of shape (B, C, H, W).
            auxiliary_inputs (dict[str, torch.Tensor] | None): Optional dict of auxiliary tensors.

        Returns:
            torch.Tensor: Predictions tensor of shape (B, num_classes, H, W) on CPU.
        """
        spatial_inputs = spatial_inputs.to(self.device)

        if auxiliary_inputs:
            # Remove 'grid' from auxiliary inputs if present, as it's already passed as spatial_inputs
            auxiliary_inputs = {k: v.to(self.device) for k, v in auxiliary_inputs.items() if k != "grid"}

        predictions = self.model(spatial_inputs, auxiliary_inputs if auxiliary_inputs else None)

        predictions = activate_target_predictions(predictions, self.target_specs)

        # Move predictions to CPU before returning
        return predictions.cpu()

    @torch.no_grad()
    def predict_named_batch(
        self,
        spatial_inputs: torch.Tensor,
        auxiliary_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[TargetName, torch.Tensor]:
        predictions = self.predict_batch(spatial_inputs, auxiliary_inputs)
        return split_target_predictions(predictions, self.target_specs)

    def __call__(
        self,
        spatial_inputs: torch.Tensor,
        auxiliary_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Shorthand for predict_batch."""
        return self.predict_batch(spatial_inputs, auxiliary_inputs)
