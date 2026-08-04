import torch
import torch.nn as nn

from src.datasets.targets import get_target_spec


class BurnProbabilityBehaviorHead(nn.Module):
    """Project shared decoder features into BP logits and conditional FI/ROS outputs."""

    def __init__(self, in_channels: int, target_names: list[str]):
        super().__init__()
        self.target_names = tuple(get_target_spec(name).name for name in target_names)
        required_targets = {"bp", "fi", "ros"}
        if len(self.target_names) != len(required_targets) or set(self.target_names) != required_targets:
            raise ValueError(f"bp_behavior output head requires exactly {sorted(required_targets)}, got {list(self.target_names)}.")

        self.bp_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.behavior_head = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        behavior = self.behavior_head(features)
        outputs = {
            "bp": self.bp_head(features),
            "fi": behavior[:, 0:1],
            "ros": behavior[:, 1:2],
        }
        return torch.cat([outputs[name] for name in self.target_names], dim=1)
