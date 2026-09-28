import torch
import torch.nn as nn

from src.datasets.targets import get_target_spec


class BurnProbabilityBehaviorHead(nn.Module):
    """Project shared decoder features into BP logits and conditional FI/ROS outputs.

    BP gets its own non-linear trunk (`bp_trunk`) ahead of the final 1x1 projection
    (`bp_head`), since its target distribution/statistics differ substantially from
    FI/ROS and a single linear projection from shared decoder features is often too
    weak to capture BP-specific structure. FI/ROS keep a single shared 1x1 conv
    (`behavior_head`) since they are smoother, well-behaved regression targets.
    """

    def __init__(
        self,
        in_channels: int,
        target_names: list[str],
        bp_head_depth: int = 2,
        bp_head_hidden_channels: int | None = None,
    ):
        super().__init__()
        self.target_names = tuple(get_target_spec(name).name for name in target_names)
        required_targets = {"bp", "fi", "ros"}
        if len(self.target_names) != len(required_targets) or set(self.target_names) != required_targets:
            raise ValueError(f"bp_behavior output head requires exactly {sorted(required_targets)}, got {list(self.target_names)}.")
        if bp_head_depth < 0:
            raise ValueError(f"bp_head_depth must be >= 0, got {bp_head_depth}.")

        hidden_channels = bp_head_hidden_channels or in_channels
        trunk_layers: list[nn.Module] = []
        trunk_in_channels = in_channels
        for _ in range(bp_head_depth):
            trunk_layers.extend(
                [
                    nn.Conv2d(trunk_in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(inplace=True),
                ]
            )
            trunk_in_channels = hidden_channels
        self.bp_trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()

        self.bp_head = nn.Conv2d(trunk_in_channels, 1, kernel_size=1)
        self.behavior_head = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor, bp_features: torch.Tensor | None = None) -> torch.Tensor:
        behavior = self.behavior_head(features)
        bp_input = bp_features if bp_features is not None else features
        outputs = {
            "bp": self.bp_head(self.bp_trunk(bp_input)),
            "fi": behavior[:, 0:1],
            "ros": behavior[:, 1:2],
        }
        return torch.cat([outputs[name] for name in self.target_names], dim=1)
