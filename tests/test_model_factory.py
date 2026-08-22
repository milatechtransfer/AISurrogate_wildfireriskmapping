import pytest
import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.heads import BurnProbabilityBehaviorHead
from src.models.interpretable_mechanistic import InterpretableMechanisticModel
from src.models.unet import BaselineUNet, MultiSourceUNet


def test_auto_factory_preserves_baseline_unet_path():
    config = ModelConfig(hidden_features=[8, 16], input_branches=["spatial"])

    model = build_model(model_config=config, spatial_input_channels=3)

    assert resolve_model_architecture(config) == "baseline_unet"
    assert isinstance(model, BaselineUNet)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 1, 32, 32)
    assert "out_conv.weight" in model.state_dict()


def test_auto_factory_preserves_multi_source_unet_path():
    config = ModelConfig(
        hidden_features=[8, 16],
        input_branches=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"tabular_weather": [8]},
        auxiliary_embed_dims={"tabular_weather": 8},
    )

    model = build_model(model_config=config, spatial_input_channels=3, auxiliary_input_dims={"tabular_weather": 5})

    assert resolve_model_architecture(config) == "multi_source_unet"
    assert isinstance(model, MultiSourceUNet)
    out = model(torch.randn(2, 3, 32, 32), {"tabular_weather": torch.randn(2, 4, 5)})
    assert out.shape == (2, 1, 32, 32)


def test_bp_behavior_head_preserves_configured_target_order():
    config = ModelConfig(
        num_classes=3,
        output_head="bp_behavior",
        hidden_features=[8, 16],
        input_branches=["spatial"],
    )
    model = build_model(
        model_config=config,
        spatial_input_channels=3,
        target_names=["fi", "bp", "ros"],
    )
    assert isinstance(model, BaselineUNet)
    assert isinstance(model.multi_output_head, BurnProbabilityBehaviorHead)

    with torch.no_grad():
        model.multi_output_head.bp_head.weight.zero_()
        model.multi_output_head.bp_head.bias.fill_(1.0)
        model.multi_output_head.behavior_head.weight.zero_()
        model.multi_output_head.behavior_head.bias.copy_(torch.tensor([2.0, 3.0]))

    out = model(torch.randn(2, 3, 32, 32))

    assert out.shape == (2, 3, 32, 32)
    assert torch.all(out[:, 0] == 2.0)
    assert torch.all(out[:, 1] == 1.0)
    assert torch.all(out[:, 2] == 3.0)
    assert "out_conv.weight" not in model.state_dict()


def test_bp_behavior_head_requires_bp_fi_and_ros():
    config = ModelConfig(num_classes=2, output_head="bp_behavior", hidden_features=[8, 16])

    with pytest.raises(ValueError, match="requires exactly"):
        build_model(model_config=config, spatial_input_channels=3, target_names=["bp", "fi"])


@pytest.mark.parametrize(
    "architecture",
    [
        "interpretable_mechanistic_v3",
        "physical_mechanistic_v3",
        "gray_box_mechanistic",
    ],
)
def test_build_model_accepts_gray_box_mechanistic_aliases(architecture: str) -> None:
    spatial_input_names = [
        "ignition_grid_human",
        "ignition_grid_lightning",
        "elevation_grid",
        "InitialSpreadIndex",
        "wind_x",
        "wind_y",
        "spatialized_fire_size/NORM_LOG_SIZE_HA_q10",
        "spatialized_fire_size/NORM_LOG_SIZE_HA_q50",
        "spatialized_fire_size/NORM_LOG_SIZE_HA_q90",
        "NORM_LOG1P_IGNITION_COUNT_MEAN",
        "NORM_IGNITION_COUNT_CV",
    ]
    config = ModelConfig(
        architecture=architecture,
        num_classes=3,
        output_head="bp_behavior",
        spatial_input_names=spatial_input_names,
        propagation_scenario_mode="fire_size",
    )

    model = build_model(
        model_config=config,
        spatial_input_channels=len(spatial_input_names),
        auxiliary_input_dims={"fuel_curve": 2 * len(config.propagation_isi_bins)},
        target_names=["bp", "fi", "ros"],
    )

    assert isinstance(model, InterpretableMechanisticModel)
    assert model.variant == "v3"
