import pytest
import torch

from inference.predictor import BurnRiskPredictor
from inference.run_ai_surrogate_model_hexel_inference import (
    create_dataset,
    get_grid_params_from_data_config,
    get_target_params_from_grid_config,
    get_target_spec_from_data_config,
    get_target_specs_from_data_config,
    resolve_target_normalization,
)
from src.datasets.targets import get_target_spec


def test_inference_target_spec_defaults_to_bp_for_old_checkpoints():
    target = get_target_spec_from_data_config({"input_sources": [{"name": "grid", "params": {}}]})

    assert target.name == "bp"
    assert target.output_type == "fire_burn_probability"


def test_inference_target_spec_uses_checkpoint_target_name():
    target = get_target_spec_from_data_config({"input_sources": [{"name": "grid", "params": {"target_name": "ros"}}]})

    assert target.name == "ros"
    assert target.output_type == "fire_ros"


def test_inference_grid_params_returns_grid_source_params():
    params = get_grid_params_from_data_config({"input_sources": [{"name": "grid", "params": {"out_norm": "log_standard"}}]})

    assert params["out_norm"] == "log_standard"


def test_inference_reads_multi_target_order_and_normalization():
    data_config = {
        "input_sources": [
            {
                "name": "grid",
                "params": {
                    "targets": [
                        {"name": "bp", "out_norm": "min_max"},
                        {"name": "fi", "out_norm": "log_standard", "log_mean": 2.0, "log_std": 0.5},
                        {"name": "ros", "out_norm": "log_standard", "log_mean": 1.0, "log_std": 0.25},
                    ]
                },
            }
        ]
    }

    targets = get_target_specs_from_data_config(data_config)
    grid_params = get_grid_params_from_data_config(data_config)

    assert [target.name for target in targets] == ["bp", "fi", "ros"]
    assert get_target_params_from_grid_config(grid_params, get_target_spec("fi")) == {
        "name": "fi",
        "out_norm": "log_standard",
        "log_mean": 2.0,
        "log_std": 0.5,
    }
    with pytest.raises(ValueError, match="Expected one inference target"):
        get_target_spec_from_data_config(data_config)


def test_inference_bp_normalization_uses_training_split_and_zero_minimum(tmp_path, monkeypatch):
    (tmp_path / "train_indices.csv").write_text("hex_id\n1\n")
    captured = {}

    def fake_range(root_dir, output_type, allowed_hex_ids, raw_data_dir):
        captured.update(
            root_dir=root_dir,
            output_type=output_type,
            allowed_hex_ids=allowed_hex_ids,
            raw_data_dir=raw_data_dir,
        )
        return 0.8, 0.1

    monkeypatch.setattr(
        "inference.run_ai_surrogate_model_hexel_inference.get_range_output_cached",
        fake_range,
    )
    normalization = resolve_target_normalization(
        {
            "root_dir": str(tmp_path),
            "raw_data_dir": str(tmp_path / "raw"),
            "train_split": "train_indices.csv",
        },
        get_target_spec("bp"),
        {"name": "bp", "out_norm": "min_max"},
        bp_nodata_as_zero=True,
    )

    assert captured["allowed_hex_ids"] == {1}
    assert captured["root_dir"] == str(tmp_path)
    assert normalization.max_value == 0.8
    assert normalization.min_value == 0.0


def test_inference_dataset_uses_training_resources_for_grid_only(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "meta_hex_01.csv").write_text("filename,valid_ratio\npatch.npy,1.0\n")
    created_roots = {}

    class FakeSource:
        def __init__(self, root_dir, params, **kwargs):
            created_roots[params["source_name"]] = (str(root_dir), kwargs)

    monkeypatch.setattr(
        "inference.run_ai_surrogate_model_hexel_inference.get_data_source_class",
        lambda _name: FakeSource,
    )
    monkeypatch.setattr(
        "inference.run_ai_surrogate_model_hexel_inference.get_data_source_param_class",
        lambda name: lambda **_params: {"source_name": name},
    )

    create_dataset(
        processed_data_dir=processed_dir,
        hex_id="01",
        config_dict={
            "root_dir": "/training/patches",
            "raw_data_dir": "/training/raw",
            "train_split": "train_indices.csv",
            "filename_col": "filename",
            "valid_mask_threshold": 0.01,
            "input_sources": [
                {"name": "grid", "params": {}},
                {"name": "spatialized_weather", "params": {}},
            ],
        },
    )

    assert created_roots["grid"] == (
        "/training/patches",
        {
            "raw_data_dir": "/training/raw",
            "train_split_csv_name": "train_indices.csv",
        },
    )
    assert created_roots["spatialized_weather"] == (str(processed_dir), {})


class ConstantModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, spatial_inputs, auxiliary_inputs=None):
        return torch.full((spatial_inputs.shape[0], 1, spatial_inputs.shape[2], spatial_inputs.shape[3]), self.value)


class MultiConstantModel(torch.nn.Module):
    def forward(self, spatial_inputs, auxiliary_inputs=None):
        values = torch.tensor([0.0, 2.0, 3.0], dtype=spatial_inputs.dtype, device=spatial_inputs.device)
        return values.view(1, 3, 1, 1).expand(spatial_inputs.shape[0], 3, spatial_inputs.shape[2], spatial_inputs.shape[3])


def test_predictor_keeps_fi_outputs_linear():
    config = {"data": {"input_sources": [{"name": "grid", "params": {"target_name": "fi"}}]}}
    predictor = BurnRiskPredictor(model=ConstantModel(2.0), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 2, 2))

    assert torch.all(predictions == 2.0)


def test_predictor_sigmoids_bp_outputs():
    config = {"data": {"input_sources": [{"name": "grid", "params": {"target_name": "bp"}}]}}
    predictor = BurnRiskPredictor(model=ConstantModel(0.0), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 2, 2))

    assert torch.allclose(predictions, torch.full((1, 1, 2, 2), 0.5))


def test_predictor_activates_and_splits_multi_target_outputs():
    config = {
        "data": {
            "input_sources": [
                {
                    "name": "grid",
                    "params": {
                        "targets": [
                            {"name": "bp", "out_norm": "min_max"},
                            {"name": "fi", "out_norm": "log_standard"},
                            {"name": "ros", "out_norm": "log_standard"},
                        ]
                    },
                }
            ]
        }
    }
    predictor = BurnRiskPredictor(model=MultiConstantModel(), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 2, 2))
    named = predictor.predict_named_batch(torch.zeros(1, 1, 2, 2))

    assert torch.allclose(predictions[:, 0], torch.full((1, 2, 2), 0.5))
    assert torch.all(predictions[:, 1] == 2.0)
    assert torch.all(predictions[:, 2] == 3.0)
    assert list(named) == ["bp", "fi", "ros"]
    assert torch.equal(named["fi"], predictions[:, 1:2])
