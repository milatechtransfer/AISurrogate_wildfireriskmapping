import pandas as pd
import pytest
import torch

from inference import run_ai_surrogate_model_hexel_inference as inference_module
from inference.predictor import BurnRiskPredictor
from inference.run_ai_surrogate_model_hexel_inference import (
    _checkpoint_data_config,
    _copy_direct_log_fire_size_artifacts,
    _copy_ignition_count_artifacts,
    _uses_direct_log_fire_size,
    create_dataset,
    get_grid_params_from_data_config,
    get_target_params_from_grid_config,
    get_target_spec_from_data_config,
    get_target_specs_from_data_config,
    prepare_hexel_data,
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


def test_inference_copies_checkpoint_ignition_count_artifacts(tmp_path):
    training_root = tmp_path / "training"
    processed_dir = tmp_path / "processed"
    training_root.mkdir()
    processed_dir.mkdir()
    (processed_dir / "ignition_count_processed.csv").write_text(
        "hex_id,GRIDCODE,NORM_LOG1P_IGNITION_COUNT_MEAN,NORM_IGNITION_COUNT_CV\n88,10,0.9,0.9\n"
    )
    (training_root / "ignition_count_processed.csv").write_text(
        "hex_id,GRIDCODE,NORM_LOG1P_IGNITION_COUNT_MEAN,NORM_IGNITION_COUNT_CV\n1,10,0.5,0.25\n"
    )
    (training_root / "ignition_count_norm_params.json").write_text('{"version": 1}\n')

    _copy_ignition_count_artifacts(
        {
            "root_dir": str(training_root),
            "input_sources": [
                {
                    "name": "spatialized_ignition_count",
                    "params": {
                        "csv_name": "ignition_count_processed.csv",
                        "hex_id_col": "hex_id",
                    },
                }
            ],
        },
        processed_dir,
        "01",
    )

    assert (processed_dir / "ignition_count_processed.csv").read_text() == (training_root / "ignition_count_processed.csv").read_text()
    assert (processed_dir / "ignition_count_norm_params.json").read_text() == '{"version": 1}\n'


def test_inference_builds_regional_ignition_count_with_training_normalization(tmp_path, monkeypatch):
    training_root = tmp_path / "training"
    processed_dir = tmp_path / "processed"
    raw_data_dir = tmp_path / "regional"
    training_root.mkdir()
    processed_dir.mkdir()
    raw_data_dir.mkdir()
    (raw_data_dir / "hex99").mkdir()
    (training_root / "ignition_count_processed.csv").write_text(
        "hex_id,GRIDCODE,IGNITION_COUNT_MEAN,IGNITION_COUNT_SD,IGNITION_COUNT_CV,"
        "LOG1P_IGNITION_COUNT_MEAN,NORM_LOG1P_IGNITION_COUNT_MEAN,NORM_IGNITION_COUNT_CV\n"
        "1,10,4,1,0.25,1.5,0.25,0.125\n"
        "99,20,9,3,0.333,2.3,0.65,0.333\n"
    )
    (processed_dir / "ignition_count_processed.csv").write_text(
        "hex_id,GRIDCODE,IGNITION_COUNT_MEAN,IGNITION_COUNT_SD,IGNITION_COUNT_CV,"
        "LOG1P_IGNITION_COUNT_MEAN,NORM_LOG1P_IGNITION_COUNT_MEAN,NORM_IGNITION_COUNT_CV\n"
        "88,30,10,2,0.2,2.4,0.7,0.0\n"
    )
    (training_root / "ignition_count_norm_params.json").write_text(
        '{"log1p_mean_minimum": 1.0, "log1p_mean_maximum": 3.0, "cv_minimum": 0.2, "cv_maximum": 0.6}\n'
    )
    captured = {}

    def fake_build(raw_root, hex_id, *, mask_scope):
        captured.update(raw_root=raw_root, hex_id=hex_id, mask_scope=mask_scope)
        return pd.DataFrame(
            {
                "hex_id": [99],
                "GRIDCODE": [20],
                "IGNITION_COUNT_MEAN": [6.0],
                "IGNITION_COUNT_SD": [2.4],
                "IGNITION_COUNT_CV": [0.4],
                "LOG1P_IGNITION_COUNT_MEAN": [2.0],
            }
        )

    monkeypatch.setattr(inference_module, "build_hex_ignition_count_rows", fake_build)
    _copy_ignition_count_artifacts(
        {
            "root_dir": str(training_root),
            "input_sources": [
                {
                    "name": "spatialized_ignition_count",
                    "params": {
                        "csv_name": "ignition_count_processed.csv",
                        "hex_id_col": "hex_id",
                    },
                }
            ],
        },
        processed_dir,
        "99",
        raw_data_dir=raw_data_dir,
        mask_scope="actual",
    )

    result = pd.read_csv(processed_dir / "ignition_count_processed.csv")
    regional = result[result["hex_id"].eq(99)].iloc[0]
    assert set(result["hex_id"]) == {1, 99}
    assert len(result[result["hex_id"].eq(99)]) == 1
    assert regional["IGNITION_COUNT_MEAN"] == pytest.approx(6.0)
    assert regional["NORM_LOG1P_IGNITION_COUNT_MEAN"] == pytest.approx(0.5)
    assert regional["NORM_IGNITION_COUNT_CV"] == pytest.approx(0.5)
    assert captured == {"raw_root": raw_data_dir, "hex_id": "99", "mask_scope": "actual"}


def test_inference_overrides_checkpoint_training_and_regional_raw_roots(tmp_path):
    training_root = tmp_path / "training"
    regional_root = tmp_path / "regional"
    training_root.mkdir()
    regional_root.mkdir()
    checkpoint = {
        "config": {
            "data": {
                "root_dir": "/old/personal/training",
                "raw_data_dir": "/old/national/raw",
                "input_sources": [],
            }
        }
    }

    data_config = _checkpoint_data_config(
        checkpoint,
        training_data_root=training_root,
        inference_raw_data_root=regional_root,
    )

    assert data_config["root_dir"] == str(training_root)
    assert data_config["raw_data_dir"] == str(regional_root)
    assert checkpoint["config"]["data"]["root_dir"] == "/old/personal/training"
    assert checkpoint["config"]["data"]["raw_data_dir"] == "/old/national/raw"


def test_inference_copies_checkpoint_direct_log_fire_size_artifacts(tmp_path):
    training_root = tmp_path / "training"
    processed_dir = tmp_path / "processed"
    training_root.mkdir()
    processed_dir.mkdir()
    (training_root / "fire_size_global_fill.csv").write_text("GRIDCODE,LOG_SIZE_HA\n1,2.0\n")
    (training_root / "fire_size_log_stats.json").write_text('{"contract": "log10_1p_hectares"}\n')

    _copy_direct_log_fire_size_artifacts(
        {
            "root_dir": str(training_root),
            "input_sources": [
                {
                    "name": "spatialized_fire_size",
                    "params": {
                        "feature_names_list": ["LOG_SIZE_HA"],
                        "global_fill_csv_name": "fire_size_global_fill.csv",
                        "include_missing_firezone_mask": True,
                    },
                }
            ],
        },
        processed_dir,
    )

    assert (processed_dir / "fire_size_global_fill.csv").read_text() == "GRIDCODE,LOG_SIZE_HA\n1,2.0\n"
    assert (processed_dir / "fire_size_log_stats.json").read_text() == '{"contract": "log10_1p_hectares"}\n'


def test_inference_does_not_treat_legacy_log_fire_size_as_v23() -> None:
    assert not _uses_direct_log_fire_size(
        {
            "input_sources": [
                {
                    "name": "spatialized_fire_size",
                    "params": {
                        "feature_names_list": ["LOG_SIZE_HA"],
                        "include_missing_firezone_mask": False,
                    },
                }
            ]
        }
    )


def test_inference_preparation_forwards_checkpoint_spatial_modes(tmp_path, monkeypatch):
    (tmp_path / "hex01").mkdir()
    captured = {}

    monkeypatch.setattr(inference_module, "build_weather_table", lambda **_kwargs: None)
    monkeypatch.setattr(inference_module, "find_hex_ids", lambda _root: ["01"])

    def fake_load_spatial_features_per_hexel(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros(1, 2, 2, 1).numpy(),
            torch.ones(1, 2, 2, dtype=torch.bool).numpy(),
            {0: (0, 0)},
        )

    monkeypatch.setattr(inference_module, "load_spatial_features_per_hexel", fake_load_spatial_features_per_hexel)
    monkeypatch.setattr(inference_module, "get_split_hexel_window", lambda **_kwargs: None)

    prepare_hexel_data(
        data_dir=tmp_path,
        hex_id="01",
        win_h=2,
        win_w=2,
        ignition_weighting="probability_mass",
        fuel_representation="raw",
        preserve_native_grid=True,
    )

    assert captured["ignition_weighting"] == "probability_mass"
    assert captured["fuel_representation"] == "raw"
    assert captured["preserve_native_grid"] is True


@pytest.mark.parametrize(
    ("build_from_raw", "expected_raw_root"),
    [(False, None), (True, "data_dir")],
)
def test_inference_preparation_only_rebuilds_ignition_count_when_requested(
    tmp_path,
    monkeypatch,
    build_from_raw,
    expected_raw_root,
):
    (tmp_path / "hex01").mkdir()
    captured = {}

    monkeypatch.setattr(inference_module, "build_weather_table", lambda **_kwargs: None)
    monkeypatch.setattr(inference_module, "find_hex_ids", lambda _root: ["01"])
    monkeypatch.setattr(
        inference_module,
        "_copy_ignition_count_artifacts",
        lambda _config, _output, _hex_id, *, raw_data_dir, mask_scope: captured.update(
            raw_data_dir=raw_data_dir,
            mask_scope=mask_scope,
        ),
    )
    monkeypatch.setattr(inference_module, "_copy_direct_log_fire_size_artifacts", lambda *_args: None)
    monkeypatch.setattr(
        inference_module,
        "load_spatial_features_per_hexel",
        lambda **_kwargs: (
            torch.zeros(1, 2, 2, 1).numpy(),
            torch.ones(1, 2, 2, dtype=torch.bool).numpy(),
            {0: (0, 0)},
        ),
    )
    monkeypatch.setattr(inference_module, "get_split_hexel_window", lambda **_kwargs: None)

    prepare_hexel_data(
        data_dir=tmp_path,
        hex_id="01",
        win_h=2,
        win_w=2,
        checkpoint_data_config={"input_sources": []},
        build_ignition_count_from_raw=build_from_raw,
    )

    expected = tmp_path if expected_raw_root else None
    assert captured == {"raw_data_dir": expected, "mask_scope": "actual"}


def test_inference_prepares_direct_log_fire_size_without_zone36_or_minmax(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    training_root = tmp_path / "training"
    (data_dir / "hex01").mkdir(parents=True)
    training_root.mkdir()
    (data_dir / "df_fire_fru.csv").write_text("GRIDCODE,SIZE_HA\n1,99\n36,99\n")
    (training_root / "fire_size_global_fill.csv").write_text("GRIDCODE,LOG_SIZE_HA\n1,2.0\n")
    (training_root / "fire_size_log_stats.json").write_text('{"contract": "log10_1p_hectares"}\n')

    monkeypatch.setattr(inference_module, "build_weather_table", lambda **_kwargs: None)
    monkeypatch.setattr(inference_module, "find_hex_ids", lambda _root: ["01"])
    monkeypatch.setattr(
        inference_module,
        "load_spatial_features_per_hexel",
        lambda **_kwargs: (
            torch.zeros(1, 2, 2, 1).numpy(),
            torch.ones(1, 2, 2, dtype=torch.bool).numpy(),
            {0: (0, 0)},
        ),
    )
    monkeypatch.setattr(inference_module, "get_split_hexel_window", lambda **_kwargs: None)

    processed_dir = prepare_hexel_data(
        data_dir=data_dir,
        hex_id="01",
        win_h=2,
        win_w=2,
        checkpoint_data_config={
            "root_dir": str(training_root),
            "input_sources": [
                {
                    "name": "spatialized_fire_size",
                    "params": {
                        "feature_names_list": ["LOG_SIZE_HA"],
                        "global_fill_csv_name": "fire_size_global_fill.csv",
                        "include_missing_firezone_mask": True,
                    },
                }
            ],
        },
    )

    fire_size = pd.read_csv(processed_dir / "df_fire_fru_processed.csv")
    assert fire_size.columns.tolist() == ["GRIDCODE", "SIZE_HA", "LOG_SIZE_HA"]
    assert set(fire_size["GRIDCODE"]) == {1}
    assert not (processed_dir / "fire_size_norm_params.json").exists()


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


def test_predictor_returns_only_configured_center_crop():
    config = {
        "data": {"input_sources": [{"name": "grid", "params": {"target_name": "bp"}}]},
        "data_prep": {"win_h": 4, "win_w": 4, "target_crop_h": 2, "target_crop_w": 2},
    }
    predictor = BurnRiskPredictor(model=ConstantModel(0.0), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 4, 4))

    assert predictions.shape == (1, 1, 2, 2)
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
