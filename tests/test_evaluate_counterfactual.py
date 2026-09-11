from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.datasets.postprocessing.counterfactual.counterfactual_base import EndpointConfig, ScenarioConfig
from src.evaluate_counterfactual import _select_endpoints, _select_scenarios, run_counterfactual_evaluation

_BASELINE = ScenarioConfig(name="baseline", kind="baseline", description="", params={})
_FUEL_A = ScenarioConfig(name="fuel_a", kind="fuel", description="", params={})
_FUEL_B = ScenarioConfig(name="fuel_b", kind="fuel", description="", params={})
_ALL_SCENARIOS = [_BASELINE, _FUEL_A, _FUEL_B]


def test_select_scenarios_returns_all_when_unfiltered() -> None:
    assert _select_scenarios(_ALL_SCENARIOS, None) == _ALL_SCENARIOS


def test_select_scenarios_always_keeps_baseline() -> None:
    selected = _select_scenarios(_ALL_SCENARIOS, {"fuel_a"})
    assert selected == [_BASELINE, _FUEL_A]


def test_select_scenarios_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match=r"Unknown scenario selection.*missing"):
        _select_scenarios(_ALL_SCENARIOS, {"missing"})


def test_select_endpoints_rejects_unknown_names() -> None:
    endpoints = {"bp": EndpointConfig(name="bp", config_path=Path("bp.yaml"))}
    with pytest.raises(ValueError, match=r"Unknown endpoint selection.*fi"):
        _select_endpoints(endpoints, {"fi"})


@dataclass
class _DataSourceParams:
    csv_name: str
    global_fill_csv_name: str | None = None
    feature_names_list: list[str] = field(default_factory=list)
    fire_weather_zone_id_col: str = "GRIDCODE"
    aggregation: str = "mean"
    hex_id_col: str | None = None
    quantiles: list[float] | None = None


@dataclass
class _DataSourceConfig:
    name: str
    params: _DataSourceParams


@dataclass
class _DataConfig:
    root_dir: str
    test_split: str = "test.csv"
    valid_mask_threshold: float = 0.5
    filename_col: str = "filename"
    num_workers: int = 4
    raw_data_dir: str = ""
    input_sources: list = field(default_factory=list)


@dataclass
class _EvaluationConfig:
    checkpoint_filename: str = "best.pt"


@dataclass
class _LoggerConfig:
    enabled: bool = True


@dataclass
class _EndpointRunConfig:
    save_dir: str
    data: _DataConfig
    evaluation: _EvaluationConfig = field(default_factory=_EvaluationConfig)
    logger: _LoggerConfig = field(default_factory=_LoggerConfig)
    modelling_approach: str = "common_input_pipeline"

    def model_copy(self, *, deep: bool) -> _EndpointRunConfig:
        return deepcopy(self) if deep else self


class _FakeFuelTransform:
    summary = pd.DataFrame([{"scenario_name": "fuel_a", "edited_pixels": 12}])
    components = pd.DataFrame()
    calls: list[dict] = []

    @classmethod
    def from_metadata(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls()


def test_run_counterfactual_evaluation_orchestrates_selected_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    data_root = project_root / "data"
    source_dir = project_root / "trained"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    source_dir.mkdir()
    (source_dir / "best.pt").write_bytes(b"checkpoint")
    pd.DataFrame(
        [
            {
                "hex_id": "16",
                "filename": "patch.npy",
                "valid_ratio": 1.0,
            }
        ]
    ).to_csv(data_root / "test.csv", index=False)
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml"}},
                "scenarios": [
                    {"name": "baseline", "kind": "baseline"},
                    {"name": "fuel_a", "kind": "fuel"},
                ],
            },
            handle,
        )

    run_config = _EndpointRunConfig(
        save_dir=str(source_dir),
        data=_DataConfig(root_dir=str(data_root)),
    )
    evaluation_calls: list[dict] = []

    def _fake_evaluate_hexels(**kwargs):
        evaluation_calls.append(kwargs)
        return {"mae": 1.5}

    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr("src.evaluate_counterfactual._fuel_channel", lambda *_: 3)
    monkeypatch.setattr("src.evaluate_counterfactual.FuelCounterfactualTransform", _FakeFuelTransform)
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", _fake_evaluate_hexels)
    _FakeFuelTransform.calls.clear()
    stale_components = save_dir / "fuel_component_replacements.csv"
    save_dir.mkdir()
    stale_components.write_text("stale\n")

    index = run_counterfactual_evaluation(
        config_path,
        endpoint_names={"bp"},
        scenario_names={"fuel_a"},
        overwrite=False,
        project_root=project_root,
    )

    assert index[["scenario", "endpoint"]].to_dict("records") == [
        {"scenario": "baseline", "endpoint": "bp"},
        {"scenario": "fuel_a", "endpoint": "bp"},
    ]
    assert len(evaluation_calls) == 2
    assert evaluation_calls[0]["patch_transform"] is None
    assert isinstance(evaluation_calls[1]["patch_transform"], _FakeFuelTransform)
    assert len(_FakeFuelTransform.calls) == 1
    assert not stale_components.exists()
    assert (save_dir / "scenario_prediction_index.csv").exists()
    assert (save_dir / "counterfactual_metrics.csv").exists()
    summary = pd.read_csv(save_dir / "fuel_edit_summary.csv")
    assert summary[["endpoint", "edited_pixels"]].to_dict("records") == [{"endpoint": "bp", "edited_pixels": 12}]


class _FakeWeatherResult:
    def __init__(self, edited_csv_path: Path) -> None:
        self.edited_csv_path = edited_csv_path
        self.summary = pd.DataFrame([{"scenario_name": "bc_mean_weather_transplant", "recipient_hex_id": "16", "donor_fwi_mean": 29.0}])


def test_run_counterfactual_evaluation_orchestrates_weather_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    data_root = project_root / "data"
    source_dir = project_root / "trained"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    source_dir.mkdir()
    (source_dir / "best.pt").write_bytes(b"checkpoint")
    pd.DataFrame([{"hex_id": "16", "filename": "patch.npy", "valid_ratio": 1.0}]).to_csv(data_root / "test.csv", index=False)
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml"}},
                "scenarios": [
                    {"name": "baseline", "kind": "baseline"},
                    {
                        "name": "bc_mean_weather_transplant",
                        "kind": "weather",
                        "params": {"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]},
                    },
                ],
            },
            handle,
        )

    weather_source = _DataSourceConfig(name="spatialized_weather", params=_DataSourceParams(csv_name="weather_table_processed.csv"))
    run_config = _EndpointRunConfig(
        save_dir=str(source_dir),
        data=_DataConfig(root_dir=str(data_root), input_sources=[weather_source]),
    )
    materialize_calls: list[dict] = []
    edited_csv_path = tmp_path / "edited_weather.csv"
    edited_csv_path.write_text("Order\n")

    def _fake_materialize_weather_scenario(**kwargs):
        materialize_calls.append(kwargs)
        return _FakeWeatherResult(edited_csv_path)

    def _fake_evaluate_hexels(**kwargs):
        evaluation_calls.append(kwargs)
        return {"mae": 1.5}

    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr("src.evaluate_counterfactual.materialize_weather_scenario", _fake_materialize_weather_scenario)
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", _fake_evaluate_hexels)

    evaluation_calls: list[dict] = []
    index = run_counterfactual_evaluation(
        config_path,
        endpoint_names={"bp"},
        scenario_names={"bc_mean_weather_transplant"},
        overwrite=False,
        project_root=project_root,
    )

    assert index[["scenario", "endpoint"]].to_dict("records") == [
        {"scenario": "baseline", "endpoint": "bp"},
        {"scenario": "bc_mean_weather_transplant", "endpoint": "bp"},
    ]
    assert len(materialize_calls) == 1
    assert materialize_calls[0]["recipient_hex_ids"] == ["16"]
    # The weather scenario's run config gets its spatialized_weather source repointed at the edited CSV.
    assert evaluation_calls[0]["config"].data.input_sources[0].params.csv_name == "weather_table_processed.csv"
    assert evaluation_calls[1]["config"].data.input_sources[0].params.csv_name == str(edited_csv_path.resolve())
    assert evaluation_calls[1]["config"].data.input_sources[0].params.global_fill_csv_name == str(
        (data_root / "weather_table_processed.csv").resolve()
    )
    weather_summary = pd.read_csv(save_dir / "weather_edit_summary.csv")
    assert weather_summary[["endpoint", "donor_fwi_mean"]].to_dict("records") == [{"endpoint": "bp", "donor_fwi_mean": 29.0}]


class _FakeFireSizeResult:
    def __init__(self, edited_csv_path: Path, global_fill_csv_path: Path) -> None:
        self.edited_csv_path = edited_csv_path
        self.global_fill_csv_path = global_fill_csv_path
        self.feature_columns = ("NORM_LOG_SIZE_HA_q10", "NORM_LOG_SIZE_HA_q50", "NORM_LOG_SIZE_HA_q90")
        self.summary = pd.DataFrame([{"scenario_name": "spread_day_fire_size", "hex_id": "16", "fire_size_multiplier_q90": 5.0625}])


def test_run_counterfactual_evaluation_orchestrates_fire_size_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    data_root = project_root / "data"
    source_dir = project_root / "trained"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    source_dir.mkdir()
    (source_dir / "best.pt").write_bytes(b"checkpoint")
    pd.DataFrame([{"hex_id": "16", "filename": "patch.npy", "valid_ratio": 1.0}]).to_csv(
        data_root / "test.csv",
        index=False,
    )
    (data_root / "df_fire_fru_processed.csv").write_text("GRIDCODE,NORM_LOG_SIZE_HA\n10,0.5\n")
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml"}},
                "scenarios": [
                    {"name": "baseline", "kind": "baseline"},
                    {
                        "name": "spread_day_fire_size",
                        "kind": "fire_size",
                        "params": {
                            "mode": "spread_day_quantile_scaling",
                            "spread_day_delta_q50_days": 0.4,
                            "spread_day_delta_q90_days": 5.0,
                            "size_scaling_exponent": 2.0,
                        },
                    },
                ],
            },
            handle,
        )

    fire_size_source = _DataSourceConfig(
        name="spatialized_fire_size",
        params=_DataSourceParams(
            csv_name="df_fire_fru_processed.csv",
            feature_names_list=["NORM_LOG_SIZE_HA"],
            quantiles=[0.1, 0.5, 0.9],
        ),
    )
    run_config = _EndpointRunConfig(
        save_dir=str(source_dir),
        data=_DataConfig(root_dir=str(data_root), input_sources=[fire_size_source]),
    )
    edited_csv_path = tmp_path / "edited_fire_size.csv"
    fill_csv_path = tmp_path / "fire_size_fill.csv"
    edited_csv_path.write_text("hex_id,GRIDCODE\n")
    fill_csv_path.write_text("hex_id\n")
    materialize_calls: list[dict] = []
    evaluation_calls: list[dict] = []

    def _fake_materialize_fire_size_scenario(**kwargs):
        materialize_calls.append(kwargs)
        return _FakeFireSizeResult(edited_csv_path, fill_csv_path)

    def _fake_evaluate_hexels(**kwargs):
        evaluation_calls.append(kwargs)
        return {"mae": 1.5}

    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr(
        "src.evaluate_counterfactual.materialize_fire_size_scenario",
        _fake_materialize_fire_size_scenario,
    )
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", _fake_evaluate_hexels)

    index = run_counterfactual_evaluation(
        config_path,
        endpoint_names={"bp"},
        scenario_names={"spread_day_fire_size"},
        overwrite=False,
        project_root=project_root,
    )

    assert index[["scenario", "endpoint"]].to_dict("records") == [
        {"scenario": "baseline", "endpoint": "bp"},
        {"scenario": "spread_day_fire_size", "endpoint": "bp"},
    ]
    assert len(materialize_calls) == 1
    assert materialize_calls[0]["feature_name"] == "NORM_LOG_SIZE_HA"
    assert materialize_calls[0]["quantiles"] == [0.1, 0.5, 0.9]
    baseline_params = evaluation_calls[0]["config"].data.input_sources[0].params
    scenario_params = evaluation_calls[1]["config"].data.input_sources[0].params
    assert baseline_params.quantiles == [0.1, 0.5, 0.9]
    assert scenario_params.csv_name == str(edited_csv_path.resolve())
    assert scenario_params.global_fill_csv_name == str(fill_csv_path.resolve())
    assert scenario_params.feature_names_list == list(_FakeFireSizeResult(edited_csv_path, fill_csv_path).feature_columns)
    assert scenario_params.quantiles is None
    assert scenario_params.hex_id_col == "hex_id"
    summary = pd.read_csv(save_dir / "fire_size_edit_summary.csv")
    assert summary[["endpoint", "fire_size_multiplier_q90"]].to_dict("records") == [{"endpoint": "bp", "fire_size_multiplier_q90": 5.0625}]


def test_run_counterfactual_evaluation_dedupes_endpoints_sharing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple endpoint names pointing at the same multi-output checkpoint (e.g. bp/fi
    aliases for one joint model) should only trigger one evaluate_hexels run per scenario,
    with every alias endpoint reusing that run's prediction_dir and metrics."""
    project_root = tmp_path
    data_root = project_root / "data"
    source_dir = project_root / "trained"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    source_dir.mkdir()
    (source_dir / "best.pt").write_bytes(b"checkpoint")
    pd.DataFrame([{"hex_id": "16", "filename": "patch.npy", "valid_ratio": 1.0}]).to_csv(data_root / "test.csv", index=False)
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {
                    "bp": {"config_path": "shared.yaml"},
                    "fi": {"config_path": "shared.yaml"},
                },
                "scenarios": [
                    {"name": "baseline", "kind": "baseline"},
                ],
            },
            handle,
        )

    run_config = _EndpointRunConfig(
        save_dir=str(source_dir),
        data=_DataConfig(root_dir=str(data_root)),
    )
    evaluation_calls: list[dict] = []

    def _fake_evaluate_hexels(**kwargs):
        evaluation_calls.append(kwargs)
        return {"bp_mae": 1.5, "fi_mae": 0.3}

    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", _fake_evaluate_hexels)

    index = run_counterfactual_evaluation(
        config_path,
        endpoint_names=None,
        scenario_names={"baseline"},
        overwrite=False,
        project_root=project_root,
    )

    assert len(evaluation_calls) == 1
    assert index[["scenario", "endpoint"]].to_dict("records") == [
        {"scenario": "baseline", "endpoint": "bp"},
        {"scenario": "baseline", "endpoint": "fi"},
    ]
    assert index["prediction_dir"].nunique() == 1
    metrics = pd.read_csv(save_dir / "counterfactual_metrics.csv")
    assert set(metrics["endpoint"]) == {"bp", "fi"}


def test_run_counterfactual_evaluation_reads_checkpoint_from_endpoint_checkpoint_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    data_root = project_root / "data"
    training_dir = project_root / "trained"
    shared_checkpoint_dir = project_root / "shared_checkpoints"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    training_dir.mkdir()
    shared_checkpoint_dir.mkdir()
    # Only the shared directory holds a checkpoint, so resolution must not fall back
    # to the endpoint config's own training save_dir.
    (shared_checkpoint_dir / "best.pt").write_bytes(b"shared-checkpoint")
    pd.DataFrame([{"hex_id": "16", "filename": "patch.npy", "valid_ratio": 1.0}]).to_csv(data_root / "test.csv", index=False)
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml", "checkpoint_dir": str(shared_checkpoint_dir)}},
                "scenarios": [{"name": "baseline", "kind": "baseline"}],
            },
            handle,
        )

    run_config = _EndpointRunConfig(save_dir=str(training_dir), data=_DataConfig(root_dir=str(data_root), input_sources=[]))
    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", lambda **_: {"mae": 1.5})

    run_counterfactual_evaluation(config_path, overwrite=False, project_root=project_root)

    copied = save_dir / "predictions" / "baseline" / "bp" / "best.pt"
    assert copied.read_bytes() == b"shared-checkpoint"


def test_run_counterfactual_evaluation_uses_seeded_checkpoint_and_output_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    data_root = project_root / "data"
    training_dir = project_root / "trained"
    shared_checkpoint_dir = project_root / "shared_checkpoints"
    seeded_checkpoint_dir = shared_checkpoint_dir / "seed_1337"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    training_dir.mkdir()
    seeded_checkpoint_dir.mkdir(parents=True)
    (seeded_checkpoint_dir / "best.pt").write_bytes(b"seeded-checkpoint")
    pd.DataFrame([{"hex_id": "16", "filename": "patch.npy", "valid_ratio": 1.0}]).to_csv(data_root / "test.csv", index=False)
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml", "checkpoint_dir": str(shared_checkpoint_dir)}},
                "scenarios": [{"name": "baseline", "kind": "baseline"}],
            },
            handle,
        )

    run_config = _EndpointRunConfig(save_dir=str(training_dir), data=_DataConfig(root_dir=str(data_root), input_sources=[]))
    evaluation_calls: list[dict] = []

    def _fake_evaluate_hexels(**kwargs):
        evaluation_calls.append(kwargs)
        return {"mae": 1.5}

    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", _fake_evaluate_hexels)

    index = run_counterfactual_evaluation(config_path, overwrite=False, run_id=1, project_root=project_root)

    seeded_save_dir = save_dir / "seed_1337"
    copied = seeded_save_dir / "predictions" / "baseline" / "bp" / "best.pt"
    assert copied.read_bytes() == b"seeded-checkpoint"
    assert evaluation_calls[0]["config"].seed == 1337
    assert index[["run_id", "seed"]].to_dict("records") == [{"run_id": 1, "seed": 1337}]
    metrics = pd.read_csv(seeded_save_dir / "counterfactual_metrics.csv")
    assert metrics[["run_id", "seed"]].to_dict("records") == [{"run_id": 1, "seed": 1337}]


def test_run_counterfactual_evaluation_rejects_unknown_run_id(tmp_path: Path) -> None:
    config_path = tmp_path / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml"}},
                "scenarios": [{"name": "baseline", "kind": "baseline"}],
            },
            handle,
        )

    with pytest.raises(ValueError, match="run_id must be between 0 and 2"):
        run_counterfactual_evaluation(config_path, run_id=3, project_root=tmp_path)
