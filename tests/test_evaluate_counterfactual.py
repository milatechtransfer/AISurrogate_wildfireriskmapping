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
