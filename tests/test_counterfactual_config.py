from pathlib import Path

import pytest
import yaml

from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    load_counterfactual_config,
    resolve_counterfactual_paths,
)


def _write_config(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def test_load_counterfactual_config_supports_custom_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["hex1", 16],
            "endpoints": {
                "hazard": {
                    "config_path": "configs/hazard.yaml",
                    "checkpoint_dir": "/checkpoints/hazard",
                }
            },
            "scenarios": [
                {"name": "baseline", "kind": "baseline"},
                {
                    "name": "remove_barriers",
                    "kind": "fuel",
                    "params": {"mode": "nonfuel_to_burnable_local_adjacent_modal"},
                },
            ],
        },
    )

    config = load_counterfactual_config(path)

    assert config.hex_ids == ["01", "16"]
    assert list(config.endpoints) == ["hazard"]
    assert config.endpoints["hazard"].checkpoint_dir == Path("/checkpoints/hazard")
    assert config.scenarios[1].fuel_edit() == {"mode": "nonfuel_to_burnable_local_adjacent_modal"}
    assert config.scenario("remove_barriers") == config.scenarios[1]


def test_load_counterfactual_config_supports_weather_scenarios(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
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
    )

    config = load_counterfactual_config(path)

    assert config.scenarios[1].weather_edit() == {"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]}
    assert config.scenarios[1].fuel_edit() is None
    with pytest.raises(KeyError, match="missing"):
        config.scenario("missing")


def test_resolve_counterfactual_paths_uses_config_defaults_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "raw",
            "save_dir": "experiment",
            "hex_ids": ["16"],
            "endpoints": {"bp": {"config_path": "bp.yaml"}},
            "scenarios": [{"name": "baseline", "kind": "baseline"}],
        },
    )
    config = load_counterfactual_config(path)

    assert resolve_counterfactual_paths(config, project_root=tmp_path) == (
        tmp_path / "experiment",
        tmp_path / "raw",
    )
    assert resolve_counterfactual_paths(
        config,
        experiment_dir=Path("override"),
        raw_data_dir=Path("/raw-override"),
        project_root=tmp_path,
    ) == (tmp_path / "override", Path("/raw-override"))


def test_load_counterfactual_config_requires_baseline(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["16"],
            "endpoints": {"bp": {"config_path": "bp.yaml"}},
            "scenarios": [{"name": "edit", "kind": "fuel"}],
        },
    )

    with pytest.raises(ValueError, match="baseline"):
        load_counterfactual_config(path)


def test_load_counterfactual_config_requires_exactly_one_baseline(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["16"],
            "endpoints": {"bp": {"config_path": "bp.yaml"}},
            "scenarios": [
                {"name": "baseline", "kind": "baseline"},
                {"name": "reference", "kind": "baseline"},
            ],
        },
    )

    with pytest.raises(ValueError, match="Exactly one baseline"):
        load_counterfactual_config(path)


def test_load_counterfactual_config_requires_standard_baseline_name(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["16"],
            "endpoints": {"bp": {"config_path": "bp.yaml"}},
            "scenarios": [{"name": "reference", "kind": "baseline"}],
        },
    )

    with pytest.raises(ValueError, match="named 'baseline'"):
        load_counterfactual_config(path)
