import re
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


def test_load_counterfactual_config_supports_fire_size_scenarios(tmp_path: Path) -> None:
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
    )

    config = load_counterfactual_config(path)

    assert config.scenarios[1].fire_size_edit() == {
        "mode": "spread_day_quantile_scaling",
        "spread_day_delta_q50_days": 0.4,
        "spread_day_delta_q90_days": 5.0,
        "size_scaling_exponent": 2.0,
    }
    assert config.scenarios[1].fuel_edit() is None
    assert config.scenarios[1].weather_edit() is None


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


def test_endpoint_checkpoint_dir_defaults_to_none_and_parses_when_set(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["16"],
            "endpoints": {
                "bp": {"config_path": "bp.yaml"},
                "fi": {"config_path": "fi.yaml", "checkpoint_dir": "/shared/checkpoints/unet"},
            },
            "scenarios": [{"name": "baseline", "kind": "baseline"}],
        },
    )

    config = load_counterfactual_config(path)

    assert config.endpoints["bp"].checkpoint_dir is None
    assert config.endpoints["fi"].checkpoint_dir == Path("/shared/checkpoints/unet")


SHARED_Q3_CHECKPOINT_DIR = Path("/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/unet_256_firesize_q3")
SHIPPED_MULTI_OUTPUT_CONFIGS = {
    Path("configs/counterfactual/counterfactual_windy_weather_zone_dependent_multi_output.yaml"): (
        "weather",
        "windy_mean_zone_dependent_transplant",
    ),
    Path("configs/counterfactual/counterfactual_wind_direction_zone_dependent_multi_output.yaml"): (
        "weather",
        "wind_direction_zone_dependent_transplant",
    ),
    Path("configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml"): (
        "fire_size",
        "spread_day_quantile_scaling",
    ),
    Path("configs/counterfactual/counterfactual_fuel_multi_output.yaml"): (
        "fuel",
        "burnable_to_burnable_fixed",
    ),
    Path("configs/counterfactual/counterfactual_mean_weather_multi_output.yaml"): (
        "weather",
        "external_mean_zone_transplant",
    ),
}


@pytest.mark.parametrize(("config_path", "expected"), sorted(SHIPPED_MULTI_OUTPUT_CONFIGS.items()))
def test_shipped_counterfactual_configs_target_shared_q3_checkpoint(config_path: Path, expected: tuple[str, str]) -> None:
    expected_kind, expected_mode = expected
    config = load_counterfactual_config(config_path)

    assert config.hex_ids == ["16"]
    assert set(config.endpoints) == {"bp", "fi", "ros"}
    for endpoint in config.endpoints.values():
        assert endpoint.config_path == Path("configs/multi_output_spatial_weather_firesize_q3.yaml")
        assert endpoint.checkpoint_dir == SHARED_Q3_CHECKPOINT_DIR

    edited = [scenario for scenario in config.scenarios if scenario.kind != "baseline"]
    assert edited, f"{config_path} defines no non-baseline scenarios."
    for scenario in edited:
        assert scenario.kind == expected_kind
        assert scenario.params["mode"] == expected_mode
        assert scenario.description


COUNTERFACTUAL_RUN_FILES_DIR = Path("run_files/counterfactual")
_SHELL_CONFIG_PATTERN = re.compile(r"configs/\S+?\.yaml")
_SHELL_SCENARIO_LIST_PATTERN = re.compile(r"^scenarios=\((.*?)^\)", re.DOTALL | re.MULTILINE)
_SHELL_SCENARIO_SCALAR_PATTERN = re.compile(r'^scenario="([^"]+)"', re.MULTILINE)


@pytest.mark.parametrize("script_path", sorted(COUNTERFACTUAL_RUN_FILES_DIR.glob("*.sh")))
def test_counterfactual_run_scripts_reference_existing_configs(script_path: Path) -> None:
    """Every config path named in a SLURM script must exist on disk."""
    text = script_path.read_text()
    referenced = set(_SHELL_CONFIG_PATTERN.findall(text))
    assert referenced, f"{script_path} references no config."
    for config_path in referenced:
        assert Path(config_path).is_file(), f"{script_path} references missing config {config_path}."


@pytest.mark.parametrize("script_path", sorted(COUNTERFACTUAL_RUN_FILES_DIR.glob("*_plots.sh")))
def test_counterfactual_plot_scripts_only_reference_declared_scenarios(script_path: Path) -> None:
    """Plot scripts must not name scenarios that their config does not define."""
    text = script_path.read_text()
    scenario_block = _SHELL_SCENARIO_LIST_PATTERN.search(text)
    scripted = set(_SHELL_SCENARIO_SCALAR_PATTERN.findall(text))
    if scenario_block is not None:
        scripted |= set(re.findall(r'"([^"]+)"', scenario_block.group(1)))
    if not scripted:
        pytest.skip(f"{script_path} names no scenarios.")

    config_paths = sorted(set(_SHELL_CONFIG_PATTERN.findall(text)))
    assert len(config_paths) == 1, f"{script_path} references {config_paths}; expected exactly one config."
    declared = {scenario.name for scenario in load_counterfactual_config(Path(config_paths[0])).scenarios}

    unknown = scripted - declared
    assert not unknown, f"{script_path} plots undeclared scenario(s) {sorted(unknown)}; config declares {sorted(declared)}."
