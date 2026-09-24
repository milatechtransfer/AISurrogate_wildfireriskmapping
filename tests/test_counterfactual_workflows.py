from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from src.datasets.postprocessing.counterfactual.counterfactual_base import load_counterfactual_config

RUN_DIR = Path("run_files/counterfactual")
SHARED_Q3_CHECKPOINT_DIR = Path("/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/unet_256_firesize_q3")
RETAINED_EXPERIMENTS = {
    "fuel": (
        Path("configs/counterfactual/counterfactual_fuel_multi_output.yaml"),
        "c2_to_mixedwood_fixed",
        "fuel",
    ),
    "fuel_polygons": (
        Path("configs/counterfactual/counterfactual_fuel_polygons_multi_output.yaml"),
        "pooled_burn_scars_to_aspen",
        "fuel",
    ),
    "mean_weather": (
        Path("configs/counterfactual/counterfactual_mean_weather_multi_output.yaml"),
        "bc_mean_weather_transplant",
        "weather",
    ),
    "fire_size": (
        Path("configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml"),
        "spread_days_q50_plus_0p7_q90_plus_5_beta2",
        "fire_size",
    ),
}


@pytest.mark.parametrize(("config_path", "scenario_name", "scenario_kind"), RETAINED_EXPERIMENTS.values())
def test_retained_configs_target_shared_q3_checkpoint(
    config_path: Path,
    scenario_name: str,
    scenario_kind: str,
) -> None:
    config = load_counterfactual_config(config_path)

    assert config.hex_ids == ["16"]
    assert set(config.endpoints) == {"bp", "fi", "ros"}
    assert {endpoint.config_path for endpoint in config.endpoints.values()} == {
        Path("configs/multi_output_spatial_weather_firesize_q3.yaml")
    }
    assert {endpoint.checkpoint_dir for endpoint in config.endpoints.values()} == {SHARED_Q3_CHECKPOINT_DIR}
    assert [(scenario.name, scenario.kind) for scenario in config.scenarios if scenario.kind != "baseline"] == [
        (scenario_name, scenario_kind)
    ]


def test_default_submitter_runs_exactly_four_multiseed_experiments() -> None:
    submitter = RUN_DIR / "submit_all_counterfactuals.sh"
    text = submitter.read_text()
    experiment_block = re.search(r"^EXPERIMENTS=\((.*?)^\)", text, re.DOTALL | re.MULTILINE)
    assert experiment_block is not None

    entries = re.findall(r'"([^"]+)"', experiment_block.group(1))
    expected = [
        f"{name}:{config_path}:{scenario_name}:{scenario_kind}"
        for name, (config_path, scenario_name, scenario_kind) in RETAINED_EXPERIMENTS.items()
    ]
    assert entries == expected

    result = subprocess.run(
        ["bash", str(submitter), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("sbatch --array=0-2") == 8
    assert result.stdout.count("counterfactual_multiseed_aggregate.sh") == 4


def test_submitter_single_seed_mode_runs_seed_42_without_aggregation() -> None:
    submitter = RUN_DIR / "submit_all_counterfactuals.sh"

    result = subprocess.run(
        ["bash", str(submitter), "--dry-run", "--single-seed"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--array=0-2" not in result.stdout
    assert result.stdout.count("sbatch --array=0 ") == 8
    assert "counterfactual_multiseed_aggregate.sh" not in result.stdout
    assert result.stdout.count("SKIPPED") == 4
