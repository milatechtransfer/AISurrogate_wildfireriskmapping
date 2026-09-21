"""Configuration models for reproducible counterfactual inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.datasets.fuel_utils import normalize_hex_id

SCENARIO_KINDS = ("baseline", "fuel", "weather")


@dataclass(frozen=True)
class EndpointConfig:
    """A trained-model endpoint (e.g. bp/fi/ros) to evaluate under each scenario."""

    name: str
    config_path: Path
    baseline_data_root: Path | None = None
    checkpoint_dir: Path | None = None

    @classmethod
    def from_mapping(cls, name: str, raw: object) -> EndpointConfig:
        """Parse an endpoint entry from the raw YAML mapping under `endpoints.<name>`."""
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint {name!r} must be a mapping.")
        if "config_path" not in raw:
            raise ValueError(f"Endpoint {name!r} is missing required key 'config_path'.")
        baseline_data_root = raw.get("baseline_data_root")
        checkpoint_dir = raw.get("checkpoint_dir")
        return cls(
            name=name,
            config_path=Path(str(raw["config_path"])),
            baseline_data_root=Path(str(baseline_data_root)) if baseline_data_root else None,
            checkpoint_dir=Path(str(checkpoint_dir)) if checkpoint_dir else None,
        )


@dataclass(frozen=True)
class ScenarioConfig:
    """A named baseline, fuel-edit, or weather-edit counterfactual scenario."""

    name: str
    kind: str
    description: str
    params: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: object) -> ScenarioConfig:
        """Parse a scenario entry from a raw YAML mapping under `scenarios`."""
        if not isinstance(raw, dict):
            raise ValueError("Each scenario must be a mapping.")
        missing = [key for key in ("name", "kind") if key not in raw]
        if missing:
            raise ValueError(f"Scenario is missing required keys: {missing}.")
        kind = str(raw["kind"])
        if kind not in SCENARIO_KINDS:
            raise ValueError(f"Scenario {raw['name']!r} has kind={kind!r}; expected one of {SCENARIO_KINDS}.")
        params = raw.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Scenario {raw['name']!r} key 'params' must be a mapping.")
        return cls(
            name=str(raw["name"]),
            kind=kind,
            description=str(raw.get("description", "")),
            params=params,
        )

    def fuel_edit(self) -> dict[str, Any] | None:
        """Return this scenario's fuel-edit params, or None for the baseline scenario."""
        return self.params if self.kind == "fuel" else None

    def weather_edit(self) -> dict[str, Any] | None:
        """Return this scenario's weather-edit params, or None otherwise."""
        return self.params if self.kind == "weather" else None


@dataclass(frozen=True)
class CounterfactualConfig:
    """Top-level counterfactual run configuration: data paths, hexels, endpoints, and scenarios."""

    raw_data_dir: Path
    save_dir: Path
    hex_ids: list[str]
    endpoints: dict[str, EndpointConfig]
    scenarios: list[ScenarioConfig]
    # Non-fuel IDs to exclude from response/delta maps for scenarios that don't edit fuel
    # themselves (e.g. weather counterfactuals), where the fuel grid is static across
    # baseline and scenario. Fuel-edit scenarios instead carry their own `nonfuel_ids` in
    # `params`, since baseline/scenario fuel differs there.
    nonfuel_ids: list[int] | None = None

    def scenario(self, name: str) -> ScenarioConfig:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise KeyError(f"Scenario {name!r} is not defined.")


def resolve_project_path(path: Path | str, project_root: Path | None = None) -> Path:
    root = (project_root or Path.cwd()).resolve()
    value = Path(path)
    return value if value.is_absolute() else root / value


def resolve_counterfactual_paths(
    config: CounterfactualConfig,
    *,
    experiment_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    return (
        resolve_project_path(experiment_dir or config.save_dir, project_root),
        resolve_project_path(raw_data_dir or config.raw_data_dir, project_root),
    )


def _parse_hex_ids(raw_hex_ids: object) -> list[str]:
    if not isinstance(raw_hex_ids, list | tuple) or not raw_hex_ids:
        raise ValueError("Counterfactual config key 'hex_ids' must be a non-empty list.")
    return [normalize_hex_id(hex_id) for hex_id in raw_hex_ids]


def _parse_endpoints(raw_endpoints: object) -> dict[str, EndpointConfig]:
    if not isinstance(raw_endpoints, dict) or not raw_endpoints:
        raise ValueError("Counterfactual config key 'endpoints' must be a non-empty mapping.")
    return {str(name): EndpointConfig.from_mapping(str(name), raw) for name, raw in raw_endpoints.items()}


def _parse_nonfuel_ids(raw_nonfuel_ids: object) -> list[int] | None:
    if raw_nonfuel_ids is None:
        return None
    if not isinstance(raw_nonfuel_ids, list | tuple):
        raise ValueError("Counterfactual config key 'nonfuel_ids' must be a list.")
    return [int(value) for value in raw_nonfuel_ids]


def _parse_scenarios(raw_scenarios: object) -> list[ScenarioConfig]:
    if not isinstance(raw_scenarios, list | tuple) or not raw_scenarios:
        raise ValueError("Counterfactual config key 'scenarios' must be a non-empty list.")
    scenarios = [ScenarioConfig.from_mapping(raw) for raw in raw_scenarios]
    names = [scenario.name for scenario in scenarios]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Scenario names must be unique; duplicates: {duplicates}.")
    baselines = [scenario for scenario in scenarios if scenario.kind == "baseline"]
    if len(baselines) != 1:
        raise ValueError(f"Exactly one baseline scenario is required; found {len(baselines)}.")
    if baselines[0].name != "baseline":
        raise ValueError("The baseline scenario must be named 'baseline'.")
    return scenarios


def load_counterfactual_config(path: Path) -> CounterfactualConfig:
    """Load and validate a counterfactual run config (e.g. `configs/counterfactual_fuel.yaml`)."""
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Counterfactual config must be a mapping, got {type(raw).__name__}.")

    required = ("raw_data_dir", "save_dir", "hex_ids", "endpoints", "scenarios")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required counterfactual config keys: {missing}.")

    return CounterfactualConfig(
        raw_data_dir=Path(str(raw["raw_data_dir"])),
        save_dir=Path(str(raw["save_dir"])),
        hex_ids=_parse_hex_ids(raw["hex_ids"]),
        endpoints=_parse_endpoints(raw["endpoints"]),
        scenarios=_parse_scenarios(raw["scenarios"]),
        nonfuel_ids=_parse_nonfuel_ids(raw.get("nonfuel_ids")),
    )
