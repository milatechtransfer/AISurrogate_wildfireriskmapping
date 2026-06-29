"""Configuration and wind-convention helpers for counterfactual analyses.

The counterfactual pipeline is intentionally split into small pieces.  This
module owns only the reproducible experiment specification and the wind
geometry convention; weather-table editing, fuel editing, model inference, and
paired metrics live in later modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from data_preparation.paths import MASK_SCOPE_CHOICES, MaskScope, normalize_mask_scope

ENDPOINTS: tuple[str, ...] = ("bp", "ros", "fi")
SCENARIO_KINDS: tuple[str, ...] = ("baseline", "wind", "fuel", "fwi", "wind_regime", "wind_direction", "composite")
FUEL_EDIT_KINDS: tuple[str, ...] = ("fuel",)
WEATHER_EDIT_KINDS: tuple[str, ...] = ("fwi", "wind_regime", "wind_direction")
SUPPORT_POLICIES: tuple[str, ...] = ("baseline", "intersection")
DEFAULT_FOCUS_HEX_ID = "16"


@dataclass(frozen=True)
class EndpointConfig:
    """Model/data assets for one prediction endpoint."""

    name: str
    config_path: Path
    baseline_data_root: Path | None = None
    enabled: bool = True

    @classmethod
    def from_mapping(cls, name: str, raw: object) -> "EndpointConfig":
        if name not in ENDPOINTS:
            raise ValueError(f"Unknown endpoint {name!r}; expected one of {ENDPOINTS}.")
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint {name!r} must be a mapping.")
        if "config_path" not in raw:
            raise ValueError(f"Endpoint {name!r} is missing required key 'config_path'.")
        baseline_data_root = raw.get("baseline_data_root")
        return cls(
            name=name,
            config_path=Path(str(raw["config_path"])),
            baseline_data_root=Path(str(baseline_data_root)) if baseline_data_root else None,
            enabled=bool(raw.get("enabled", True)),
        )


def _validate_composite_params(name: str, params: dict[str, Any]) -> None:
    edits = params.get("edits")
    if not isinstance(edits, list | tuple) or not edits:
        raise ValueError(f"Composite scenario {name!r} must define a non-empty 'edits' list.")
    seen_groups: set[str] = set()
    for edit in edits:
        if not isinstance(edit, dict) or "kind" not in edit:
            raise ValueError(f"Composite scenario {name!r} has an edit without a 'kind'.")
        edit_kind = str(edit["kind"])
        if edit_kind not in (*FUEL_EDIT_KINDS, *WEATHER_EDIT_KINDS):
            raise ValueError(
                f"Composite scenario {name!r} edit kind={edit_kind!r} must be one of " f"{(*FUEL_EDIT_KINDS, *WEATHER_EDIT_KINDS)}."
            )
        group = "fuel" if edit_kind in FUEL_EDIT_KINDS else "weather"
        if group in seen_groups:
            raise ValueError(f"Composite scenario {name!r} has more than one {group} edit; at most one is supported.")
        seen_groups.add(group)


@dataclass(frozen=True)
class ScenarioConfig:
    """One counterfactual scenario definition."""

    name: str
    kind: str
    description: str
    params: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: object) -> "ScenarioConfig":
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
        if kind == "composite":
            _validate_composite_params(str(raw["name"]), params)
        return cls(
            name=str(raw["name"]),
            kind=kind,
            description=str(raw.get("description", "")),
            params=params,
        )

    def edit_specs(self) -> list[tuple[str, dict[str, Any]]]:
        """Constituent (kind, params) edits; a composite expands to its sub-edits."""

        if self.kind != "composite":
            return [(self.kind, self.params)]
        return [(str(edit["kind"]), dict(edit.get("params", {}))) for edit in self.params["edits"]]

    def fuel_edit(self) -> dict[str, Any] | None:
        """Params of the fuel-grid edit this scenario applies, or None."""

        return next((params for kind, params in self.edit_specs() if kind in FUEL_EDIT_KINDS), None)

    def weather_edit(self) -> tuple[str, dict[str, Any]] | None:
        """The (kind, params) of the weather-table edit this scenario applies, or None."""

        return next(((kind, params) for kind, params in self.edit_specs() if kind in WEATHER_EDIT_KINDS), None)


@dataclass(frozen=True)
class CounterfactualConfig:
    """Top-level counterfactual experiment specification."""

    raw_data_dir: Path
    save_dir: Path
    hex_ids: list[str]
    focus_hex_id: str
    mask_scope: MaskScope
    support_policy: str
    endpoints: dict[str, EndpointConfig]
    scenarios: list[ScenarioConfig]
    seed: int

    @property
    def enabled_endpoints(self) -> dict[str, EndpointConfig]:
        return {name: endpoint for name, endpoint in self.endpoints.items() if endpoint.enabled}


def _parse_hex_ids(raw_hex_ids: object) -> list[str]:
    if raw_hex_ids in (None, "hex16", "16"):
        return [DEFAULT_FOCUS_HEX_ID]
    if raw_hex_ids == "all":
        raise ValueError("Counterfactual runs require an explicit hex list; use ['16'] for the first smoke test.")
    if not isinstance(raw_hex_ids, list | tuple) or not raw_hex_ids:
        raise ValueError("Counterfactual config key 'hex_ids' must be a non-empty list.")
    return [str(hex_id).zfill(2) for hex_id in raw_hex_ids]


def _parse_endpoints(raw_endpoints: object) -> dict[str, EndpointConfig]:
    if not isinstance(raw_endpoints, dict) or not raw_endpoints:
        raise ValueError("Counterfactual config key 'endpoints' must be a non-empty mapping.")
    endpoints = {str(name): EndpointConfig.from_mapping(str(name), endpoint_raw) for name, endpoint_raw in raw_endpoints.items()}
    if "bp" not in endpoints or not endpoints["bp"].enabled:
        raise ValueError("BP must be configured and enabled because it is the primary barrier-response endpoint.")
    return endpoints


def _parse_scenarios(raw_scenarios: object) -> list[ScenarioConfig]:
    if not isinstance(raw_scenarios, list | tuple) or not raw_scenarios:
        raise ValueError("Counterfactual config key 'scenarios' must be a non-empty list.")
    scenarios = [ScenarioConfig.from_mapping(raw) for raw in raw_scenarios]
    names = [scenario.name for scenario in scenarios]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Scenario names must be unique; duplicates: {duplicate_names}.")
    if not any(scenario.kind == "baseline" for scenario in scenarios):
        raise ValueError("At least one baseline scenario is required for paired deltas.")
    return scenarios


def load_counterfactual_config(path: Path) -> CounterfactualConfig:
    """Load and validate a YAML counterfactual experiment config."""

    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Counterfactual config must be a mapping, got {type(raw).__name__}.")

    required = ("raw_data_dir", "save_dir", "endpoints", "scenarios")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required counterfactual config keys: {missing}.")

    hex_ids = _parse_hex_ids(raw.get("hex_ids", [DEFAULT_FOCUS_HEX_ID]))
    focus_hex_id = str(raw.get("focus_hex_id", hex_ids[0])).zfill(2)
    if focus_hex_id not in hex_ids:
        raise ValueError(f"focus_hex_id={focus_hex_id!r} must be included in hex_ids={hex_ids}.")

    support_policy = str(raw.get("support_policy", "baseline"))
    if support_policy not in SUPPORT_POLICIES:
        raise ValueError(f"support_policy={support_policy!r}; expected one of {SUPPORT_POLICIES}.")

    mask_scope = normalize_mask_scope(str(raw.get("mask_scope", "actual")))
    if mask_scope not in MASK_SCOPE_CHOICES:
        raise ValueError(f"mask_scope={mask_scope!r}; expected one of {MASK_SCOPE_CHOICES}.")

    return CounterfactualConfig(
        raw_data_dir=Path(str(raw["raw_data_dir"])),
        save_dir=Path(str(raw["save_dir"])),
        hex_ids=hex_ids,
        focus_hex_id=focus_hex_id,
        mask_scope=mask_scope,
        support_policy=support_policy,
        endpoints=_parse_endpoints(raw["endpoints"]),
        scenarios=_parse_scenarios(raw["scenarios"]),
        seed=int(raw.get("seed", 42)),
    )


def bearing_to_components(speed: float | np.ndarray, bearing_deg: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a compass bearing into x/y components.

    Bearing convention: 0 degrees = north, 90 degrees = east.
    """

    speed_arr = np.asarray(speed, dtype=np.float64)
    radians = np.deg2rad(np.asarray(bearing_deg, dtype=np.float64))
    return speed_arr * np.sin(radians), speed_arr * np.cos(radians)


def encoded_components_from_from_bearing(
    wind_speed: float | np.ndarray,
    wind_from_bearing_deg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current model-encoded wind components for a from-bearing.

    BurnP3+/NRCan WindDirection is treated as a meteorological from-bearing.
    The existing feature encoding therefore points toward the wind source
    (upwind), not toward physical flow/downwind.
    """

    return bearing_to_components(wind_speed, wind_from_bearing_deg)
