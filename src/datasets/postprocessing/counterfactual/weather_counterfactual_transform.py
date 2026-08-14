"""Materialize an edited weather lookup table for one weather counterfactual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.counterfactual_weather import (
    WEATHER_NORM_PARAMS_FILENAME,
    apply_weather_edit,
    load_all_raw_weather_with_hex_ids,
)

WEATHER_INTERVENTION_CSV_NAME = "weather_table_processed.csv"


def weather_intervention_csv_path(prediction_dir: Path) -> Path:
    return prediction_dir / "weather_intervention" / WEATHER_INTERVENTION_CSV_NAME


@dataclass(frozen=True)
class WeatherCounterfactualResult:
    """The edited weather table written for one scenario, plus its edit summary."""

    edited_csv_path: Path
    summary: pd.DataFrame


def materialize_weather_scenario(
    *,
    scenario: ScenarioConfig,
    raw_data_dir: Path,
    processed_weather_csv: Path,
    recipient_hex_ids: list[str],
    prediction_dir: Path,
) -> WeatherCounterfactualResult:
    """Apply a configured weather edit and write its lookup table under `prediction_dir`.

    Loads `processed_weather_csv` (the endpoint's shared, unedited
    `weather_table_processed.csv`) and the raw per-hexel weather tables it was built
    from, applies the scenario's edit while preserving one lookup row per
    `(hex_id, WeatherZone)`, and writes the edited table to
    `weather_intervention_csv_path(prediction_dir)` - ready to be pointed at by
    overriding the endpoint's `spatialized_weather` source `csv_name` with an
    absolute path.
    """
    params = dict(scenario.weather_edit() or {})
    mode = params.pop("mode", None)
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"Weather scenario {scenario.name!r} must define an explicit non-empty mode.")

    processed = pd.read_csv(processed_weather_csv)
    raw_features = load_all_raw_weather_with_hex_ids(raw_data_dir)
    norm_params_path = processed_weather_csv.parent / WEATHER_NORM_PARAMS_FILENAME
    edited, reports = apply_weather_edit(
        raw_features,
        processed,
        mode=mode,
        scenario_name=scenario.name,
        recipient_hex_ids=[normalize_hex_id(hex_id) for hex_id in recipient_hex_ids],
        params=params,
        norm_params_path=norm_params_path if norm_params_path.exists() else None,
    )

    out_path = weather_intervention_csv_path(prediction_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edited.to_csv(out_path, index=False)
    summary = pd.DataFrame([report.__dict__ for report in reports])
    return WeatherCounterfactualResult(edited_csv_path=out_path, summary=summary)
