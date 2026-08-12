"""Run fixed checkpoints against baseline and counterfactual patch transforms."""

from __future__ import annotations

import argparse
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    EndpointConfig,
    ScenarioConfig,
    load_counterfactual_config,
    resolve_counterfactual_paths,
    resolve_project_path,
)
from src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform import FuelCounterfactualTransform
from src.datasets.postprocessing.counterfactual.weather_counterfactual_transform import materialize_weather_scenario
from src.evaluate_hexels import load_config
from src.evaluate_hexels import main as evaluate_hexels

SPATIALIZED_WEATHER_SOURCE_NAME = "spatialized_weather"


def _validate_requested_names(requested: set[str] | None, available: set[str], *, label: str) -> None:
    if requested is None:
        return
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"Unknown {label} selection(s): {unknown}; available {label}s: {sorted(available)}.")


def _select_endpoints(
    endpoints: dict[str, EndpointConfig],
    endpoint_names: set[str] | None,
) -> list[EndpointConfig]:
    _validate_requested_names(endpoint_names, set(endpoints), label="endpoint")
    if endpoint_names is None:
        return list(endpoints.values())
    return [endpoint for name, endpoint in endpoints.items() if name in endpoint_names]


def _select_scenarios(scenarios: list[ScenarioConfig], scenario_names: set[str] | None) -> list[ScenarioConfig]:
    """Filter to the requested scenarios, always keeping the baseline scenario.

    Downstream plotting scripts diff every fuel scenario against baseline, so baseline
    predictions must exist even if `scenario_names` doesn't request it explicitly.
    """
    if scenario_names is None:
        return list(scenarios)
    _validate_requested_names(scenario_names, {scenario.name for scenario in scenarios}, label="scenario")
    return [scenario for scenario in scenarios if scenario.kind == "baseline" or scenario.name in scenario_names]


def _filter_hex_ids(metadata: pd.DataFrame, hex_ids: set[str]) -> pd.DataFrame:
    normalized = metadata["hex_id"].astype(str).map(normalize_hex_id)
    return metadata.loc[normalized.isin(hex_ids)]


def _fuel_channel(data_root: Path, modelling_approach: str) -> int:
    import json

    path = data_root / f"feature_channel_map_{modelling_approach}.json"
    with path.open() as handle:
        channel_map = json.load(handle)
    channels = channel_map.get("fuel_grid")
    if not channels:
        raise ValueError(f"{path} has no fuel_grid channel.")
    return int(channels[0])


def _override_spatialized_weather_csv(
    run_config,
    edited_csv_path: Path,
    *,
    baseline_csv_path: Path,
) -> None:
    """Point every `spatialized_weather` input source at an absolute edited CSV path.

    `SpatializedTabularSource` resolves its CSV as `os.path.join(root_dir, csv_name)`,
    which returns `csv_name` unchanged when it is already absolute - so this doesn't
    require duplicating the rest of `data.root_dir`.
    """
    matched = False
    for source in run_config.data.input_sources:
        if source.name == SPATIALIZED_WEATHER_SOURCE_NAME:
            source.params.csv_name = str(edited_csv_path.resolve())
            source.params.global_fill_csv_name = str(baseline_csv_path.resolve())
            matched = True
    if not matched:
        raise ValueError(f"No {SPATIALIZED_WEATHER_SOURCE_NAME!r} input source configured; cannot apply a weather scenario.")


def _spatialized_weather_csv_name(base_config) -> str:
    for source in base_config.data.input_sources:
        if source.name == SPATIALIZED_WEATHER_SOURCE_NAME:
            return str(source.params.csv_name)
    raise ValueError(f"No {SPATIALIZED_WEATHER_SOURCE_NAME!r} input source configured; cannot apply a weather scenario.")


def _prepare_prediction_dir(
    *,
    source_save_dir: Path,
    prediction_dir: Path,
    checkpoint_filename: str,
    overwrite: bool,
) -> None:
    if prediction_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{prediction_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True)
    source_checkpoint = source_save_dir / checkpoint_filename
    destination_checkpoint = prediction_dir / checkpoint_filename
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    try:
        os.link(source_checkpoint, destination_checkpoint)
    except OSError:
        shutil.copy2(source_checkpoint, destination_checkpoint)


def _evaluation_args() -> argparse.Namespace:
    return argparse.Namespace(
        config="",
        visualize_predictions=False,
        save_visualizations=False,
        metrics_only=False,
        skip_hexel_plots=True,
        no_save_predictions=True,
        robust_plot_percentile=None,
        stitch_mode="mean",
        mask_scope="actual",
        run_id=None,
    )


def _write_optional_frame(frame: pd.DataFrame | None, path: Path) -> None:
    if frame is None or frame.empty:
        path.unlink(missing_ok=True)
        return
    frame.to_csv(path, index=False)


def run_counterfactual_evaluation(
    config_path: Path,
    *,
    endpoint_names: set[str] | None = None,
    scenario_names: set[str] | None = None,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Evaluate each selected endpoint under each selected scenario for the configured hexels.

    For every (endpoint, scenario) pair, this loads the endpoint's checkpoint config,
    filters test metadata to `config.hex_ids`, applies the scenario's edit - a fuel-edit
    patch transform, an edited `weather_table_processed.csv` for `weather` scenarios, or a
    no-op for the baseline scenario - and runs `evaluate_hexels` to write predicted hexel
    rasters under `<save_dir>/predictions/<scenario>/<endpoint>/`. Also writes, under
    `save_dir`: `scenario_prediction_index.csv` (returned), `counterfactual_metrics.csv`,
    and (for `fuel` scenarios) `fuel_edit_summary.csv` / `fuel_component_replacements.csv`,
    or (for `weather` scenarios) `weather_edit_summary.csv`.

    The baseline scenario is always evaluated regardless of `scenario_names`, since
    downstream plotting scripts diff each scenario against it.

    Returns:
        The scenario/endpoint -> prediction_dir index, as also written to
        `scenario_prediction_index.csv`.
    """
    project_root = (project_root or Path.cwd()).resolve()
    config = load_counterfactual_config(config_path)
    save_dir, raw_data_dir = resolve_counterfactual_paths(config, project_root=project_root)
    hex_ids = set(config.hex_ids)
    endpoints = _select_endpoints(config.endpoints, endpoint_names)
    scenarios = _select_scenarios(config.scenarios, scenario_names)
    if not endpoints:
        raise ValueError("No enabled endpoints selected.")
    if not scenarios:
        raise ValueError("No scenarios selected.")

    index_rows = []
    metric_rows: list[dict[str, str | float]] = []
    summary_frames = []
    component_frames = []
    weather_summary_frames = []
    # Cache one materialized run per (scenario, resolved config_path, resolved data_root).
    # Multiple logical endpoint names (e.g. "bp"/"fi"/"ros") can point at the exact same
    # multi-output checkpoint so that existing per-target plotting scripts keep working
    # unchanged; in that case the underlying model only needs to run inference once per
    # scenario, and every alias endpoint reuses that run's prediction_dir/metrics/summaries.
    run_cache: dict[tuple[str, Path, Path], dict[str, Any]] = {}
    for endpoint in endpoints:
        endpoint_config_path = resolve_project_path(endpoint.config_path, project_root)
        base_config = load_config(str(endpoint_config_path))
        source_save_dir = resolve_project_path(base_config.save_dir, project_root)
        data_root = resolve_project_path(endpoint.baseline_data_root or base_config.data.root_dir, project_root)
        metadata = pd.read_csv(data_root / base_config.data.test_split)
        if "valid_ratio" in metadata.columns:
            metadata = metadata.loc[metadata["valid_ratio"] > base_config.data.valid_mask_threshold]
        metadata = _filter_hex_ids(metadata, hex_ids).reset_index(drop=True)
        if metadata.empty:
            raise ValueError(f"No test metadata found for hex_ids={sorted(hex_ids)} and endpoint={endpoint.name!r}.")

        for scenario in scenarios:
            cache_key = (scenario.name, endpoint_config_path.resolve(), data_root.resolve())
            cached = run_cache.get(cache_key)
            if cached is not None:
                index_rows.append(
                    {
                        "scenario": scenario.name,
                        "endpoint": endpoint.name,
                        "prediction_dir": str(cached["prediction_dir"].resolve()),
                    }
                )
                metric_rows.extend(
                    {"scenario": scenario.name, "endpoint": endpoint.name, "metric": metric, "value": value}
                    for metric, value in cached["metrics"].items()
                )
                if cached["fuel_summary"] is not None:
                    summary = cached["fuel_summary"].copy()
                    summary["endpoint"] = endpoint.name
                    summary_frames.append(summary)
                if cached["fuel_components"] is not None:
                    components = cached["fuel_components"].copy()
                    components["endpoint"] = endpoint.name
                    component_frames.append(components)
                if cached["weather_summary"] is not None:
                    summary = cached["weather_summary"].copy()
                    summary["endpoint"] = endpoint.name
                    weather_summary_frames.append(summary)
                continue

            prediction_dir = save_dir / "predictions" / scenario.name / endpoint.name
            checkpoint_filename = base_config.evaluation.checkpoint_filename
            _prepare_prediction_dir(
                source_save_dir=source_save_dir,
                prediction_dir=prediction_dir,
                checkpoint_filename=checkpoint_filename,
                overwrite=overwrite,
            )

            run_config = base_config.model_copy(deep=True)
            run_config.save_dir = str(prediction_dir)
            run_config.data.root_dir = str(data_root)
            run_config.data.raw_data_dir = str(raw_data_dir)
            run_config.data.num_workers = 0
            run_config.logger.enabled = False

            patch_transform = None
            fuel_summary = None
            fuel_components = None
            weather_summary = None
            if scenario.kind == "fuel":
                patch_transform = FuelCounterfactualTransform.from_metadata(
                    metadata=metadata,
                    fuel_channel=_fuel_channel(data_root, run_config.modelling_approach),
                    scenario=scenario,
                    filename_col=base_config.data.filename_col,
                    mask_scope=_evaluation_args().mask_scope,
                    prediction_dir=prediction_dir,
                    raw_data_dir=raw_data_dir,
                )
                fuel_summary = patch_transform.summary.copy()
                summary = fuel_summary.copy()
                summary.insert(0, "endpoint", endpoint.name)
                summary_frames.append(summary)
                if not patch_transform.components.empty:
                    fuel_components = patch_transform.components.copy()
                    components = fuel_components.copy()
                    components.insert(0, "endpoint", endpoint.name)
                    component_frames.append(components)
            elif scenario.kind == "weather":
                baseline_weather_csv = data_root / _spatialized_weather_csv_name(base_config)
                weather_result = materialize_weather_scenario(
                    scenario=scenario,
                    raw_data_dir=raw_data_dir,
                    processed_weather_csv=baseline_weather_csv,
                    recipient_hex_ids=sorted(hex_ids),
                    prediction_dir=prediction_dir,
                )
                _override_spatialized_weather_csv(
                    run_config,
                    weather_result.edited_csv_path,
                    baseline_csv_path=baseline_weather_csv,
                )
                weather_summary = weather_result.summary.copy()
                summary = weather_summary.copy()
                summary.insert(0, "endpoint", endpoint.name)
                weather_summary_frames.append(summary)

            metrics = evaluate_hexels(
                args=_evaluation_args(),
                config=run_config,
                patch_transform=patch_transform,
                metadata_filter=partial(_filter_hex_ids, hex_ids=hex_ids),
            )
            metric_rows.extend(
                {
                    "scenario": scenario.name,
                    "endpoint": endpoint.name,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            )
            index_rows.append(
                {
                    "scenario": scenario.name,
                    "endpoint": endpoint.name,
                    "prediction_dir": str(prediction_dir.resolve()),
                }
            )
            run_cache[cache_key] = {
                "prediction_dir": prediction_dir,
                "metrics": metrics,
                "fuel_summary": fuel_summary,
                "fuel_components": fuel_components,
                "weather_summary": weather_summary,
            }

    save_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(index_rows)
    index.to_csv(save_dir / "scenario_prediction_index.csv", index=False)
    _write_optional_frame(pd.DataFrame(metric_rows), save_dir / "counterfactual_metrics.csv")
    _write_optional_frame(
        pd.concat(summary_frames, ignore_index=True) if summary_frames else None,
        save_dir / "fuel_edit_summary.csv",
    )
    _write_optional_frame(
        pd.concat(component_frames, ignore_index=True) if component_frames else None,
        save_dir / "fuel_component_replacements.csv",
    )
    _write_optional_frame(
        pd.concat(weather_summary_frames, ignore_index=True) if weather_summary_frames else None,
        save_dir / "weather_edit_summary.csv",
    )
    return index


def _parse_name_set(values: list[str] | None) -> set[str] | None:
    return set(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_fuel.yaml"))
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = run_counterfactual_evaluation(
        args.config,
        endpoint_names=_parse_name_set(args.endpoints),
        scenario_names=_parse_name_set(args.scenarios),
        overwrite=args.overwrite,
    )
    print(f"Completed {len(index)} counterfactual evaluations.")


if __name__ == "__main__":
    main()
