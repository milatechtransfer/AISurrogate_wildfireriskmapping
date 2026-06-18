"""Materialize isolated data roots for fixed-model counterfactual inference."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data_preparation.paths import Paths
from data_preparation.spatial.utils import FUEL_GROUP_MAP
from src.datasets.postprocessing.bp_restricted_zero import bp_nonfuel_restricted_ids
from src.datasets.postprocessing.counterfactual import (
    CounterfactualConfig,
    EndpointConfig,
    ScenarioConfig,
    load_counterfactual_config,
)
from src.datasets.postprocessing.counterfactual_fuel import (
    burnable_mask,
    modal_adjacent_burnable_fuel_across_grids,
    nonfuel_mask,
    replace_nonfuel_components_with_adjacent_modal,
    replace_nonfuel_with_burnable,
)
from src.datasets.postprocessing.counterfactual_weather import (
    apply_wind_scenario_to_processed_weather,
    normalize_raw_weather_ids,
    raw_weather_path,
    validate_wind_roundtrip,
)
from src.datasets.sources.spatialized_tabular import write_train_global_fill_stats
from src.datasets.utils import SPATIALIZED_TABULAR_SOURCE_NAMES


@dataclass(frozen=True)
class MaterializedScenario:
    scenario: str
    scenario_kind: str
    endpoint: str
    data_root: Path
    prediction_dir: Path
    generated_config_path: Path
    n_metadata_rows: int
    n_unique_patch_files: int


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root / resolved


def _replace_existing(path: Path, *, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _link_or_copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _replace_existing(dst, overwrite=overwrite)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _replace_existing(dst, overwrite=overwrite)
    shutil.copy2(src, dst)


def _write_csv(df: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _replace_existing(path, overwrite=overwrite)
    df.to_csv(path, index=False)


def _write_yaml(data: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _replace_existing(path, overwrite=overwrite)
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return raw


def _hex_mask(metadata: pd.DataFrame, hex_ids: list[str]) -> pd.Series:
    normalized_hex_ids = {str(hex_id).zfill(2) for hex_id in hex_ids}
    return metadata["hex_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(2).isin(normalized_hex_ids)


def _relative_file_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"Patch metadata filename must be relative to the data root, got {path}.")
    return path


def _load_channel_map(data_root: Path, modelling_approach: str) -> dict[str, list[int]]:
    path = data_root / f"feature_channel_map_{modelling_approach}.json"
    with path.open() as handle:
        channel_map = json.load(handle)
    if "fuel_grid" not in channel_map:
        raise ValueError(f"{path} is missing required key 'fuel_grid'.")
    return channel_map


def _weather_csv_name(endpoint_config: dict[str, Any]) -> str | None:
    for source in endpoint_config.get("data", {}).get("input_sources", []):
        if not isinstance(source, dict):
            continue
        if "weather" not in str(source.get("name", "")).lower():
            continue
        params = source.get("params", {})
        if isinstance(params, dict) and "csv_name" in params:
            return str(params["csv_name"])
    return None


def _spatialized_tabular_imputation_stats(
    *,
    baseline_root: Path,
    scenario_root: Path,
    endpoint_config: dict[str, Any],
    modelling_approach: str,
    overwrite: bool,
) -> dict[str, str]:
    """Write frozen training-derived imputation stats for spatialized tabular sources."""

    data_cfg = endpoint_config.get("data", {})
    train_split = str(data_cfg.get("train_split", "train_indices.csv"))
    filename_col = str(data_cfg.get("filename_col", "filename"))
    valid_mask_threshold = float(data_cfg.get("valid_mask_threshold", 0.01))
    stats_paths: dict[str, str] = {}
    for source in data_cfg.get("input_sources", []):
        if not isinstance(source, dict) or source.get("name") not in SPATIALIZED_TABULAR_SOURCE_NAMES:
            continue
        source_name = str(source["name"])
        params = source.get("params", {})
        if not isinstance(params, dict) or str(params.get("missing_value_strategy", "global_mean")).lower() != "global_mean":
            continue
        output_name = f"{source_name}_train_imputation_stats.json"
        output_path = scenario_root / output_name
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
        write_train_global_fill_stats(
            root_dir=baseline_root,
            output_path=output_path,
            source_name=source_name,
            csv_name=str(params["csv_name"]),
            feature_names_list=[str(feature) for feature in params["feature_names_list"]],
            zone_id_col=str(params.get("fire_weather_zone_id_col", "WeatherZone")),
            zone_channel_key=str(params.get("zone_channel_key", "firezones_grid")),
            train_split_csv_name=train_split,
            filename_col=filename_col,
            valid_mask_threshold=valid_mask_threshold,
            modelling_approach=modelling_approach,
        )
        stats_paths[source_name] = output_name
    return stats_paths


def _source_csv_names(endpoint_config: dict[str, Any]) -> set[str]:
    csv_names: set[str] = set()
    for source in endpoint_config.get("data", {}).get("input_sources", []):
        if not isinstance(source, dict):
            continue
        params = source.get("params", {})
        if isinstance(params, dict) and "csv_name" in params:
            csv_names.add(str(params["csv_name"]))
    return csv_names


@lru_cache(maxsize=None)
def _processed_weather_hex_slice(raw_data_dir: Path, hex_id: str) -> slice:
    target_path = raw_weather_path(raw_data_dir, hex_id)
    files = sorted(raw_data_dir.glob("hex*/tabular/hex*_DailyWeather.csv"))
    offset = 0
    for path in files:
        count = int(sum(1 for _ in path.open()) - 1)
        if path == target_path:
            return slice(offset, offset + count)
        offset += count
    raise FileNotFoundError(f"Missing raw weather table for hex{hex_id}: {target_path}")


def _copy_static_root_assets(
    *,
    baseline_root: Path,
    scenario_root: Path,
    endpoint_config: dict[str, Any],
    weather_csv_name: str | None,
    overwrite: bool,
) -> None:
    for path in baseline_root.glob("feature_channel_map_*.json"):
        _copy_file(path, scenario_root / path.name, overwrite=overwrite)
    target_log_stats = baseline_root / "target_log_stats.json"
    if target_log_stats.exists():
        _copy_file(target_log_stats, scenario_root / target_log_stats.name, overwrite=overwrite)

    for csv_name in sorted(_source_csv_names(endpoint_config)):
        if csv_name == weather_csv_name:
            continue
        _copy_file(baseline_root / csv_name, scenario_root / csv_name, overwrite=overwrite)


def _write_split_files(
    *,
    baseline_root: Path,
    scenario_root: Path,
    endpoint_config: dict[str, Any],
    hex_ids: list[str],
    overwrite: bool,
) -> pd.DataFrame:
    data_cfg = endpoint_config["data"]
    test_split = str(data_cfg.get("test_split", "test_indices.csv"))
    train_split = str(data_cfg.get("train_split", "train_indices.csv"))
    val_split = str(data_cfg.get("val_split", "val_indices.csv"))

    metadata = pd.read_csv(baseline_root / test_split)
    scenario_metadata = metadata.loc[_hex_mask(metadata, hex_ids)].copy()
    if scenario_metadata.empty:
        raise ValueError(f"No test metadata rows found for hex_ids={hex_ids} in {baseline_root / test_split}.")

    _write_csv(scenario_metadata, scenario_root / test_split, overwrite=overwrite)
    empty = scenario_metadata.iloc[0:0].copy()
    _write_csv(empty, scenario_root / train_split, overwrite=overwrite)
    _write_csv(empty, scenario_root / val_split, overwrite=overwrite)
    return scenario_metadata


def _patch_paths(baseline_root: Path, metadata: pd.DataFrame) -> list[tuple[Path, Path]]:
    rel_paths = [_relative_file_path(value) for value in metadata["filename"].drop_duplicates()]
    return [(baseline_root / rel_path, rel_path) for rel_path in rel_paths]


def _patch_records(baseline_root: Path, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"filename", "row", "col"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Patch metadata is missing columns required for local fuel edits: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for _, item in metadata.drop_duplicates("filename").iterrows():
        rel_path = _relative_file_path(item["filename"])
        records.append(
            {
                "src": baseline_root / rel_path,
                "rel_path": rel_path,
                "row": int(item["row"]),
                "col": int(item["col"]),
            }
        )
    return records


def _fuel_grid_iter(patch_paths: list[Path], fuel_channel: int):
    for path in patch_paths:
        patch = np.load(path, mmap_mode="r")
        if fuel_channel >= patch.shape[2]:
            raise ValueError(f"Fuel channel {fuel_channel} is out of bounds for patch {path} with shape {patch.shape}.")
        yield patch[:, :, fuel_channel]


def _load_stitched_patch_fuel(records: list[dict[str, Any]], fuel_channel: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Stitch patch fuel channels into one global array using metadata windows."""

    max_row = 0
    max_col = 0
    enriched: list[dict[str, Any]] = []
    for record in records:
        patch = np.load(record["src"], mmap_mode="r")
        if fuel_channel >= patch.shape[2]:
            raise ValueError(f"Fuel channel {fuel_channel} is out of bounds for patch {record['src']} with shape {patch.shape}.")
        height, width = int(patch.shape[0]), int(patch.shape[1])
        enriched_record = dict(record)
        enriched_record.update({"height": height, "width": width})
        enriched.append(enriched_record)
        max_row = max(max_row, int(record["row"]) + height)
        max_col = max(max_col, int(record["col"]) + width)

    stitched = np.full((max_row, max_col), np.nan, dtype=np.float32)
    for record in enriched:
        patch = np.load(record["src"], mmap_mode="r")
        fuel = np.asarray(patch[:, :, fuel_channel], dtype=np.float32)
        row = int(record["row"])
        col = int(record["col"])
        height = int(record["height"])
        width = int(record["width"])
        window = stitched[row : row + height, col : col + width]
        finite_fuel = np.isfinite(fuel)
        finite_window = np.isfinite(window)
        conflict = finite_fuel & finite_window & (fuel != window)
        if conflict.any():
            raise ValueError(
                f"Conflicting fuel values in overlapping patch windows for {record['rel_path']} " f"({int(conflict.sum())} pixels)."
            )
        window[finite_fuel & ~finite_window] = fuel[finite_fuel & ~finite_window]
        stitched[row : row + height, col : col + width] = window

    return stitched, enriched


def prepared_nonfuel_ids(raw_nonfuel_ids: list[int] | tuple[int, ...]) -> list[int]:
    """Map raw FBP non-fuel IDs to the grouped fuel IDs stored in prepared patches."""

    missing = sorted(set(raw_nonfuel_ids) - set(FUEL_GROUP_MAP))
    if missing:
        raise ValueError(f"Raw non-fuel IDs are missing from FUEL_GROUP_MAP: {missing}")
    mapped = sorted({int(FUEL_GROUP_MAP[int(fuel_id)]) for fuel_id in raw_nonfuel_ids})
    if not mapped:
        raise ValueError("No prepared non-fuel IDs could be derived from raw non-fuel IDs.")
    return mapped


def _merge_materialized_index(existing: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Return a scenario index with updated rows replacing matching existing rows."""

    if existing.empty:
        return updates
    required = {"scenario", "endpoint"}
    missing_existing = required - set(existing.columns)
    missing_updates = required - set(updates.columns)
    if missing_existing or missing_updates:
        raise ValueError("Materialized indexes must include scenario and endpoint columns.")

    update_keys = set(zip(updates["scenario"].astype(str), updates["endpoint"].astype(str), strict=False))
    keep_existing = ~existing.apply(
        lambda row: (str(row["scenario"]), str(row["endpoint"])) in update_keys,
        axis=1,
    )
    return pd.concat([existing.loc[keep_existing], updates], ignore_index=True, sort=False)


def _materialize_patch_files(
    *,
    baseline_root: Path,
    scenario_root: Path,
    metadata: pd.DataFrame,
    scenario: ScenarioConfig,
    raw_data_dir: Path,
    hex_id: str,
    fuel_channel: int,
    overwrite: bool,
) -> pd.DataFrame:
    src_rel_paths = _patch_paths(baseline_root, metadata)
    if scenario.kind != "fuel":
        for src, rel_path in src_rel_paths:
            _link_or_copy_file(src, scenario_root / rel_path, overwrite=overwrite)
        return pd.DataFrame()

    mode = str(scenario.params.get("mode", "nonfuel_to_burnable_adjacent_modal"))
    supported_modes = {
        "nonfuel_to_burnable_adjacent_modal",
        "nonfuel_to_burnable_local_adjacent_modal",
        "nonfuel_to_burnable_global_adjacent_modal",
    }
    if mode not in supported_modes:
        raise ValueError(f"Unsupported fuel scenario mode {mode!r}.")

    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    _, raw_nonfuel_ids, restricted_names = bp_nonfuel_restricted_ids(paths, hex_id)
    patch_nonfuel_ids = prepared_nonfuel_ids(raw_nonfuel_ids)

    rows: list[dict[str, Any]] = []
    if mode == "nonfuel_to_burnable_local_adjacent_modal":
        records = _patch_records(baseline_root, metadata)
        stitched_fuel, records = _load_stitched_patch_fuel(records, fuel_channel)
        edited_stitched_fuel, _, full_report, component_report = replace_nonfuel_components_with_adjacent_modal(
            stitched_fuel,
            patch_nonfuel_ids,
            scenario_name=scenario.name,
        )
        if not component_report.empty:
            component_report = component_report.copy()
            component_report.insert(0, "scenario_name", scenario.name)
            component_report.insert(1, "mode", full_report.mode)
            _write_csv(component_report, scenario_root / "fuel_component_replacements.csv", overwrite=overwrite)
        full_replacement_ids = (
            ";".join(map(str, sorted(component_report["replacement_fuel_id"].dropna().astype(int).unique())))
            if not component_report.empty
            else ""
        )

        for record in records:
            src = record["src"]
            rel_path = record["rel_path"]
            dst = scenario_root / rel_path
            if dst.exists():
                _replace_existing(dst, overwrite=overwrite)
            dst.parent.mkdir(parents=True, exist_ok=True)

            patch = np.load(src)
            row_start = int(record["row"])
            col_start = int(record["col"])
            height = int(record["height"])
            width = int(record["width"])
            edited_fuel = edited_stitched_fuel[row_start : row_start + height, col_start : col_start + width]
            baseline_fuel = patch[:, :, fuel_channel]
            patch_edit_mask = nonfuel_mask(baseline_fuel, patch_nonfuel_ids)
            patch_burnable_mask = burnable_mask(baseline_fuel, patch_nonfuel_ids)
            replacement_values = edited_fuel[patch_edit_mask]
            unique_replacements = sorted({int(value) for value in replacement_values[np.isfinite(replacement_values)]})

            edited_patch = np.array(patch, copy=True)
            edited_patch[:, :, fuel_channel] = edited_fuel.astype(edited_patch.dtype, copy=False)
            np.save(dst, edited_patch)

            row = asdict(full_report)
            row.update(
                {
                    "endpoint": "",
                    "patch_file": str(rel_path),
                    "edited_pixels": int(patch_edit_mask.sum()),
                    "original_nonfuel_pixels": int(patch_edit_mask.sum()),
                    "original_burnable_pixels": int(patch_burnable_mask.sum()),
                    "replacement_fuel_id": unique_replacements[0] if len(unique_replacements) == 1 else full_report.replacement_fuel_id,
                    "replacement_fuel_ids": full_replacement_ids,
                    "patch_replacement_fuel_ids": ";".join(map(str, unique_replacements)),
                    "n_components": int(len(component_report)) if not component_report.empty else 0,
                    "full_grid_edited_pixels": full_report.edited_pixels,
                    "full_grid_nonfuel_pixels": full_report.original_nonfuel_pixels,
                    "full_grid_burnable_pixels": full_report.original_burnable_pixels,
                    "raw_nonfuel_ids": ";".join(map(str, raw_nonfuel_ids)),
                    "prepared_nonfuel_ids": ";".join(map(str, patch_nonfuel_ids)),
                    "restricted_names": "; ".join(restricted_names),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    patch_src_paths = [src for src, _ in src_rel_paths]
    replacement, candidate_pixels, note = modal_adjacent_burnable_fuel_across_grids(
        _fuel_grid_iter(patch_src_paths, fuel_channel),
        patch_nonfuel_ids,
    )

    for src, rel_path in src_rel_paths:
        dst = scenario_root / rel_path
        if dst.exists():
            _replace_existing(dst, overwrite=overwrite)
        dst.parent.mkdir(parents=True, exist_ok=True)

        patch = np.load(src)
        if fuel_channel >= patch.shape[2]:
            raise ValueError(f"Fuel channel {fuel_channel} is out of bounds for patch {src} with shape {patch.shape}.")
        edited_fuel, _, report = replace_nonfuel_with_burnable(
            patch[:, :, fuel_channel],
            patch_nonfuel_ids,
            replacement,
            scenario_name=scenario.name,
            candidate_pixels=candidate_pixels,
            note=note,
        )
        edited_patch = np.array(patch, copy=True)
        edited_patch[:, :, fuel_channel] = edited_fuel.astype(edited_patch.dtype, copy=False)
        np.save(dst, edited_patch)

        row = asdict(report)
        row.update(
            {
                "endpoint": "",
                "patch_file": str(rel_path),
                "replacement_fuel_ids": str(report.replacement_fuel_id),
                "n_components": 0,
                "full_grid_edited_pixels": pd.NA,
                "full_grid_nonfuel_pixels": pd.NA,
                "full_grid_burnable_pixels": pd.NA,
                "raw_nonfuel_ids": ";".join(map(str, raw_nonfuel_ids)),
                "prepared_nonfuel_ids": ";".join(map(str, patch_nonfuel_ids)),
                "restricted_names": "; ".join(restricted_names),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _write_weather_table(
    *,
    baseline_root: Path,
    scenario_root: Path,
    weather_csv_name: str | None,
    scenario: ScenarioConfig,
    raw_data_dir: Path,
    hex_id: str,
    seed: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weather_csv_name is None:
        return pd.DataFrame(), pd.DataFrame()

    src = baseline_root / weather_csv_name
    if scenario.kind != "wind":
        _copy_file(src, scenario_root / weather_csv_name, overwrite=overwrite)
        return pd.DataFrame(), pd.DataFrame()

    full_processed = pd.read_csv(src)
    hex_slice = _processed_weather_hex_slice(raw_data_dir, hex_id)
    processed_hex = full_processed.iloc[hex_slice].copy()
    raw_weather = normalize_raw_weather_ids(pd.read_csv(raw_weather_path(raw_data_dir, hex_id)))

    stats, validation = validate_wind_roundtrip(raw_weather=raw_weather, processed_weather=processed_hex)
    scenario_processed_hex, diagnostics = apply_wind_scenario_to_processed_weather(
        raw_weather,
        processed_hex,
        stats,
        scenario.params,
        seed=seed,
    )

    full_scenario = full_processed.copy()
    for column in scenario_processed_hex.columns:
        full_scenario.iloc[hex_slice, full_scenario.columns.get_loc(column)] = scenario_processed_hex[column].to_numpy()

    _write_csv(full_scenario, scenario_root / weather_csv_name, overwrite=overwrite)
    return validation, diagnostics


def build_scenario_endpoint_config(
    endpoint_config: dict[str, Any],
    *,
    data_root: Path,
    prediction_dir: Path,
    raw_data_dir: Path,
    imputation_stats_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return an evaluator config pointed at a materialized scenario root."""

    scenario_config = copy.deepcopy(endpoint_config)
    scenario_config["save_dir"] = str(prediction_dir)
    scenario_config.setdefault("logger", {})["enabled"] = False
    scenario_config["data"]["root_dir"] = str(data_root)
    scenario_config["data"]["raw_data_dir"] = str(raw_data_dir)
    scenario_config["data"]["num_workers"] = 0
    scenario_config["data"]["train_split"] = str(scenario_config["data"].get("train_split", "train_indices.csv"))
    scenario_config["data"]["val_split"] = str(scenario_config["data"].get("val_split", "val_indices.csv"))
    scenario_config["data"]["test_split"] = str(scenario_config["data"].get("test_split", "test_indices.csv"))
    for source in scenario_config["data"].get("input_sources", []):
        if not isinstance(source, dict) or not isinstance(source.get("params"), dict):
            continue
        source_name = str(source.get("name", ""))
        if imputation_stats_paths and source_name in imputation_stats_paths:
            source["params"]["imputation_stats_path"] = imputation_stats_paths[source_name]
    return scenario_config


def _link_checkpoint(
    *,
    endpoint_config: dict[str, Any],
    prediction_dir: Path,
    project_root: Path,
    overwrite: bool,
) -> None:
    checkpoint_filename = str(endpoint_config.get("evaluation", {}).get("checkpoint_filename", "best.pth"))
    source_save_dir = _resolve_path(endpoint_config["save_dir"], project_root)
    _link_or_copy_file(source_save_dir / checkpoint_filename, prediction_dir / checkpoint_filename, overwrite=overwrite)


def _materialize_one(
    *,
    cfg: CounterfactualConfig,
    endpoint: EndpointConfig,
    scenario: ScenarioConfig,
    project_root: Path,
    experiment_dir: Path,
    overwrite: bool,
) -> tuple[MaterializedScenario, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoint_config_path = _resolve_path(endpoint.config_path, project_root)
    endpoint_config = _read_yaml(endpoint_config_path)
    baseline_root = endpoint.baseline_data_root or Path(endpoint_config["data"]["root_dir"])
    baseline_root = _resolve_path(baseline_root, project_root)
    modelling_approach = str(endpoint_config.get("modelling_approach", "1"))

    scenario_root = experiment_dir / "scenario_data" / scenario.name / endpoint.name
    prediction_dir = experiment_dir / "predictions" / scenario.name / endpoint.name
    generated_config_path = experiment_dir / "generated_configs" / f"{scenario.name}_{endpoint.name}.yaml"
    scenario_root.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    weather_csv_name = _weather_csv_name(endpoint_config)
    _copy_static_root_assets(
        baseline_root=baseline_root,
        scenario_root=scenario_root,
        endpoint_config=endpoint_config,
        weather_csv_name=weather_csv_name,
        overwrite=overwrite,
    )
    imputation_stats_paths = _spatialized_tabular_imputation_stats(
        baseline_root=baseline_root,
        scenario_root=scenario_root,
        endpoint_config=endpoint_config,
        modelling_approach=modelling_approach,
        overwrite=overwrite,
    )
    metadata = _write_split_files(
        baseline_root=baseline_root,
        scenario_root=scenario_root,
        endpoint_config=endpoint_config,
        hex_ids=cfg.hex_ids,
        overwrite=overwrite,
    )

    channel_map = _load_channel_map(scenario_root, modelling_approach)
    fuel_channel = int(channel_map["fuel_grid"][0])
    fuel_report = _materialize_patch_files(
        baseline_root=baseline_root,
        scenario_root=scenario_root,
        metadata=metadata,
        scenario=scenario,
        raw_data_dir=cfg.raw_data_dir,
        hex_id=cfg.focus_hex_id,
        fuel_channel=fuel_channel,
        overwrite=overwrite,
    )
    if not fuel_report.empty:
        fuel_report["endpoint"] = endpoint.name

    weather_validation, weather_diagnostics = _write_weather_table(
        baseline_root=baseline_root,
        scenario_root=scenario_root,
        weather_csv_name=weather_csv_name,
        scenario=scenario,
        raw_data_dir=cfg.raw_data_dir,
        hex_id=cfg.focus_hex_id,
        seed=cfg.seed,
        overwrite=overwrite,
    )
    for frame in (weather_validation, weather_diagnostics):
        if not frame.empty:
            frame.insert(0, "endpoint", endpoint.name)
            frame.insert(0, "scenario", scenario.name)

    generated_config = build_scenario_endpoint_config(
        endpoint_config,
        data_root=scenario_root,
        prediction_dir=prediction_dir,
        raw_data_dir=cfg.raw_data_dir,
        imputation_stats_paths=imputation_stats_paths,
    )
    _write_yaml(generated_config, generated_config_path, overwrite=overwrite)
    _link_checkpoint(
        endpoint_config=endpoint_config,
        prediction_dir=prediction_dir,
        project_root=project_root,
        overwrite=overwrite,
    )

    materialized = MaterializedScenario(
        scenario=scenario.name,
        scenario_kind=scenario.kind,
        endpoint=endpoint.name,
        data_root=scenario_root,
        prediction_dir=prediction_dir,
        generated_config_path=generated_config_path,
        n_metadata_rows=int(len(metadata)),
        n_unique_patch_files=int(metadata["filename"].nunique()),
    )
    return materialized, weather_validation, weather_diagnostics, fuel_report


def materialize_counterfactual_inputs(
    config_path: Path,
    *,
    endpoint_names: set[str] | None = None,
    scenario_names: set[str] | None = None,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> list[MaterializedScenario]:
    """Materialize scenario-specific data roots and generated evaluator configs."""

    project_root = (project_root or Path.cwd()).resolve()
    cfg = load_counterfactual_config(config_path)
    cfg = CounterfactualConfig(
        raw_data_dir=_resolve_path(cfg.raw_data_dir, project_root),
        save_dir=_resolve_path(cfg.save_dir, project_root),
        hex_ids=cfg.hex_ids,
        focus_hex_id=cfg.focus_hex_id,
        mask_scope=cfg.mask_scope,
        support_policy=cfg.support_policy,
        endpoints=cfg.endpoints,
        scenarios=cfg.scenarios,
        seed=cfg.seed,
    )

    endpoints = [endpoint for endpoint in cfg.enabled_endpoints.values() if endpoint_names is None or endpoint.name in endpoint_names]
    scenarios = [scenario for scenario in cfg.scenarios if scenario_names is None or scenario.name in scenario_names]
    if not endpoints:
        raise ValueError("No enabled endpoints selected.")
    if not scenarios:
        raise ValueError("No scenarios selected.")

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    materialized_rows: list[dict[str, Any]] = []
    weather_validation_rows: list[pd.DataFrame] = []
    weather_diagnostic_rows: list[pd.DataFrame] = []
    fuel_report_rows: list[pd.DataFrame] = []
    materialized: list[MaterializedScenario] = []

    for scenario in scenarios:
        for endpoint in endpoints:
            result, weather_validation, weather_diagnostics, fuel_report = _materialize_one(
                cfg=cfg,
                endpoint=endpoint,
                scenario=scenario,
                project_root=project_root,
                experiment_dir=cfg.save_dir,
                overwrite=overwrite,
            )
            materialized.append(result)
            materialized_rows.append({key: str(value) if isinstance(value, Path) else value for key, value in asdict(result).items()})
            if not weather_validation.empty:
                weather_validation_rows.append(weather_validation)
            if not weather_diagnostics.empty:
                weather_diagnostic_rows.append(weather_diagnostics)
            if not fuel_report.empty:
                fuel_report_rows.append(fuel_report)

    materialized_index = pd.DataFrame(materialized_rows)
    index_path = cfg.save_dir / "scenario_prediction_index.csv"
    if index_path.exists() and (endpoint_names is not None or scenario_names is not None):
        materialized_index = _merge_materialized_index(pd.read_csv(index_path), materialized_index)
    _write_csv(materialized_index, index_path, overwrite=overwrite)
    if weather_validation_rows:
        _write_csv(
            pd.concat(weather_validation_rows, ignore_index=True), cfg.save_dir / "weather_encoding_validation.csv", overwrite=overwrite
        )
    if weather_diagnostic_rows:
        _write_csv(
            pd.concat(weather_diagnostic_rows, ignore_index=True), cfg.save_dir / "scenario_ood_diagnostics.csv", overwrite=overwrite
        )
    if fuel_report_rows:
        fuel_reports = pd.concat(fuel_report_rows, ignore_index=True)
        summary = (
            fuel_reports.groupby(["scenario_name", "endpoint"], dropna=False)
            .agg(
                patch_files=("patch_file", "nunique"),
                patch_edited_pixels=("edited_pixels", "sum"),
                full_grid_edited_pixels=("full_grid_edited_pixels", "first"),
                replacement_fuel_id=("replacement_fuel_id", "first"),
                replacement_fuel_ids=("replacement_fuel_ids", "first"),
                n_components=("n_components", "first"),
                candidate_pixels=("candidate_pixels", "first"),
                patch_original_nonfuel_pixels=("original_nonfuel_pixels", "sum"),
                patch_original_burnable_pixels=("original_burnable_pixels", "sum"),
                full_grid_nonfuel_pixels=("full_grid_nonfuel_pixels", "first"),
                full_grid_burnable_pixels=("full_grid_burnable_pixels", "first"),
                raw_nonfuel_ids=("raw_nonfuel_ids", "first"),
                prepared_nonfuel_ids=("prepared_nonfuel_ids", "first"),
                restricted_names=("restricted_names", "first"),
                note=("note", "first"),
            )
            .reset_index()
        )
        summary["edited_pixels"] = summary["full_grid_edited_pixels"].where(
            summary["full_grid_edited_pixels"].notna(), summary["patch_edited_pixels"]
        )
        summary["original_nonfuel_pixels"] = summary["full_grid_nonfuel_pixels"].where(
            summary["full_grid_nonfuel_pixels"].notna(), summary["patch_original_nonfuel_pixels"]
        )
        summary["original_burnable_pixels"] = summary["full_grid_burnable_pixels"].where(
            summary["full_grid_burnable_pixels"].notna(), summary["patch_original_burnable_pixels"]
        )
        _write_csv(summary, cfg.save_dir / "fuel_edit_summary.csv", overwrite=overwrite)

    manifest = {
        "config_path": str(config_path),
        "raw_data_dir": str(cfg.raw_data_dir),
        "save_dir": str(cfg.save_dir),
        "hex_ids": cfg.hex_ids,
        "focus_hex_id": cfg.focus_hex_id,
        "mask_scope": cfg.mask_scope,
        "support_policy": cfg.support_policy,
        "endpoints": [endpoint.name for endpoint in endpoints],
        "scenarios": [scenario.name for scenario in scenarios],
        "generated_rows": len(materialized_rows),
    }
    _write_yaml(manifest, cfg.save_dir / "scenario_manifest.yaml", overwrite=overwrite)
    return materialized


def _parse_name_set(values: list[str] | None) -> set[str] | None:
    return set(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize fixed-model counterfactual data roots.")
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_hex16.yaml"))
    parser.add_argument("--endpoint", action="append", dest="endpoints", help="Endpoint to materialize; may be repeated.")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario to materialize; may be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated roots/configs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialized = materialize_counterfactual_inputs(
        args.config,
        endpoint_names=_parse_name_set(args.endpoints),
        scenario_names=_parse_name_set(args.scenarios),
        overwrite=args.overwrite,
    )
    print(f"Materialized {len(materialized)} endpoint/scenario data roots.")
    for item in materialized:
        print(f"{item.scenario}/{item.endpoint}: data_root={item.data_root} config={item.generated_config_path}")


if __name__ == "__main__":
    main()
