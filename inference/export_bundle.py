"""
Export a training checkpoint into a portable model bundle.

Example (Mila cluster):

    uv run python -m inference.export_bundle \
        --checkpoint /network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/unet_256_firesize_q3/seed_1337/best.pth \
        --out_dir bundles/nrcan-surrogate-bp-fi-ros-v1.0 \
        --name nrcan-surrogate-bp-fi-ros --version 1.0.0 \
        --hazard_denominator_json experiments/hazard_eval_multi_output_spatial_weather_firesize_q3/seed_1337/hazard_scale_denominator.json \
        --fire_size_training_table "/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/df_fire_fru_25ha_1970_2023.csv" \
        --selection_note "Seed 1337 of {42, 1337, 2024}: best mean validation hexel CCC over BP/FI/ROS."

Normalization statistics and lookup tables are read from the checkpoint's training ``data.root_dir``
unless ``--data_root`` or per-file overrides are given.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from inference.bundle import (
    BUNDLE_FORMAT_VERSION,
    CHECKSUMS_FILENAME,
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    RESOURCE_DATASET_NORM_STATS,
    RESOURCE_FEATURE_CHANNEL_MAP,
    RESOURCE_FIRE_SIZE_NORM_PARAMS,
    RESOURCE_FUEL_CURVES,
    RESOURCE_WEATHER_NORM_PARAMS,
    WEIGHTS_FILENAME,
    BundleError,
    BundleManifest,
    EvaluationEntry,
    HazardEntry,
    InferenceDataEntry,
    InputSpecEntry,
    ModelIOEntry,
    ProvenanceEntry,
    ReferenceTableEntry,
    ResourceEntry,
    build_model_from_manifest,
    compute_model_io_dims,
    get_grid_params,
    load_bundle,
    resolve_targets,
    sha256_file,
    utc_timestamp,
)
from src.config import Config
from src.datasets.fuel_utils import FUEL_CURVE_ENCODINGS
from src.datasets.postprocessing.hazard import DEFAULT_FI_CAP, DEFAULT_HAZARD_BIN_THRESHOLDS, DEFAULT_SCALE_TO

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_METRICS = ("ccc", "spearman", "mae", "auc_iou_top10")


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def _summary_metrics(metrics_csv: Path, prefix: str, target_names: Sequence[str]) -> dict[str, float]:
    """Pick the all-hexel summary metrics (e.g. ``test_hexel/all/bp_ccc``) from a metrics CSV."""
    if not metrics_csv.is_file():
        return {}
    with open(metrics_csv) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    summary = {}
    for target in target_names:
        for metric in SUMMARY_METRICS:
            value = row.get(f"{prefix}_hexel/all/{target}_{metric}")
            if value:
                summary[f"{target}_{metric}"] = float(value)
    return summary


def _reference_table(path: Path, note: str) -> ReferenceTableEntry:
    df = pd.read_csv(path)
    columns = [str(column) for column in df.columns if not str(column).startswith("Unnamed:")]
    return ReferenceTableEntry(filename=path.name, sha256=sha256_file(path), num_rows=len(df), columns=columns, note=note)


def _copy_resource(src: Path, bundle_dir: Path, rel_path: str, description: str) -> ResourceEntry:
    if not src.is_file():
        raise BundleError(f"Required resource not found: {src}")
    dst = bundle_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return ResourceEntry(path=rel_path, sha256=sha256_file(dst), description=description)


def _verify_fuel_curve_buffers(state_dict: dict[str, torch.Tensor], norm_stats: dict[str, Any], encoding: str) -> None:
    """The FuelCurveEncoder buffers are baked into the weights; they must agree with the shipped norm stats."""
    stats = norm_stats.get(f"fuel_curve_{encoding}", {})
    for key, stat_name in (("fuel_curve_encoder.curve_mean", "log_mean"), ("fuel_curve_encoder.curve_std", "log_std")):
        if key in state_dict and stats.get(stat_name) is not None:
            buffer_value = float(state_dict[key].flatten()[0])
            if abs(buffer_value - float(stats[stat_name])) > 1e-4 * max(1.0, abs(buffer_value)):
                raise BundleError(
                    f"{key}={buffer_value} in the checkpoint disagrees with fuel_curve_{encoding}.{stat_name}={stats[stat_name]} "
                    "in the normalization stats. The stats file does not belong to this checkpoint."
                )


def _write_model_card(bundle_dir: Path, manifest: BundleManifest) -> None:
    lines = [
        f"# Model card: {manifest.name} v{manifest.version}",
        "",
        manifest.description or "U-Net surrogate of BurnP3+ trained on the national BurnP3+ hexel dataset.",
        "",
        "## Outputs",
        "",
        "| Target | Label | Units |",
        "|---|---|---|",
        *[f"| `{t.name}` | {t.label} | {t.units} |" for t in manifest.targets],
        "",
        "## Inputs",
        "",
        f"- Rasters are reprojected to `{manifest.inputs.crs}`; the model expects ~{manifest.inputs.resolution_m:g} m cells.",
        f"- Patches of {manifest.data_prep.win_h}x{manifest.data_prep.win_w} px with overlap {manifest.data_prep.overlap_ratio}.",
    ]
    table = manifest.inputs.fire_size_training_table
    if table is not None:
        lines.append(
            f"- Fire-size table is **not included**; users provide their own (`{', '.join(table.columns)}`). "
            f"Training used `{table.filename}` ({table.num_rows} rows). {table.note}"
        )
    lines += ["", "## Training and selection", ""]
    prov = manifest.provenance
    lines += [
        f"- Source checkpoint: `{prov.source_checkpoint}` (sha256 `{prov.source_checkpoint_sha256[:12]}...`), epoch {prov.epoch}, seed {prov.seed}.",
        f"- Code commit: `{prov.code_git_commit}`{' (uncommitted changes)' if prov.code_git_dirty else ''}.",
    ]
    if prov.selection_note:
        lines.append(f"- Selection: {prov.selection_note}")
    if prov.metrics:
        metric_names = sorted({name for split in prov.metrics.values() for name in split})
        lines += ["", "## Metrics (all evaluated hexels, vs BurnP3+)", "", "| Metric | " + " | ".join(prov.metrics) + " |"]
        lines.append("|---|" + "---|" * len(prov.metrics))
        for name in metric_names:
            lines.append(f"| {name} | " + " | ".join(f"{prov.metrics[split].get(name, float('nan')):.4f}" for split in prov.metrics) + " |")
    lines += [
        "",
        "## Limitations",
        "",
        "- Emulates BurnP3+ outputs; it inherits BurnP3+ assumptions and does not replace validation against simulations.",
        "- Accuracy is lower for fuels, weather, or fire regimes not represented in the national training hexels.",
        "- Fuel codes absent from the bundled fuel-curve table cannot be predicted.",
        "",
    ]
    (bundle_dir / MODEL_CARD_FILENAME).write_text("\n".join(lines))


def export_bundle(
    checkpoint_path: Path,
    out_dir: Path,
    name: str,
    version: str,
    description: str = "",
    data_root: Path | None = None,
    dataset_norm_stats_path: Path | None = None,
    weather_norm_params_path: Path | None = None,
    fire_size_norm_params_path: Path | None = None,
    fuel_curves_path: Path | None = None,
    feature_channel_map_path: Path | None = None,
    hazard_denominator_json: Path | None = None,
    fire_size_training_table: Path | None = None,
    metrics_dir: Path | None = None,
    selection_note: str = "",
    overwrite: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise BundleError(f"Output directory {out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    # The training checkpoint is produced by this project; full unpickling is needed for its config dict.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    grid_params = get_grid_params(config.data.input_sources)
    data_root = Path(data_root) if data_root is not None else Path(config.data.root_dir)
    approach = config.data_prep.modelling_approach

    sources = {
        RESOURCE_DATASET_NORM_STATS: (
            dataset_norm_stats_path or data_root / config.data.norm_stats_filename,
            "norm/dataset_norm_stats.json",
            "Training-split normalization stats (elevation, targets, fuel curves).",
        ),
        RESOURCE_WEATHER_NORM_PARAMS: (
            weather_norm_params_path or data_root / "weather_norm_params.json",
            "norm/weather_norm_params.json",
            "Weather min-max / z-score parameters fitted on training hexels.",
        ),
        RESOURCE_FIRE_SIZE_NORM_PARAMS: (
            fire_size_norm_params_path or data_root / "fire_size_norm_params.json",
            "norm/fire_size_norm_params.json",
            "log10(fire size + 1) min-max parameters fitted on training fire zones.",
        ),
        RESOURCE_FEATURE_CHANNEL_MAP: (
            feature_channel_map_path or data_root / f"feature_channel_map_{approach}.json",
            f"lookups/feature_channel_map_{approach}.json",
            "Channel layout of the prepared patches used in training.",
        ),
    }
    if "fuel_grid" in grid_params.feature_names_list and grid_params.fuel_feats_encoding in FUEL_CURVE_ENCODINGS:
        curves_src = fuel_curves_path or data_root / grid_params.fuel_curves_filename
        sources[RESOURCE_FUEL_CURVES] = (curves_src, f"lookups/{Path(curves_src).name}", "FBP fuel curves (fbp_code x SeasonState x ISI).")

    missing = [str(src) for src, _, _ in sources.values() if not Path(src).is_file()]
    if missing:
        raise BundleError("Cannot export bundle, missing resources:\n  " + "\n  ".join(missing))

    with open(sources[RESOURCE_DATASET_NORM_STATS][0]) as handle:
        norm_stats = json.load(handle)
    with open(sources[RESOURCE_FEATURE_CHANNEL_MAP][0]) as handle:
        feature_channel_map = json.load(handle)

    fuel_curves_src = sources[RESOURCE_FUEL_CURVES][0] if RESOURCE_FUEL_CURVES in sources else None
    spatial_channels, auxiliary_dims = compute_model_io_dims(config.data.input_sources, feature_channel_map, fuel_curves_src)
    state_dict = checkpoint["model_state"]
    if grid_params.fuel_feats_encoding in FUEL_CURVE_ENCODINGS:
        _verify_fuel_curve_buffers(state_dict, norm_stats, grid_params.fuel_feats_encoding)

    hazard = HazardEntry(
        fi_cap=config.evaluation.hazard_fi_cap if config.evaluation.hazard_fi_cap is not None else DEFAULT_FI_CAP,
        scale_to=DEFAULT_SCALE_TO,
        bin_thresholds=list(DEFAULT_HAZARD_BIN_THRESHOLDS),
    )
    if hazard_denominator_json is not None:
        with open(hazard_denominator_json) as handle:
            denominator = json.load(handle)
        hazard.scale_denominator = float(denominator["scale_denominator"])
        hazard.scale_denominator_source = denominator.get("scale_denominator_source")

    metrics_dir = Path(metrics_dir) if metrics_dir is not None else checkpoint_path.parent
    target_names = [target.name for target in grid_params.resolved_targets()]
    metrics = {
        split: values
        for split, values in {
            "validation": _summary_metrics(metrics_dir / "val_results.csv", "val", target_names),
            "test": _summary_metrics(metrics_dir / "test_metrics.csv", "test", target_names),
        }.items()
        if values
    }
    git_commit, git_dirty = _git_state()
    best_metrics = checkpoint.get("metric_value") if isinstance(checkpoint.get("metric_value"), dict) else {}

    out_dir.mkdir(parents=True)
    weights_path = out_dir / WEIGHTS_FILENAME
    torch.save({key: value.detach().cpu().contiguous() for key, value in state_dict.items()}, weights_path)
    resources = {key: _copy_resource(Path(src), out_dir, rel, desc) for key, (src, rel, desc) in sources.items()}

    manifest = BundleManifest(
        bundle_format_version=BUNDLE_FORMAT_VERSION,
        name=name,
        version=version,
        description=description,
        targets=resolve_targets(grid_params, norm_stats, bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero),
        model=config.model,
        model_io=ModelIOEntry(
            spatial_input_channels=spatial_channels,
            auxiliary_input_dims=auxiliary_dims,
            feature_channel_map=feature_channel_map,
        ),
        data=InferenceDataEntry(
            filename_col=config.data.filename_col,
            valid_mask_threshold=config.data.valid_mask_threshold,
            input_sources=config.data.input_sources,
        ),
        data_prep=config.data_prep,
        evaluation=EvaluationEntry(
            bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero,
            prediction_support_policy=config.evaluation.prediction_support_policy,
        ),
        hazard=hazard,
        inputs=InputSpecEntry(
            resolution_m=grid_params.terrain_cell_size_m,
            fire_size_training_table=(
                _reference_table(
                    Path(fire_size_training_table),
                    note="Not redistributed for licensing reasons; supply an equivalent table for your study area.",
                )
                if fire_size_training_table is not None
                else None
            ),
        ),
        weights=ResourceEntry(path=WEIGHTS_FILENAME, sha256=sha256_file(weights_path), description="Model weights (state_dict)."),
        resources=resources,
        provenance=ProvenanceEntry(
            exported_at=utc_timestamp(),
            source_checkpoint=str(checkpoint_path),
            source_checkpoint_sha256=sha256_file(checkpoint_path),
            epoch=checkpoint.get("epoch"),
            seed=config.seed,
            best_checkpoint_metrics={k: float(v) for k, v in best_metrics.items()},
            selection_note=selection_note,
            code_git_commit=git_commit,
            code_git_dirty=git_dirty,
            training_data_root=str(config.data.root_dir),
            metrics=metrics,
        ),
    )

    # Verify the manifest fully describes the architecture before publishing it.
    model = build_model_from_manifest(manifest)
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    except RuntimeError as exc:
        shutil.rmtree(out_dir)
        raise BundleError(f"Exported manifest does not reproduce the checkpoint architecture: {exc}") from exc

    with open(out_dir / MANIFEST_FILENAME, "w") as handle:
        yaml.safe_dump(manifest.model_dump(mode="json"), handle, sort_keys=False)
    _write_model_card(out_dir, manifest)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != CHECKSUMS_FILENAME)
    (out_dir / CHECKSUMS_FILENAME).write_text("".join(f"{sha256_file(p)}  {p.relative_to(out_dir).as_posix()}\n" for p in files))

    load_bundle(out_dir)  # round-trip validation
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a training checkpoint into a portable model bundle.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint (best.pth).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Bundle directory to create.")
    parser.add_argument("--name", required=True, help="Bundle name, e.g. nrcan-surrogate-bp-fi-ros.")
    parser.add_argument("--version", required=True, help="Bundle version, e.g. 1.0.0.")
    parser.add_argument("--description", default="", help="One-paragraph description for the manifest and model card.")
    parser.add_argument("--data_root", type=Path, default=None, help="Training data_samples dir (default: checkpoint data.root_dir).")
    parser.add_argument("--dataset_norm_stats", type=Path, default=None, help="Override path to dataset_norm_stats.json.")
    parser.add_argument("--weather_norm_params", type=Path, default=None, help="Override path to weather_norm_params.json.")
    parser.add_argument("--fire_size_norm_params", type=Path, default=None, help="Override path to fire_size_norm_params.json.")
    parser.add_argument("--fuel_curves", type=Path, default=None, help="Override path to the fuel curves CSV.")
    parser.add_argument("--feature_channel_map", type=Path, default=None, help="Override path to feature_channel_map_<N>.json.")
    parser.add_argument("--hazard_denominator_json", type=Path, default=None, help="hazard_scale_denominator.json from evaluate_hazard.")
    parser.add_argument(
        "--fire_size_training_table", type=Path, default=None, help="Raw fire-size table used in training (recorded, not copied)."
    )
    parser.add_argument("--metrics_dir", type=Path, default=None, help="Dir with val_results.csv/test_metrics.csv (default: ckpt dir).")
    parser.add_argument("--selection_note", default="", help="How this checkpoint was selected (recorded in the model card).")
    parser.add_argument("--overwrite", action="store_true", help="Replace --out_dir if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = export_bundle(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        name=args.name,
        version=args.version,
        description=args.description,
        data_root=args.data_root,
        dataset_norm_stats_path=args.dataset_norm_stats,
        weather_norm_params_path=args.weather_norm_params,
        fire_size_norm_params_path=args.fire_size_norm_params,
        fuel_curves_path=args.fuel_curves,
        feature_channel_map_path=args.feature_channel_map,
        hazard_denominator_json=args.hazard_denominator_json,
        fire_size_training_table=args.fire_size_training_table,
        metrics_dir=args.metrics_dir,
        selection_note=args.selection_note,
        overwrite=args.overwrite,
    )
    bundle = load_bundle(out_dir, verify_checksums=False)
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"Bundle {bundle.manifest.name} v{bundle.manifest.version} written to {out_dir} ({size_mb:.1f} MB).")
    for target in bundle.manifest.targets:
        print(f"  {target.name}: {target.label} [{target.units}], out_norm={target.out_norm}")


if __name__ == "__main__":
    main()
