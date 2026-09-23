"""
Evaluate a model bundle against BurnP3+ results: predict (or reuse predictions), then compare hexel by hexel.

The project needs the BurnP3+ inputs *and* outputs of every hexel, e.g.
``hexNN/results/burnP3Plus_OutputBurnProbability/burnProbability-sn2.tif`` (national layout) or
``hexNN/results/<scenario>/burnP3Plus_OutputBurnProbability_<scenario>_All.tif`` with ``--scenario_name``.
Metrics are computed per hexel on the stitched 100 m rasters, as in the model's own evaluation, then
averaged over hexels. Hazard classes are compared with the bundle's national hazard scaling.

Examples:
    # Predict and evaluate every hexel of a project
    python -m inference.evaluate --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \\
        --output evaluation/

    # Score predictions made earlier by inference.predict (the model is not run again)
    python -m inference.evaluate --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \\
        --predictions predictions/ --output evaluation/

Outputs in ``--output``: ``metrics_per_hexel.csv``, ``metrics_summary.csv``, ``hazard_metrics_per_hexel.csv``,
``hazard_metrics_summary.csv``, ``hazard_confusion_matrix.{csv,png}``, ``metrics_per_firezone.csv`` (with
``--by_firezone``), ``plots/hexNN/`` (maps, scatter and histogram per target), ``predictions/`` (when the model
was run), ``evaluation_manifest.json`` and ``evaluate.log``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch

from data_preparation.paths import Paths
from data_preparation.utils import find_hex_ids
from inference.bundle import (
    NO_MASK_SCOPE,
    BundleError,
    ModelBundle,
    data_mask_scope,
    load_bundle,
    resolve_device,
    resolve_mask_scope,
    resolve_scenario_name,
    utc_timestamp,
    validate_scenario_name,
)
from inference.check import CheckReport, check_project, resolve_hex_ids
from inference.predict import (
    PREDICT_MASK_SCOPES,
    RUN_MANIFEST_FILENAME,
    PredictError,
    console_logging,
    discover_hex_ids,
    git_commit,
    log_to_file,
    output_overlap_error,
    run_predict,
)
from src.datasets.postprocessing.hazard import compute_raw_hazard
from src.datasets.postprocessing.hazard_metrics import flatten_hazard_class_metrics
from src.datasets.postprocessing.hazard_pipeline import compute_hazard_hexel, write_confusion_matrix_csv, write_confusion_matrix_plot
from src.datasets.postprocessing.hexel_reconstruction import StitchedHexel
from src.datasets.postprocessing.utils import (
    _actual_area_mask,
    calculate_hexel_metrics_pytorch,
    load_firezone_ids,
    load_target_grid_for_mask_scope,
    mask_grids_by_support,
)
from src.datasets.postprocessing.visualize_predictions import plot_hexbin_distribution, plot_histogram_distribution, visualize_target_grids
from src.datasets.targets import TargetSpec, get_target_spec
from src.utils import AVAILABLE_METRICS

logger = logging.getLogger("inference.evaluate")

# The hexel-level metrics reported for the released model.
DEFAULT_METRICS = (
    "ccc",
    "spearman",
    "mae",
    "mse",
    "normalized_mae",
    "bias",
    "normalized_bias",
    "mae_top01",
    "mae_top02",
    "mae_top05",
    "mae_top10",
    "iou_top005",
    "iou_top01",
    "iou_top02",
    "iou_top05",
    "iou_top10",
    "auc_iou_top10",
    "auc_iou_full",
)
# Shown in the console summary when computed.
HEADLINE_METRICS = ("ccc", "spearman", "mae", "bias")
HAZARD_HEADLINE_METRICS = ("exact_accuracy", "within_1_accuracy", "macro_f1")
# Continuous hazard (BP x capped FI) is scored like a target under this name.
HAZARD_TARGET = "hazard"

PREDICTIONS_DIRNAME = "predictions"
PLOTS_DIRNAME = "plots"
LOG_FILENAME = "evaluate.log"
EVALUATION_MANIFEST_FILENAME = "evaluation_manifest.json"
METRICS_PER_HEXEL_FILENAME = "metrics_per_hexel.csv"
METRICS_SUMMARY_FILENAME = "metrics_summary.csv"
METRICS_PER_FIREZONE_FILENAME = "metrics_per_firezone.csv"
HAZARD_PER_HEXEL_FILENAME = "hazard_metrics_per_hexel.csv"
HAZARD_SUMMARY_FILENAME = "hazard_metrics_summary.csv"
OWNED_OUTPUTS = frozenset(
    {
        PREDICTIONS_DIRNAME,
        PLOTS_DIRNAME,
        LOG_FILENAME,
        EVALUATION_MANIFEST_FILENAME,
        METRICS_PER_HEXEL_FILENAME,
        METRICS_SUMMARY_FILENAME,
        METRICS_PER_FIREZONE_FILENAME,
        HAZARD_PER_HEXEL_FILENAME,
        HAZARD_SUMMARY_FILENAME,
        "hazard_confusion_matrix.csv",
        "hazard_confusion_matrix.png",
    }
)


class EvaluateError(RuntimeError):
    """A problem the user can fix (missing predictions, bad options, input errors)."""


@dataclass
class EvaluateRun:
    output_dir: Path
    predictions_dir: Path
    mask_scope: str = "actual"
    hexels: list[str] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    firezone_metrics: pd.DataFrame | None = None
    hazard_metrics: pd.DataFrame | None = None
    hazard_summary: dict[str, float] | None = None
    input_check: CheckReport | None = None
    fire_size_note: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class _HexelScores:
    rows: list[dict[str, Any]] = field(default_factory=list)
    firezone_rows: list[dict[str, Any]] = field(default_factory=list)
    hazard_row: dict[str, Any] | None = None
    confusion: np.ndarray | None = None


def resolve_metric_functions(names: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    names = list(names or DEFAULT_METRICS)
    unknown = [name for name in names if name not in AVAILABLE_METRICS]
    if unknown:
        raise EvaluateError(f"Unknown metric(s) {unknown}. Available: {', '.join(sorted(AVAILABLE_METRICS))}.")
    return {name: AVAILABLE_METRICS[name] for name in names}


def prediction_path(predictions_dir: Path, hex_id: str, name: str) -> Path:
    return predictions_dir / f"hex{hex_id}" / f"hex{hex_id}_{name}.tif"


def read_prediction(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """A prediction raster written by inference.predict, with nodata as NaN."""
    with rasterio.open(path) as src:
        grid = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        return grid, dict(src.profile)


def score_grids(gt: np.ndarray, pred: np.ndarray, metric_functions: dict[str, Any], device: torch.device) -> dict[str, float]:
    """Metrics over the pixels where both grids are finite and the target is non-negative."""
    valid = np.isfinite(gt) & np.isfinite(pred) & (np.nan_to_num(gt, nan=-1.0) >= 0.0)
    num_pixels = int(valid.sum())
    if num_pixels == 0:
        return {"n_pixels": 0, **{name: float("nan") for name in metric_functions}}
    return {"n_pixels": num_pixels, **calculate_hexel_metrics_pytorch(gt, pred, device, metric_functions)}


def _read_predictions_manifest(predictions_dir: Path) -> dict[str, Any] | None:
    path = predictions_dir / RUN_MANIFEST_FILENAME
    if not path.is_file():
        return None
    with open(path) as handle:
        return json.load(handle)


def _hexels_with_predictions(predictions_dir: Path, bundle: ModelBundle) -> list[str]:
    first_target = bundle.manifest.targets[0].name
    return sorted(
        hex_id for hex_id in find_hex_ids(str(predictions_dir)) if prediction_path(predictions_dir, hex_id, first_target).is_file()
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool, keep: Path | None) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise EvaluateError(f"Output folder {output_dir} is not empty. Choose another --output or pass --overwrite.")
        for child in output_dir.iterdir():
            if child.name not in OWNED_OUTPUTS or (keep is not None and child.resolve() == keep):
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _plot_target(
    stitched: StitchedHexel, label: str, plots_dir: Path, show_support_outline: bool, run_warnings: list[str] | None = None
) -> None:
    """Maps (surrogate, BurnP3+, difference), scatter and histogram, saved as plots/hexNN/hexNN_<target>_*.png."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    hex_id, name = stitched.hex_id, stitched.target.name
    hexel_dir = plots_dir / f"hex{hex_id}"
    raw_dir = hexel_dir / "predicted_hexels_plot"
    renames = {
        f"hexel_{hex_id}_{name}_predicted.png": f"hex{hex_id}_{name}_maps.png",
        f"hexbin_hex_{hex_id}_{name}.png": f"hex{hex_id}_{name}_scatter.png",
        f"hist_hex_{hex_id}_{name}.png": f"hex{hex_id}_{name}_histogram.png",
    }
    try:
        # The plotting helpers print the pre-rename paths; keep the console to our own log lines.
        with contextlib.redirect_stdout(io.StringIO()):
            visualize_target_grids(
                gt_grid=stitched.gt_grid,
                pred_grid=stitched.pred_grid,
                hex_id=hex_id,
                save_dir=str(hexel_dir),
                target_label=label,
                target_name=name,
                actual_support_mask=stitched.actual_support_mask,
                buffer_support_mask=stitched.buffer_support_mask,
                prediction_support_label="surrogate",
                show_prediction_support_outline=show_support_outline,
                gt_title="BurnP3+",
                diff_title="Surrogate - BurnP3+",
            )
            for plot in (plot_hexbin_distribution, plot_histogram_distribution):
                plot(
                    gt_grid=stitched.gt_grid,
                    pred_grid=stitched.pred_grid,
                    hex_id=hex_id,
                    save_dir=str(hexel_dir),
                    target_label=label,
                    probability_scale=stitched.target.probability_scale,
                    target_name=name,
                )
    except ValueError as exc:
        message = f"hex{hex_id}: could not plot {name}: {exc}"
        logger.warning("%s", message)
        if run_warnings is not None:
            run_warnings.append(message)
    finally:
        plt.close("all")
        for old, new in renames.items():
            if (raw_dir / old).is_file():
                (raw_dir / old).replace(hexel_dir / new)
        shutil.rmtree(raw_dir, ignore_errors=True)


def _area_rows(
    hex_id: str,
    target: str,
    scope: str,
    gt: np.ndarray,
    pred: np.ndarray,
    actual_support: np.ndarray | None,
    metric_functions: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Scores over the predicted area; buffer runs also report the inner hexel and the ring around it."""
    rows = [{"hexel": f"hex{hex_id}", "target": target, "area": scope, **score_grids(gt, pred, metric_functions, device)}]
    if scope == "buffer" and actual_support is not None:
        for area, support in (("actual", actual_support), ("buffer_only", np.isfinite(pred) & ~actual_support)):
            area_gt, area_pred = mask_grids_by_support(gt, pred, support)
            rows.append(
                {"hexel": f"hex{hex_id}", "target": target, "area": area, **score_grids(area_gt, area_pred, metric_functions, device)}
            )
    return rows


def evaluate_hexel(
    bundle: ModelBundle,
    project_dir: Path,
    predictions_dir: Path,
    hex_id: str,
    scope: str,
    metric_functions: dict[str, Any],
    device: torch.device,
    by_firezone: bool = False,
    plots_dir: Path | None = None,
    scenario_name: str | None = None,
    run_warnings: list[str] | None = None,
) -> _HexelScores:
    manifest = bundle.manifest
    paths = Paths(hex_id=hex_id, root_dir=project_dir)
    scores = _HexelScores()
    kept: dict[str, StitchedHexel] = {}
    firezone_ids: np.ndarray | None = None
    for target in manifest.targets:
        spec: TargetSpec = get_target_spec(target.name)
        path = prediction_path(predictions_dir, hex_id, target.name)
        if not path.is_file():
            raise EvaluateError(f"Missing prediction {path}. Run inference.predict for this hexel or drop --predictions.")
        pred, profile = read_prediction(path)
        gt, pred = load_target_grid_for_mask_scope(
            paths=paths,
            target=spec,
            pred_grid=pred,
            profile=profile,
            mask_scope=data_mask_scope(scope),
            hex_id=hex_id,
            bp_nodata_as_zero=manifest.evaluation.bp_nodata_as_zero,
            scenario_name=scenario_name,
        )
        actual_support = _actual_area_mask(paths.mask_grid_actual(hex_id=hex_id), profile, pred.shape) if scope != NO_MASK_SCOPE else None
        buffer_support = (
            _actual_area_mask(paths.mask_grid(hex_id=hex_id, mask_scope=scope), profile, pred.shape) if scope == "buffer" else None
        )
        stitched = StitchedHexel(hex_id, spec, gt, pred, profile, actual_support, buffer_support)
        scores.rows += _area_rows(hex_id, target.name, scope, gt, pred, actual_support, metric_functions, device)

        if by_firezone:
            if firezone_ids is None:
                firezone_ids = load_firezone_ids(paths, hex_id, profile)
            if firezone_ids is None:
                message = f"hex{hex_id}: no fire-zone raster, skipping per-fire-zone metrics."
                logger.warning("%s", message)
                if run_warnings is not None:
                    run_warnings.append(message)
                by_firezone = False
            else:
                for zone in np.unique(firezone_ids[np.isfinite(firezone_ids) & np.isfinite(pred)]):
                    in_zone = firezone_ids == zone
                    zone_scores = score_grids(np.where(in_zone, gt, np.nan), np.where(in_zone, pred, np.nan), metric_functions, device)
                    scores.firezone_rows.append({"hexel": f"hex{hex_id}", "target": target.name, "firezone": int(zone), **zone_scores})

        if plots_dir is not None:
            label = f"{target.label} ({target.units})" if target.units else target.label
            _plot_target(stitched, label, plots_dir, show_support_outline=scope == "actual", run_warnings=run_warnings)
        if target.name in {"bp", "fi"}:
            kept[target.name] = stitched

    hazard = manifest.hazard
    if {"bp", "fi"} <= set(kept):
        bp, fi = kept["bp"], kept["fi"]
        gt_raw = compute_raw_hazard(bp.gt_grid, fi.gt_grid, hazard.fi_cap)
        pred_raw = compute_raw_hazard(bp.pred_grid, fi.pred_grid, hazard.fi_cap)
        scores.rows += _area_rows(hex_id, HAZARD_TARGET, scope, gt_raw, pred_raw, bp.actual_support_mask, metric_functions, device)
        if hazard.scale_denominator is not None:
            try:
                result = compute_hazard_hexel(
                    bp,
                    fi,
                    denominator=hazard.scale_denominator,
                    fi_cap=hazard.fi_cap,
                    scale_to=hazard.scale_to,
                    bin_thresholds=list(hazard.bin_thresholds),
                )
            except ValueError as exc:
                message = f"hex{hex_id}: hazard classes not compared: {exc}"
                logger.warning("%s", message)
                if run_warnings is not None:
                    run_warnings.append(message)
            else:
                scores.hazard_row = {"hexel": f"hex{hex_id}", **flatten_hazard_class_metrics(result.metrics)}
                scores.confusion = np.asarray(result.metrics["confusion_matrix"], dtype=np.int64)
    return scores


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric over hexels (NaN ignored), per target and area."""
    metric_columns = [column for column in metrics.columns if column not in {"hexel", "target", "area", "n_pixels"}]
    grouped = metrics.groupby(["target", "area"], sort=False)
    summary = grouped[metric_columns].mean()
    summary.insert(0, "n_hexels", grouped["hexel"].nunique())
    return summary.reset_index()


def _summary_dict(summary: pd.DataFrame, scope: str) -> dict[str, Any]:
    """{target: {metric: mean}}; the extra areas of buffer runs appear as "<target>/<area>"."""
    result: dict[str, Any] = {}
    for record in summary.to_dict(orient="records"):
        target, area = record.pop("target"), record.pop("area")
        key = target if area == scope else f"{target}/{area}"
        result.setdefault(key, {}).update({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in record.items()})
    return result


def _write_manifest(
    run: EvaluateRun,
    bundle: ModelBundle,
    project_dir: Path,
    options: dict[str, Any],
    predictions_info: dict[str, Any],
    started_at: str,
    status: str,
    error: str | None = None,
) -> None:
    hazard = bundle.manifest.hazard
    manifest = {
        "status": status,
        "error": error,
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "bundle": {
            "name": bundle.manifest.name,
            "version": bundle.manifest.version,
            "path": str(bundle.root),
            "weights_sha256": bundle.manifest.weights.sha256,
        },
        "project_dir": str(project_dir),
        "predictions": predictions_info,
        "options": options,
        "hazard": {
            "fi_cap": hazard.fi_cap,
            "scale_to": hazard.scale_to,
            "scale_denominator": hazard.scale_denominator,
            "scale_denominator_source": hazard.scale_denominator_source,
            "bin_thresholds": list(hazard.bin_thresholds),
        },
        "input_check": (
            {"errors": [asdict(f) for f in run.input_check.errors], "warnings": [asdict(f) for f in run.input_check.warnings]}
            if run.input_check is not None
            else None
        ),
        "warnings": run.warnings,
        "hexels": run.hexels,
        "summary": _summary_dict(run.summary, run.mask_scope) if not run.summary.empty else None,
        "hazard_summary": run.hazard_summary,
        "software": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    with open(run.output_dir / EVALUATION_MANIFEST_FILENAME, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def _resolve_scope(requested: str | None, bundle: ModelBundle) -> str:
    try:
        return resolve_mask_scope(requested, bundle)
    except (BundleError, ValueError) as exc:
        raise EvaluateError(str(exc)) from exc


def _resolve_scenario(requested: str | None, bundle: ModelBundle, predicted_options: dict[str, Any] | None = None) -> str | None:
    """The scenario to score: the one reused predictions were made for, else ``requested`` or the bundle's."""
    try:
        if predicted_options is None or "scenario_name" not in predicted_options:
            return resolve_scenario_name(requested, bundle)
        predicted = validate_scenario_name(predicted_options["scenario_name"])
        requested = validate_scenario_name(requested)
    except ValueError as exc:
        raise EvaluateError(str(exc)) from exc
    if requested and requested != predicted:
        made_for = f"scenario {predicted!r}" if predicted else "the national rasters (no scenario)"
        hint = f"--scenario_name {predicted}" if predicted else "no --scenario_name"
        raise EvaluateError(f"The predictions were made for {made_for}; evaluate them with {hint}.")
    return predicted


def run_evaluate(
    bundle_dir: str | Path,
    project_dir: str | Path,
    output_dir: str | Path,
    predictions_dir: str | Path | None = None,
    hex_ids: list[str] | None = None,
    metrics: list[str] | None = None,
    mask_scope: str | None = None,
    by_firezone: bool = False,
    plots: bool = True,
    fire_size_table: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    scenario_name: str | None = None,
    overwrite: bool = False,
    verify_checksums: bool = True,
    check_inputs: bool = True,
) -> EvaluateRun:
    """Compare the bundle's predictions for a project with its BurnP3+ outputs; returns the metric tables."""
    started_at = utc_timestamp()
    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise EvaluateError(f"Project folder not found: {project_dir}")
    overlap = output_overlap_error(output_dir, project_dir)
    if overlap:
        raise EvaluateError(overlap)
    bundle = load_bundle(bundle_dir, verify_checksums=verify_checksums)
    metric_functions = resolve_metric_functions(metrics)

    reuse = predictions_dir is not None
    predictions_manifest: dict[str, Any] | None = None
    if reuse:
        assert predictions_dir is not None
        pred_dir = Path(predictions_dir).expanduser().resolve()
        if not pred_dir.is_dir():
            raise EvaluateError(f"Predictions folder not found: {pred_dir}")
        predictions_manifest = _read_predictions_manifest(pred_dir)
        predicted_options = (predictions_manifest or {}).get("options", {})
        predicted_scope = predicted_options.get("mask_scope")
        scope = _resolve_scope(mask_scope or predicted_scope, bundle)
        if predicted_scope and scope != predicted_scope:
            raise EvaluateError(f"The predictions cover the {predicted_scope!r} area; evaluate them with --mask_scope {predicted_scope}.")
        scenario_name = _resolve_scenario(scenario_name, bundle, predicted_options)
        available = _hexels_with_predictions(pred_dir, bundle)
        if not available:
            raise EvaluateError(f"No predictions (hexNN/hexNN_<target>.tif) found in {pred_dir}.")
        selected = available
        if hex_ids:
            selected, unknown = resolve_hex_ids(hex_ids, available)
            if unknown:
                raise EvaluateError(f"No predictions for hexel(s) {unknown} in {pred_dir}. Available: {available}.")
        project_hexels = find_hex_ids(str(project_dir))
        missing = [hex_id for hex_id in selected if hex_id not in project_hexels]
        if missing:
            raise EvaluateError(f"Hexel(s) {['hex' + h for h in missing]} have predictions but no folder in the project {project_dir}.")
    else:
        pred_dir = output_dir / PREDICTIONS_DIRNAME
        scope = _resolve_scope(mask_scope, bundle)
        scenario_name = _resolve_scenario(scenario_name, bundle)
        try:
            selected = discover_hex_ids(project_dir, hex_ids)
        except PredictError as exc:
            raise EvaluateError(str(exc)) from exc

    _prepare_output_dir(output_dir, overwrite, keep=pred_dir if reuse else None)
    resolved_device = resolve_device(device)
    options = {
        "hex_ids": selected,
        "mask_scope": scope,
        "metrics": list(metric_functions),
        "by_firezone": by_firezone,
        "plots": plots,
        "device": str(resolved_device),
        "scenario_name": scenario_name,
        "check_inputs": check_inputs,
    }
    predictions_info: dict[str, Any] = {"path": str(pred_dir), "made_by_this_run": not reuse}
    if predictions_manifest is not None:
        predictions_info["bundle"] = predictions_manifest.get("bundle")
    run = EvaluateRun(output_dir=output_dir, predictions_dir=pred_dir, mask_scope=scope, hexels=[f"hex{hex_id}" for hex_id in selected])

    with log_to_file(output_dir / LOG_FILENAME) as log_handler:
        try:
            logger.info(
                "Evaluating bundle %s v%s on %d hexel(s) (mask: %s)", bundle.manifest.name, bundle.manifest.version, len(selected), scope
            )
            if reuse:
                logger.info("Scoring existing predictions in %s", pred_dir)
                if predictions_manifest is None:
                    run.warnings.append(f"{pred_dir} has no {RUN_MANIFEST_FILENAME}; cannot confirm which model made these predictions.")
                elif (predictions_manifest.get("bundle") or {}).get("weights_sha256") != bundle.manifest.weights.sha256:
                    made_by = predictions_manifest.get("bundle") or {}
                    run.warnings.append(
                        f"The predictions were made with {made_by.get('name')} v{made_by.get('version')}, not with this bundle; "
                        "the metrics describe those predictions."
                    )
                for message in run.warnings:
                    logger.warning("%s", message)
            if check_inputs:
                run.input_check = check_project(
                    bundle,
                    project_dir,
                    hex_ids=selected,
                    fire_size_table=fire_size_table,
                    mask_scope=scope,
                    scenario_name=scenario_name,
                    inputs=not reuse,
                    outputs=True,
                )
                for finding in run.input_check.findings:
                    if finding.level == "warning":
                        logger.warning("Input check: %s", finding.describe())
                if run.input_check.errors:
                    listed = "\n".join(f"  - {finding.describe()}" for finding in run.input_check.errors)
                    raise EvaluateError(
                        f"The project has {len(run.input_check.errors)} problem(s); nothing was evaluated. "
                        f"Fix these (or run `python -m inference.check --outputs` for the full report):\n{listed}"
                    )
            if not reuse:
                logger.info("Predicting into %s", pred_dir)
                predict_run = run_predict(
                    bundle_dir=bundle.root,
                    project_dir=project_dir,
                    output_dir=pred_dir,
                    fire_size_table=fire_size_table,
                    hex_ids=selected,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    mask_scope=scope,
                    scenario_name=scenario_name,
                    verify_checksums=False,
                    check_inputs=False,
                )
                run.fire_size_note = predict_run.fire_size_note

            plots_dir = output_dir / PLOTS_DIRNAME if plots else None
            rows: list[dict[str, Any]] = []
            firezone_rows: list[dict[str, Any]] = []
            hazard_rows: list[dict[str, Any]] = []
            confusion: np.ndarray | None = None
            for hex_id in selected:
                start = time.perf_counter()
                logger.info("hex%s: comparing with BurnP3+", hex_id)
                scores = evaluate_hexel(
                    bundle,
                    project_dir,
                    pred_dir,
                    hex_id,
                    scope,
                    metric_functions,
                    resolved_device,
                    by_firezone=by_firezone,
                    plots_dir=plots_dir,
                    scenario_name=scenario_name,
                    run_warnings=run.warnings,
                )
                rows += scores.rows
                firezone_rows += scores.firezone_rows
                if scores.hazard_row is not None:
                    hazard_rows.append(scores.hazard_row)
                if scores.confusion is not None:
                    confusion = scores.confusion if confusion is None else confusion + scores.confusion
                logger.info("hex%s: evaluated in %.1f s", hex_id, time.perf_counter() - start)

            run.metrics = pd.DataFrame(rows)
            run.metrics.to_csv(output_dir / METRICS_PER_HEXEL_FILENAME, index=False)
            run.summary = summarize(run.metrics)
            run.summary.to_csv(output_dir / METRICS_SUMMARY_FILENAME, index=False)
            if by_firezone and firezone_rows:
                run.firezone_metrics = pd.DataFrame(firezone_rows)
                run.firezone_metrics.to_csv(output_dir / METRICS_PER_FIREZONE_FILENAME, index=False)
            if hazard_rows:
                run.hazard_metrics = pd.DataFrame(hazard_rows)
                run.hazard_metrics.to_csv(output_dir / HAZARD_PER_HEXEL_FILENAME, index=False)
                means = run.hazard_metrics.drop(columns=["hexel"]).mean()
                run.hazard_summary = {"n_hexels": len(hazard_rows), **{k: float(v) for k, v in means.items() if np.isfinite(v)}}
                pd.DataFrame([run.hazard_summary]).to_csv(output_dir / HAZARD_SUMMARY_FILENAME, index=False)
            if confusion is not None:
                write_confusion_matrix_csv(confusion, str(output_dir))
                write_confusion_matrix_plot(confusion, str(output_dir))
            _write_manifest(run, bundle, project_dir, options, predictions_info, started_at, status="success")
            logger.info("Wrote evaluation of %d hexel(s) to %s", len(selected), output_dir)
            return run
        except Exception as exc:
            log_handler.stream.write(traceback.format_exc())
            _write_manifest(run, bundle, project_dir, options, predictions_info, started_at, status="failed", error=str(exc))
            raise


def format_summary(run: EvaluateRun) -> str:
    """Short console table of the headline metrics."""
    summary = run.summary
    lines = [f"Evaluated {len(run.hexels)} hexel(s) against BurnP3+ (area: {run.mask_scope}). Mean over hexels:"]
    if not summary.empty:
        columns = [name for name in HEADLINE_METRICS if name in summary.columns]
        main_areas = summary[summary["area"] == run.mask_scope]
        lines.append("  " + f"{'target':<8}" + "".join(f"{name:>12}" for name in columns))
        for record in main_areas.to_dict(orient="records"):
            lines.append("  " + f"{record['target']:<8}" + "".join(f"{record[name]:>12.4g}" for name in columns))
    if run.hazard_summary:
        parts = [
            f"{name.replace('_', ' ')} {run.hazard_summary[name]:.3f}" for name in HAZARD_HEADLINE_METRICS if name in run.hazard_summary
        ]
        lines.append("Hazard classes: " + ", ".join(parts))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference.evaluate",
        description="Evaluate a model bundle against the BurnP3+ outputs of a project (predicts first unless --predictions is given).",
    )
    parser.add_argument("--bundle", required=True, help="Model bundle folder (contains manifest.yaml).")
    parser.add_argument("--project", required=True, help="Project folder with hexNN/ folders holding BurnP3+ inputs and results.")
    parser.add_argument("--output", required=True, help="Folder to write metrics, plots (and predictions) to.")
    parser.add_argument("--predictions", default=None, help="Score this inference.predict output instead of running the model.")
    parser.add_argument("--hex_ids", nargs="+", default=None, help="Hexels to evaluate, e.g. 12 or hex12 (default: all).")
    parser.add_argument(
        "--metrics", nargs="+", default=None, help=f"Metrics to compute (default: {' '.join(DEFAULT_METRICS)}).", metavar="METRIC"
    )
    parser.add_argument(
        "--mask_scope",
        choices=PREDICT_MASK_SCOPES,
        default=None,
        help="Area to evaluate: actual hexel mask (default), buffer, or none (whole raster extent).",
    )
    parser.add_argument("--by_firezone", action="store_true", help="Also write metrics per fire zone (slower).")
    parser.add_argument("--no_plots", action="store_true", help="Skip the per-hexel plots (faster).")
    parser.add_argument(
        "--fire_size_table", default=None, help="Fire-size CSV (GRIDCODE, SIZE_HA) to use instead of the bundle's national table."
    )
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, cuda:1 or mps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Patches per model call (lower it if memory runs out).")
    parser.add_argument("--num_workers", type=int, default=0, help="Data-loading worker processes (0 is safest on Windows/macOS).")
    parser.add_argument(
        "--scenario_name",
        default=None,
        help="Use hexNN_fbp_<name>.tif and results/<name>/ BurnP3+ outputs (default: the scenario of --predictions, else the model's).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a previous evaluation in --output.")
    parser.add_argument("--skip_checksums", action="store_true", help="Skip bundle checksum verification (faster start-up).")
    parser.add_argument("--skip_check", action="store_true", help="Do not check the project first (see inference.check --outputs).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        with console_logging():
            run = run_evaluate(
                bundle_dir=args.bundle,
                project_dir=args.project,
                output_dir=args.output,
                predictions_dir=args.predictions,
                hex_ids=args.hex_ids,
                metrics=args.metrics,
                mask_scope=args.mask_scope,
                by_firezone=args.by_firezone,
                plots=not args.no_plots,
                fire_size_table=args.fire_size_table,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scenario_name=args.scenario_name,
                overwrite=args.overwrite,
                verify_checksums=not args.skip_checksums,
                check_inputs=not args.skip_check,
            )
    except (EvaluateError, PredictError, BundleError, FileNotFoundError, ValueError, KeyError) as exc:
        log_path = Path(args.output) / LOG_FILENAME
        details = f"\n(Full details in {log_path})" if log_path.is_file() else ""
        print(f"\nError: {exc}{details}", file=sys.stderr)
        return 2
    print("\n" + format_summary(run))
    if run.fire_size_note:
        print(run.fire_size_note)
    num_warnings = len(run.warnings) + (len(run.input_check.warnings) if run.input_check is not None else 0)
    if num_warnings:
        print(f"{num_warnings} warning(s) were logged; see {run.output_dir / LOG_FILENAME}.")
    print(
        f"Results in {run.output_dir} ({METRICS_PER_HEXEL_FILENAME}, {METRICS_SUMMARY_FILENAME}{', plots/' if not args.no_plots else ''})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
