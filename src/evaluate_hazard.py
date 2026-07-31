"""Hazard evaluation CLI for a single multi-output BP/FI checkpoint.

This entrypoint reuses the standard model inference stack and shared hexel
reconstruction, then applies the hazard-specific denominator, raster, and
class-metric logic. The model is expected to jointly predict both ``bp`` and
``fi`` (and optionally other targets, e.g. ``ros``) from one checkpoint.

python -m src.evaluate_hazard --config configs/hazard_eval_common_input_pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Collection, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_raster, read_split_hex_ids
from data_preparation.utils import find_hex_ids
from src.config import Config, HazardEvalConfig, HazardModelEntry
from src.datasets.dataset import get_test_dataloader
from src.datasets.postprocessing.hazard import compute_raw_hazard, max_finite_hazard
from src.datasets.postprocessing.hazard_metrics import flatten_hazard_class_metrics
from src.datasets.postprocessing.hazard_pipeline import (
    compute_hazard_hexel,
    pair_stitched_hexels,
    save_hazard_hexel_artifacts,
)
from src.datasets.postprocessing.hexel_reconstruction import (
    StitchedHexel,
    reconstruct_denormalized_hexels,
)
from src.datasets.postprocessing.utils import get_config_grid_params
from src.datasets.postprocessing.visualize_predictions import as_float_array_with_nan
from src.datasets.targets import get_target_spec
from src.datasets.utils import get_dataset_dimensions
from src.evaluate_hexels import load_config
from src.trainer import Trainer
from src.utils import seed_everything

DENOMINATOR_JSON_FILENAME = "hazard_scale_denominator.json"
PER_HEX_CSV_FILENAME = "hazard_metrics_per_hex.csv"
SUMMARY_JSON_FILENAME = "hazard_metrics_summary.json"
CONFUSION_MATRIX_CSV_FILENAME = "hazard_confusion_matrix.csv"
CONFUSION_MATRIX_PNG_FILENAME = "hazard_confusion_matrix.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate combined BP x FI hazard product from a single multi-output model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hazard_eval_common_input_pipeline.yaml",
        help="Path to hazard evaluation YAML config file.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Compute hazard metrics without writing GeoTIFF or plot artifacts.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Write hazard GeoTIFFs but skip per-hexel PNG plots.",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Override model-entry save_predictions flags and never write patch predictions.",
    )
    parser.add_argument(
        "--self_normalized_prediction",
        action="store_true",
        help="Scale predicted hazard by the max raw hazard over this run's predictions; ground truth keeps the configured denominator.",
    )
    parser.add_argument(
        "--mask_scope",
        type=str,
        choices=["actual", "buffer", "buffer_only"],
        default=None,
        help="Override hazard_config.mask_scope for this run.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override hazard_config.save_dir for this run (avoids output collisions).",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Override hazard_config.root_dir for this run.",
    )
    parser.add_argument(
        "--stitch_mode",
        type=str,
        choices=["mean", "max"],
        default=None,
        help="Override hazard_config.stitch_mode for this run.",
    )
    return parser.parse_args()


def load_hazard_config(path: str) -> HazardEvalConfig:
    """Load and validate a :class:`HazardEvalConfig` from YAML."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hazard config file not found: {path}")
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    return HazardEvalConfig(**raw)


def get_model_out_norm(model_config: Config) -> str:
    """Return the grid source ``out_norm`` for a model config, falling back to ``min_max``."""
    grid_params = get_config_grid_params(model_config)
    return grid_params.out_norm if grid_params is not None else "min_max"


def prepare_model_config_for_hazard(
    model_config: Config,
    hazard_config: HazardEvalConfig,
    entry: HazardModelEntry,
    required_targets: Collection[str] = ("bp", "fi"),
) -> Config:
    """Override a model config's data/checkpoint/logger settings for hazard eval.

    Validates that the model's grid source jointly predicts every target in
    ``required_targets`` (by default ``bp`` and ``fi``) from a single checkpoint.
    """
    model_config.data.root_dir = hazard_config.root_dir
    model_config.data.raw_data_dir = hazard_config.raw_data_dir
    model_config.data.test_split = hazard_config.test_split
    model_config.data.valid_mask_threshold = hazard_config.valid_mask_threshold
    model_config.evaluation.checkpoint_filename = entry.checkpoint_filename
    model_config.logger.enabled = False

    grid_params = get_config_grid_params(model_config)
    if grid_params is None:
        raise ValueError("Model config has no grid input source; cannot validate hazard targets.")
    actual_targets = {target.name for target in grid_params.resolved_targets()}
    expected_targets = {get_target_spec(name).name for name in required_targets}
    missing_targets = expected_targets - actual_targets
    if missing_targets:
        raise ValueError(
            f"Hazard eval requires a multi-output model predicting {sorted(expected_targets)}, "
            f"but the configured targets are {sorted(actual_targets)} (missing {sorted(missing_targets)})."
        )
    return model_config


def run_test_inference(model_config: Config, seed: int) -> tuple[np.ndarray, str]:
    """Run test inference; return patch predictions and the grid ``out_norm``."""
    test_loader = get_test_dataloader(
        config=model_config.data,
        modelling_approach=model_config.modelling_approach,
        seed=seed,
    )
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(test_loader.dataset)
    trainer = Trainer(
        model_config,
        spatial_input_channels=spatial_channels,
        auxiliary_input_dims=auxiliary_input_dims,
    )
    trainer.load_model(filename=model_config.evaluation.checkpoint_filename)
    _, test_predictions = trainer.test(test_loader, return_predictions=True)
    if not isinstance(test_predictions, np.ndarray):
        raise TypeError(f"Expected ndarray predictions, got {type(test_predictions)!r}")
    return test_predictions, get_model_out_norm(model_config)


def read_reference_denominator(path: str) -> float:
    """Read a positive, finite denominator from a JSON reference file.

    Accepts either a bare number or an object with a ``scale_denominator`` or
    ``denominator`` key.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference denominator file not found: {path}")
    with open(path) as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("scale_denominator", "denominator"):
            if key in payload:
                value = payload[key]
                break
        else:
            raise KeyError(f"Reference denominator file {path!r} is missing a 'scale_denominator' or 'denominator' key.")
    elif isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(f"Reference denominator file {path!r} must contain a number or an object with 'scale_denominator'/'denominator'.")
    else:
        value = payload

    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"reference denominator must be a positive finite number, got {value!r}")
    return numeric


def raw_ground_truth_denominator(
    raw_data_dir: str,
    fi_cap: float | None,
    allowed_hex_ids: Collection[int] | None = None,
) -> float:
    """Compute the max raw hazard over raw BP/FI rasters in ``raw_data_dir``.

    When ``allowed_hex_ids`` is provided, only those hexes contribute (e.g. a
    training split), so the denominator is not derived from held-out data. The
    per-hex maxima are accumulated incrementally to avoid holding every raw
    raster in memory at once.
    """
    hex_ids = find_hex_ids(raw_data_dir)
    if allowed_hex_ids is not None:
        allowed = {int(value) for value in allowed_hex_ids}
        hex_ids = [hex_id for hex_id in hex_ids if int(hex_id) in allowed]
    if not hex_ids:
        raise ValueError(f"No raw hex directories found under {raw_data_dir!r} for the requested hex ids.")

    running_max = -np.inf
    for hex_id in hex_ids:
        paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
        bp_grid = as_float_array_with_nan(load_raster(str(paths.output_burn_prob())))
        fi_grid = as_float_array_with_nan(load_raster(str(paths.output_fire_intensity())))
        raw = compute_raw_hazard(bp_grid, fi_grid, fi_cap)
        finite = raw[np.isfinite(raw)]
        if finite.size:
            running_max = max(running_max, float(finite.max()))
    return max_finite_hazard(np.asarray([running_max]))


def _pairs_raw_hazard_denominator(
    pairs: Iterable[tuple[StitchedHexel, StitchedHexel]],
    fi_cap: float | None,
    use_prediction: bool,
) -> float:
    """Max raw hazard over reconstructed GT (or prediction) BP/FI hexel pairs."""
    denominator = float("-inf")
    saw_pair = False
    for bp_hexel, fi_hexel in pairs:
        saw_pair = True
        raw_grid = compute_raw_hazard(
            bp_hexel.pred_grid if use_prediction else bp_hexel.gt_grid,
            fi_hexel.pred_grid if use_prediction else fi_hexel.gt_grid,
            fi_cap,
        )
        finite = raw_grid[np.isfinite(raw_grid)]
        if finite.size:
            denominator = max(denominator, float(finite.max()))
    if not saw_pair:
        raise ValueError("No BP/FI hexel pairs available to compute a denominator.")
    return max_finite_hazard(np.asarray([denominator]))


def resolve_hazard_denominator(
    hazard_config: HazardEvalConfig,
    pairs: Iterable[tuple[StitchedHexel, StitchedHexel]] | None = None,
    *,
    model_config: Config | None = None,
    save_dir: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve a single scale denominator and return it with metadata.

    An explicit ``scale_denominator`` always wins; otherwise the configured
    ``scale_denominator_source`` determines how the value is derived.
    """
    meta: dict[str, Any] = {"source": hazard_config.scale_denominator_source}

    if hazard_config.scale_denominator is not None:
        denominator = float(hazard_config.scale_denominator)
        meta["source"] = "explicit"
    elif hazard_config.scale_denominator_source == "reference_file":
        if hazard_config.reference_denominator_path is None:
            raise ValueError("reference_denominator_path is required for scale_denominator_source='reference_file'.")
        denominator = read_reference_denominator(hazard_config.reference_denominator_path)
        meta["reference_denominator_path"] = hazard_config.reference_denominator_path
    elif hazard_config.scale_denominator_source == "all_raw_ground_truth":
        denominator = raw_ground_truth_denominator(hazard_config.raw_data_dir, hazard_config.fi_cap)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            json_path = os.path.join(save_dir, DENOMINATOR_JSON_FILENAME)
            with open(json_path, "w") as handle:
                json.dump({"scale_denominator": denominator, "scale_denominator_source": meta["source"]}, handle, indent=2)
            meta["denominator_json"] = json_path
    elif hazard_config.scale_denominator_source == "train_ground_truth":
        if model_config is None:
            raise ValueError("model_config is required for scale_denominator_source='train_ground_truth'.")
        train_split_path = os.path.join(model_config.data.root_dir, model_config.data.train_split)
        denominator = raw_ground_truth_denominator(hazard_config.raw_data_dir, hazard_config.fi_cap, read_split_hex_ids(train_split_path))
        meta["train_split"] = train_split_path
    elif hazard_config.scale_denominator_source == "eval_ground_truth":
        if pairs is None:
            raise ValueError("BP/FI hexel pairs are required for scale_denominator_source='eval_ground_truth'.")
        denominator = _pairs_raw_hazard_denominator(pairs, hazard_config.fi_cap, use_prediction=False)
    elif hazard_config.scale_denominator_source == "prediction":
        if pairs is None:
            raise ValueError("BP/FI hexel pairs are required for scale_denominator_source='prediction'.")
        denominator = _pairs_raw_hazard_denominator(pairs, hazard_config.fi_cap, use_prediction=True)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported scale_denominator_source={hazard_config.scale_denominator_source!r}.")

    denominator = float(denominator)
    meta["value"] = denominator
    return denominator, meta


def _write_hazard_metric_summaries_from_records(
    metric_records: Sequence[dict[str, Any]],
    confusion_matrices: Sequence[np.ndarray],
    save_dir: str,
    denominator: float,
    denominator_metadata: dict[str, Any],
    prediction_denominator: float | None = None,
    prediction_denominator_metadata: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, float]]:
    if not metric_records:
        raise ValueError("No hazard hexel results to summarize.")
    os.makedirs(save_dir, exist_ok=True)

    per_hex_df = pd.DataFrame(metric_records)
    csv_path = os.path.join(save_dir, PER_HEX_CSV_FILENAME)
    per_hex_df.to_csv(csv_path, index=False)

    numeric_df = per_hex_df.drop(columns=["hex_id"], errors="ignore").select_dtypes(include=[np.number])
    aggregate: dict[str, float] = {}
    for column in numeric_df.columns:
        values = numeric_df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        aggregate[column] = float(finite.mean()) if finite.size else float("nan")

    confusion_matrix = np.sum(confusion_matrices, axis=0) if confusion_matrices else None
    confusion_csv_path = None
    confusion_png_path = None
    if confusion_matrix is not None:
        confusion_csv_path = write_confusion_matrix_csv(confusion_matrix, save_dir)
        confusion_png_path = write_confusion_matrix_plot(confusion_matrix, save_dir)

    json_metrics = {key: (value if np.isfinite(value) else None) for key, value in aggregate.items()}
    summary: dict[str, Any] = {
        "num_hexels": len(metric_records),
        "denominator": float(denominator),
        "denominator_metadata": denominator_metadata,
        "metrics": json_metrics,
    }
    if prediction_denominator is not None:
        summary["prediction_denominator"] = float(prediction_denominator)
        summary["prediction_denominator_metadata"] = prediction_denominator_metadata or {
            "source": "prediction",
            "value": float(prediction_denominator),
        }
    if confusion_matrix is not None:
        summary["confusion_matrix"] = confusion_matrix.tolist()
        summary["confusion_matrix_csv"] = confusion_csv_path
        summary["confusion_matrix_plot"] = confusion_png_path
    json_path = os.path.join(save_dir, SUMMARY_JSON_FILENAME)
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    return csv_path, json_path, aggregate


def write_confusion_matrix_csv(confusion_matrix: np.ndarray, save_dir: str) -> str:
    """Write an aggregate hazard confusion matrix with rows=GT class and columns=predicted class."""
    class_labels = [str(index) for index in range(1, confusion_matrix.shape[0] + 1)]
    confusion_df = pd.DataFrame(confusion_matrix, index=class_labels, columns=class_labels)
    confusion_df.index.name = "gt_class"
    csv_path = os.path.join(save_dir, CONFUSION_MATRIX_CSV_FILENAME)
    confusion_df.to_csv(csv_path)
    return csv_path


def row_normalized_confusion_percentages(confusion_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion_matrix, dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix * 100.0, row_totals, out=np.full_like(matrix, np.nan, dtype=float), where=row_totals > 0.0)


def write_confusion_matrix_plot(confusion_matrix: np.ndarray, save_dir: str) -> str:
    """Write an aggregate hazard confusion matrix plot with rows=GT class and columns=predicted class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percentages = row_normalized_confusion_percentages(confusion_matrix)
    row_support = confusion_matrix.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="lightgray")
    image = ax.imshow(np.ma.masked_invalid(percentages), cmap=cmap, vmin=0.0, vmax=100.0)
    labels = np.arange(1, confusion_matrix.shape[0] + 1)
    ax.set_xticks(np.arange(confusion_matrix.shape[1]), labels=labels)
    ax.set_yticks(
        np.arange(confusion_matrix.shape[0]),
        labels=[f"{label}\n(n={int(support):,})" for label, support in zip(labels, row_support, strict=True)],
    )
    ax.set_xlabel("Predicted hazard class")
    ax.set_ylabel("Ground-truth hazard class")
    ax.set_title("Hazard class confusion matrix\nRow-normalized by ground-truth class")
    fig.colorbar(image, ax=ax, label="Pixels within GT class (%)")

    if confusion_matrix.shape[0] <= 15 and confusion_matrix.shape[1] <= 15:
        for row in range(confusion_matrix.shape[0]):
            for col in range(confusion_matrix.shape[1]):
                value = percentages[row, col]
                if not np.isfinite(value) or (value == 0.0 and confusion_matrix[row, col] == 0):
                    continue
                color = "white" if value >= 50.0 else "black"
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", color=color, fontsize=6)

    ax.text(
        0.5,
        -0.12,
        f"Rows sum to 100% for classes with support; raw pixel counts are in {CONFUSION_MATRIX_CSV_FILENAME}.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    png_path = os.path.join(save_dir, CONFUSION_MATRIX_PNG_FILENAME)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    start_time = time.time()
    hazard_config = load_hazard_config(args.config)

    overrides = {
        key: value
        for key, value in (
            ("mask_scope", args.mask_scope),
            ("save_dir", args.save_dir),
            ("root_dir", args.root_dir),
            ("stitch_mode", args.stitch_mode),
            ("self_normalized_prediction", True if args.self_normalized_prediction else None),
        )
        if value is not None
    }
    if overrides:
        hazard_config = hazard_config.model_copy(update=overrides)

    os.makedirs(hazard_config.save_dir, exist_ok=True)

    model_config = prepare_model_config_for_hazard(load_config(hazard_config.model.config_path), hazard_config, hazard_config.model)

    seed = getattr(model_config, "seed", 42)
    deterministic = getattr(model_config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    print("\n[Hazard] Running multi-output BP/FI test inference...")
    predictions, out_norm = run_test_inference(model_config, seed)

    if hazard_config.model.save_predictions and not args.no_save_predictions:
        np.save(os.path.join(hazard_config.save_dir, "test_predictions.npy"), predictions)

    def make_pairs() -> Iterable[tuple[StitchedHexel, StitchedHexel]]:
        """Return a fresh streamed BP/FI reconstruction pass from the single multi-output model."""
        hexels = reconstruct_denormalized_hexels(
            test_predictions=predictions,
            config=model_config,
            out_norm=out_norm,
            stitch_mode=hazard_config.stitch_mode,
            mask_scope=hazard_config.mask_scope,
        )
        return pair_stitched_hexels(hexels)

    # Denominator passes consume the streamed reconstructions. Rebuilding them
    # is slower than caching full rasters, but keeps buffer-scale runs within memory.
    denominator_pairs = None
    if hazard_config.scale_denominator_source in {"eval_ground_truth", "prediction"}:
        denominator_pairs = make_pairs()
    denominator, denominator_metadata = resolve_hazard_denominator(
        hazard_config,
        denominator_pairs,
        model_config=model_config,
        save_dir=hazard_config.save_dir,
    )
    print(f"[Hazard] Resolved scale denominator={denominator} (source={denominator_metadata['source']})")
    prediction_denominator = None
    prediction_denominator_metadata = None
    if hazard_config.self_normalized_prediction:
        print("[Hazard] Resolving prediction denominator from reconstructed predictions...", flush=True)
        prediction_denominator = _pairs_raw_hazard_denominator(make_pairs(), hazard_config.fi_cap, use_prediction=True)
        prediction_denominator_metadata = {"source": "prediction", "value": prediction_denominator}
        print(f"[Hazard] Resolved prediction denominator={prediction_denominator} (source=prediction)")

    metric_records: list[dict[str, Any]] = []
    confusion_matrices: list[np.ndarray] = []
    for index, (bp_hexel, fi_hexel) in enumerate(make_pairs(), start=1):
        print(f"[Hazard] Computing hazard hex {bp_hexel.hex_id} ({index})...", flush=True)
        result = compute_hazard_hexel(
            bp_hexel,
            fi_hexel,
            denominator=denominator,
            pred_denominator=prediction_denominator,
            fi_cap=hazard_config.fi_cap,
            scale_to=hazard_config.scale_to,
            bin_thresholds=hazard_config.bin_thresholds,
        )
        if not args.metrics_only and hazard_config.save_hazard_map:
            save_hazard_hexel_artifacts(result, hazard_config.save_dir, save_plots=not args.skip_plots)
        metric_records.append({"hex_id": result.hex_id, **flatten_hazard_class_metrics(result.metrics)})
        if "confusion_matrix" in result.metrics:
            confusion_matrices.append(np.asarray(result.metrics["confusion_matrix"], dtype=np.int64))

    _, _, aggregate = _write_hazard_metric_summaries_from_records(
        metric_records,
        confusion_matrices,
        hazard_config.save_dir,
        denominator,
        denominator_metadata,
        prediction_denominator=prediction_denominator,
        prediction_denominator_metadata=prediction_denominator_metadata,
    )

    print("\n===== Hazard Evaluation Summary =====")
    print(f"Scale denominator: {denominator} (source={denominator_metadata['source']})")
    if prediction_denominator is not None:
        print(f"Prediction denominator: {prediction_denominator} (source=prediction)")
    print(f"Number of hexels:  {len(metric_records)}")
    print("Aggregate metrics:")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.6f}")
    print(f"===== Total time {round(time.time() - start_time, 3)}s =====")


if __name__ == "__main__":
    main()
