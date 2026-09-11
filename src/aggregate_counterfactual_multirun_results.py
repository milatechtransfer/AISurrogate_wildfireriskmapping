"""Aggregate seeded counterfactual responses and evaluation metrics.

Each run is expected under ``<config.save_dir>/seed_<seed>`` as produced by
``src.evaluate_counterfactual --run_id``. The aggregator writes per-seed and
mean/std CSVs, pixelwise ensemble mean/std response rasters, and a 2x3
BP/FI/ROS mean/std response figure for every requested hexel.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize

from src.config import SEEDS
from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    CounterfactualConfig,
    load_counterfactual_config,
    resolve_counterfactual_paths,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps import (
    ENDPOINT_SPECS,
    load_endpoint_response,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    delta_norm,
    downsample_for_display,
    finite_values,
    prediction_dirs_from_index,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENDPOINTS = ("bp", "fi", "ros")
RASTER_NODATA = -9999.0
EXTENT_ABSOLUTE_TOLERANCE_M = 1e-6


@dataclass(frozen=True)
class SeedRun:
    run_id: int
    seed: int
    experiment_dir: Path


def _validate_run_ids(run_ids: list[int]) -> None:
    if not run_ids:
        raise ValueError("At least one run_id is required.")
    invalid = [run_id for run_id in run_ids if not 0 <= run_id < len(SEEDS)]
    if invalid:
        raise ValueError(f"run_ids must be between 0 and {len(SEEDS) - 1}; received {invalid}.")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"run_ids must be unique; received {run_ids}.")


def resolve_seed_runs(base_experiment_dir: Path, run_ids: list[int]) -> list[SeedRun]:
    _validate_run_ids(run_ids)
    runs = [SeedRun(run_id=run_id, seed=SEEDS[run_id], experiment_dir=base_experiment_dir / f"seed_{SEEDS[run_id]}") for run_id in run_ids]
    missing = [
        run.experiment_dir / "scenario_prediction_index.csv"
        for run in runs
        if not (run.experiment_dir / "scenario_prediction_index.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing seeded counterfactual output(s): {missing}")
    return runs


def ensemble_mean_std(arrays: list[np.ma.MaskedArray]) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    if not arrays:
        raise ValueError("At least one response array is required.")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError(f"Response arrays must share one shape; received {[array.shape for array in arrays]}.")

    stack = np.stack([np.asarray(np.ma.asarray(array).filled(np.nan), dtype=np.float64) for array in arrays])
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    totals = np.where(valid, stack, 0.0).sum(axis=0)
    means = np.divide(totals, counts, out=np.full(shape, np.nan, dtype=np.float64), where=counts > 0)

    centered = np.zeros_like(stack)
    np.subtract(stack, means, out=centered, where=valid)
    squared = np.square(centered).sum(axis=0)
    stds = np.sqrt(
        np.divide(
            squared,
            counts - 1,
            out=np.full(shape, np.nan, dtype=np.float64),
            where=counts > 1,
        )
    )
    return np.ma.masked_invalid(means), np.ma.masked_invalid(stds)


def extents_match(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return bool(np.allclose(first, second, rtol=0.0, atol=EXTENT_ABSOLUTE_TOLERANCE_M))


def summarize_response_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "scenario", "endpoint", "mean_delta", "mean_absolute_delta"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Response rows are missing required columns: {missing}.")
    return (
        rows.groupby(["scenario", "endpoint"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_delta=("mean_delta", "mean"),
            std_delta_across_seeds=("mean_delta", "std"),
            mean_absolute_delta=("mean_absolute_delta", "mean"),
            std_absolute_delta_across_seeds=("mean_absolute_delta", "std"),
        )
        .sort_values(["scenario", "endpoint"])
        .reset_index(drop=True)
    )


def aggregate_metric_frames(runs: list[SeedRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for run in runs:
        path = run.experiment_dir / "counterfactual_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["run_id"] = run.run_id
        frame["seed"] = run.seed
        frames.append(frame)
    by_seed = pd.concat(frames, ignore_index=True)
    summary = (
        by_seed.groupby(["scenario", "endpoint", "metric"], as_index=False)
        .agg(seed_count=("seed", "nunique"), mean=("value", "mean"), std=("value", "std"))
        .sort_values(["scenario", "endpoint", "metric"])
        .reset_index(drop=True)
    )
    return by_seed, summary


def _scenario_support_kwargs(config: CounterfactualConfig, scenario_name: str) -> dict[str, list[int] | None]:
    scenario = config.scenario(scenario_name)
    fuel_edit = scenario.fuel_edit()
    if fuel_edit is not None:
        return {
            "nonfuel_ids": [int(value) for value in fuel_edit["nonfuel_ids"]],
            "static_nonfuel_ids": None,
        }
    return {
        "nonfuel_ids": None,
        "static_nonfuel_ids": config.nonfuel_ids,
    }


def _load_seed_delta(
    *,
    run: SeedRun,
    config: CounterfactualConfig,
    raw_data_dir: Path,
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float], dict]:
    prediction_dirs = prediction_dirs_from_index(run.experiment_dir)
    _, _, _, delta, extent, profile = load_endpoint_response(
        run.experiment_dir,
        prediction_dirs,
        hex_id,
        raw_data_dir,
        scenario=scenario,
        endpoint=endpoint,
        **_scenario_support_kwargs(config, scenario),
    )
    return delta, extent, profile


def _write_raster(path: Path, array: np.ma.MaskedArray, profile: dict) -> None:
    write_profile = profile.copy()
    write_profile.update(count=1, dtype="float32", nodata=RASTER_NODATA, compress="lzw")
    values = np.asarray(np.ma.asarray(array).filled(RASTER_NODATA), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **write_profile) as dst:
        dst.write(values, 1)


def _extent_km(extent_m: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent_m
    return (0.0, (right - left) / 1000.0, 0.0, (top - bottom) / 1000.0)


def _plot_mean_std(
    *,
    means: dict[str, np.ma.MaskedArray],
    stds: dict[str, np.ma.MaskedArray],
    extent: tuple[float, float, float, float],
    scenario: str,
    hex_id: str,
    output_dir: Path,
    downsample: int,
) -> None:
    figure, axes = plt.subplots(2, len(ENDPOINTS), figsize=(15.0, 10.0), constrained_layout=True)
    display_extent = _extent_km(extent)
    for column, endpoint in enumerate(ENDPOINTS):
        spec = ENDPOINT_SPECS[endpoint]
        mean_image = axes[0, column].imshow(
            downsample_for_display(means[endpoint], downsample),
            cmap="RdBu_r",
            norm=delta_norm([means[endpoint]], percentile=99.0),
            extent=display_extent,
            origin="upper",
            interpolation="nearest",
        )
        axes[0, column].set_title(f"Mean {spec.label} response")
        mean_colorbar = figure.colorbar(mean_image, ax=axes[0, column], orientation="horizontal", fraction=0.055, pad=0.04)
        mean_colorbar.set_label(f"Mean delta {spec.units}")

        std_values = finite_values(stds[endpoint])
        std_vmax = float(np.percentile(std_values, 99.0)) if std_values.size else 1.0
        std_image = axes[1, column].imshow(
            downsample_for_display(stds[endpoint], downsample),
            cmap="magma",
            norm=Normalize(vmin=0.0, vmax=max(std_vmax, 1e-9)),
            extent=display_extent,
            origin="upper",
            interpolation="nearest",
        )
        axes[1, column].set_title(f"Across-seed SD of {spec.label} response")
        std_colorbar = figure.colorbar(std_image, ax=axes[1, column], orientation="horizontal", fraction=0.055, pad=0.04)
        std_colorbar.set_label(f"SD delta {spec.units}")

    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(f"{scenario} ensemble response (hex {hex_id})")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    basename = f"hex{hex_id}_{scenario}_response_mean_std"
    figure.savefig(figure_dir / f"{basename}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(figure_dir / f"{basename}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def aggregate_counterfactual_runs(
    config_path: Path,
    *,
    scenario: str,
    run_ids: list[int],
    hex_ids: list[str] | None = None,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    downsample: int = 3,
) -> Path:
    project_root = (project_root or Path.cwd()).resolve()
    config = load_counterfactual_config(config_path)
    config.scenario(scenario)
    missing_endpoints = sorted(set(ENDPOINTS) - set(config.endpoints))
    if missing_endpoints:
        raise ValueError(f"Counterfactual config is missing required endpoint(s): {missing_endpoints}.")
    base_experiment_dir, raw_data_dir = resolve_counterfactual_paths(config, project_root=project_root)
    runs = resolve_seed_runs(base_experiment_dir, run_ids)
    selected_hex_ids = [normalize_hex_id(hex_id) for hex_id in (hex_ids or config.hex_ids)]
    result_dir = output_dir or base_experiment_dir / "multiseed"
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_seed, metrics_summary = aggregate_metric_frames(runs)
    metrics_by_seed.to_csv(result_dir / "counterfactual_metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(result_dir / "counterfactual_metrics_summary.csv", index=False)

    response_rows = []
    plot_means: dict[tuple[str, str], np.ma.MaskedArray] = {}
    plot_stds: dict[tuple[str, str], np.ma.MaskedArray] = {}
    plot_extents: dict[str, tuple[float, float, float, float]] = {}
    for endpoint in ENDPOINTS:
        pooled_by_seed: dict[int, list[np.ndarray]] = {run.seed: [] for run in runs}
        for hex_id in selected_hex_ids:
            seed_deltas = []
            reference_extent = None
            reference_profile = None
            for run in runs:
                delta, extent, profile = _load_seed_delta(
                    run=run,
                    config=config,
                    raw_data_dir=raw_data_dir,
                    scenario=scenario,
                    endpoint=endpoint,
                    hex_id=hex_id,
                )
                if reference_extent is not None and not extents_match(extent, reference_extent):
                    raise ValueError(f"Seed {run.seed} {endpoint.upper()} extent differs for hex_id={hex_id}.")
                reference_extent = extent
                reference_profile = profile
                seed_deltas.append(delta)
                pooled_by_seed[run.seed].append(finite_values(delta))

            if reference_extent is None or reference_profile is None:
                raise RuntimeError(f"No seeded responses loaded for endpoint={endpoint!r}, hex_id={hex_id!r}.")
            mean_delta, std_delta = ensemble_mean_std(seed_deltas)
            raster_dir = result_dir / "rasters" / f"hex{hex_id}"
            _write_raster(raster_dir / f"{scenario}_{endpoint}_delta_mean.tif", mean_delta, reference_profile)
            _write_raster(raster_dir / f"{scenario}_{endpoint}_delta_std.tif", std_delta, reference_profile)

            if hex_id in plot_extents and not extents_match(plot_extents[hex_id], reference_extent):
                raise ValueError(f"{endpoint.upper()} prediction extent differs from the other endpoints for hex_id={hex_id}.")
            plot_means[(hex_id, endpoint)] = mean_delta
            plot_stds[(hex_id, endpoint)] = std_delta
            plot_extents[hex_id] = reference_extent

        for run in runs:
            pooled = np.concatenate(pooled_by_seed[run.seed]) if pooled_by_seed[run.seed] else np.array([], dtype=np.float64)
            response_rows.append(
                {
                    "run_id": run.run_id,
                    "seed": run.seed,
                    "scenario": scenario,
                    "endpoint": endpoint,
                    "hex_ids": ",".join(selected_hex_ids),
                    "pixel_count": int(pooled.size),
                    "mean_delta": float(np.mean(pooled)) if pooled.size else float("nan"),
                    "mean_absolute_delta": float(np.mean(np.abs(pooled))) if pooled.size else float("nan"),
                }
            )

    response_by_seed = pd.DataFrame(response_rows).sort_values(["endpoint", "seed"]).reset_index(drop=True)
    response_summary = summarize_response_rows(response_by_seed)
    response_by_seed.to_csv(result_dir / "counterfactual_response_by_seed.csv", index=False)
    response_summary.to_csv(result_dir / "counterfactual_response_summary.csv", index=False)

    for hex_id in selected_hex_ids:
        _plot_mean_std(
            means={endpoint: plot_means[(hex_id, endpoint)] for endpoint in ENDPOINTS},
            stds={endpoint: plot_stds[(hex_id, endpoint)] for endpoint in ENDPOINTS},
            extent=plot_extents[hex_id],
            scenario=scenario,
            hex_id=hex_id,
            output_dir=result_dir,
            downsample=downsample,
        )
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run_ids", type=int, nargs="+", default=list(range(len(SEEDS))))
    parser.add_argument("--hex_ids", nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--downsample", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = aggregate_counterfactual_runs(
        args.config,
        scenario=args.scenario,
        run_ids=args.run_ids,
        hex_ids=args.hex_ids,
        output_dir=args.output_dir,
        downsample=args.downsample,
    )
    print(f"Wrote multiseed counterfactual aggregation to: {output_dir}")


if __name__ == "__main__":
    main()
