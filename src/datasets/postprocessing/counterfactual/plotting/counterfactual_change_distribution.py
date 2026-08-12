"""Summarize how counterfactual prediction changes are distributed in space."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    load_counterfactual_config,
    resolve_counterfactual_paths,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map import (
    burnable_fuel_support,
    load_evaluated_fuel_pair,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    abs_share_at,
    build_endpoint_response,
    cumulative_abs_share,
    load_baseline_scenario_pair,
    pixel_fraction_for_share,
    prediction_dirs_from_index,
    prediction_raster_path,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO = "remove_barriers_adjacent_modal"
DISTANCE_BIN_EDGES_M = (100.0, 250.0, 500.0, 1000.0, 2000.0, np.inf)
CONCENTRATION_FRACTIONS = (0.001, 0.01, 0.05, 0.10, 0.50)


@dataclass(frozen=True)
class ChangeSummary:
    """Per-hexel statistics on how much a scenario's prediction differs from baseline."""

    scenario: str
    endpoint: str
    hex_id: str
    n_response_pixels: int
    delta_mean: float
    delta_median: float
    delta_abs_mean: float
    delta_sum: float
    delta_abs_sum: float
    frac_delta_positive: float
    frac_delta_negative: float
    edited_pixel_abs_change_share: float
    off_edit_abs_change_share: float
    top_0_1pct_abs_change_share: float
    top_1pct_abs_change_share: float
    top_5pct_abs_change_share: float
    top_10pct_abs_change_share: float
    top_50pct_abs_change_share: float
    pixel_share_for_50pct_abs_change: float
    pixel_share_for_80pct_abs_change: float
    pixel_share_for_90pct_abs_change: float


def prediction_response_change(
    experiment_dir: Path,
    *,
    scenario: str,
    endpoint: str,
    hex_id: str,
    nonfuel_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the support-aware prediction response used by all counterfactual plots.

    Returns:
        delta: scenario minus baseline with non-burnable support contributing zero.
        support: pixels burnable in the baseline, scenario, or both.
        profile: the baseline raster's rasterio profile (for grid/transform reuse).
    """
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline_dir = prediction_dirs.get(("baseline", endpoint))
    scenario_dir = prediction_dirs.get((scenario, endpoint))
    if baseline_dir is None or scenario_dir is None:
        raise KeyError(f"Missing baseline/scenario predictions for endpoint={endpoint!r}, scenario={scenario!r}.")

    baseline_path = prediction_raster_path(baseline_dir, hex_id, target_name=endpoint)
    baseline, scenario_values = load_baseline_scenario_pair(
        prediction_dirs,
        hex_id,
        endpoint=endpoint,
        scenario=scenario,
    )
    baseline_fuel, scenario_fuel = load_evaluated_fuel_pair(
        experiment_dir=experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    response = build_endpoint_response(
        baseline,
        scenario_values,
        baseline_support=burnable_fuel_support(baseline_fuel, nonfuel_ids),
        scenario_support=burnable_fuel_support(scenario_fuel, nonfuel_ids),
    )
    delta = np.asarray(response.delta.filled(np.nan), dtype=np.float32)
    with rasterio.open(baseline_path) as src:
        profile = src.profile.copy()
    return delta, response.response_support, profile


def evaluated_fuel_edit_mask(
    *,
    experiment_dir: Path,
    scenario: str,
    endpoint: str,
    hex_id: str,
    support: np.ndarray,
) -> np.ndarray:
    """Return pixels whose persisted scenario fuel differs from persisted baseline fuel."""

    baseline_fuel, scenario_fuel = load_evaluated_fuel_pair(
        experiment_dir=experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    baseline_values = np.asarray(baseline_fuel.filled(np.nan), dtype=np.float64)
    scenario_values = np.asarray(scenario_fuel.filled(np.nan), dtype=np.float64)
    finite = np.isfinite(baseline_values) & np.isfinite(scenario_values)
    return support & finite & (baseline_values != scenario_values)


def summarize_change(
    delta: np.ndarray,
    support: np.ndarray,
    edit_mask: np.ndarray,
    *,
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> ChangeSummary:
    """Compute magnitude/sign/concentration statistics for a hexel's prediction change."""
    values = np.asarray(delta[support], dtype=np.float64)
    if values.size == 0:
        raise ValueError("No supported counterfactual response pixels found.")
    abs_values = np.abs(values)
    pixel_fraction, cumulative_share = cumulative_abs_share(values)
    total_abs = float(abs_values.sum())
    edited_abs = float(np.abs(delta[edit_mask]).sum())
    edited_share = 0.0 if np.isclose(total_abs, 0.0) else edited_abs / total_abs

    top_shares = {fraction: abs_share_at(pixel_fraction, cumulative_share, fraction) for fraction in CONCENTRATION_FRACTIONS}
    return ChangeSummary(
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
        n_response_pixels=int(values.size),
        delta_mean=float(values.mean()),
        delta_median=float(np.median(values)),
        delta_abs_mean=float(abs_values.mean()),
        delta_sum=float(values.sum()),
        delta_abs_sum=total_abs,
        frac_delta_positive=float(np.mean(values > 0.0)),
        frac_delta_negative=float(np.mean(values < 0.0)),
        edited_pixel_abs_change_share=float(edited_share),
        off_edit_abs_change_share=float(1.0 - edited_share),
        top_0_1pct_abs_change_share=top_shares[0.001],
        top_1pct_abs_change_share=top_shares[0.01],
        top_5pct_abs_change_share=top_shares[0.05],
        top_10pct_abs_change_share=top_shares[0.10],
        top_50pct_abs_change_share=top_shares[0.50],
        pixel_share_for_50pct_abs_change=pixel_fraction_for_share(pixel_fraction, cumulative_share, 0.50),
        pixel_share_for_80pct_abs_change=pixel_fraction_for_share(pixel_fraction, cumulative_share, 0.80),
        pixel_share_for_90pct_abs_change=pixel_fraction_for_share(pixel_fraction, cumulative_share, 0.90),
    )


def distance_bin_summary(
    delta: np.ndarray,
    support: np.ndarray,
    edit_mask: np.ndarray,
    *,
    pixel_height_m: float,
    pixel_width_m: float,
) -> pd.DataFrame:
    """Bin off-edit pixels by Euclidean distance to the nearest edited fuel pixel."""
    if not edit_mask.any():
        raise ValueError("No edited fuel pixels found.")
    distance_m = distance_transform_edt(
        ~edit_mask,
        sampling=(pixel_height_m, pixel_width_m),
    )
    rows = []
    lower_bounds = DISTANCE_BIN_EDGES_M[:-1]
    upper_bounds = DISTANCE_BIN_EDGES_M[1:]
    for lower, upper in zip(lower_bounds, upper_bounds, strict=True):
        in_bin = support & ~edit_mask & (distance_m >= lower)
        if np.isfinite(upper):
            in_bin &= distance_m < upper
            label = f"{lower:g}-{upper:g} m"
        else:
            label = f">{lower:g} m"
        values = np.asarray(delta[in_bin], dtype=np.float64)
        if values.size == 0:
            continue
        rows.append(
            {
                "distance_bin": label,
                "distance_min_m": lower,
                "distance_max_m": upper,
                "n_pixels": int(values.size),
                "delta_mean": float(values.mean()),
                "delta_median": float(np.median(values)),
                "delta_abs_mean": float(np.abs(values).mean()),
                "delta_p25": float(np.percentile(values, 25)),
                "delta_p75": float(np.percentile(values, 75)),
                "frac_delta_positive": float(np.mean(values > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_distance_summary(
    summary: pd.DataFrame,
    *,
    endpoint: str,
    scenario: str,
    hex_id: str,
    out_path: Path,
) -> None:
    """Bar-plot mean/median Δ per edit-distance bin."""
    x = np.arange(len(summary))
    means = summary["delta_mean"].to_numpy(dtype=float)
    medians = summary["delta_median"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    bars = ax.bar(x, means, color="#4c78a8")
    ax.scatter(x, medians, color="black", marker="D", s=28, label="Median", zorder=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, summary["distance_bin"])
    ax.set_ylabel(f"Mean Δ{endpoint.upper()} (scenario − baseline)")
    ax.set_xlabel("Distance from nearest edited fuel pixel")
    ax.set_title(f"Hex {hex_id} mean {endpoint.upper()} change by fuel-edit distance\n{scenario.replace('_', ' ')}")
    ax.bar_label(bars, labels=[f"n={count:,}" for count in summary["n_pixels"]], padding=4, fontsize=8)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_change_distribution(
    *,
    experiment_dir: Path,
    config_path: Path,
    scenario: str,
    endpoint: str,
    hex_id: str,
    out_dir: Path | None = None,
) -> list[Path]:
    """Compute response concentration and decay from the evaluated fuel edit.

    Writes the mean-change-by-edit-distance plot to `out_dir` (default:
    `<experiment_dir>/figures/fuel_change_distribution`), and the change-concentration and
    edit-distance summary CSVs to `experiment_dir`.

    Returns:
        The three paths written: [distance_plot, change_summary_csv, distance_summary_csv].
    """
    config = load_counterfactual_config(config_path)
    try:
        scenario_config = config.scenario(scenario)
    except KeyError as error:
        raise KeyError(f"Scenario {scenario!r} not found in {config_path}.") from error
    fuel_edit = scenario_config.fuel_edit()
    if fuel_edit is None:
        raise ValueError(f"Scenario {scenario!r} is not a fuel intervention.")
    nonfuel_ids = fuel_edit.get("nonfuel_ids")
    if not isinstance(nonfuel_ids, list | tuple) or not nonfuel_ids:
        raise ValueError(f"Fuel scenario {scenario!r} must define nonfuel_ids.")

    delta, support, profile = prediction_response_change(
        experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
        nonfuel_ids=[int(value) for value in nonfuel_ids],
    )
    edit_mask = evaluated_fuel_edit_mask(
        experiment_dir=experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
        support=support,
    )
    summary = summarize_change(
        delta,
        support,
        edit_mask,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    distance_summary = distance_bin_summary(
        delta,
        support,
        edit_mask,
        pixel_height_m=abs(float(profile["transform"].e)),
        pixel_width_m=abs(float(profile["transform"].a)),
    )

    out_dir = out_dir or experiment_dir / "figures" / "fuel_change_distribution"
    prefix = f"hex{int(hex_id):02d}_{scenario}_{endpoint}"
    distance_path = out_dir / f"{prefix}_mean_change_by_edit_distance.png"
    summary_path = experiment_dir / f"counterfactual_{scenario}_{endpoint}_change_concentration.csv"
    distance_summary_path = experiment_dir / f"counterfactual_{scenario}_{endpoint}_edit_distance_summary.csv"

    plot_distance_summary(
        distance_summary,
        endpoint=endpoint,
        scenario=scenario,
        hex_id=hex_id,
        out_path=distance_path,
    )
    pd.DataFrame([asdict(summary)]).to_csv(summary_path, index=False)
    distance_summary.to_csv(distance_summary_path, index=False)
    return [distance_path, summary_path, distance_summary_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_fuel.yaml"))
    parser.add_argument("--experiment_dir", type=Path, default=None, help="Overrides save_dir from --config.")
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_counterfactual_config(args.config)
    experiment_dir, _ = resolve_counterfactual_paths(config, experiment_dir=args.experiment_dir)
    paths = write_change_distribution(
        experiment_dir=experiment_dir,
        config_path=args.config,
        scenario=args.scenario,
        endpoint=args.endpoint,
        hex_id=str(args.hex_id).zfill(2),
        out_dir=args.out_dir,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
