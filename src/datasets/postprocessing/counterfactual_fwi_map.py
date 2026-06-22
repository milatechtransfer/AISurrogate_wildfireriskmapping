"""FI response figures for the daily FWI-regime counterfactual.

Generates scientist-facing figures for the two daily weather-regime swaps:
before/after/change FI maps, the per-pixel delta distribution, and the
per-zone FWI intervention that produced the response.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.colors import Normalize, TwoSlopeNorm

from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_hazard_map import (
    downsample_for_display,
    finite_values,
    prediction_dirs_from_index,
    prediction_raster_path,
    read_prediction,
    read_prediction_extent,
    restrict_to_support,
    symmetric_percentile_limit,
    values_and_valid,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GT_RELATIVE_PATH = "results/burnP3Plus_OutputFireIntensitySummaryMap/fbpSummary-FireIntensity-Average.tif"
DAILY_SCENARIOS = ("fwi_daily_low_to_high", "fwi_daily_high_to_low")
SCENARIO_TITLES = {
    "fwi_daily_low_to_high": "Low\u2192High FWI (intensified regime)",
    "fwi_daily_high_to_low": "High\u2192Low FWI (calmed regime)",
}
SCENARIO_SHORT = {
    "fwi_daily_low_to_high": "Low\u2192High",
    "fwi_daily_high_to_low": "High\u2192Low",
}
DELTA_COLORS = {
    "fwi_daily_low_to_high": "#b2182b",
    "fwi_daily_high_to_low": "#2166ac",
}


def _extent_km(extent_m: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent_m
    origin_x, origin_y = left, bottom
    return (
        (left - origin_x) / 1000.0,
        (right - origin_x) / 1000.0,
        (bottom - origin_y) / 1000.0,
        (top - origin_y) / 1000.0,
    )


def _raw_data_dir_from_config(experiment_dir: Path) -> Path:
    config_path = experiment_dir / "generated_configs" / "baseline_fi.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return Path(config["data"]["raw_data_dir"])


def load_ground_truth(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ma.MaskedArray:
    gt_path = raw_data_dir / f"hex{int(hex_id):02d}" / GT_RELATIVE_PATH
    gt, _ = load_spatial_raster(path=gt_path, reference_profile=reference_profile)
    return gt


def load_response(prediction_dirs: dict[tuple[str, str], Path], hex_id: str, raw_data_dir: Path):
    baseline_dir = prediction_dirs.get(("baseline", "fi"))
    if baseline_dir is None:
        raise KeyError("Missing baseline FI prediction directory.")
    baseline_path = prediction_raster_path(baseline_dir, hex_id)
    baseline = read_prediction(baseline_path)
    extent = read_prediction_extent(baseline_path)
    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()

    ground_truth = load_ground_truth(raw_data_dir, hex_id, reference_profile)
    if ground_truth.shape != baseline.shape:
        raise ValueError(f"Ground-truth shape {ground_truth.shape} does not match prediction grid {baseline.shape}.")
    ground_truth = restrict_to_support(ground_truth, ~np.ma.getmaskarray(baseline))

    scenarios: dict[str, dict[str, np.ma.MaskedArray]] = {}
    for scenario in DAILY_SCENARIOS:
        scenario_dir = prediction_dirs.get((scenario, "fi"))
        if scenario_dir is None:
            raise KeyError(f"Missing FI prediction directory for scenario={scenario!r}.")
        scenario_fi = read_prediction(prediction_raster_path(scenario_dir, hex_id))
        delta = scenario_fi - baseline
        scenarios[scenario] = {"fi": scenario_fi, "delta": delta}
    return ground_truth, baseline, scenarios, extent


def plot_before_after_change(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenarios: dict[str, dict[str, np.ma.MaskedArray]],
    extent_m: tuple[float, float, float, float],
    *,
    out_path: Path,
    downsample: int,
) -> None:
    pooled_fi = [finite_values(ground_truth), finite_values(baseline)] + [finite_values(scenarios[s]["fi"]) for s in DAILY_SCENARIOS]
    fi_vmax = float(np.percentile(np.concatenate(pooled_fi), 99.0))
    fi_norm = Normalize(vmin=0.0, vmax=fi_vmax)
    delta_limit = symmetric_percentile_limit([scenarios[s]["delta"] for s in DAILY_SCENARIOS], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    extent = _extent_km(extent_m)

    fig, axes = plt.subplots(2, 4, figsize=(21.0, 11.0))
    gt_display = downsample_for_display(ground_truth, downsample)
    baseline_display = downsample_for_display(baseline, downsample)
    for row, scenario in enumerate(DAILY_SCENARIOS):
        panels = [
            (gt_display, "Ground truth FI (BurnP3+)", "inferno", fi_norm, "Fire intensity (kW/m)"),
            (baseline_display, "Baseline FI (model)", "inferno", fi_norm, "Fire intensity (kW/m)"),
            (
                downsample_for_display(scenarios[scenario]["fi"], downsample),
                f"Scenario FI (model) \u2014 {SCENARIO_SHORT[scenario]}",
                "inferno",
                fi_norm,
                "Fire intensity (kW/m)",
            ),
            (
                downsample_for_display(scenarios[scenario]["delta"], downsample),
                f"\u0394FI (model, scenario \u2212 baseline) \u2014 {SCENARIO_SHORT[scenario]}",
                "RdBu_r",
                delta_norm,
                "\u0394 Fire intensity (kW/m)",
            ),
        ]
        for col, (data, title, cmap, norm, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation="nearest")
            ax.set_title(title, fontsize=10.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label, fontsize=8.5)
        axes[row, 0].set_ylabel(SCENARIO_TITLES[scenario], fontsize=12, labelpad=12)

    fig.suptitle("FI response to daily FWI-regime counterfactuals (hex 16)", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_distribution(
    scenarios: dict[str, dict[str, np.ma.MaskedArray]],
    *,
    out_path: Path,
) -> None:
    deltas = {scenario: finite_values(scenarios[scenario]["delta"]) for scenario in DAILY_SCENARIOS}
    pooled = np.concatenate(list(deltas.values()))
    limit = float(np.max(np.abs(pooled)))
    bins = np.linspace(-limit, limit, 201).tolist()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for scenario in DAILY_SCENARIOS:
        values = deltas[scenario]
        mean = float(np.mean(values))
        frac_expected = float(np.mean(values > 0)) if scenario == "fwi_daily_low_to_high" else float(np.mean(values < 0))
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=DELTA_COLORS[scenario],
            label=f"{SCENARIO_SHORT[scenario]}: mean \u0394={mean:+.0f}, {frac_expected:.0%} pixels expected sign",
        )
        ax.axvline(mean, color=DELTA_COLORS[scenario], linestyle="--", linewidth=1.2)
    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel("\u0394FI per pixel (scenario \u2212 baseline, kW/m)")
    ax.set_ylabel("Pixel count")
    ax.set_yscale("log")
    ax.set_title("Per-pixel FI change under daily FWI-regime swaps (hex 16)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_zone_fwi_shift(
    edit_summary_path: Path,
    *,
    out_path: Path,
) -> None:
    summary = pd.read_csv(edit_summary_path)
    summary = summary[summary["scenario"].isin(DAILY_SCENARIOS)]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, scenario in zip(axes, DAILY_SCENARIOS, strict=True):
        rows = summary[summary["scenario"] == scenario].sort_values("zone")
        zones = rows["zone"].astype(int).to_numpy()
        positions = np.arange(zones.size)
        width = 0.4
        ax.bar(positions - width / 2, rows["baseline_fwi_mean"], width, label="Baseline", color="0.6")
        ax.bar(positions + width / 2, rows["scenario_fwi_mean"], width, label="Scenario", color=DELTA_COLORS[scenario])
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(zone) for zone in zones])
        ax.set_xlabel("Weather zone")
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Zone-mean normalized FWI")
    fig.suptitle("Input intervention: per-zone FWI shift driving the FI response (hex 16)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Defaults to baseline config data.raw_data_dir.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/fwi_daily.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / "fwi_daily"
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else _raw_data_dir_from_config(args.experiment_dir)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)
    ground_truth, baseline, scenarios, extent = load_response(prediction_dirs, args.hex_id, raw_data_dir)

    plot_before_after_change(
        ground_truth,
        baseline,
        scenarios,
        extent,
        out_path=out_dir / "fwi_daily_fi_response_maps.png",
        downsample=args.downsample,
    )
    plot_delta_distribution(scenarios, out_path=out_dir / "fwi_daily_delta_distribution.png")

    edit_summary_path = args.experiment_dir / "fwi_edit_summary.csv"
    if edit_summary_path.exists():
        plot_zone_fwi_shift(edit_summary_path, out_path=out_dir / "fwi_daily_zone_fwi_shift.png")

    for scenario in DAILY_SCENARIOS:
        delta_values = finite_values(scenarios[scenario]["delta"])
        _, valid = values_and_valid(scenarios[scenario]["delta"])
        print(
            f"{scenario}: n={int(valid.sum())} mean_dFI={float(np.mean(delta_values)):+.1f} "
            f"median={float(np.median(delta_values)):+.1f}"
        )
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
