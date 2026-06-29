"""FI response figures for the daily FWI-regime counterfactual.

Generates scientist-facing figures for the two daily weather-regime swaps:
before/after/change FI maps, the per-pixel delta distribution, the per-zone
FWI intervention, the zone-level dose-response, the delta-vs-baseline
relationship, the hex-wide FI distribution shift, multivariate driver
verification, and high-response patch zoom-ins.
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

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.counterfactual_fwi import THERMO_SWAP_COLUMNS
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
from src.datasets.postprocessing.counterfactual_weather import (
    normalize_raw_weather_ids,
    processed_weather_for_hex,
    raw_weather_path,
    recover_wind_encoding_stats,
)
from src.datasets.postprocessing.fuel_barrier_geometry import parse_fuel_barrier_info

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 17,
    }
)

GT_RELATIVE_PATH = "results/burnP3Plus_OutputFireIntensitySummaryMap/fbpSummary-FireIntensity-Average.tif"
FIREZONES_RELATIVE_PATH = "spatial/hex{hex_int:02d}_firezones.tif"
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


def load_burnable_support(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ndarray:
    """Boolean mask of burnable pixels (non-fuel/water excluded) on the prediction grid."""
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel_ma, _ = load_spatial_raster(path=paths.fuel_grid(hex_id), reference_profile=reference_profile)
    fuel_values = np.ma.asarray(fuel_ma).filled(-32768).astype(np.int32)
    fuel_info = parse_fuel_barrier_info(paths, hex_id)
    return ~np.isin(fuel_values, fuel_info.nonfuel_ids)


def load_response(prediction_dirs: dict[tuple[str, str], Path], hex_id: str, raw_data_dir: Path):
    baseline_dir = prediction_dirs.get(("baseline", "fi"))
    if baseline_dir is None:
        raise KeyError("Missing baseline FI prediction directory.")
    baseline_path = prediction_raster_path(baseline_dir, hex_id)
    baseline = read_prediction(baseline_path)
    extent = read_prediction_extent(baseline_path)
    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()

    burnable = load_burnable_support(raw_data_dir, hex_id, reference_profile)
    support = burnable & ~np.ma.getmaskarray(baseline)

    ground_truth = load_ground_truth(raw_data_dir, hex_id, reference_profile)
    if ground_truth.shape != baseline.shape:
        raise ValueError(f"Ground-truth shape {ground_truth.shape} does not match prediction grid {baseline.shape}.")
    ground_truth = restrict_to_support(ground_truth, support)
    baseline = restrict_to_support(baseline, support)

    scenarios: dict[str, dict[str, np.ma.MaskedArray]] = {}
    for scenario in DAILY_SCENARIOS:
        scenario_dir = prediction_dirs.get((scenario, "fi"))
        if scenario_dir is None:
            raise KeyError(f"Missing FI prediction directory for scenario={scenario!r}.")
        scenario_fi = read_prediction(prediction_raster_path(scenario_dir, hex_id))
        if scenario_fi.shape != baseline.shape:
            raise ValueError(f"Scenario {scenario!r} FI shape {scenario_fi.shape} does not match baseline grid {baseline.shape}.")
        scenario_fi = restrict_to_support(scenario_fi, support)
        delta = restrict_to_support(scenario_fi - baseline, support)
        scenarios[scenario] = {"fi": scenario_fi, "delta": delta}
    return ground_truth, baseline, scenarios, extent, reference_profile


def load_zone_labels(raw_data_dir: Path, hex_id: str, reference_profile: dict) -> np.ma.MaskedArray:
    firezones_path = raw_data_dir / f"hex{int(hex_id):02d}" / FIREZONES_RELATIVE_PATH.format(hex_int=int(hex_id))
    zones, _ = load_spatial_raster(path=firezones_path, reference_profile=reference_profile)
    return zones


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
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label, fontsize=14)
        axes[row, 0].set_ylabel(SCENARIO_TITLES[scenario], labelpad=12)

    fig.suptitle("FI response to daily FWI-regime counterfactuals (hex 16)", y=0.99)
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
    ax.legend(loc="upper left")
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
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Weather zone", fontsize=16)
        ax.set_title(SCENARIO_TITLES[scenario], fontsize=15)
        ax.legend(fontsize=14)
    axes[0].set_ylabel("Zone-mean normalized FWI", fontsize=16)
    fig.suptitle("Input intervention: per-zone FWI shift driving the FI response (hex 16)", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _paint_zone_choropleth(zone_labels: np.ma.MaskedArray, zone_fwi: dict[int, float]) -> np.ma.MaskedArray:
    """Return a float array with each pixel painted by its zone's FWI value."""
    out = np.full(zone_labels.shape, np.nan, dtype=np.float64)
    label_data = np.ma.filled(zone_labels, -1).astype(np.int64)
    for zone, value in zone_fwi.items():
        out[label_data == zone] = value
    mask = np.ma.getmaskarray(zone_labels) | ~np.isfinite(out)
    return np.ma.array(out, mask=mask)


def _fwi_zone_means_raw(
    raw_data_dir: Path,
    experiment_dir: Path,
    hex_id: str,
    summary: pd.DataFrame,
    zone_column: str = "WeatherZone",
) -> tuple[dict[int, float], dict[str, dict[int, float]], float, float]:
    """Return zone-mean raw FWI for baseline and each scenario, plus shared vmin/vmax."""
    raw_weather = normalize_raw_weather_ids(pd.read_csv(raw_weather_path(raw_data_dir, hex_id)))
    baseline_processed_path = experiment_dir / "scenario_data" / "baseline" / "fi" / "weather_table_processed.csv"
    processed_hex = processed_weather_for_hex(raw_data_dir, baseline_processed_path, hex_id)
    fwi_stats = recover_wind_encoding_stats(
        raw_weather[["FireWeatherIndex"]].rename(columns={"FireWeatherIndex": "FireWeatherIndex"}),
        processed_hex[["FireWeatherIndex"]].rename(columns={"FireWeatherIndex": "FireWeatherIndex"}),
        columns=("FireWeatherIndex",),
    )
    stat = fwi_stats.by_name()["FireWeatherIndex"]

    raw_weather["_zone"] = raw_weather[zone_column].astype(int)
    baseline_means: dict[int, float] = raw_weather.groupby("_zone")["FireWeatherIndex"].mean().to_dict()

    scenario_means: dict[str, dict[int, float]] = {}
    for scenario in DAILY_SCENARIOS:
        rows = summary[summary["scenario"] == scenario]
        scenario_means[scenario] = {int(r["zone"]): float(r["scenario_fwi_mean"]) * stat.std + stat.mean for _, r in rows.iterrows()}

    all_vals = list(baseline_means.values()) + [v for d in scenario_means.values() for v in d.values()]
    return baseline_means, scenario_means, float(min(all_vals)), float(max(all_vals))


def plot_zone_fwi_map(
    zone_labels: np.ma.MaskedArray,
    edit_summary_path: Path,
    extent_m: tuple[float, float, float, float],
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    hex_id: str,
    support_mask: np.ndarray,
    out_path: Path,
    downsample: int = 3,
) -> None:
    """Choropleth map: each firezone coloured by its zone-mean raw FWI, baseline vs intervention."""
    summary = pd.read_csv(edit_summary_path)
    summary = summary[summary["scenario"].isin(DAILY_SCENARIOS)]
    extent = _extent_km(extent_m)

    baseline_means, scenario_means, vmin, vmax = _fwi_zone_means_raw(raw_data_dir, experiment_dir, hex_id, summary)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0))
    for row, scenario in enumerate(DAILY_SCENARIOS):
        panels = [
            (
                restrict_to_support(_paint_zone_choropleth(zone_labels, baseline_means), support_mask),
                "Baseline zone-mean FWI",
            ),
            (
                restrict_to_support(_paint_zone_choropleth(zone_labels, scenario_means[scenario]), support_mask),
                f"Scenario zone-mean FWI \u2014 {SCENARIO_SHORT[scenario]}",
            ),
        ]
        for col, (data, title) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(
                downsample_for_display(data, downsample),
                cmap="YlOrRd",
                norm=norm,
                extent=extent,
                origin="upper",
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=14)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Zone-mean FWI", fontsize=13)
            cbar.ax.tick_params(labelsize=12)
        axes[row, 0].set_ylabel(SCENARIO_TITLES[scenario], fontsize=15, labelpad=12)

    fig.suptitle("Firezone FWI regime: baseline vs intervention (hex 16)", fontsize=17, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _zone_mean_delta_fi(delta: np.ma.MaskedArray, zone_labels: np.ma.MaskedArray, zone: int) -> tuple[float, int]:
    mask = (np.ma.filled(zone_labels, -1).astype(np.int64) == zone) & ~np.ma.getmaskarray(delta)
    values = np.ma.getdata(delta)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), 0
    return float(np.mean(values)), int(values.size)


def plot_zone_dose_response(
    scenarios: dict[str, dict[str, np.ma.MaskedArray]],
    zone_labels: np.ma.MaskedArray,
    edit_summary_path: Path,
    *,
    out_path: Path,
) -> None:
    summary = pd.read_csv(edit_summary_path)
    summary = summary[summary["scenario"].isin(DAILY_SCENARIOS)]

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for scenario in DAILY_SCENARIOS:
        rows = summary[summary["scenario"] == scenario].sort_values("zone")
        xs: list[float] = []
        ys: list[float] = []
        sizes: list[float] = []
        zones: list[int] = []
        for row in rows.itertuples(index=False):
            delta_fwi = float(row.scenario_fwi_mean) - float(row.baseline_fwi_mean)
            delta_fi, n_pixels = _zone_mean_delta_fi(scenarios[scenario]["delta"], zone_labels, int(row.zone))
            if n_pixels == 0 or not np.isfinite(delta_fi):
                continue
            xs.append(delta_fwi)
            ys.append(delta_fi)
            sizes.append(n_pixels)
            zones.append(int(row.zone))
        sizes_arr = np.asarray(sizes, dtype=np.float64)
        marker_sizes = 60.0 + 240.0 * (sizes_arr / sizes_arr.max()) if sizes_arr.size else sizes_arr
        ax.scatter(
            xs,
            ys,
            s=marker_sizes,
            color=DELTA_COLORS[scenario],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=SCENARIO_SHORT[scenario],
            zorder=3,
        )
        for x, y, zone in zip(xs, ys, zones, strict=True):
            ax.annotate(f"z{zone}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=12, color="0.25")
        pooled_x.extend(xs)
        pooled_y.extend(ys)

    if len(pooled_x) >= 2:
        coeffs = np.polyfit(pooled_x, pooled_y, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        corr = float(np.corrcoef(pooled_x, pooled_y)[0, 1])
        x_line = np.linspace(min(pooled_x), max(pooled_x), 100)
        ax.plot(x_line, slope * x_line + intercept, color="0.4", linestyle="--", linewidth=1.2, zorder=2)
        ax.text(0.04, 0.94, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=14)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.axvline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("Induced \u0394 zone-mean normalized FWI (scenario \u2212 baseline)")
    ax.set_ylabel("Resulting \u0394 zone-mean FI (kW/m)")
    ax.set_title("Zone-level dose-response: FWI shift vs FI response (hex 16)\nmarker size \u221d zone pixel count")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_baseline(
    baseline: np.ma.MaskedArray,
    scenarios: dict[str, dict[str, np.ma.MaskedArray]],
    *,
    out_path: Path,
) -> None:
    baseline_values, baseline_valid = values_and_valid(baseline)
    delta_limit = symmetric_percentile_limit([scenarios[s]["delta"] for s in DAILY_SCENARIOS], percentile=99.5)
    base_hi = float(np.percentile(baseline_values[baseline_valid], 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True, sharey=True)
    for ax, scenario in zip(axes, DAILY_SCENARIOS, strict=True):
        delta_values, delta_valid = values_and_valid(scenarios[scenario]["delta"])
        valid = baseline_valid & delta_valid
        x = baseline_values[valid]
        y = delta_values[valid]
        hexbin = ax.hexbin(x, y, gridsize=70, bins="log", cmap="viridis", extent=(0, base_hi, -delta_limit, delta_limit))
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        edges = np.linspace(0, base_hi, 31)
        idx = np.clip(np.searchsorted(edges, x_sorted, side="right") - 1, 0, edges.size - 2)
        centers = 0.5 * (edges[:-1] + edges[1:])
        median_line = np.array([np.median(y_sorted[idx == b]) if np.any(idx == b) else np.nan for b in range(edges.size - 1)])
        ax.plot(centers, median_line, color="#d62728", linewidth=2.0, label="Binned median \u0394FI")
        ax.axhline(0.0, color="white", linewidth=1.0)
        ax.set_xlabel("Baseline FI (model, kW/m)")
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.legend(loc="upper left")
        fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.02, label="Pixel count (log)")
    axes[0].set_ylabel("\u0394FI (scenario \u2212 baseline, kW/m)")
    fig.suptitle("FI response scales with baseline intensity (hex 16)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ecdf(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)
    return np.searchsorted(ordered, query, side="right") / ordered.size


def plot_fi_distribution_shift(
    ground_truth: np.ma.MaskedArray,
    baseline: np.ma.MaskedArray,
    scenarios: dict[str, dict[str, np.ma.MaskedArray]],
    *,
    out_path: Path,
) -> None:
    series = {
        "Ground truth (BurnP3+)": (finite_values(ground_truth), "#2ca02c"),
        "Baseline (model)": (finite_values(baseline), "0.35"),
        SCENARIO_SHORT["fwi_daily_low_to_high"]: (
            finite_values(scenarios["fwi_daily_low_to_high"]["fi"]),
            DELTA_COLORS["fwi_daily_low_to_high"],
        ),
        SCENARIO_SHORT["fwi_daily_high_to_low"]: (
            finite_values(scenarios["fwi_daily_high_to_low"]["fi"]),
            DELTA_COLORS["fwi_daily_high_to_low"],
        ),
    }
    pooled = np.concatenate([values for values, _ in series.values()])
    low = max(float(np.percentile(pooled, 0.5)), 1.0)
    high = float(np.percentile(pooled, 99.5))
    query = np.logspace(np.log10(low), np.log10(high), 400)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for label, (values, color) in series.items():
        median = float(np.median(values))
        ax.plot(query, _ecdf(values, query), color=color, linewidth=1.9, label=f"{label} (median {median:.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("FI (kW/m, log scale)")
    ax.set_ylabel("Cumulative fraction of pixels")
    ax.set_title("Hex-wide FI distribution shift under daily FWI swaps (hex 16)")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _zone_driver_means(table_path: Path, zones: list[int]) -> pd.DataFrame:
    columns = ["WeatherZone", *THERMO_SWAP_COLUMNS]
    table = pd.read_csv(table_path, usecols=columns)
    table = table[table["WeatherZone"].isin(zones)]
    return table.groupby("WeatherZone")[list(THERMO_SWAP_COLUMNS)].mean()


def plot_driver_verification(
    experiment_dir: Path,
    zones: list[int],
    *,
    out_path: Path,
) -> None:
    baseline_means = _zone_driver_means(experiment_dir / "scenario_data" / "baseline" / "fi" / "weather_table_processed.csv", zones)
    limit = 0.0
    deltas: dict[str, pd.DataFrame] = {}
    for scenario in DAILY_SCENARIOS:
        scenario_means = _zone_driver_means(experiment_dir / "scenario_data" / scenario / "fi" / "weather_table_processed.csv", zones)
        delta = (scenario_means - baseline_means).reindex(index=baseline_means.index)
        deltas[scenario] = delta
        limit = max(limit, float(np.nanmax(np.abs(delta.to_numpy()))))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, scenario in zip(axes, DAILY_SCENARIOS, strict=True):
        delta = deltas[scenario]
        matrix = delta[list(THERMO_SWAP_COLUMNS)].to_numpy().T
        image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), aspect="auto")
        ax.set_xticks(np.arange(delta.index.size))
        ax.set_xticklabels([f"z{int(zone)}" for zone in delta.index])
        ax.set_yticks(np.arange(len(THERMO_SWAP_COLUMNS)))
        ax.set_yticklabels(list(THERMO_SWAP_COLUMNS) if ax is axes[0] else [])
        ax.set_xlabel("Weather zone")
        ax.set_title(SCENARIO_TITLES[scenario])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=10, color="0.15")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="\u0394 mean (standardized)")
    fig.suptitle("Intervention transplants a coherent multivariate regime, not FWI alone (hex 16)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _block_response(delta: np.ma.MaskedArray, block: int) -> np.ndarray:
    abs_delta = np.abs(np.ma.filled(delta, 0.0))
    valid = (~np.ma.getmaskarray(delta)).astype(np.float64)
    rows = (abs_delta.shape[0] // block) * block
    cols = (abs_delta.shape[1] // block) * block
    summed = abs_delta[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    counts = valid[:rows, :cols].reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    return np.where(counts > 0, summed / np.maximum(counts, 1.0), 0.0)


def _shared_hotspot_centers(deltas: list[np.ma.MaskedArray], block: int, *, count: int, window: int) -> list[tuple[int, int]]:
    scores = _block_response(np.ma.abs(np.ma.stack(deltas)).sum(axis=0), block)
    suppress = max(window // block, 1)
    centers: list[tuple[int, int]] = []
    for _ in range(count):
        if not np.any(scores > 0):
            break
        flat_index = int(np.argmax(scores))
        row_block, col_block = (int(idx) for idx in np.unravel_index(flat_index, scores.shape))
        centers.append((row_block * block + block // 2, col_block * block + block // 2))
        scores[
            max(row_block - suppress, 0) : row_block + suppress + 1,
            max(col_block - suppress, 0) : col_block + suppress + 1,
        ] = 0.0
    return centers


def plot_patch_zoom(
    baseline: np.ma.MaskedArray,
    scenarios: dict[str, dict[str, np.ma.MaskedArray]],
    *,
    out_path: Path,
    window: int,
    center: tuple[int, int],
    rank: int,
) -> None:
    pooled_fi = [finite_values(baseline)] + [finite_values(scenarios[s]["fi"]) for s in DAILY_SCENARIOS]
    fi_norm = Normalize(vmin=0.0, vmax=float(np.percentile(np.concatenate(pooled_fi), 99.0)))
    delta_limit = symmetric_percentile_limit([scenarios[s]["delta"] for s in DAILY_SCENARIOS], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    half = window // 2

    center_row, center_col = center
    r0 = max(center_row - half, 0)
    c0 = max(center_col - half, 0)
    r1 = min(r0 + window, baseline.shape[0])
    c1 = min(c0 + window, baseline.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.0))
    for row, scenario in enumerate(DAILY_SCENARIOS):
        panels = [
            (baseline[r0:r1, c0:c1], "Baseline FI (model)", "inferno", fi_norm, "FI (kW/m)"),
            (scenarios[scenario]["fi"][r0:r1, c0:c1], f"Scenario FI \u2014 {SCENARIO_SHORT[scenario]}", "inferno", fi_norm, "FI (kW/m)"),
            (
                scenarios[scenario]["delta"][r0:r1, c0:c1],
                f"\u0394FI \u2014 {SCENARIO_SHORT[scenario]}",
                "RdBu_r",
                delta_norm,
                "\u0394FI (kW/m)",
            ),
        ]
        for col, (data, title, cmap, norm, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(data, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label, fontsize=14)
        axes[row, 0].set_ylabel(SCENARIO_TITLES[scenario], labelpad=12)
    fig.suptitle(f"High-response window #{rank} ({window}\u00d7{window} px, hex 16)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Defaults to baseline config data.raw_data_dir.")
    parser.add_argument("--patch_window", type=int, default=400, help="Side length (pixels) of patch zoom windows.")
    parser.add_argument("--hotspot_block", type=int, default=64, help="Block size for locating high-response windows.")
    parser.add_argument("--patch_count", type=int, default=3, help="Number of distinct high-response windows to render.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/fwi_daily.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / "fwi_daily"
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else _raw_data_dir_from_config(args.experiment_dir)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)
    ground_truth, baseline, scenarios, extent, reference_profile = load_response(prediction_dirs, args.hex_id, raw_data_dir)
    zone_labels = load_zone_labels(raw_data_dir, args.hex_id, reference_profile)

    plot_before_after_change(
        ground_truth,
        baseline,
        scenarios,
        extent,
        out_path=out_dir / "fwi_daily_fi_response_maps.png",
        downsample=args.downsample,
    )
    plot_delta_distribution(scenarios, out_path=out_dir / "fwi_daily_delta_distribution.png")
    plot_delta_vs_baseline(baseline, scenarios, out_path=out_dir / "fwi_daily_delta_vs_baseline.png")
    plot_fi_distribution_shift(ground_truth, baseline, scenarios, out_path=out_dir / "fwi_daily_fi_distribution_shift.png")
    patch_centers = _shared_hotspot_centers(
        [scenarios[s]["delta"] for s in DAILY_SCENARIOS],
        args.hotspot_block,
        count=args.patch_count,
        window=args.patch_window,
    )
    for rank, center in enumerate(patch_centers, start=1):
        plot_patch_zoom(
            baseline,
            scenarios,
            out_path=out_dir / f"fwi_daily_patch_zoom_{rank}.png",
            window=args.patch_window,
            center=center,
            rank=rank,
        )

    edit_summary_path = args.experiment_dir / "fwi_edit_summary.csv"
    if edit_summary_path.exists():
        plot_zone_fwi_shift(edit_summary_path, out_path=out_dir / "fwi_daily_zone_fwi_shift.png")
        plot_zone_fwi_map(
            zone_labels,
            edit_summary_path,
            extent,
            experiment_dir=args.experiment_dir,
            raw_data_dir=raw_data_dir,
            hex_id=args.hex_id,
            support_mask=~np.ma.getmaskarray(baseline),
            out_path=out_dir / "fwi_daily_zone_fwi_map.png",
            downsample=args.downsample,
        )
        zones = sorted(int(z) for z in pd.read_csv(edit_summary_path)["zone"].unique())
        plot_zone_dose_response(scenarios, zone_labels, edit_summary_path, out_path=out_dir / "fwi_daily_zone_dose_response.png")
        plot_driver_verification(args.experiment_dir, zones, out_path=out_dir / "fwi_daily_driver_verification.png")

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
