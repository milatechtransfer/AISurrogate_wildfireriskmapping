"""Fuel intervention maps for the local non-fuel barrier counterfactual."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.datasets.postprocessing.counterfactual_fuel import replace_nonfuel_components_with_adjacent_modal
from src.datasets.postprocessing.counterfactual_materialize import (
    _load_stitched_patch_fuel,
    _patch_records,
)
from src.datasets.postprocessing.counterfactual_viz import (
    DEFAULT_ZONE_OVERLAY_ALPHA,
    DEFAULT_ZONE_OVERLAY_COLOR,
    DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    add_zone_overlay_args,
    overlay_zone_boundaries,
    prediction_dirs_from_index,
    prediction_raster_path,
    prediction_reference_profile,
    read_prediction,
)
from src.datasets.postprocessing.counterfactual_weather_maps import load_zone_labels
from src.datasets.postprocessing.fuel_barrier_geometry import load_grouped_fuel_on_prediction_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO = "remove_barriers_adjacent_modal"
NONFUEL_GROUP = 0

FUEL_GROUP_LABELS: dict[int, str] = {
    0: "Non-fuel barrier",
    1: "C-1 spruce-lichen",
    2: "C-2 boreal spruce",
    3: "C-3 mature pine",
    4: "C-4 immature pine",
    5: "C-5 red/white pine",
    6: "C-6 plantation",
    7: "C-7 ponderosa/Douglas-fir",
    8: "D-1/D-2 aspen",
    9: "S-1/S-2/S-3 slash",
    10: "O-1 grass",
    11: "M-1/M-2 mixedwood",
    12: "M-1 mixedwood, % conifer",
    13: "M-2 mixedwood, % conifer",
    14: "M-1/M-2 mixedwood, % conifer",
    15: "M-3/M-4 dead balsam",
    16: "M-3 dead fir, % dead fir",
    17: "M-4 dead fir, % dead fir",
    18: "M-3/M-4 dead fir, % dead fir",
}

FUEL_GROUP_COLOURS: dict[int, str] = {
    0: "#111111",
    1: "#7fc97f",
    2: "#1b9e77",
    3: "#66a61e",
    4: "#a6d854",
    5: "#b2df8a",
    6: "#33a02c",
    7: "#e6ab02",
    8: "#7570b3",
    9: "#e7298a",
    10: "#d95f02",
    11: "#8c510a",
    12: "#1f78b4",
    13: "#6a3d9a",
    14: "#a6761d",
    15: "#b15928",
    16: "#80cdc1",
    17: "#dfc27d",
    18: "#bf812d",
}


@dataclass(frozen=True)
class FuelInterventionSummary:
    scenario: str
    endpoint: str
    hex_id: str
    n_valid_pixels: int
    n_original_nonfuel_pixels: int
    n_original_burnable_pixels: int
    n_replaced_pixels: int
    n_unexpected_burnable_changes: int
    replacement_fuel_ids: str
    replacement_fuel_labels: str


def paired_prediction_support(
    *,
    experiment_dir: Path,
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> np.ndarray:
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline = read_prediction(prediction_raster_path(prediction_dirs[("baseline", endpoint)], hex_id))
    scenario_values = read_prediction(prediction_raster_path(prediction_dirs[(scenario, endpoint)], hex_id))
    baseline_values = np.asarray(baseline.filled(np.nan), dtype=np.float64)
    scenario_values_arr = np.asarray(scenario_values.filled(np.nan), dtype=np.float64)
    return np.isfinite(baseline_values) & np.isfinite(scenario_values_arr)


def _load_fuel_channel(data_root: Path, endpoint: str) -> int:
    channel_paths = sorted(data_root.glob("feature_channel_map_*.json"))
    if not channel_paths:
        raise FileNotFoundError(f"No feature_channel_map_*.json found in {data_root}")
    with channel_paths[0].open() as handle:
        channel_map = json.load(handle)
    if "fuel_grid" not in channel_map:
        raise ValueError(f"{channel_paths[0]} is missing required key 'fuel_grid'.")
    fuel_channels = channel_map["fuel_grid"]
    if not fuel_channels:
        raise ValueError(f"{channel_paths[0]} has an empty fuel_grid channel list.")
    return int(fuel_channels[0])


def load_stitched_fuel(
    *,
    experiment_dir: Path,
    scenario: str,
    endpoint: str,
) -> np.ndarray:
    """Load a stitched prepared fuel-group raster from materialized patch files."""

    data_root = experiment_dir / "scenario_data" / scenario / endpoint
    metadata_path = data_root / "test_indices.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path)
    fuel_channel = _load_fuel_channel(data_root, endpoint)
    fuel, _ = _load_stitched_patch_fuel(_patch_records(data_root, metadata), fuel_channel)
    return fuel


def intervention_layers_on_prediction_grid(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return grouped fuel and intervention layers on paired prediction support."""

    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    reference_profile = prediction_reference_profile(prediction_dirs, hex_id)
    grouped_fuel = load_grouped_fuel_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        reference_profile=reference_profile,
        hex_id=hex_id,
    )
    support = paired_prediction_support(
        experiment_dir=experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    edited, original_nonfuel, _, _ = replace_nonfuel_components_with_adjacent_modal(
        grouped_fuel,
        [NONFUEL_GROUP],
        scenario_name=scenario,
    )
    supported_fuel = np.where(support, grouped_fuel, np.nan)
    supported_nonfuel = original_nonfuel & support
    replacement_map = np.full(grouped_fuel.shape, np.nan, dtype=np.float32)
    replacement_map[supported_nonfuel] = edited[supported_nonfuel]
    unexpected_burnable_changes = np.zeros(grouped_fuel.shape, dtype=bool)
    return supported_fuel, supported_nonfuel, replacement_map, unexpected_burnable_changes


def intervention_layers(
    baseline_fuel: np.ndarray,
    scenario_fuel: np.ndarray,
    *,
    nonfuel_group: int = NONFUEL_GROUP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original non-fuel mask, replacement-only map, and changed burnable mask."""

    if baseline_fuel.shape != scenario_fuel.shape:
        raise ValueError(f"Fuel arrays have different shapes: {baseline_fuel.shape} vs {scenario_fuel.shape}")
    baseline = np.asarray(baseline_fuel)
    scenario = np.asarray(scenario_fuel)
    finite = np.isfinite(baseline) & np.isfinite(scenario)
    baseline_int = np.full(baseline.shape, -9999, dtype=np.int32)
    scenario_int = np.full(scenario.shape, -9999, dtype=np.int32)
    baseline_int[finite] = baseline[finite].astype(np.int32)
    scenario_int[finite] = scenario[finite].astype(np.int32)
    original_nonfuel = finite & (baseline_int == int(nonfuel_group))
    replacement_mask = original_nonfuel & (scenario_int != int(nonfuel_group))
    replacement_map = np.full(baseline.shape, np.nan, dtype=np.float32)
    replacement_map[replacement_mask] = scenario[replacement_mask]
    unexpected_burnable_changes = finite & ~original_nonfuel & (baseline != scenario)
    return original_nonfuel, replacement_map, unexpected_burnable_changes


def _downsample(data: np.ndarray | np.ma.MaskedArray, factor: int) -> np.ndarray | np.ma.MaskedArray:
    if factor <= 1:
        return data
    return data[::factor, ::factor]


def _categorical_codes(values: np.ndarray, categories: list[int]) -> np.ma.MaskedArray:
    values_arr = np.asarray(values)
    codes = np.full(values_arr.shape, -1, dtype=np.int16)
    finite = np.isfinite(values_arr)
    values_int = np.full(values_arr.shape, -9999, dtype=np.int32)
    values_int[finite] = values_arr[finite].astype(np.int32)
    for idx, category in enumerate(categories):
        codes[finite & (values_int == category)] = idx
    return np.ma.masked_where(codes < 0, codes)


def _legend_handles(categories: list[int], *, include_nonfuel: bool) -> list[Patch]:
    handles = []
    for category in categories:
        if category == NONFUEL_GROUP and not include_nonfuel:
            continue
        label = FUEL_GROUP_LABELS.get(category, f"Group {category}")
        if category != NONFUEL_GROUP:
            label = f"{category}: {label}"
        handles.append(Patch(facecolor=FUEL_GROUP_COLOURS.get(category, "#999999"), edgecolor="none", label=label))
    return handles


def summarize_intervention(
    *,
    scenario: str,
    endpoint: str,
    hex_id: str,
    baseline_fuel: np.ndarray,
    replacement_map: np.ndarray,
    original_nonfuel: np.ndarray,
    unexpected_burnable_changes: np.ndarray,
) -> FuelInterventionSummary:
    baseline = np.asarray(baseline_fuel)
    valid = np.isfinite(baseline)
    original_burnable = valid & ~original_nonfuel
    replacement_values = replacement_map[np.isfinite(replacement_map)].astype(np.int32)
    replacement_ids = sorted(map(int, np.unique(replacement_values))) if replacement_values.size else []
    return FuelInterventionSummary(
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
        n_valid_pixels=int(valid.sum()),
        n_original_nonfuel_pixels=int(original_nonfuel.sum()),
        n_original_burnable_pixels=int(original_burnable.sum()),
        n_replaced_pixels=int(replacement_values.size),
        n_unexpected_burnable_changes=int(unexpected_burnable_changes.sum()),
        replacement_fuel_ids=";".join(map(str, replacement_ids)),
        replacement_fuel_labels="; ".join(FUEL_GROUP_LABELS.get(group, f"Group {group}") for group in replacement_ids),
    )


def plot_intervention_map(
    *,
    baseline_fuel: np.ndarray,
    replacement_map: np.ndarray,
    original_nonfuel: np.ndarray,
    scenario: str,
    hex_id: str,
    out_path: Path,
    downsample: int = 2,
    zone_labels: np.ma.MaskedArray | None = None,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Write the two-panel original fuel/replacement intervention map."""

    baseline = np.asarray(baseline_fuel)
    valid = np.isfinite(baseline)
    baseline_categories = sorted(map(int, np.unique(baseline[valid].astype(np.int32))))
    replacement_values = replacement_map[np.isfinite(replacement_map)].astype(np.int32)
    replacement_categories = sorted(map(int, np.unique(replacement_values))) if replacement_values.size else []

    baseline_codes = _categorical_codes(baseline, baseline_categories)
    burnable_codes = _categorical_codes(np.where(original_nonfuel, np.nan, baseline), baseline_categories)
    nonfuel_mask = np.ma.masked_where(~original_nonfuel, np.ones(baseline.shape, dtype=np.float32))
    replacement_codes = _categorical_codes(replacement_map, replacement_categories)
    support_context = np.ma.masked_where(~valid, np.ones(baseline.shape, dtype=np.float32))

    baseline_cmap = ListedColormap([FUEL_GROUP_COLOURS.get(category, "#999999") for category in baseline_categories])
    replacement_cmap = ListedColormap([FUEL_GROUP_COLOURS.get(category, "#999999") for category in replacement_categories])
    support_cmap = ListedColormap(["#eeeeee"])
    nonfuel_cmap = ListedColormap([FUEL_GROUP_COLOURS[NONFUEL_GROUP]])
    baseline_cmap.set_bad(color="white", alpha=0.0)
    replacement_cmap.set_bad(color="white", alpha=0.0)
    nonfuel_cmap.set_bad(color="white", alpha=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.5), constrained_layout=True, sharex=True, sharey=True)
    ax_original, ax_replacement = axes
    zone_display = _downsample(zone_labels, downsample) if zone_labels is not None else None

    ax_original.imshow(
        _downsample(support_context, downsample),
        cmap=support_cmap,
        interpolation="nearest",
        origin="upper",
    )
    ax_original.imshow(
        _downsample(burnable_codes, downsample),
        cmap=baseline_cmap,
        interpolation="nearest",
        origin="upper",
        alpha=0.75,
    )
    ax_original.imshow(
        _downsample(nonfuel_mask, downsample),
        cmap=nonfuel_cmap,
        interpolation="nearest",
        origin="upper",
    )
    ax_original.set_title("A. Original grouped fuel map", fontsize=15)
    overlay_zone_boundaries(
        ax_original,
        zone_display,
        color=zone_overlay_color,
        linewidth=zone_overlay_linewidth,
        alpha=zone_overlay_alpha,
    )
    ax_original.set_xticks([])
    ax_original.set_yticks([])
    ax_original.set_aspect("equal")

    ax_replacement.imshow(
        _downsample(support_context, downsample),
        cmap=support_cmap,
        interpolation="nearest",
        origin="upper",
    )
    if replacement_categories:
        ax_replacement.imshow(
            _downsample(replacement_codes, downsample),
            cmap=replacement_cmap,
            interpolation="nearest",
            origin="upper",
        )
    ax_replacement.set_title("B. Counterfactual replacement map", fontsize=15)
    overlay_zone_boundaries(
        ax_replacement,
        zone_display,
        color=zone_overlay_color,
        linewidth=zone_overlay_linewidth,
        alpha=zone_overlay_alpha,
    )
    ax_replacement.set_xticks([])
    ax_replacement.set_yticks([])
    ax_replacement.set_aspect("equal")

    combined_categories = sorted(set(baseline_categories) | set(replacement_categories))
    combined_handles = _legend_handles(combined_categories, include_nonfuel=NONFUEL_GROUP in combined_categories)
    fig.legend(
        handles=combined_handles,
        title="Fuel groups",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        ncol=6,
    )

    fig.suptitle(f"Hex{int(hex_id):02d} local adjacent-modal fuel intervention", y=1.03, fontsize=18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_fuel_intervention_map(
    *,
    experiment_dir: Path,
    raw_data_dir: Path | None = None,
    scenario: str = SCENARIO,
    endpoint: str = "bp",
    hex_id: str = "16",
    downsample: int = 2,
    support_policy: str = "prediction",
    zone_overlay: bool = False,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    if support_policy not in {"prediction", "prepared_patch"}:
        raise ValueError("support_policy must be one of {'prediction', 'prepared_patch'}.")
    if support_policy == "prediction":
        if raw_data_dir is None:
            raise ValueError("raw_data_dir is required for support_policy='prediction'.")
        baseline_fuel, original_nonfuel, replacement_map, unexpected_burnable_changes = intervention_layers_on_prediction_grid(
            experiment_dir=experiment_dir,
            raw_data_dir=raw_data_dir,
            scenario=scenario,
            endpoint=endpoint,
            hex_id=hex_id,
        )
    else:
        baseline_fuel = load_stitched_fuel(experiment_dir=experiment_dir, scenario="baseline", endpoint=endpoint)
        scenario_fuel = load_stitched_fuel(experiment_dir=experiment_dir, scenario=scenario, endpoint=endpoint)
        original_nonfuel, replacement_map, unexpected_burnable_changes = intervention_layers(
            baseline_fuel,
            scenario_fuel,
            nonfuel_group=NONFUEL_GROUP,
        )
    summary = summarize_intervention(
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
        baseline_fuel=baseline_fuel,
        replacement_map=replacement_map,
        original_nonfuel=original_nonfuel,
        unexpected_burnable_changes=unexpected_burnable_changes,
    )

    out_dir = out_dir if out_dir is not None else experiment_dir / "figures" / "fuel_intervention"
    plot_path = out_dir / f"hex{int(hex_id):02d}_{scenario}_fuel_intervention_map.png"
    summary_path = experiment_dir / f"counterfactual_{scenario}_fuel_intervention_summary.csv"
    zone_labels = None
    if zone_overlay and raw_data_dir is not None:
        prediction_dirs = prediction_dirs_from_index(experiment_dir)
        zone_labels = load_zone_labels(
            raw_data_dir,
            hex_id,
            prediction_reference_profile(prediction_dirs, hex_id),
            support=np.isfinite(np.asarray(baseline_fuel)),
        )
    plot_intervention_map(
        baseline_fuel=baseline_fuel,
        replacement_map=replacement_map,
        original_nonfuel=original_nonfuel,
        scenario=scenario,
        hex_id=hex_id,
        out_path=plot_path,
        downsample=max(1, int(downsample)),
        zone_labels=zone_labels,
        zone_overlay_color=zone_overlay_color,
        zone_overlay_linewidth=zone_overlay_linewidth,
        zone_overlay_alpha=zone_overlay_alpha,
    )
    pd.DataFrame([asdict(summary)]).to_csv(summary_path, index=False)
    return plot_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the fuel intervention for the local-modal counterfactual.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--endpoint", default="bp", choices=("bp", "ros", "fi"))
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--support_policy", choices=("prediction", "prepared_patch"), default="prediction")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/fuel_intervention.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_path, summary_path = write_fuel_intervention_map(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        scenario=args.scenario,
        endpoint=args.endpoint,
        hex_id=str(args.hex_id).zfill(2),
        downsample=args.downsample,
        support_policy=args.support_policy,
        zone_overlay=args.zone_overlay,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
        out_dir=args.out_dir,
    )
    print(f"Wrote fuel intervention map: {plot_path}")
    print(f"Wrote fuel intervention summary: {summary_path}")


if __name__ == "__main__":
    main()
