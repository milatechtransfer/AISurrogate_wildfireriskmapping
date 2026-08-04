"""
Used to generate visuals for AAAI paper.
Aggregate BP/FI/ROS response for one counterfactual fuel-edit scenario.

Single-purpose script mirroring `weather_scenario_mean_delta.py`, but for a
fuel-edit run (e.g. `final_results/counterfactual_fuel_multi_output_hex16`,
scenario `remove_barriers_fixed_c2`). It reads the already-materialized
baseline/scenario prediction rasters and the persisted baseline/scenario fuel
rasters (`predictions/<scenario>/<subdir>/fuel_intervention/hexel_XX_{baseline,scenario}_fuel.tif`,
written by `evaluate_counterfactual.py` for every fuel-edit scenario), and:

  1. Prints a one-row aggregate table like:

         Intervention                  delta_BP   delta_FI   delta_ROS
         remove_barriers_fixed_c2       -0.0013      -2312      -2.18

     where each delta is the mean of the support-aware (scenario - baseline)
     response over all finite pixels (pooled across `--hex_ids`), reusing the
     same `burnable_fuel_support`/`build_endpoint_response` logic as
     `counterfactual_response_maps.py::load_endpoint_response`'s `nonfuel_ids`
     path, so pixels outside burnable land in *either* baseline or scenario
     fuel don't dilute the mean. The response is additionally restricted to
     the actual-fire-occurrence footprint for the hex
     (`<experiment_dir>/GT/hex<hex_id>_actual.shp`), the same "actual" mask
     scope `load_target_grid_for_mask_scope` applies in
     `src/datasets/postprocessing/utils.py`.

  2. Optionally (`--out_path`) renders a 1x4 figure for one hex_id: a changed-
     pixel fuel-intervention mask panel, followed by BP/FI/ROS response-delta
     panels (RdBu_r, zero-centered), matching the style of
     `counterfactual_response_maps.py::plot_response_maps`'s delta panel.

Used to generate: Figure 3 and Table 4.

Usage:
    python -m src.paper_utils.fuel_scenario_aggregate_response \
        --experiment_dir final_results/counterfactual_fuel_multi_output_hex16 \
        --config configs/counterfactual_fuel_multi_output.yaml \
        --scenario c2_to_mixedwood_fixed \
        --hex_ids 16 \
        --out_path final_results/counterfactual_fuel_multi_output_hex16/c2_to_mixedwood_fixed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

from src.datasets.postprocessing.counterfactual.counterfactual_base import load_counterfactual_config
from src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform import fuel_intervention_raster_path
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map import burnable_fuel_support
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    build_endpoint_response,
    delta_norm,
    downsample_for_display,
    find_local_prediction_dir,
    finite_values,
    prediction_raster_path,
    read_prediction,
)

ENDPOINTS = ("bp", "fi", "ros")
DEFAULT_GT_SUBDIR = "GT"
# Non-fuel IDs used by every fuel-edit scenario in configs/counterfactual_fuel*.yaml.
DEFAULT_NONFUEL_IDS = [100, 101, 102, 105, 106, 110]
PANEL_TITLES: dict[str, str] = {
    "bp": "Burn probability response",
    "fi": "Fire intensity response",
    "ros": "Rate of spread response",
}
PANEL_CBAR_LABELS: dict[str, str] = {
    "bp": "\u0394 burn probability",
    "fi": "\u0394 fire intensity (kW m$^{-1}$)",
    "ros": "\u0394 rate of spread (m min$^{-1}$)",
}


def actual_fire_mask_path(experiment_dir: Path, hex_id: str, gt_dir: Path | None = None) -> Path:
    """Path to the actual-fire-occurrence mask shapefile for `hex_id`, e.g. GT/hex16_actual.shp.

    Matches `Paths.mask_grid_actual()`'s naming convention; the raster form
    (`hex16_actual.tif`) alongside it is a rasterized copy of the same footprint.
    """

    return (gt_dir if gt_dir is not None else experiment_dir / DEFAULT_GT_SUBDIR) / f"hex{hex_id}_actual.shp"


def load_actual_fire_valid_mask(
    experiment_dir: Path,
    hex_id: str,
    reference_profile: dict,
    *,
    gt_dir: Path | None = None,
) -> np.ndarray:
    """Boolean mask, True within the actual-fire-occurrence footprint for `hex_id`, on the prediction grid.

    Same pattern as `_actual_area_mask` in `src/datasets/postprocessing/utils.py`:
    rasterize the mask polygon onto the prediction grid via `geometry_mask`.
    """

    mask_path = actual_fire_mask_path(experiment_dir, hex_id, gt_dir)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing actual-fire mask for hex_id={hex_id!r}: {mask_path}")
    mask_gdf = gpd.read_file(mask_path)
    if mask_gdf.empty:
        raise ValueError(f"Actual-fire mask contains no geometries: {mask_path}")
    return geometry_mask(
        mask_gdf.to_crs(reference_profile["crs"]).geometry,
        out_shape=(reference_profile["height"], reference_profile["width"]),
        transform=reference_profile["transform"],
        invert=True,
    )


def load_fuel_pair(experiment_dir: Path, scenario: str, hex_id: str, endpoint: str) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Read the persisted baseline/scenario fuel rasters for one hex_id/endpoint's fuel-edit run."""

    scenario_dir = find_local_prediction_dir(experiment_dir, scenario, hex_id, endpoint)
    baseline_fuel = read_prediction(fuel_intervention_raster_path(scenario_dir, hex_id, "baseline"))
    scenario_fuel = read_prediction(fuel_intervention_raster_path(scenario_dir, hex_id, "scenario"))
    if baseline_fuel.shape != scenario_fuel.shape:
        raise ValueError(f"Fuel intervention rasters have different shapes: {baseline_fuel.shape} vs {scenario_fuel.shape}.")
    return baseline_fuel, scenario_fuel


def load_endpoint_delta(
    experiment_dir: Path,
    scenario: str,
    hex_id: str,
    endpoint: str,
    *,
    nonfuel_ids: list[int],
    gt_dir: Path | None = None,
) -> np.ma.MaskedArray:
    """Support-aware (scenario - baseline) response for one hex_id/endpoint, restricted to burnable land
    and to the actual-fire-occurrence footprint for `hex_id`."""

    baseline_dir = find_local_prediction_dir(experiment_dir, "baseline", hex_id, endpoint)
    scenario_dir = find_local_prediction_dir(experiment_dir, scenario, hex_id, endpoint)
    baseline_path = prediction_raster_path(baseline_dir, hex_id, target_name=endpoint)
    baseline = read_prediction(baseline_path)
    scenario_values = read_prediction(prediction_raster_path(scenario_dir, hex_id, target_name=endpoint))
    if scenario_values.shape != baseline.shape:
        raise ValueError(
            f"Scenario {endpoint.upper()} shape {scenario_values.shape} does not match baseline grid {baseline.shape} for hex_id={hex_id!r}."
        )

    baseline_fuel, scenario_fuel = load_fuel_pair(experiment_dir, scenario, hex_id, endpoint)
    response = build_endpoint_response(
        baseline,
        scenario_values,
        baseline_support=burnable_fuel_support(baseline_fuel, nonfuel_ids),
        scenario_support=burnable_fuel_support(scenario_fuel, nonfuel_ids),
    )
    delta = response.delta

    with rasterio.open(baseline_path) as src:
        reference_profile = src.profile.copy()
    valid = load_actual_fire_valid_mask(experiment_dir, hex_id, reference_profile, gt_dir=gt_dir)
    if valid.shape != delta.shape:
        raise ValueError(f"Actual-fire grid {valid.shape} does not match prediction grid {delta.shape} for hex_id={hex_id!r}.")
    return np.ma.masked_where(~valid | np.ma.getmaskarray(delta), delta)


def mean_delta_for_endpoint(
    experiment_dir: Path,
    scenario: str,
    hex_ids: list[str],
    endpoint: str,
    *,
    nonfuel_ids: list[int],
    gt_dir: Path | None = None,
) -> float:
    """Pooled mean of the support-aware response across all finite pixels of all `hex_ids`."""

    pooled = [
        finite_values(load_endpoint_delta(experiment_dir, scenario, hex_id, endpoint, nonfuel_ids=nonfuel_ids, gt_dir=gt_dir))
        for hex_id in hex_ids
    ]
    all_values = np.concatenate(pooled) if pooled else np.array([], dtype=np.float64)
    if all_values.size == 0:
        return float("nan")
    return float(np.mean(all_values))


def changed_fuel_mask(baseline_fuel: np.ma.MaskedArray, scenario_fuel: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Boolean-as-float mask (1.0 where fuel changed, masked elsewhere) for the intervention panel."""

    baseline_values, baseline_valid = np.ma.filled(baseline_fuel, np.nan), ~np.ma.getmaskarray(baseline_fuel)
    scenario_values, scenario_valid = np.ma.filled(scenario_fuel, np.nan), ~np.ma.getmaskarray(scenario_fuel)
    valid = baseline_valid & scenario_valid & np.isfinite(baseline_values) & np.isfinite(scenario_values)
    changed = valid & (baseline_values != scenario_values)
    return np.ma.masked_where(~changed, changed.astype(np.float32))


def plot_aggregate_response_figure(
    experiment_dir: Path,
    scenario: str,
    hex_id: str,
    out_path: Path,
    *,
    nonfuel_ids: list[int],
    title: str,
    downsample: int = 1,
    gt_dir: Path | None = None,
    changed_pixel_label: str = "Changed pixels",
) -> None:
    """Render a 1x4 figure: fuel-intervention changed-pixel mask + BP/FI/ROS response-delta panels."""

    baseline_fuel, scenario_fuel = load_fuel_pair(experiment_dir, scenario, hex_id, "bp")
    mask = changed_fuel_mask(baseline_fuel, scenario_fuel)
    deltas = {
        endpoint: load_endpoint_delta(experiment_dir, scenario, hex_id, endpoint, nonfuel_ids=nonfuel_ids, gt_dir=gt_dir)
        for endpoint in ENDPOINTS
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.25)

    mask_ax = axes[0]
    image = mask_ax.imshow(
        downsample_for_display(mask, downsample),
        cmap=ListedColormap(["#6a1b73"]),
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
    )
    mask_ax.set_title(f"Fuel intervention: {title}")
    mask_ax.set_aspect("equal")
    mask_ax.axis("off")
    # Reserve the same vertical space below the map as the colorbar strip in the
    # other 3 panels (fraction=0.05, pad=0.04), so all 4 map panels stay the same
    # size/height; the legend replaces the (hidden) colorbar in that reserved axis.
    legend_bar = fig.colorbar(image, ax=mask_ax, orientation="horizontal", fraction=0.05, pad=0.04)
    legend_bar.ax.set_visible(False)
    mask_ax.legend(
        handles=[Patch(facecolor="#6a1b73", label=changed_pixel_label)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        ncols=1,
    )

    for ax, endpoint in zip(axes[1:], ENDPOINTS, strict=True):
        norm = delta_norm([deltas[endpoint]])
        image = ax.imshow(
            downsample_for_display(deltas[endpoint], downsample),
            cmap="RdBu_r",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(PANEL_TITLES[endpoint])
        ax.set_aspect("equal")
        ax.axis("off")
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cbar.set_label(PANEL_CBAR_LABELS[endpoint])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        required=True,
        help="Materialized counterfactual fuel run directory, e.g. final_results/counterfactual_fuel_multi_output_hex16.",
    )
    parser.add_argument("--scenario", type=str, required=True, help="Fuel-edit scenario name, e.g. remove_barriers_fixed_c2.")
    parser.add_argument("--hex_ids", type=str, nargs="+", required=True, help="Hex id(s) to pool the mean delta over, e.g. 16.")
    parser.add_argument("--label", type=str, default=None, help="Optional row/panel label. Defaults to the scenario config's description.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional counterfactual config (e.g. configs/counterfactual_fuel_multi_output.yaml) to look up nonfuel_ids/description for --scenario.",
    )
    parser.add_argument(
        "--nonfuel_ids",
        type=int,
        nargs="+",
        default=None,
        help=f"Non-fuel IDs to exclude from burnable support. Defaults to {DEFAULT_NONFUEL_IDS} (or --config's value for --scenario).",
    )
    parser.add_argument("--out_path", type=Path, default=None, help="Optional output image path for the 1x4 aggregate response figure.")
    parser.add_argument("--downsample", type=int, default=0, help="Stride factor to downsample rasters before plotting (for speed/size).")
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=None,
        help="Directory with the actual-fire-occurrence mask (hex<hex_id>_actual.shp) used to restrict the "
        "response to the observed fire extent. Defaults to <experiment_dir>/GT.",
    )
    parser.add_argument(
        "--panel_title",
        type=str,
        default="C-2 \u2192 M-1",
        help="Short title suffix for the fuel-intervention panel, e.g. 'C-2 \u2192 M-1' "
        "(rendered as 'Fuel intervention: <panel_title>'). Defaults to the scenario's --label/description.",
    )
    parser.add_argument(
        "--changed_pixel_label",
        type=str,
        default="Changed pixels",
        help="Legend label for the changed-pixel swatch in the fuel-intervention panel, e.g. 'Changed C-2 pixels'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nonfuel_ids = args.nonfuel_ids
    description = args.scenario
    if args.config is not None:
        scenario_config = load_counterfactual_config(args.config).scenario(args.scenario)
        description = scenario_config.description or args.scenario
        if nonfuel_ids is None:
            fuel_edit = scenario_config.fuel_edit()
            if fuel_edit is not None and "nonfuel_ids" in fuel_edit:
                nonfuel_ids = [int(value) for value in fuel_edit["nonfuel_ids"]]
    if nonfuel_ids is None:
        nonfuel_ids = DEFAULT_NONFUEL_IDS
    label = args.label or args.scenario

    deltas = {
        endpoint: mean_delta_for_endpoint(
            args.experiment_dir, args.scenario, args.hex_ids, endpoint, nonfuel_ids=nonfuel_ids, gt_dir=args.gt_dir
        )
        for endpoint in ENDPOINTS
    }

    print(f"{'Intervention':<28}{'delta_BP':>12}{'delta_FI':>12}{'delta_ROS':>12}")
    print(f"{label:<28}{deltas['bp']:>12.4f}{deltas['fi']:>12.0f}{deltas['ros']:>12.2f}")

    if args.out_path is not None:
        plot_aggregate_response_figure(
            args.experiment_dir,
            args.scenario,
            args.hex_ids[0],
            args.out_path,
            nonfuel_ids=nonfuel_ids,
            title=args.panel_title or args.label or description,
            downsample=args.downsample,
            gt_dir=args.gt_dir,
            changed_pixel_label=args.changed_pixel_label,
        )
        print(f"Wrote fuel-intervention aggregate response figure: {args.out_path}")


if __name__ == "__main__":
    main()
