"""Generic ROS response figures for a single counterfactual scenario.

Renders the standard ground-truth/baseline/scenario/delta ROS maps, the
high-response patch zoom-ins, and the per-pixel ROS change distribution for any
one ``ros``-endpoint scenario in ``scenario_prediction_index.csv``.  Scenario
scripts with bespoke panels (per-zone wind shift, paired direction mirror) keep
their own modules; this covers the common case so a new scenario does not need a
new figure script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.datasets.postprocessing.counterfactual import load_counterfactual_config
from src.datasets.postprocessing.counterfactual_fuel_intervention_map import intervention_layers_on_prediction_grid
from src.datasets.postprocessing.counterfactual_ros_maps import (
    ENDPOINT,
    load_ros_response,
    plot_ros_patch_zoom,
    plot_ros_response_maps,
)
from src.datasets.postprocessing.counterfactual_viz import (
    add_zone_overlay_args,
    finite_values,
    plot_delta_histogram,
    prediction_dirs_from_index,
    prediction_footprint,
)
from src.datasets.postprocessing.counterfactual_weather_maps import load_zone_labels, raw_data_dir_from_config

DELTA_COLOR = "#b2182b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="Scenario name from scenario_prediction_index.csv.")
    parser.add_argument("--label", default=None, help="Human-readable scenario label for titles (defaults to the name).")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_hex16.yaml"))
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Defaults to baseline config data.raw_data_dir.")
    parser.add_argument("--patch_window", type=int, default=400, help="Side length (pixels) of patch zoom windows.")
    parser.add_argument("--hotspot_block", type=int, default=64, help="Block size for locating high-response windows.")
    parser.add_argument("--patch_count", type=int, default=3, help="Number of distinct high-response windows to render.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/<scenario>.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label if args.label is not None else args.scenario
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / args.scenario
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else raw_data_dir_from_config(args.experiment_dir, endpoint=ENDPOINT)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)

    config = load_counterfactual_config(args.config)
    scenario_cfg = next((s for s in config.scenarios if s.name == args.scenario), None)
    if scenario_cfg is None:
        raise KeyError(f"Scenario {args.scenario!r} not found in {args.config}.")

    fuel_filled_mask = None
    if scenario_cfg.fuel_edit() is not None:
        _, fuel_filled_mask, _, _ = intervention_layers_on_prediction_grid(
            experiment_dir=args.experiment_dir,
            raw_data_dir=raw_data_dir,
            scenario=args.scenario,
            endpoint=ENDPOINT,
            hex_id=args.hex_id,
        )

    ground_truth, baseline, scenario_ros, delta, extent, reference_profile = load_ros_response(
        prediction_dirs, args.hex_id, raw_data_dir, scenario=args.scenario, fuel_filled_mask=fuel_filled_mask
    )
    zone_labels = None
    if args.zone_overlay:
        zone_labels = load_zone_labels(
            raw_data_dir,
            args.hex_id,
            reference_profile,
            support=prediction_footprint(prediction_dirs, args.hex_id, endpoint=ENDPOINT),
        )

    plot_ros_response_maps(
        ground_truth,
        baseline,
        scenario_ros,
        delta,
        extent,
        out_path=out_dir / f"{args.scenario}_ros_response_maps.png",
        scenario_label=label,
        suptitle=f"ROS response \u2014 {label} (hex 16)",
        downsample=args.downsample,
        zone_labels=zone_labels,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
    )
    plot_ros_patch_zoom(
        ground_truth,
        baseline,
        scenario_ros,
        delta,
        out_path=out_dir / f"{args.scenario}_patch_zoom.png",
        scenario_label=label,
        suptitle=f"High-response {args.patch_window}\u00d7{args.patch_window}-pixel windows \u2014 {label} (hex 16)",
        window=args.patch_window,
        hotspot_block=args.hotspot_block,
        patch_count=args.patch_count,
        zone_labels=zone_labels,
        zone_overlay_color=args.zone_overlay_color,
        zone_overlay_linewidth=args.zone_overlay_linewidth,
        zone_overlay_alpha=args.zone_overlay_alpha,
    )
    plot_delta_histogram(
        delta,
        out_path=out_dir / f"{args.scenario}_ros_delta_distribution.png",
        xlabel="\u0394ROS per pixel (scenario \u2212 baseline, m/min)",
        title=f"Per-pixel ROS change \u2014 {label} (hex 16)",
        color=DELTA_COLOR,
    )

    values = finite_values(delta)
    print(f"{args.scenario}: n={values.size} mean_dROS={float(np.mean(values)):+.3f} median={float(np.median(values)):+.3f}")
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
