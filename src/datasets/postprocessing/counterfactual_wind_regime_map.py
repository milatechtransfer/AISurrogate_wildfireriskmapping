"""ROS response figures for the per-zone peak-wind regime counterfactual.

Generates scientist-facing figures for the ``wind_zone_peak_day`` scenario,
which transplants each weather zone's single highest-WindSpeed day onto every
day in the zone: the ground-truth/baseline/scenario/delta ROS maps, high-response
patch zoom-ins, the per-zone wind-speed shift driving the response, and the
hex-wide per-pixel ROS change distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO = "wind_zone_peak_day"
SCENARIO_LABEL = "peak-wind regime"
DELTA_COLOR = "#b2182b"


def plot_zone_wind_shift(edit_summary_path: Path, *, out_path: Path) -> None:
    summary = pd.read_csv(edit_summary_path)
    summary = summary[summary["scenario"] == SCENARIO].sort_values("zone")
    zones = summary["zone"].astype(int).to_numpy()
    positions = np.arange(zones.size)
    width = 0.4

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(positions - width / 2, summary["baseline_speed_mean"], width, label="Baseline zone-mean", color="0.6")
    ax.bar(positions + width / 2, summary["scenario_speed_mean"], width, label="Peak-wind day", color=DELTA_COLOR)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(zone) for zone in zones])
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Weather zone", fontsize=16)
    ax.set_ylabel("WindSpeed (km/h)", fontsize=16)
    ax.set_title("Input intervention: per-zone wind-speed shift driving the ROS response (hex 16)", fontsize=15)
    ax.legend(fontsize=14)
    fig.tight_layout()
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
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/wind_zone_peak.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / "wind_zone_peak"
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else raw_data_dir_from_config(args.experiment_dir, endpoint=ENDPOINT)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)
    ground_truth, baseline, scenario_ros, delta, extent, reference_profile = load_ros_response(
        prediction_dirs, args.hex_id, raw_data_dir, scenario=SCENARIO
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
        out_path=out_dir / "wind_zone_peak_ros_response_maps.png",
        scenario_label=SCENARIO_LABEL,
        suptitle="ROS response to per-zone peak-wind regime transplant (hex 16)",
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
        out_path=out_dir / "wind_zone_peak_patch_zoom.png",
        scenario_label=SCENARIO_LABEL,
        suptitle=f"High-response {args.patch_window}\u00d7{args.patch_window}-pixel windows \u2014 {SCENARIO_LABEL} (hex 16)",
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
        out_path=out_dir / "wind_zone_peak_ros_delta_distribution.png",
        xlabel="\u0394ROS per pixel (scenario \u2212 baseline, m/min)",
        title="Per-pixel ROS change under peak-wind transplant (hex 16)",
        color=DELTA_COLOR,
    )

    edit_summary_path = args.experiment_dir / "wind_edit_summary.csv"
    if edit_summary_path.exists():
        plot_zone_wind_shift(edit_summary_path, out_path=out_dir / "wind_zone_peak_zone_wind_shift.png")

    values = finite_values(delta)
    print(f"{SCENARIO}: n={values.size} mean_dROS={float(np.mean(values)):+.3f} median={float(np.median(values)):+.3f}")
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
