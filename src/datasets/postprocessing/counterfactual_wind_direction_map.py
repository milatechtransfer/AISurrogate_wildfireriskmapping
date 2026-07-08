"""ROS response figures for the uniform wind-direction counterfactual.

For the paired ``wind_dir_dominant`` / ``wind_dir_dominant_opposite`` scenarios --
every day set to the hex's dominant from-bearing, then its 180-degree opposite --
this renders, per direction, the ground-truth/baseline/scenario/delta ROS maps,
the high-response patch zoom-ins, and the per-pixel ROS change distribution.  A
final side-by-side delta map probes whether the spatial response mirrors when the
wind is reversed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.datasets.postprocessing.counterfactual_ros_maps import (
    ENDPOINT,
    hotspot_centers,
    load_ros_response,
    plot_ros_patch_zoom,
    plot_ros_response_maps,
)
from src.datasets.postprocessing.counterfactual_viz import (
    DEFAULT_ZONE_OVERLAY_ALPHA,
    DEFAULT_ZONE_OVERLAY_COLOR,
    DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    add_zone_overlay_args,
    downsample_for_display,
    finite_values,
    overlay_zone_boundaries,
    plot_delta_histogram,
    prediction_dirs_from_index,
    prediction_footprint,
    symmetric_percentile_limit,
)
from src.datasets.postprocessing.counterfactual_weather_maps import extent_km, load_zone_labels, raw_data_dir_from_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DELTA_COLOR = "#b2182b"
DIRECTION_SCOPES = {
    "hex": (
        ("wind_dir_dominant", "dominant direction", "dominant"),
        ("wind_dir_dominant_opposite", "opposite direction", "opposite"),
    ),
    "zone": (
        ("wind_zone_dir_dominant", "per-zone dominant directions", "zone_dominant"),
        ("wind_zone_dir_dominant_opposite", "per-zone opposite directions", "zone_opposite"),
    ),
}


def _bearing_label(edit_summary: pd.DataFrame | None, scenario: str, fallback: str) -> str:
    if edit_summary is None:
        return fallback
    rows = edit_summary[edit_summary["scenario"] == scenario]
    if rows.empty:
        return fallback
    if len(rows) > 1:
        return f"{fallback} ({len(rows)} zones)"
    return f"{fallback} ({float(rows['from_bearing_deg'].iloc[0]):.0f}\u00b0 from-bearing)"


def plot_direction_pair_delta(
    delta_dominant: np.ma.MaskedArray,
    delta_opposite: np.ma.MaskedArray,
    extent_m: tuple[float, float, float, float],
    *,
    out_path: Path,
    downsample: int,
    suptitle: str,
    dominant_title: str = "\u0394ROS \u2014 dominant direction",
    opposite_title: str = "\u0394ROS \u2014 opposite direction",
    zone_labels: np.ma.MaskedArray | None = None,
    zone_overlay_color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    zone_overlay_linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    zone_overlay_alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Side-by-side ΔROS maps for the dominant and opposite wind directions."""

    delta_limit = symmetric_percentile_limit([delta_dominant, delta_opposite], percentile=99.0)
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit)
    extent = extent_km(extent_m)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8))
    for ax, (data, title) in zip(
        axes,
        [(delta_dominant, dominant_title), (delta_opposite, opposite_title)],
        strict=True,
    ):
        image = ax.imshow(
            downsample_for_display(data, downsample),
            cmap="RdBu_r",
            norm=delta_norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        overlay_zone_boundaries(
            ax,
            zone_labels,
            extent=extent,
            color=zone_overlay_color,
            linewidth=zone_overlay_linewidth,
            alpha=zone_overlay_alpha,
        )
        ax.set_title(title, pad=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("\u0394ROS (m/min)", fontsize=14)
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
    parser.add_argument(
        "--direction_scope",
        choices=tuple(DIRECTION_SCOPES),
        default="hex",
        help="Render the hex-wide or per-zone wind-direction intervention pair.",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/wind_direction.")
    add_zone_overlay_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directions = DIRECTION_SCOPES[args.direction_scope]
    default_out_dir = "wind_direction" if args.direction_scope == "hex" else "wind_direction_zone"
    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir / "figures" / default_out_dir
    raw_data_dir = args.raw_data_dir if args.raw_data_dir is not None else raw_data_dir_from_config(args.experiment_dir, endpoint=ENDPOINT)
    prediction_dirs = prediction_dirs_from_index(args.experiment_dir)

    edit_summary_path = args.experiment_dir / "wind_direction_edit_summary.csv"
    edit_summary = pd.read_csv(edit_summary_path) if edit_summary_path.exists() else None

    responses: dict[str, tuple] = {}
    labels: dict[str, str] = {}
    extent: tuple[float, float, float, float] | None = None
    zone_labels: np.ma.MaskedArray | None = None
    for scenario, fallback_label, slug in directions:
        labels[slug] = _bearing_label(edit_summary, scenario, fallback_label)
        ground_truth, baseline, scenario_ros, delta, extent, reference_profile = load_ros_response(
            prediction_dirs, args.hex_id, raw_data_dir, scenario=scenario
        )
        responses[slug] = (ground_truth, baseline, scenario_ros, delta)
        if args.zone_overlay and zone_labels is None:
            zone_labels = load_zone_labels(
                raw_data_dir,
                args.hex_id,
                reference_profile,
                support=prediction_footprint(prediction_dirs, args.hex_id, endpoint=ENDPOINT),
            )

    deltas = {slug: response[3] for slug, response in responses.items()}
    if extent is None:
        raise RuntimeError("No wind-direction scenarios were loaded.")
    dominant_key = directions[0][2]
    opposite_key = directions[1][2]
    combined_abs_delta = np.ma.maximum(np.ma.abs(deltas[dominant_key]), np.ma.abs(deltas[opposite_key]))
    shared_centers = hotspot_centers(combined_abs_delta, args.hotspot_block, count=args.patch_count, window=args.patch_window)

    for scenario, _fallback_label, slug in directions:
        label = labels[slug]
        ground_truth, baseline, scenario_ros, delta = responses[slug]
        scenario_dir = out_dir / slug

        plot_ros_response_maps(
            ground_truth,
            baseline,
            scenario_ros,
            delta,
            extent,
            out_path=scenario_dir / f"{scenario}_ros_response_maps.png",
            scenario_label=label,
            suptitle=f"ROS response to uniform wind direction \u2014 {label} (hex 16)",
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
            out_path=scenario_dir / f"{scenario}_patch_zoom.png",
            scenario_label=label,
            suptitle=f"Shared high-response {args.patch_window}\u00d7{args.patch_window}-pixel windows \u2014 {label} (hex 16)",
            window=args.patch_window,
            hotspot_block=args.hotspot_block,
            patch_count=args.patch_count,
            centers=shared_centers,
            zone_labels=zone_labels,
            zone_overlay_color=args.zone_overlay_color,
            zone_overlay_linewidth=args.zone_overlay_linewidth,
            zone_overlay_alpha=args.zone_overlay_alpha,
        )
        plot_delta_histogram(
            delta,
            out_path=scenario_dir / f"{scenario}_ros_delta_distribution.png",
            xlabel="\u0394ROS per pixel (scenario \u2212 baseline, m/min)",
            title=f"Per-pixel ROS change \u2014 {label} (hex 16)",
            color=DELTA_COLOR,
        )
        values = finite_values(delta)
        print(f"{scenario}: n={values.size} mean_dROS={float(np.mean(values)):+.3f} median={float(np.median(values)):+.3f}")

    if extent is not None and {"dominant", "opposite"} <= deltas.keys():
        pair_out_name = "wind_direction_pair_ros_delta.png"
        pair_suptitle = "Spatial ROS response under reversed wind direction (hex 16)"
        dominant_title = "\u0394ROS \u2014 dominant direction"
        opposite_title = "\u0394ROS \u2014 opposite direction"
    elif extent is not None and {"zone_dominant", "zone_opposite"} <= deltas.keys():
        pair_out_name = "wind_zone_direction_pair_ros_delta.png"
        pair_suptitle = "Spatial ROS response under reversed per-zone wind directions (hex 16)"
        dominant_title = "\u0394ROS \u2014 per-zone dominant directions"
        opposite_title = "\u0394ROS \u2014 per-zone opposite directions"
    else:
        pair_out_name = ""

    if pair_out_name:
        plot_direction_pair_delta(
            deltas[dominant_key],
            deltas[opposite_key],
            extent,
            out_path=out_dir / pair_out_name,
            downsample=args.downsample,
            suptitle=pair_suptitle,
            dominant_title=dominant_title,
            opposite_title=opposite_title,
            zone_labels=zone_labels,
            zone_overlay_color=args.zone_overlay_color,
            zone_overlay_linewidth=args.zone_overlay_linewidth,
            zone_overlay_alpha=args.zone_overlay_alpha,
        )
    print(f"Figures written under: {out_dir}")


if __name__ == "__main__":
    main()
