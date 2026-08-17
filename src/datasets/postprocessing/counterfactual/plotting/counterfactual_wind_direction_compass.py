"""Compass-arranged \u0394 (diff) maps for the wind-direction sweep counterfactual.

Lays the eight unique `wind_dir_*` scenario diff maps (the last panel of
`counterfactual_response_maps.plot_response_maps`) out around a circle so that
0\u00b0 is at the top (north) and bearings increase clockwise (45\u00b0 = NE, 90\u00b0 = E,
...), mirroring a compass rose. One figure per endpoint (bp/fi/ros), with a
shared \u0394 colour scale across all eight panels for direct visual comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import (
    load_counterfactual_config,
    resolve_counterfactual_paths,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps import (
    ENDPOINT_SPECS,
    _extent_km,
    load_endpoint_response,
)
from src.datasets.postprocessing.counterfactual.plotting.counterfactual_viz import (
    delta_norm,
    downsample_for_display,
    prediction_dirs_from_index,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The sweep also includes wind_dir_360, numerically identical to wind_dir_000 by
# construction; it is a closing point for the config, not a distinct compass panel.
DIRECTIONS_DEG: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
COMPASS_LABELS: dict[int, str] = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}


def scenario_name_for_direction(direction_degrees: int) -> str:
    return f"wind_dir_{direction_degrees:03d}"


def _panel_rect(direction_degrees: float, *, radius: float, panel_size: float) -> tuple[float, float, float, float]:
    """Figure-fraction `(left, bottom, width, height)` for a compass bearing (0=N, clockwise)."""
    theta = np.radians(direction_degrees)
    cx = 0.5 + radius * np.sin(theta)
    cy = 0.5 + radius * np.cos(theta)
    return (cx - panel_size / 2.0, cy - panel_size / 2.0, panel_size, panel_size)


def plot_compass_diff_maps(
    deltas: dict[int, np.ma.MaskedArray],
    extent: tuple[float, float, float, float],
    *,
    endpoint: str,
    out_path: Path,
    suptitle: str,
    downsample: int,
    radius: float = 0.35,
    panel_size: float = 0.20,
) -> None:
    """One \u0394{endpoint} panel per compass bearing, arranged on a circle (0\u00b0 = N, up)."""
    spec = ENDPOINT_SPECS[endpoint]
    norm = delta_norm(list(deltas.values()), percentile=99.0)

    fig = plt.figure(figsize=(12.0, 12.0))
    image = None
    for direction_degrees, delta in deltas.items():
        left, bottom, width, height = _panel_rect(direction_degrees, radius=radius, panel_size=panel_size)
        ax = fig.add_axes((left, bottom, width, height))
        image = ax.imshow(
            downsample_for_display(delta, downsample),
            cmap="RdBu_r",
            norm=norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(1.0)
        ax.set_title(f"{direction_degrees}\u00b0 {COMPASS_LABELS[direction_degrees]}", fontsize=13, pad=6)

    center_size = panel_size * 0.7
    center_ax = fig.add_axes((0.5 - center_size / 2.0, 0.5 - center_size / 2.0, center_size, center_size))
    center_ax.axis("off")
    center_ax.text(0.5, 0.5, f"\u0394{spec.label}\nby wind\ndirection", ha="center", va="center", fontsize=13, fontweight="bold")

    if image is not None:
        cbar_ax = fig.add_axes((0.965, 0.15, 0.02, 0.7))
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label(f"\u0394 {spec.units} (scenario \u2212 baseline)", fontsize=13)

    fig.suptitle(suptitle, y=1.02, fontsize=17)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, choices=sorted(ENDPOINT_SPECS), help="Prediction endpoint to render.")
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual/counterfactual_wind_direction_multi_output.yaml"))
    parser.add_argument("--experiment_dir", type=Path, default=None, help="Overrides save_dir from --config.")
    parser.add_argument("--hex_id", type=str, default="16")
    parser.add_argument("--downsample", type=int, default=3, help="Stride factor for map display only.")
    parser.add_argument("--raw_data_dir", type=Path, default=None, help="Overrides raw_data_dir from --config.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Defaults to experiment_dir/figures/compass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.hex_id = normalize_hex_id(args.hex_id)
    config = load_counterfactual_config(args.config)
    experiment_dir, raw_data_dir = resolve_counterfactual_paths(config, experiment_dir=args.experiment_dir, raw_data_dir=args.raw_data_dir)
    out_dir = args.out_dir if args.out_dir is not None else experiment_dir / "figures" / "compass"
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    static_nonfuel_ids = config.nonfuel_ids if config.nonfuel_ids else None

    deltas: dict[int, np.ma.MaskedArray] = {}
    extent_m: tuple[float, float, float, float] | None = None
    for direction_degrees in DIRECTIONS_DEG:
        _, _, _, delta, extent_m, _ = load_endpoint_response(
            experiment_dir,
            prediction_dirs,
            args.hex_id,
            raw_data_dir,
            scenario=scenario_name_for_direction(direction_degrees),
            endpoint=args.endpoint,
            static_nonfuel_ids=static_nonfuel_ids,
        )
        deltas[direction_degrees] = delta
    assert extent_m is not None

    spec = ENDPOINT_SPECS[args.endpoint]
    out_path = out_dir / f"wind_direction_compass_{args.endpoint}.png"
    plot_compass_diff_maps(
        deltas,
        _extent_km(extent_m),
        endpoint=args.endpoint,
        out_path=out_path,
        suptitle=f"{spec.label} \u0394 by forced wind direction \u2014 hex {args.hex_id}",
        downsample=args.downsample,
    )
    print(f"Compass diff map written to {out_path}")


if __name__ == "__main__":
    main()
