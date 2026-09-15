"""
One-off diagnostic for the "mosaicking looks wrong for bp/fi but fine for ros" issue.

Since `generate_national_mosaics` runs the exact same `mosaic_predicted_hexels` code path for
every target, if only some targets show gaps in the mosaic, the difference must come from the
per-target predicted hexel `.tif` inputs themselves (not the mosaicking logic). This script
inspects one hex_id across all configured targets to pin down exactly what differs:

  1. Per-target predicted hexel raster: shape, transform, CRS, nodata value, nodata pixel
     count/fraction, and valid-value range.
  2. Per-target reference (GT) raster: same metadata, to check for resolution/alignment
     mismatches between it and the predicted hexel.
  3. Saves a side-by-side PNG of the 3 (or however many configured) targets' single-hexel
     rasters so gaps are directly visible/comparable in isolation from mosaicking.

Usage:
    python -m src.full_map.diagnose_hexel_gap --config path/to/config.yaml \\
        --pred-root experiments/full_map/train --hex-id 12345 \\
        --output diagnose_hex_12345.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

from src.config import Config
from src.full_map.utils import group_predicted_hexel_files_by_target, mask_nodata, valid_pixel_mask


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _describe_raster(path: str, label: str) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        valid_mask = valid_pixel_mask(band, nodata)
        n_nodata = int((~valid_mask).sum())
        valid = band[valid_mask]
        print(f"[{label}] path={path}")
        print(f"  shape={band.shape} transform={src.transform} crs={src.crs} nodata={src.nodata}")
        print(f"  nodata px: {n_nodata}/{band.size} ({100 * n_nodata / band.size:.1f}%)")
        if valid.size:
            print(f"  valid value range: [{valid.min():.4g}, {valid.max():.4g}], mean={valid.mean():.4g}")
        else:
            print("  no valid (non-nodata) pixels!")
    return band


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one hex_id's predicted rasters across targets to isolate a mosaicking gap.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--pred-root", type=str, required=True, help="Directory containing predicted_hexels/ (e.g. experiments/full_map/train)."
    )
    parser.add_argument("--hex-id", type=int, required=True, help="hex_id to inspect (pick one whose mosaic shows a gap).")
    parser.add_argument(
        "--pattern", type=str, default="predicted_hexels/hexel_*_predicted.tif", help="Glob pattern (relative to --pred-root)."
    )
    parser.add_argument("--output", type=str, default="diagnose_hexel_gap.png", help="Path to save the comparison PNG to.")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped = group_predicted_hexel_files_by_target(Path(args.pred_root), args.pattern)
    gt_paths = config.full_map.national_gt_raster_paths or {}

    targets = sorted(grouped.keys())
    if not targets:
        raise ValueError(f"No predicted hexel files found under {args.pred_root} matching {args.pattern!r}.")

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets, strict=True):
        file_map = grouped[target]
        if args.hex_id not in file_map:
            print(f"[{target}] hex_id={args.hex_id} not found among {len(file_map)} predicted hexels for this target -- skipping.")
            ax.set_title(f"{target}: hex {args.hex_id} missing")
            ax.axis("off")
            continue

        print(f"\n=== target={target}, hex_id={args.hex_id} ===")
        band = _describe_raster(str(file_map[args.hex_id]), label=f"{target} predicted hexel")

        if target in gt_paths:
            _describe_raster(gt_paths[target], label=f"{target} national GT (reference grid)")

        with rasterio.open(file_map[args.hex_id]) as src:
            nodata = src.nodata if src.nodata is not None else -9999.0
        masked = mask_nodata(band, nodata)
        im = ax.imshow(masked)
        ax.set_title(f"{target} (hex {args.hex_id})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"\nSaved side-by-side comparison to {args.output}")


if __name__ == "__main__":
    main()
