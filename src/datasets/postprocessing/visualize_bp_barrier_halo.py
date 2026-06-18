"""Presentation-quality visualisation of the BP barrier halo effect.

Default output:

  Spatialized distance-bin BP map (saved as halo_distance_bin_mean_map.png,
  and as halo_figure.png for compatibility): a zoomed map around a water/non-
  fuel barrier in the focal hexel/zone. All valid burnable pixels are shown,
  but they are coloured by the whole-zone mean BP of their distance-to-barrier
  bin. This intentionally visualizes the measured distance-decay statistic
  rather than noisy raw pixel-level BP.

Legacy outputs from the previous multi-panel workflow can still be produced
with --render_legacy_figures.

Usage:
    python -m src.datasets.postprocessing.visualize_bp_barrier_halo \
        --diag_dir experiments/halo_diagnostic \
        --raw_data_dir /home/mila/o/olutayot/copilot-work/canada_bp3+_2026_MILA \
        --out_dir experiments/halo_diagnostic/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, label

from data_preparation.paths import Paths
from data_preparation.spatial.utils import FUEL_GROUP_MAP
from src.datasets.postprocessing.bp_restricted_zero import bp_nonfuel_restricted_ids

matplotlib.use("Agg")

# ── Constants ────────────────────────────────────────────────────────────────

FOCAL_HEX = "10"
FOCAL_ZONE = 26

# Pixel size of the rasters (metres)
PIXEL_M = 100.0

# Distance bands used in the diagnostic
NEAR_M = 500.0
FAR_M = 2000.0

# Fuel groups to show in the standalone Panel A candidate figure.
# Group 2 = C-2, group 8 = M-1/M-2 (most common in hex10 zone 26).
SPATIAL_PANEL_FUEL_GROUPS: list[int] = [2, 8]
SPATIAL_PANEL_MAIN_FUEL_GROUP = 8

# Reverse lookup: fuel raw ID → group index
_FUEL_ID_TO_GROUP: dict[int, int] = {fid: grp for fid, grp in FUEL_GROUP_MAP.items()}

# Distance bins used by diagnose_bp_barrier_halo.py outputs.
DIST_BIN_EDGES_M: tuple[float, ...] = (100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)

# Bin labels → representative centre value (m) for profile x-axis
BIN_CENTRES: dict[str, float] = {
    "100-250m": 175.0,
    "250-500m": 375.0,
    "500-1000m": 750.0,
    "1000-2000m": 1500.0,
    "2000-5000m": 3500.0,
    ">5000m": 6500.0,
}

BIN_ORDER = list(BIN_CENTRES.keys())

# FBP fuel-group labels (group index → short label)
FUEL_GROUP_LABELS: dict[int, str] = {
    1: "C-1",
    2: "C-2",
    3: "C-3",
    4: "C-4",
    5: "C-5",
    6: "C-6",
    7: "C-7",
    8: "M-1/M-2",
    9: "M-3/M-4",
    10: "O-1",
    11: "S",
    12: "M-1 (% conifer)",
    13: "M-2 (% conifer)",
    14: "M-1/M-2 (% conifer)",
}

# Fuel groups to plot as coloured lines in the profile panel.
# All groups present in the data are shown; percent-conifer mixedwood variants
# (12-14) use dashed lines to distinguish them from fixed FBP classes (1-11).
HIGHLIGHTED_GROUPS = [1, 2, 3, 4, 7, 8, 10, 12, 13, 14]

# Whether each group uses a dashed line style (percent-conifer mixedwood variants)
DASHED_GROUPS: set[int] = {12, 13, 14}

# Colours for highlighted fuel groups (tab10 palette)
_TAB10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
FUEL_COLOURS: dict[int, tuple] = {fg: _TAB10[i % 10] for i, fg in enumerate(HIGHLIGHTED_GROUPS)}

# Sector display order and labels for shadow figure
SECTOR_ORDER = ["towards_wind_vector", "crosswind_left", "against_wind_vector", "crosswind_right"]
SECTOR_LABELS = ["Towards\nwind", "Crosswind\nleft", "Against\nwind", "Crosswind\nright"]

# ── Matplotlib style ─────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Helper functions ─────────────────────────────────────────────────────────


def _load_hex_rasters(
    hex_id: str,
    raw_data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Load BP, fuel, and firezones rasters for one hexel."""
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    with rasterio.open(paths.output_burn_prob()) as src:
        bp_raw = src.read(1).astype(np.float32)
        bp_nodata = float(src.nodata) if src.nodata is not None else -9999.0
        transform = src.transform
        crs = src.crs
    with rasterio.open(paths.fuel_grid(hex_id)) as src:
        fuel = src.read(1)
    with rasterio.open(paths.firezones_grid(hex_id)) as src:
        zones = src.read(1)

    bp = np.where(bp_raw == bp_nodata, np.nan, bp_raw)
    return bp, fuel, zones, transform, crs


def _nonfuel_mask(hex_id: str, raw_data_dir: str, fuel: np.ndarray) -> np.ndarray:
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    _, nonfuel_ids, _ = bp_nonfuel_restricted_ids(paths, hex_id)
    return np.isin(fuel, nonfuel_ids)


def _aggregate_profile(group_df: pd.DataFrame) -> pd.DataFrame:
    """Weighted-average all fuel groups into a single all-fuels profile row per bin."""
    rows = []
    for bin_label in BIN_ORDER:
        sub = group_df[group_df["dist_bin"] == bin_label]
        if sub.empty:
            continue
        w = sub["n_pixels"].values.astype(float)
        rows.append(
            {
                "dist_bin": bin_label,
                "bp_mean": np.average(sub["bp_mean"].values, weights=w),
                "bp_p25": np.average(sub["bp_p25"].values, weights=w),
                "bp_p75": np.average(sub["bp_p75"].values, weights=w),
                "n_pixels": w.sum(),
            }
        )
    return pd.DataFrame(rows)


def _fuel_group_array(fuel: np.ndarray) -> np.ndarray:
    """Map raw fuel IDs to grouped FBP fuel classes."""
    return np.vectorize(_FUEL_ID_TO_GROUP.get)(fuel, -1).astype(np.int16)


def _distance_bin_labels_for_pixels(dist_m: np.ndarray) -> np.ndarray:
    """Return diagnostic distance-bin labels for every pixel."""
    labels = np.empty(dist_m.shape, dtype=object)
    edges = [0.0, *DIST_BIN_EDGES_M, np.inf]
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        if np.isinf(hi):
            mask = dist_m >= lo
            label_str = f">{lo:.0f}m"
        else:
            mask = (dist_m >= lo) & (dist_m < hi)
            label_str = f"{lo:.0f}-{hi:.0f}m"
        labels[mask] = label_str
    return labels


def _profile_for_fuel_group(
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
    fuel_group: int,
) -> pd.DataFrame:
    sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id) & (profiles["fuel_group"] == fuel_group)].copy()
    sub["_order"] = sub["dist_bin"].map({b: i for i, b in enumerate(BIN_ORDER)})
    return sub.sort_values("_order")


def _weighted_bp_for_bins(profile: pd.DataFrame, bins: tuple[str, ...]) -> float:
    sub = profile[profile["dist_bin"].isin(bins)]
    if sub.empty:
        return float("nan")
    return float(np.average(sub["bp_mean"], weights=sub["n_pixels"]))


def _compute_profiles_and_contrasts_from_bp(
    bp: np.ndarray,
    fuel: np.ndarray,
    zones: np.ndarray,
    nonfuel: np.ndarray,
    hex_id: int,
    valid_support: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all-fuel/fuel-group distance profiles and near/far contrasts from one BP raster."""
    dist_m = distance_transform_edt(~nonfuel, sampling=(PIXEL_M, PIXEL_M))
    bin_labels = ["0-100m"] + BIN_ORDER
    bin_ids = np.digitize(dist_m, np.asarray(DIST_BIN_EDGES_M), right=False).astype(np.int8)
    fuel_group = _fuel_group_array(fuel)
    valid = (~nonfuel) & np.isfinite(bp) & (bp >= 0.0) & (zones > 0)
    if valid_support is not None:
        valid = valid & valid_support

    if not valid.any():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(
        {
            "zone_id": zones[valid].astype(np.int16, copy=False),
            "fuel_group": fuel_group[valid].astype(np.int16, copy=False),
            "bin_id": bin_ids[valid],
            "bp": bp[valid].astype(np.float32, copy=False),
        }
    )

    grouped = df.groupby(["zone_id", "fuel_group", "bin_id"], sort=True)["bp"]
    profiles = grouped.agg(bp_mean="mean", n_pixels="size").reset_index()
    quantiles = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quantiles = quantiles.rename(columns={0.25: "bp_p25", 0.75: "bp_p75"})
    profiles = profiles.merge(quantiles, on=["zone_id", "fuel_group", "bin_id"], how="left")
    profiles["hex_id"] = int(hex_id)
    profiles["dist_bin"] = profiles["bin_id"].map({i: label for i, label in enumerate(bin_labels)})
    profiles = profiles[["hex_id", "zone_id", "fuel_group", "dist_bin", "bp_mean", "bp_p25", "bp_p75", "n_pixels"]].copy()

    def _band_stats(band_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        stats = (
            band_df.groupby(["zone_id", "fuel_group"], sort=True)["bp"]
            .agg(**{f"n_{prefix}": "size", f"mean_bp_{prefix}": "mean", f"std_bp_{prefix}": "std"})
            .reset_index()
        )
        stats[f"std_bp_{prefix}"] = stats[f"std_bp_{prefix}"].fillna(0.0)
        return stats

    near_stats = _band_stats(df[df["bin_id"].isin([1, 2])], "near")
    far_stats = _band_stats(df[df["bin_id"].isin([5, 6])], "far")
    contrasts = near_stats.merge(far_stats, on=["zone_id", "fuel_group"], how="inner")
    if contrasts.empty:
        contrasts["hex_id"] = []
    else:
        contrasts["hex_id"] = int(hex_id)
        contrasts["halo_abs"] = contrasts["mean_bp_far"] - contrasts["mean_bp_near"]
        contrasts["halo_rel"] = np.where(
            contrasts["mean_bp_far"] > 0,
            contrasts["halo_abs"] / contrasts["mean_bp_far"],
            np.nan,
        )
        contrasts = contrasts[
            [
                "hex_id",
                "zone_id",
                "fuel_group",
                "n_near",
                "n_far",
                "mean_bp_near",
                "mean_bp_far",
                "std_bp_near",
                "std_bp_far",
                "halo_abs",
                "halo_rel",
            ]
        ].copy()

    return profiles, contrasts


# ── Panel A: spatial map ─────────────────────────────────────────────────────


def _find_crop_window(
    nonfuel_mask: np.ndarray,
    zone_mask: np.ndarray,
    half_width_px: int = 200,
    valid_bp_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Return (row_lo, row_hi, col_lo, col_hi) centred on the best barrier patch.

    By default picks the largest fitting connected component.  When *valid_bp_mask*
    is provided (boolean array, True where BP is valid), picks the fitting component
    whose surrounding crop window contains the most valid BP pixels, so the window
    is always within the fire-simulation extent.
    """
    from scipy.ndimage import find_objects
    from scipy.ndimage import label as sci_label

    barriers_in_zone = nonfuel_mask & zone_mask
    labeled, _ = sci_label(barriers_in_zone)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    sorted_lbls = np.argsort(sizes)[::-1]
    objects = find_objects(labeled)

    H, W = nonfuel_mask.shape
    crop_side = 2 * half_width_px
    margin_needed = int(crop_side * 0.20)

    # Collect fitting candidates, scored by valid-BP coverage.
    # When no valid_bp_mask is provided: use fast early-exit (first fitting by size).
    candidates: list[tuple[int, int, int, int]] = []  # (score, lbl_size, cy, cx)
    for lbl in sorted_lbls:
        if sizes[lbl] < 50:
            break
        if lbl == 0 or lbl - 1 >= len(objects):
            continue
        obj = objects[lbl - 1]
        if obj is None:
            continue
        row_slice, col_slice = obj
        bbox_h = int(row_slice.stop - row_slice.start)
        bbox_w = int(col_slice.stop - col_slice.start)
        if bbox_h <= crop_side - 2 * margin_needed and bbox_w <= crop_side - 2 * margin_needed:
            cy_c = int((row_slice.start + row_slice.stop) / 2)
            cx_c = int((col_slice.start + col_slice.stop) / 2)
            if valid_bp_mask is not None:
                r0c = max(0, cy_c - half_width_px)
                r1c = min(H, cy_c + half_width_px)
                c0c = max(0, cx_c - half_width_px)
                c1c = min(W, cx_c + half_width_px)
                score = int(valid_bp_mask[r0c:r1c, c0c:c1c].sum())
                candidates.append((score, int(sizes[lbl]), cy_c, cx_c))
            else:
                # Fast path: just take the largest fitting component
                candidates.append((int(sizes[lbl]), int(sizes[lbl]), cy_c, cx_c))
                break  # no need to scan further

    cy, cx = H // 2, W // 2  # fallback
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sz, cy, cx = candidates[0]
        if valid_bp_mask is not None:
            print(f"  Crop centred on barrier ({best_sz:,} px, " f"{best_score:,} valid BP px in window)")
        else:
            print(f"  Crop centred on barrier of size {best_sz:,} px")

    r0 = max(0, cy - half_width_px)
    r1 = min(H, cy + half_width_px)
    c0 = max(0, cx - half_width_px)
    c1 = min(W, cx + half_width_px)
    return r0, r1, c0, c1


def _pick_fuel_group_for_crop(
    bp: np.ndarray,
    nonfuel: np.ndarray,
    fuel: np.ndarray,
    zone_mask: np.ndarray,
    crop: tuple[int, int, int, int],
    profiles: pd.DataFrame | None,
    hex_id: int,
    zone_id: int,
    fallback: int = SPATIAL_PANEL_MAIN_FUEL_GROUP,
) -> int:
    """Return the fuel group with the most valid target pixels inside the crop window.

    When *profiles* is provided the search is restricted to groups that have
    pre-computed profile data for this hex/zone.  When *profiles* is None all
    groups present in the crop are considered.
    """
    r0, r1, c0, c1 = crop
    nf_crop = nonfuel[r0:r1, c0:c1] & zone_mask[r0:r1, c0:c1]
    zm_crop = zone_mask[r0:r1, c0:c1]
    fg_crop = _fuel_group_array(fuel[r0:r1, c0:c1])
    bp_crop = bp[r0:r1, c0:c1]

    if profiles is not None:
        valid_groups: set[int] = set(
            int(g) for g in profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id)]["fuel_group"].unique()
        )
    else:
        # No pre-computed profiles — consider every group present in the crop
        valid_groups = set(int(g) for g in np.unique(fg_crop) if g >= 0)

    best_group, best_count = fallback, 0
    for fg in valid_groups:
        count = int((zm_crop & ~nf_crop & (fg_crop == fg) & ~np.isnan(bp_crop)).sum())
        if count > best_count:
            best_count, best_group = count, int(fg)

    return best_group


def _draw_spatial_panel(
    ax: plt.Axes,
    bp: np.ndarray,
    nonfuel: np.ndarray,
    fuel: np.ndarray,
    zone_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crop: tuple[int, int, int, int],
    profiles: pd.DataFrame | None,
    hex_id: int,
    zone_id: int,
    fuel_group: int | None,
    *,
    vmax_override: float | None = None,
    title_prefix: str = "A",
) -> None:
    from matplotlib.patches import FancyBboxPatch

    r0, r1, c0, c1 = crop
    bp_crop = bp[r0:r1, c0:c1]
    nf_crop = nonfuel[r0:r1, c0:c1] & zone_mask[r0:r1, c0:c1]
    zm_crop = zone_mask[r0:r1, c0:c1]
    fg_crop = _fuel_group_array(fuel[r0:r1, c0:c1])

    # Auto-detect dominant fuel group in the crop window if not specified
    if fuel_group is None:
        fuel_group = _pick_fuel_group_for_crop(bp, nonfuel, fuel, zone_mask, crop, profiles, hex_id, zone_id)
        print(
            f"  Auto-selected fuel group {fuel_group} "
            f"({FUEL_GROUP_LABELS.get(fuel_group, f'group {fuel_group}')}) "
            f"for hex{hex_id}/zone{zone_id}"
        )

    # Pixel-centre distance to the nearest true non-fuel/water barrier.
    dist_m = distance_transform_edt(~nonfuel, sampling=(PIXEL_M, PIXEL_M))
    dist_labels = _distance_bin_labels_for_pixels(dist_m[r0:r1, c0:c1])

    target_mask = zm_crop & ~nf_crop & (fg_crop == fuel_group) & ~np.isnan(bp_crop)

    # Build distance-bin → mean BP mapping.
    # Use pre-computed profiles when available, otherwise compute on-the-fly
    # from the crop itself (enables use with any BP array, e.g. predictions).
    _ALL_BIN_LABELS = ["0-100m"] + BIN_ORDER  # include 0-100m for completeness
    if profiles is not None:
        profile = _profile_for_fuel_group(profiles, hex_id, zone_id, fuel_group)
        bp_by_bin = dict(zip(profile["dist_bin"], profile["bp_mean"]))
        if not bp_by_bin:
            raise ValueError(f"No profile data for hex{hex_id}, zone {zone_id}, fuel group {fuel_group}")
    else:
        bp_by_bin = {}
        for bin_label in _ALL_BIN_LABELS:
            bin_pix = bp_crop[target_mask & (dist_labels == bin_label)]
            if bin_pix.size > 0:
                bp_by_bin[bin_label] = float(np.nanmean(bin_pix))
        if not bp_by_bin:
            raise ValueError(f"No target pixels in crop for hex{hex_id}/zone{zone_id}/group{fuel_group}")

    display_bp = np.full(bp_crop.shape, np.nan, dtype=np.float32)
    for bin_label, bp_mean in bp_by_bin.items():
        display_bp[target_mask & (dist_labels == bin_label)] = float(bp_mean)

    valid_vals = display_bp[~np.isnan(display_bp)]
    vmax = vmax_override if vmax_override is not None else (float(np.nanmax(valid_vals) * 1.05) if valid_vals.size else 0.05)
    # Guard against degenerate colourbar (all-zero data)
    vmax = max(float(vmax), 1e-5)

    # White background: non-target pixels blend into the figure and disappear.
    _BG = "white"
    ax.set_facecolor(_BG)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(_BG)

    img = ax.imshow(
        np.ma.masked_invalid(display_bp),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )

    # Barriers: dark charcoal
    barrier_rgba = np.zeros((*nf_crop.shape, 4), dtype=np.float32)
    barrier_rgba[nf_crop] = [0.15, 0.15, 0.15, 1.0]
    ax.imshow(barrier_rgba, origin="upper", interpolation="none")

    # --- Scale bar (inside axes, lower left, white backing box) ----------
    ax_w_px = c1 - c0
    scale_km = 10
    scale_frac = (scale_km * 1000 / PIXEL_M) / ax_w_px
    sb_x0, sb_y = 0.05, 0.055
    pad_x, pad_y = 0.012, 0.018
    ax.add_patch(
        FancyBboxPatch(
            (sb_x0 - pad_x, sb_y - pad_y),
            scale_frac + 2 * pad_x,
            0.075,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.8,
            zorder=3,
            clip_on=False,
        )
    )
    ax.plot([sb_x0, sb_x0 + scale_frac], [sb_y, sb_y], transform=ax.transAxes, color="k", lw=2.5, solid_capstyle="butt", zorder=4)
    for xv in [sb_x0, sb_x0 + scale_frac]:
        ax.plot([xv, xv], [sb_y - 0.012, sb_y + 0.012], transform=ax.transAxes, color="k", lw=1.5, zorder=4)
    ax.text(
        sb_x0 + scale_frac / 2,
        sb_y + 0.022,
        f"{scale_km} km",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="k",
        fontsize=10,
        fontweight="bold",
        zorder=5,
    )

    # --- Colourbar -------------------------------------------------------
    cb = plt.colorbar(img, ax=ax, fraction=0.035, pad=0.025, shrink=0.88)
    cb.set_label("Mean burn probability", fontsize=10, labelpad=10)
    cb.ax.tick_params(labelsize=9.5)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))

    # --- Legend ----------------------------------------------------------
    # Strip footnote marker so it reads cleanly in a map title/legend
    fuel_label = FUEL_GROUP_LABELS.get(fuel_group, f"Group {fuel_group}").replace("†", "").strip()
    legend_elements = [
        Line2D([0], [0], color="0.18", lw=7, label="Water / non-fuel barrier"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        framealpha=1.0,
        edgecolor="0.55",
        facecolor="white",
        handlelength=2.0,
        borderpad=0.6,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # --- Near/far ratio annotation (lower-right text box) ----------------
    _NEAR_B = ("100-250m", "250-500m")
    _FAR_B = ("2000-5000m", ">5000m")
    near_vals = [bp_by_bin[b] for b in _NEAR_B if b in bp_by_bin]
    far_vals = [bp_by_bin[b] for b in _FAR_B if b in bp_by_bin]
    if near_vals and far_vals:
        near_bp = float(np.mean(near_vals))
        far_bp = float(np.mean(far_vals))
        ratio = far_bp / near_bp if near_bp > 0 else float("nan")
        if not np.isnan(ratio):
            ax.text(
                0.97,
                0.06,
                f"BP < 500 m:   {near_bp:.4f}\n" f"BP > 2 km:    {far_bp:.4f}\n" f"Ratio (far/near):  {ratio:.1f}×",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#111111",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.6", alpha=0.95),
            )

    # --- Title -----------------------------------------------------------
    ax.set_title(
        f"{title_prefix}  Burn probability gradient near spread barriers"
        f" — {fuel_label} pixels\n"
        f"Hex {hex_id}, Zone {zone_id}  ·  Each pixel colored by its "
        f"distance-band mean burn probability",
        loc="left",
        pad=8,
        fontsize=10,
        color="#111111",
    )


def _draw_distance_bin_mean_map(
    ax: plt.Axes,
    bp: np.ndarray,
    nonfuel: np.ndarray,
    zone_mask: np.ndarray,
    crop: tuple[int, int, int, int],
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
    *,
    title_prefix: str = "Distance-band mean burn probability near spread barriers",
    vmax_override: float | None = None,
    valid_support: np.ndarray | None = None,
) -> None:
    """Draw all valid burnable pixels coloured by whole-zone distance-bin mean BP."""
    from matplotlib.patches import FancyBboxPatch

    r0, r1, c0, c1 = crop
    bp_crop = bp[r0:r1, c0:c1]
    zm_crop = zone_mask[r0:r1, c0:c1]
    nf_crop = nonfuel[r0:r1, c0:c1] & zm_crop

    burnable_bp_mask = zm_crop & ~nf_crop & np.isfinite(bp_crop) & (bp_crop >= 0.0)
    if valid_support is not None:
        burnable_bp_mask &= valid_support[r0:r1, c0:c1]
    display_bp = np.full(bp_crop.shape, np.nan, dtype=np.float32)

    sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id)].copy()
    if sub.empty:
        raise ValueError(f"No profile rows for hex{hex_id}/zone{zone_id}")

    bp_by_bin: dict[str, float] = {}
    n_by_bin: dict[str, int] = {}
    for bin_label, rows in sub.groupby("dist_bin"):
        weights = rows["n_pixels"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            continue
        bp_by_bin[str(bin_label)] = float(np.average(rows["bp_mean"], weights=weights))
        n_by_bin[str(bin_label)] = int(weights.sum())

    dist_m = distance_transform_edt(~nonfuel, sampling=(PIXEL_M, PIXEL_M))[r0:r1, c0:c1]
    dist_labels = _distance_bin_labels_for_pixels(dist_m)
    for bin_label, bp_mean in bp_by_bin.items():
        display_bp[burnable_bp_mask & (dist_labels == bin_label)] = bp_mean

    valid_vals = display_bp[np.isfinite(display_bp)]
    if valid_vals.size == 0:
        raise ValueError(f"No displayable distance-bin pixels in crop for hex{hex_id}/zone{zone_id}")

    vmax = max(float(vmax_override if vmax_override is not None else np.nanmax(valid_vals)), 1e-5)

    bg = "white"
    ax.set_facecolor(bg)
    # Truncate the near-white end so low-but-valid BP pixels remain visible;
    # pure white is reserved for pixels that are not rendered.
    base_cmap = plt.get_cmap("YlOrRd")
    cmap = LinearSegmentedColormap.from_list(
        "YlOrRd_visible_low",
        base_cmap(np.linspace(0.16, 1.0, 256)),
    )
    cmap.set_bad(bg)

    img = ax.imshow(
        np.ma.masked_invalid(display_bp),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )

    barrier_rgba = np.zeros((*nf_crop.shape, 4), dtype=np.float32)
    barrier_rgba[nf_crop] = [0.15, 0.15, 0.15, 1.0]
    ax.imshow(barrier_rgba, origin="upper", interpolation="none")

    # Scale bar with opaque backing so it remains readable over the raster.
    ax_w_px = c1 - c0
    scale_km = 10
    scale_frac = (scale_km * 1000 / PIXEL_M) / ax_w_px
    sb_x0, sb_y = 0.05, 0.055
    pad_x, pad_y = 0.012, 0.018
    ax.add_patch(
        FancyBboxPatch(
            (sb_x0 - pad_x, sb_y - pad_y),
            scale_frac + 2 * pad_x,
            0.075,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.8,
            zorder=3,
            clip_on=False,
        )
    )
    ax.plot([sb_x0, sb_x0 + scale_frac], [sb_y, sb_y], transform=ax.transAxes, color="k", lw=2.5, solid_capstyle="butt", zorder=4)
    for xv in [sb_x0, sb_x0 + scale_frac]:
        ax.plot([xv, xv], [sb_y - 0.012, sb_y + 0.012], transform=ax.transAxes, color="k", lw=1.5, zorder=4)
    ax.text(
        sb_x0 + scale_frac / 2,
        sb_y + 0.022,
        f"{scale_km} km",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="k",
        fontsize=10,
        fontweight="bold",
        zorder=5,
    )

    cb = plt.colorbar(img, ax=ax, fraction=0.035, pad=0.025, shrink=0.88)
    cb.set_label("Mean burn probability by distance band", fontsize=10, labelpad=10)
    cb.ax.tick_params(labelsize=9.5)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))

    ax.legend(
        handles=[Line2D([0], [0], color="0.18", lw=7, label="Water / non-fuel barrier")],
        loc="upper right",
        fontsize=10,
        framealpha=1.0,
        edgecolor="0.55",
        facecolor="white",
        handlelength=2.0,
        borderpad=0.6,
    )

    near_bins = ("100-250m", "250-500m")
    far_bins = ("2000-5000m", ">5000m")

    def _weighted_mean_for(bin_names: tuple[str, ...]) -> tuple[float, int]:
        vals = [(bp_by_bin[b], n_by_bin[b]) for b in bin_names if b in bp_by_bin and b in n_by_bin]
        if not vals:
            return float("nan"), 0
        means, counts = zip(*vals)
        return float(np.average(means, weights=counts)), int(sum(counts))

    near_bp, n_near = _weighted_mean_for(near_bins)
    far_bp, n_far = _weighted_mean_for(far_bins)
    ratio = far_bp / near_bp if np.isfinite(near_bp) and near_bp > 0 else float("nan")
    ratio_str = f"{ratio:.1f}x" if np.isfinite(ratio) else "n/a"
    ax.text(
        0.97,
        0.06,
        "Colour: whole-zone distance-band mean BP\n"
        f"Mean BP < 500 m: {near_bp:.4f} (n={n_near:,})\n"
        f"Mean BP > 2 km:  {far_bp:.4f} (n={n_far:,})\n"
        f"Far / near ratio: {ratio_str}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#111111",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.6", alpha=0.95),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title_prefix}\n" f"Hex {hex_id}, Zone {zone_id}  ·  all valid burnable pixels; colour summarizes mean BP by barrier distance",
        loc="left",
        pad=8,
        fontsize=11,
        color="#111111",
    )


def _draw_distance_bin_mean_map_figure(
    bp: np.ndarray,
    nonfuel: np.ndarray,
    zone_mask: np.ndarray,
    crop: tuple[int, int, int, int],
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
    out_paths: list[Path],
    title_prefix: str = "Distance-band mean burn probability near spread barriers",
    vmax_override: float | None = None,
    valid_support: np.ndarray | None = None,
) -> None:
    """Render and save the standalone all-fuel distance-bin map."""
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 8.2))
    _draw_distance_bin_mean_map(
        ax,
        bp,
        nonfuel,
        zone_mask,
        crop,
        profiles,
        hex_id,
        zone_id,
        title_prefix=title_prefix,
        vmax_override=vmax_override,
        valid_support=valid_support,
    )
    fig.subplots_adjust(left=0.02, right=0.96, top=0.91, bottom=0.02)
    for out_path in out_paths:
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"  Saved distance-bin BP map -> {out_path}")
    plt.close(fig)


def _draw_spatial_panel_candidates(
    bp: np.ndarray,
    nonfuel: np.ndarray,
    fuel: np.ndarray,
    zone_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crop: tuple[int, int, int, int],
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
    out_path: Path,
) -> None:
    """Render standalone Panel A candidates for a few fuel groups."""
    ncols = len(SPATIAL_PANEL_FUEL_GROUPS)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.8), squeeze=False)
    for ax, fuel_group in zip(axes[0], SPATIAL_PANEL_FUEL_GROUPS):
        _draw_spatial_panel(
            ax,
            bp,
            nonfuel,
            fuel,
            zone_mask,
            transform,
            crop,
            profiles,
            hex_id,
            zone_id,
            fuel_group,
        )
    fig.suptitle(
        "Panel A candidates: spatialized distance-decay profile by fuel type",
        y=1.02,
        fontsize=13,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved Panel A candidates → {out_path}")


# ── Panel B: distance-decay profile ─────────────────────────────────────────


def _draw_profile_panel(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
) -> None:
    sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id)].copy()

    # sort bins by defined order
    sub["_order"] = sub["dist_bin"].map({b: i for i, b in enumerate(BIN_ORDER)})
    sub = sub.sort_values("_order")

    # Aggregate all-fuels line
    agg = _aggregate_profile(sub)
    agg["x"] = agg["dist_bin"].map(BIN_CENTRES)

    # --- Shaded bands for "near" and "far" --------------------------------
    near_x = BIN_CENTRES["250-500m"]
    far_x = BIN_CENTRES["2000-5000m"]
    ax.axvspan(0, NEAR_M, color="#FDDBC7", alpha=0.45, zorder=0, label="_near_shade")
    ax.axvspan(FAR_M, 8000, color="#D1E5F0", alpha=0.45, zorder=0, label="_far_shade")
    ax.axvline(NEAR_M, color="#D6604D", lw=1.0, ls="--", zorder=1)
    ax.axvline(FAR_M, color="#4393C3", lw=1.0, ls="--", zorder=1)

    # --- Per-fuel-group thin lines ----------------------------------------
    groups_present = sorted(sub["fuel_group"].unique())
    for fg in groups_present:
        fg_sub = sub[sub["fuel_group"] == fg].copy()
        fg_sub["x"] = fg_sub["dist_bin"].map(BIN_CENTRES)
        fg_sub = fg_sub.dropna(subset=["x"])
        colour = FUEL_COLOURS.get(int(fg), "#AAAAAA")
        ls = "--" if int(fg) in DASHED_GROUPS else "-"
        ax.plot(
            fg_sub["x"],
            fg_sub["bp_mean"],
            color=colour,
            lw=1.0,
            ls=ls,
            alpha=0.55,
            zorder=2,
        )

    # --- All-fuels aggregate (bold) — no ribbon to avoid visual clutter -------
    ax.plot(
        agg["x"],
        agg["bp_mean"],
        color="#222222",
        lw=2.5,
        zorder=4,
        label="All fuels (mean ± IQR)",
    )

    # --- Band annotations (placed at fixed axes-fraction height) ----------
    ax.text(80, 0.0, "Near\n(<500 m)", ha="left", va="bottom", fontsize=8, color="#D6604D", alpha=0.9, transform=ax.get_xaxis_transform())
    ax.text(
        FAR_M * 1.05,
        0.0,
        "Far\n(>2 000 m)",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4393C3",
        alpha=0.9,
        transform=ax.get_xaxis_transform(),
    )

    # --- Fuel group legend ------------------------------------------------
    fg_handles = []
    for fg in groups_present:
        colour = FUEL_COLOURS.get(int(fg), "#AAAAAA")
        label_str = FUEL_GROUP_LABELS.get(int(fg), f"Group {int(fg)}")
        ls = "--" if int(fg) in DASHED_GROUPS else "-"
        fg_handles.append(Line2D([0], [0], color=colour, lw=2, ls=ls, label=label_str))
    fg_handles.append(Line2D([0], [0], color="#222222", lw=2.5, label="All fuels (mean)"))

    ax.legend(handles=fg_handles, fontsize=7, loc="lower right", framealpha=0.9, ncol=2)

    ax.set_xlim(0, 8000)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    ax.set_xlabel("Distance to nearest barrier (km)")
    ax.set_ylabel("Mean Burn Probability")
    ax.set_title(
        "B  Distance-Decay Profile  (Hex 10 · Zone 26)\n" "Dashed groups are mixedwood percent-conifer variants.",
        loc="left",
        pad=6,
        fontsize=10,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))


def _draw_standalone_distance_profile(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
    *,
    source_label: str | None = None,
    ymax_override: float | None = None,
) -> None:
    """Single clean Panel B: all-fuels mean BP by distance to barrier."""
    sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id)].copy()
    if sub.empty:
        raise ValueError(f"No profile rows for hex{hex_id}/zone{zone_id}")

    agg = _aggregate_profile(sub)
    agg["x_m"] = agg["dist_bin"].map(BIN_CENTRES)
    agg = agg.dropna(subset=["x_m"]).sort_values("x_m")
    if agg.empty:
        raise ValueError(f"No plottable profile bins for hex{hex_id}/zone{zone_id}")

    ax.axvspan(0, NEAR_M, color="#FDDBC7", alpha=0.38, zorder=0)
    ax.axvspan(FAR_M, 8000, color="#D1E5F0", alpha=0.38, zorder=0)
    ax.axvline(NEAR_M, color="#D6604D", lw=1.0, ls="--", zorder=1)
    ax.axvline(FAR_M, color="#4393C3", lw=1.0, ls="--", zorder=1)

    ax.plot(
        agg["x_m"],
        agg["bp_mean"],
        color="#222222",
        lw=2.8,
        marker="o",
        markersize=6.5,
        markerfacecolor="#222222",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )

    if {"bp_p25", "bp_p75"}.issubset(agg.columns):
        ax.fill_between(
            agg["x_m"].to_numpy(dtype=float),
            agg["bp_p25"].to_numpy(dtype=float),
            agg["bp_p75"].to_numpy(dtype=float),
            color="#222222",
            alpha=0.12,
            linewidth=0,
            zorder=2,
            label="Weighted interquartile range",
        )

    near = agg[agg["dist_bin"].isin(("100-250m", "250-500m"))]
    far = agg[agg["dist_bin"].isin(("2000-5000m", ">5000m"))]
    near_bp = float(np.average(near["bp_mean"], weights=near["n_pixels"])) if not near.empty else float("nan")
    far_bp = float(np.average(far["bp_mean"], weights=far["n_pixels"])) if not far.empty else float("nan")
    n_near = int(near["n_pixels"].sum()) if not near.empty else 0
    n_far = int(far["n_pixels"].sum()) if not far.empty else 0
    ratio = far_bp / near_bp if np.isfinite(near_bp) and near_bp > 0 else float("nan")
    ratio_str = f"{ratio:.1f}x" if np.isfinite(ratio) else "n/a"

    ymax = max(float(ymax_override if ymax_override is not None else agg["bp_mean"].max() * 1.35), 1e-4)
    ax.text(
        0.04,
        0.96,
        f"Far / near mean BP: {ratio_str}\n" f"<500 m: {near_bp:.4f}  (n={n_near:,})\n" f">2 km:  {far_bp:.4f}  (n={n_far:,})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.65", alpha=0.96),
        zorder=6,
    )

    ax.text(
        NEAR_M * 0.48,
        ymax * 0.05,
        "Near\n(<500 m)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#B2182B",
        zorder=5,
    )
    ax.text(
        FAR_M * 1.55,
        ymax * 0.05,
        "Far reference\n(>2 km)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#2166AC",
        zorder=5,
    )

    ax.set_xlim(0, 7000)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Distance to nearest water / non-fuel barrier (km)")
    ax.set_ylabel("Mean burn probability")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    title_head = "Mean burn probability increases with distance from spread barriers"
    if source_label:
        title_head = f"{source_label}: {title_head[0].lower()}{title_head[1:]}"
    ax.set_title(
        f"{title_head}\n" f"Hex {hex_id}, Zone {zone_id}  ·  all valid burnable pixels",
        loc="left",
        pad=10,
        fontsize=12,
        color="#111111",
    )


def _draw_standalone_distance_profile_figure(
    profiles: pd.DataFrame,
    hex_id: int,
    zone_id: int,
    out_path: Path,
    *,
    source_label: str | None = None,
    ymax_override: float | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 6.4))
    _draw_standalone_distance_profile(
        ax,
        profiles,
        hex_id,
        zone_id,
        source_label=source_label,
        ymax_override=ymax_override,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.13)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved standalone distance profile -> {out_path}")


# ── Panel B (multi-zone): all-zones overlay for one hexel ────────────────────


def _draw_profile_panel_multizones(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    hex_id: int,
) -> None:
    """Panel B variant: one all-fuels aggregate line per zone, overlaid on the same axes."""
    _NEAR_BINS = {"0-100m", "100-250m", "250-500m"}
    _FAR_BINS = {"2000-5000m", ">5000m"}

    zone_ids = sorted(profiles[profiles["hex_id"] == hex_id]["zone_id"].unique())

    # Build per-zone aggregate + ratio
    zone_data: list[tuple[int, pd.DataFrame, float]] = []
    for zid in zone_ids:
        sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zid)]
        agg = _aggregate_profile(sub)
        if agg.empty:
            continue
        agg["x"] = agg["dist_bin"].map(BIN_CENTRES)
        near_r = agg[agg["dist_bin"].isin(_NEAR_BINS)]
        far_r = agg[agg["dist_bin"].isin(_FAR_BINS)]
        if near_r.empty or far_r.empty or near_r["n_pixels"].sum() < 200:
            ratio = float("nan")
        else:
            bp_near = np.average(near_r["bp_mean"], weights=near_r["n_pixels"])
            bp_far = np.average(far_r["bp_mean"], weights=far_r["n_pixels"])
            ratio = bp_far / bp_near if bp_near > 0 else float("nan")
        zone_data.append((zid, agg, ratio))

    # Sort weakest→strongest so the strongest zone is drawn last (on top)
    zone_data.sort(key=lambda t: t[2] if not np.isnan(t[2]) else 0.0)
    best_ratio = max((t[2] for t in zone_data if not np.isnan(t[2])), default=float("nan"))

    # Near / far shaded bands
    ax.axvspan(0, NEAR_M, color="#FDDBC7", alpha=0.45, zorder=0)
    ax.axvspan(FAR_M, 8000, color="#D1E5F0", alpha=0.45, zorder=0)
    ax.axvline(NEAR_M, color="#D6604D", lw=1.0, ls="--", zorder=1)
    ax.axvline(FAR_M, color="#4393C3", lw=1.0, ls="--", zorder=1)

    colours = cast(Any, plt.cm).tab10.colors
    for i, (zid, agg, ratio) in enumerate(zone_data):
        colour = colours[i % 10]
        is_best = (not np.isnan(ratio)) and ratio == best_ratio
        lw = 2.8 if is_best else 1.4
        alpha = 1.0 if is_best else 0.70
        ratio_str = f"{ratio:.1f}×" if not np.isnan(ratio) else "—"
        label = f"Zone {zid}  ({ratio_str})" + ("  ★" if is_best else "")
        ax.plot(agg["x"], agg["bp_mean"], color=colour, lw=lw, alpha=alpha, zorder=3 + i, label=label)

    # Band labels — same positioning as single-zone panel
    ax.text(80, 0.0, "Near\n(<500 m)", ha="left", va="bottom", fontsize=8, color="#D6604D", transform=ax.get_xaxis_transform())
    ax.text(FAR_M * 1.05, 0.0, "Far\n(>2 000 m)", ha="left", va="bottom", fontsize=8, color="#4393C3", transform=ax.get_xaxis_transform())

    ax.legend(fontsize=8, loc="upper left", framealpha=0.95, ncol=1, title="Zone (far/near ratio)", title_fontsize=8)
    ax.set_xlim(0, 8000)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))
    ax.set_xlabel("Distance to nearest barrier (km)")
    ax.set_ylabel("Mean Burn Probability")
    ax.set_title(
        f"B  Distance-Decay Profiles by Zone — Hex {hex_id}\n" "All-fuels weighted aggregate  ·  ★ = zone with strongest gradient",
        loc="left",
        pad=8,
        fontsize=10,
    )
    ax.grid(axis="y", lw=0.4, alpha=0.4)


# ── Panel C: zone-level halo contrast ────────────────────────────────────────


def _draw_zone_contrast_panel(
    ax: plt.Axes,
    contrasts: pd.DataFrame,
    hex_id: int,
) -> None:
    sub = contrasts[contrasts["hex_id"] == hex_id].copy()

    # Weighted average of near/far across fuel groups per zone
    def _zone_agg(g: pd.DataFrame) -> pd.Series:
        w_near = g["n_near"].values.astype(float)
        w_far = g["n_far"].values.astype(float)
        return pd.Series(
            {
                "mean_bp_near": np.average(g["mean_bp_near"], weights=w_near),
                "mean_bp_far": np.average(g["mean_bp_far"], weights=w_far),
                "n_near": w_near.sum(),
                "n_far": w_far.sum(),
            }
        )

    zone_df = sub.groupby("zone_id").apply(_zone_agg, include_groups=False).reset_index()
    zone_df["halo_abs"] = zone_df["mean_bp_far"] - zone_df["mean_bp_near"]
    zone_df = zone_df.sort_values("halo_abs", ascending=False).reset_index(drop=True)

    y_pos = np.arange(len(zone_df))

    for i, row in zone_df.iterrows():
        colour = "#D6604D" if row.halo_abs > 0 else "#92C5DE"
        ax.plot([row.mean_bp_near, row.mean_bp_far], [i, i], color=colour, lw=1.5, alpha=0.7, zorder=2)
        ax.scatter(row.mean_bp_near, i, color=colour, s=55, zorder=3, marker="o", label="_near" if i == 0 else "")
        ax.scatter(row.mean_bp_far, i, color=colour, s=55, zorder=3, marker="D", label="_far" if i == 0 else "")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Zone {int(r.zone_id)}" for _, r in zone_df.iterrows()], fontsize=8)
    ax.set_xlabel("Mean Burn Probability")
    ax.set_title(f"C  Near vs Far BP by Zone (Hex {hex_id})\nCircle = near (<500 m)  ◆ = far (>2 000 m)", loc="left", pad=6)
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    legend_elements = [
        Line2D([0], [0], color="#D6604D", marker="o", lw=1.5, ms=6, label="Near   (positive halo)"),
        Line2D([0], [0], color="#92C5DE", marker="o", lw=1.5, ms=6, label="Near   (null / negative)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")


def _aggregate_zone_contrasts(
    contrasts: pd.DataFrame,
    *,
    min_near_pixels: int = 10_000,
    min_far_pixels: int = 1_000,
) -> pd.DataFrame:
    """Aggregate fuel-group near/far contrasts to one all-fuels row per hex/zone."""
    rows: list[dict[str, float | int]] = []
    for (hex_id, zone_id), group in contrasts.groupby(["hex_id", "zone_id"]):
        n_near = group["n_near"].to_numpy(dtype=float)
        n_far = group["n_far"].to_numpy(dtype=float)
        total_near = int(n_near.sum())
        total_far = int(n_far.sum())
        if total_near < min_near_pixels or total_far < min_far_pixels:
            continue

        mean_near = float(np.average(group["mean_bp_near"], weights=n_near))
        mean_far = float(np.average(group["mean_bp_far"], weights=n_far))
        halo_abs = mean_far - mean_near
        halo_ratio = mean_far / mean_near if mean_near > 0 else float("nan")
        rows.append(
            {
                "hex_id": int(hex_id),
                "zone_id": int(zone_id),
                "n_near": total_near,
                "n_far": total_far,
                "mean_bp_near": mean_near,
                "mean_bp_far": mean_far,
                "halo_abs": halo_abs,
                "halo_ratio": halo_ratio,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("halo_ratio", ascending=False).reset_index(drop=True)


def _draw_standalone_zone_contrast_summary(
    ax: plt.Axes,
    contrasts: pd.DataFrame,
    focal_hex: int,
    focal_zone: int,
    *,
    source_label: str | None = None,
    xmax_override: float | None = None,
) -> None:
    """Single clean Panel C: near-vs-far BP contrast across hex/zone cases."""
    zone_df = _aggregate_zone_contrasts(contrasts)
    if zone_df.empty:
        raise ValueError("No hex/zone contrasts passed the support filter.")

    y = np.arange(len(zone_df))
    near_color = "#D6604D"
    far_color = "#2166AC"

    for idx, row in zone_df.iterrows():
        is_focal = int(row.hex_id) == focal_hex and int(row.zone_id) == focal_zone
        line_color = "#111111" if is_focal else "#A0A0A0"
        line_width = 2.8 if is_focal else 1.35
        alpha = 1.0 if is_focal else 0.78
        ax.plot(
            [row.mean_bp_near, row.mean_bp_far],
            [idx, idx],
            color=line_color,
            lw=line_width,
            alpha=alpha,
            zorder=2,
        )
        ax.scatter(
            row.mean_bp_near,
            idx,
            marker="o",
            s=82 if is_focal else 56,
            color=near_color,
            edgecolor="#111111" if is_focal else "white",
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            row.mean_bp_far,
            idx,
            marker="D",
            s=82 if is_focal else 56,
            color=far_color,
            edgecolor="#111111" if is_focal else "white",
            linewidth=0.9,
            zorder=3,
        )

    labels = []
    for _, row in zone_df.iterrows():
        prefix = "★ " if int(row.hex_id) == focal_hex and int(row.zone_id) == focal_zone else ""
        labels.append(f"{prefix}Hex {int(row.hex_id)}, Zone {int(row.zone_id)}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    xmax = float(xmax_override if xmax_override is not None else max(zone_df["mean_bp_near"].max(), zone_df["mean_bp_far"].max()) * 1.24)
    ax.set_xlim(0, max(xmax, 0.001))
    ratio_x = ax.get_xlim()[1] * 0.985
    for idx, row in zone_df.iterrows():
        ratio = row.halo_ratio
        ratio_label = f"{ratio:.1f}x" if np.isfinite(ratio) else "n/a"
        ax.text(
            ratio_x,
            idx,
            ratio_label,
            va="center",
            ha="right",
            fontsize=8.5,
            color="#111111",
            family="monospace",
        )

    ax.text(
        ratio_x,
        -0.75,
        "Far / near",
        va="bottom",
        ha="right",
        fontsize=8.5,
        color="#444444",
        family="monospace",
    )

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=near_color, markeredgecolor="white", markersize=8, label="Near (<500 m)"
        ),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=far_color, markeredgecolor="white", markersize=8, label="Far (>2 km)"),
        Line2D([0], [0], color="#111111", lw=2.8, label="Focal example"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=9)

    ax.grid(axis="x", lw=0.5, alpha=0.35)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.set_xlabel("Mean burn probability")
    title_head = "Near-vs-far burn probability across fire zones"
    if source_label:
        title_head = f"{source_label}: {title_head[0].lower()}{title_head[1:]}"
    ax.set_title(
        f"{title_head}\n" "All-fuels weighted mean; rows require >=10k near pixels and >=1k far pixels",
        loc="left",
        pad=10,
        fontsize=12,
        color="#111111",
    )


def _draw_standalone_zone_contrast_summary_figure(
    contrasts: pd.DataFrame,
    focal_hex: int,
    focal_zone: int,
    out_path: Path,
    *,
    source_label: str | None = None,
    xmax_override: float | None = None,
) -> None:
    zone_df = _aggregate_zone_contrasts(contrasts)
    fig_h = max(5.6, 0.46 * len(zone_df) + 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(9.8, fig_h))
    _draw_standalone_zone_contrast_summary(
        ax,
        contrasts,
        focal_hex,
        focal_zone,
        source_label=source_label,
        xmax_override=xmax_override,
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.11)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved standalone zone contrast summary -> {out_path}")


# ── Figure: Ground Truth vs Model Prediction comparison ─────────────────────


def _load_pred_bp(pred_path: Path) -> np.ndarray:
    """Load a model-predicted BP TIF and return a float32 array (nodata → NaN)."""
    with rasterio.open(pred_path) as src:
        raw = src.read(1).astype(np.float32)
        nd = float(src.nodata) if src.nodata is not None else -9999.0
    return np.where(raw == nd, np.nan, raw)


def _load_and_align_pred_bp(
    pred_path: Path,
    ref_crs: rasterio.crs.CRS,
    ref_transform: rasterio.transform.Affine,
    ref_height: int,
    ref_width: int,
) -> np.ndarray:
    """Load predicted BP TIF and reproject/resample to the reference GT grid."""
    from rasterio.warp import Resampling, reproject

    with rasterio.open(pred_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nd = float(src.nodata) if src.nodata is not None else -9999.0
        src_data = np.where(src_data == src_nd, np.nan, src_data)
        src_crs = src.crs
        src_transform = src.transform

    dest = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dest


def _profile_vmax_for_zone(profiles: pd.DataFrame, hex_id: int, zone_id: int) -> float:
    sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id)]
    if sub.empty:
        return 1e-5
    return max(float(sub["bp_mean"].max()), 1e-5)


def _profile_ymax_for_zone(profiles: pd.DataFrame, hex_id: int, zone_id: int) -> float:
    sub = profiles[(profiles["hex_id"] == hex_id) & (profiles["zone_id"] == zone_id)]
    if sub.empty:
        return 1e-4
    agg = _aggregate_profile(sub)
    if agg.empty:
        return 1e-4
    return max(float(agg["bp_mean"].max()) * 1.35, 1e-4)


def _contrast_xmax(contrasts: pd.DataFrame) -> float:
    zone_df = _aggregate_zone_contrasts(contrasts)
    if zone_df.empty:
        return 0.001
    return max(float(max(zone_df["mean_bp_near"].max(), zone_df["mean_bp_far"].max()) * 1.24), 0.001)


def _render_gt_pred_halo_set(
    raw_data_dir: str,
    pred_dir: Path,
    out_dir: Path,
    hex_id: str,
    zone_id: int,
    crop_half_px: int,
) -> None:
    """Render GT and prediction standalone Panel A/B/C figures for one held-out hex/zone."""
    hid_int = int(hex_id)
    case_dir = out_dir / f"hex{hid_int:02d}_zone{zone_id}_gt_pred"
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering GT/pred halo set for hex{hid_int:02d}, zone {zone_id}")
    print(f"  Prediction directory: {pred_dir}")

    bp_gt, fuel, zones, transform, crs = _load_hex_rasters(f"{hid_int:02d}", raw_data_dir)
    nonfuel = _nonfuel_mask(f"{hid_int:02d}", raw_data_dir, fuel)
    zone_mask = zones == zone_id

    pred_path = pred_dir / f"hexel_{hid_int:02d}_predicted.tif"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction TIF not found: {pred_path}")
    bp_pred = _load_and_align_pred_bp(pred_path, crs, transform, bp_gt.shape[0], bp_gt.shape[1])
    shared_support = np.isfinite(bp_gt) & np.isfinite(bp_pred) & (bp_gt >= 0.0) & (~nonfuel) & (zones > 0)
    bp_pred = np.where(shared_support, bp_pred, np.nan)

    crop = _find_crop_window(
        nonfuel,
        zone_mask,
        half_width_px=crop_half_px,
        valid_bp_mask=shared_support,
    )
    print(f"  Crop window: rows {crop[0]}-{crop[1]}, cols {crop[2]}-{crop[3]}")

    print("  Computing GT profiles/contrasts ...")
    gt_profiles, gt_contrasts = _compute_profiles_and_contrasts_from_bp(bp_gt, fuel, zones, nonfuel, hid_int, valid_support=shared_support)
    print("  Computing prediction profiles/contrasts ...")
    pred_profiles, pred_contrasts = _compute_profiles_and_contrasts_from_bp(
        bp_pred, fuel, zones, nonfuel, hid_int, valid_support=shared_support
    )

    gt_profiles.to_csv(case_dir / f"hex{hid_int:02d}_gt_halo_distance_profiles.csv", index=False)
    pred_profiles.to_csv(case_dir / f"hex{hid_int:02d}_pred_halo_distance_profiles.csv", index=False)
    gt_contrasts.to_csv(case_dir / f"hex{hid_int:02d}_gt_halo_zone_contrasts.csv", index=False)
    pred_contrasts.to_csv(case_dir / f"hex{hid_int:02d}_pred_halo_zone_contrasts.csv", index=False)

    shared_map_vmax = max(
        _profile_vmax_for_zone(gt_profiles, hid_int, zone_id),
        _profile_vmax_for_zone(pred_profiles, hid_int, zone_id),
    )
    shared_profile_ymax = max(
        _profile_ymax_for_zone(gt_profiles, hid_int, zone_id),
        _profile_ymax_for_zone(pred_profiles, hid_int, zone_id),
    )
    shared_contrast_xmax = max(_contrast_xmax(gt_contrasts), _contrast_xmax(pred_contrasts))

    _draw_distance_bin_mean_map_figure(
        bp=bp_gt,
        nonfuel=nonfuel,
        zone_mask=zone_mask,
        crop=crop,
        profiles=gt_profiles,
        hex_id=hid_int,
        zone_id=zone_id,
        out_paths=[case_dir / f"hex{hid_int:02d}_zone{zone_id}_gt_distance_bin_mean_map.png"],
        title_prefix="Ground truth distance-band mean burn probability near spread barriers",
        vmax_override=shared_map_vmax,
        valid_support=shared_support,
    )
    _draw_distance_bin_mean_map_figure(
        bp=bp_pred,
        nonfuel=nonfuel,
        zone_mask=zone_mask,
        crop=crop,
        profiles=pred_profiles,
        hex_id=hid_int,
        zone_id=zone_id,
        out_paths=[case_dir / f"hex{hid_int:02d}_zone{zone_id}_pred_distance_bin_mean_map.png"],
        title_prefix="Model prediction distance-band mean burn probability near spread barriers",
        vmax_override=shared_map_vmax,
        valid_support=shared_support,
    )

    _draw_standalone_distance_profile_figure(
        profiles=gt_profiles,
        hex_id=hid_int,
        zone_id=zone_id,
        out_path=case_dir / f"hex{hid_int:02d}_zone{zone_id}_gt_distance_profile.png",
        source_label="Ground truth",
        ymax_override=shared_profile_ymax,
    )
    _draw_standalone_distance_profile_figure(
        profiles=pred_profiles,
        hex_id=hid_int,
        zone_id=zone_id,
        out_path=case_dir / f"hex{hid_int:02d}_zone{zone_id}_pred_distance_profile.png",
        source_label="Model prediction",
        ymax_override=shared_profile_ymax,
    )

    _draw_standalone_zone_contrast_summary_figure(
        contrasts=gt_contrasts,
        focal_hex=hid_int,
        focal_zone=zone_id,
        out_path=case_dir / f"hex{hid_int:02d}_gt_zone_contrast_summary.png",
        source_label="Ground truth",
        xmax_override=shared_contrast_xmax,
    )
    _draw_standalone_zone_contrast_summary_figure(
        contrasts=pred_contrasts,
        focal_hex=hid_int,
        focal_zone=zone_id,
        out_path=case_dir / f"hex{hid_int:02d}_pred_zone_contrast_summary.png",
        source_label="Model prediction",
        xmax_override=shared_contrast_xmax,
    )

    print(f"  Saved GT/pred halo figures -> {case_dir}")


def _draw_gt_vs_pred_comparison(
    cases: list[tuple[str, int]],  # [(hex_id_str, zone_id), ...]
    raw_data_dir: str,
    pred_dir: Path,
    crop_half_px: int,
    out_path: Path,
) -> None:
    """2 × 2 grid: rows = hexels, cols = Ground Truth | Model Prediction."""
    nrows = len(cases)
    ncols = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11 * ncols, 9 * nrows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_idx, (hid_str, zone_id) in enumerate(cases):
        hid_int = int(hid_str)
        print(f"  GT vs Pred: loading hex{hid_str} / zone {zone_id} …")

        # Load GT data
        bp_gt, fuel, zones, tf, gt_crs = _load_hex_rasters(hid_str, raw_data_dir)
        nonfuel = _nonfuel_mask(hid_str, raw_data_dir, fuel)
        zone_mask = zones == zone_id
        # Use the original (pre-fill) BP to guide crop selection: only pick barriers
        # with actual simulation-extent pixels nearby (valid BP > 0 in crop window).
        valid_bp_mask = ~np.isnan(bp_gt)
        crop = _find_crop_window(nonfuel, zone_mask, half_width_px=crop_half_px, valid_bp_mask=valid_bp_mask)
        print(f"    Crop: rows {crop[0]}–{crop[1]}, cols {crop[2]}–{crop[3]}")

        # Detect dominant fuel group from GT crop (no profiles available)
        fuel_group = _pick_fuel_group_for_crop(bp_gt, nonfuel, fuel, zone_mask, crop, None, hid_int, zone_id)
        print(f"    Auto fuel group: {fuel_group} " f"({FUEL_GROUP_LABELS.get(fuel_group, f'group {fuel_group}')})")

        # Load predicted BP, reprojected to GT grid
        pred_tif = pred_dir / f"hexel_{hid_int:02d}_predicted.tif"
        h_gt, w_gt = bp_gt.shape
        bp_pred = _load_and_align_pred_bp(pred_tif, gt_crs, tf, h_gt, w_gt)

        # Treat GT nodata as zero — consistent with bp_nodata_as_zero=True training
        # convention (NRCAN confirms burnable pixels outside simulation extent → BP=0)
        bp_gt = np.where(np.isnan(bp_gt), 0.0, bp_gt)
        # Treat prediction nodata as zero too (outside model coverage → 0)
        bp_pred = np.where(np.isnan(bp_pred), 0.0, bp_pred)

        # Compute a shared vmax so GT and Pred use the same colour scale
        # Build a temporary bp_by_bin for GT to find display max
        r0, r1, c0, c1 = crop
        dist_m = distance_transform_edt(~nonfuel, sampling=(PIXEL_M, PIXEL_M))
        dist_labels = _distance_bin_labels_for_pixels(dist_m[r0:r1, c0:c1])
        nf_crop = nonfuel[r0:r1, c0:c1] & zone_mask[r0:r1, c0:c1]
        zm_crop = zone_mask[r0:r1, c0:c1]
        fg_crop = _fuel_group_array(fuel[r0:r1, c0:c1])
        _ALL_BINS = ["0-100m"] + BIN_ORDER

        def _bin_means(bp_arr: np.ndarray) -> dict[str, float]:
            bp_c = bp_arr[r0:r1, c0:c1]
            tmask = zm_crop & ~nf_crop & (fg_crop == fuel_group) & ~np.isnan(bp_c)
            d: dict[str, float] = {}
            for bl in _ALL_BINS:
                pix = bp_c[tmask & (dist_labels == bl)]
                if pix.size > 0:
                    d[bl] = float(np.nanmean(pix))
            return d

        bpb_gt = _bin_means(bp_gt)
        bpb_pred = _bin_means(bp_pred)

        # Shared vmax (5% headroom over whichever is larger)
        all_vals = list(bpb_gt.values()) + list(bpb_pred.values())
        vmax = float(max(all_vals) * 1.05) if all_vals else 0.05

        # Draw GT panel
        _draw_spatial_panel(
            axes[row_idx][0],
            bp_gt,
            nonfuel,
            fuel,
            zone_mask,
            tf,
            crop,
            None,
            hid_int,
            zone_id,
            fuel_group,
            vmax_override=vmax,
            title_prefix="Ground Truth",
        )
        # Draw Pred panel
        _draw_spatial_panel(
            axes[row_idx][1],
            bp_pred,
            nonfuel,
            fuel,
            zone_mask,
            tf,
            crop,
            None,
            hid_int,
            zone_id,
            fuel_group,
            vmax_override=vmax,
            title_prefix="Model Prediction",
        )

    fig.suptitle(
        "Halo effect: ground truth vs. model prediction (held-out test hexels)",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, hspace=0.15, wspace=0.06)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved GT vs Pred figure → {out_path}")


def _draw_shadow_figure(
    shadow: pd.DataFrame,
    wind: pd.DataFrame,
    out_path: Path,
) -> None:
    # Only zones that appear in shadow (already filtered by consistency threshold)
    zone_keys = shadow[["hex_id", "zone_id"]].drop_duplicates()
    n = len(zone_keys)
    if n == 0:
        print("  No shadow rows to plot.")
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        "Directional BP Contrast Near Spread Barriers\n" "(near-barrier pixels segmented by sector relative to dominant wind)",
        fontsize=13,
        y=1.02,
    )

    colours = ["#D6604D", "#4393C3", "#74ADD1", "#FDAE61"]

    for idx, (_, key) in enumerate(zone_keys.iterrows()):
        ax = axes[idx // ncols][idx % ncols]
        hid, zid = key.hex_id, key.zone_id
        sub = shadow[(shadow.hex_id == hid) & (shadow.zone_id == zid)].copy()
        sub = sub.set_index("sector").reindex(SECTOR_ORDER)

        bp_vals = sub["bp_mean"].fillna(0).values
        bars = ax.bar(
            np.arange(4),
            bp_vals,
            color=colours,
            edgecolor="white",
            lw=0.8,
            alpha=0.9,
        )
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(SECTOR_LABELS, fontsize=8)
        ax.set_ylabel("Mean Burn Probability", fontsize=9)
        ymax = max(bp_vals.max() * 1.35, 1e-4)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))

        # Annotate relative range (max−min) / mean as % to quantify anisotropy
        if bp_vals.mean() > 0:
            rel_range = (bp_vals.max() - bp_vals.min()) / bp_vals.mean() * 100
            ax.text(
                0.97, 0.97, f"Range: {rel_range:.1f}% of mean", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#555555"
            )

        # Wind metadata from wind table
        wrow = wind[(wind.hex_id == hid) & (wind.zone_id == zid)]
        if not wrow.empty:
            cons = wrow.iloc[0].consistency
            dom_dir = wrow.iloc[0].dominant_direction_deg
            ax.set_title(
                f"Hex {int(hid)} · Zone {int(zid)}\n" f"consistency={cons:.2f}, dom. dir={dom_dir:.0f}°",
                fontsize=9,
                loc="left",
            )
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shadow figure → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise BP barrier halo diagnostic outputs.")
    p.add_argument("--diag_dir", default="experiments/halo_diagnostic", help="Directory produced by diagnose_bp_barrier_halo.py")
    p.add_argument(
        "--raw_data_dir",
        default="/home/mila/o/olutayot/copilot-work/canada_bp3+_2026_MILA",
        help="Raw hexel data root (same as diagnose_bp_barrier_halo config)",
    )
    p.add_argument("--out_dir", default=None, help="Output directory for figures (default: <diag_dir>/plots)")
    p.add_argument("--focal_hex", default=FOCAL_HEX, help=f"Hexel ID for spatial map and profile panels (default: {FOCAL_HEX})")
    p.add_argument("--focal_zone", type=int, default=FOCAL_ZONE, help=f"Zone ID for spatial map and profile panels (default: {FOCAL_ZONE})")
    p.add_argument("--crop_half_km", type=float, default=20.0, help="Half-width of spatial crop centred on largest lake (km)")
    p.add_argument(
        "--pred_dir",
        default=("experiments/" "bp_full_config_v3_kl_ccc_hexpairrank_fullsupport_bpzero_wind_vector_dist_ignition/" "predicted_hexels"),
        help="Directory containing hexel_NN_predicted.tif files (default: latest best BP model).",
    )
    p.add_argument(
        "--render_gt_pred_halo_set",
        action="store_true",
        help="Render GT and prediction Panel A/B/C figures for --focal_hex/--focal_zone, then exit.",
    )
    p.add_argument(
        "--render_legacy_figures",
        action="store_true",
        help="Also render the older multi-panel/profile/shadow figures.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir) if args.out_dir else diag_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    focal_hex_id = args.focal_hex
    focal_zone = args.focal_zone
    crop_half_px = int(args.crop_half_km * 1000 / PIXEL_M)

    if args.render_gt_pred_halo_set:
        _render_gt_pred_halo_set(
            raw_data_dir=args.raw_data_dir,
            pred_dir=Path(args.pred_dir),
            out_dir=out_dir,
            hex_id=focal_hex_id,
            zone_id=focal_zone,
            crop_half_px=crop_half_px,
        )
        return

    # ── Load profile table for the default spatialized distance-bin map ───
    print("Loading diagnostic CSVs ...")
    profiles = pd.read_csv(diag_dir / "halo_distance_profiles.csv")
    contrasts = pd.read_csv(diag_dir / "halo_matched_contrasts.csv")

    # ── Load rasters for standalone distance-bin map ─────────────────────
    print(f"Loading rasters for hex{focal_hex_id} ...")
    bp, fuel, zones, transform, _crs = _load_hex_rasters(focal_hex_id, args.raw_data_dir)
    nonfuel = _nonfuel_mask(focal_hex_id, args.raw_data_dir, fuel)
    zone_mask = zones == focal_zone

    # Find best crop window around largest barrier patch in focal zone
    crop = _find_crop_window(nonfuel, zone_mask, half_width_px=crop_half_px)
    print(f"  Crop window: rows {crop[0]}–{crop[1]}, cols {crop[2]}–{crop[3]}")

    print("Rendering standalone distance-bin BP map ...")
    _draw_distance_bin_mean_map_figure(
        bp=bp,
        nonfuel=nonfuel,
        zone_mask=zone_mask,
        crop=crop,
        profiles=profiles,
        hex_id=int(focal_hex_id),
        zone_id=focal_zone,
        out_paths=[
            out_dir / "halo_distance_bin_mean_map.png",
            out_dir / "halo_figure.png",
        ],
    )

    print("Rendering standalone distance profile ...")
    _draw_standalone_distance_profile_figure(
        profiles=profiles,
        hex_id=int(focal_hex_id),
        zone_id=focal_zone,
        out_path=out_dir / "halo_distance_profile.png",
    )

    print("Rendering standalone zone contrast summary ...")
    _draw_standalone_zone_contrast_summary_figure(
        contrasts=contrasts,
        focal_hex=int(focal_hex_id),
        focal_zone=focal_zone,
        out_path=out_dir / "halo_zone_contrast_summary.png",
    )

    if not args.render_legacy_figures:
        return

    # ── Load remaining diagnostic CSVs for legacy figures ────────────────
    print("Loading diagnostic CSVs ...")
    wind = pd.read_csv(diag_dir / "wind_consistency_by_zone.csv")
    shadow = pd.read_csv(diag_dir / "directional_shadow_contrasts.csv")

    # ── Figure 1: halo evidence (3 panels) ───────────────────────────────
    print("Rendering Figure 1 (halo evidence) ...")
    fig = plt.figure(figsize=(19, 6.5))
    # Panel A gets ~47% of width; B and C share the rest.
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.0, 1.2, 1.05],
        wspace=0.40,
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.11,
    )
    ax_map = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1])
    ax_cont = fig.add_subplot(gs[2])

    _draw_spatial_panel(
        ax_map,
        bp,
        nonfuel,
        fuel,
        zone_mask,
        transform,
        crop,
        profiles,
        int(focal_hex_id),
        focal_zone,
        SPATIAL_PANEL_MAIN_FUEL_GROUP,
    )
    _draw_profile_panel(ax_prof, profiles, int(focal_hex_id), focal_zone)
    _draw_zone_contrast_panel(ax_cont, contrasts, int(focal_hex_id))

    fig.savefig(out_dir / "halo_figure.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved Figure 1 → {out_dir / 'halo_figure.png'}")

    _draw_spatial_panel_candidates(
        bp,
        nonfuel,
        fuel,
        zone_mask,
        transform,
        crop,
        profiles,
        int(focal_hex_id),
        focal_zone,
        out_dir / "halo_panel_a_spatialized_fuel_groups.png",
    )

    # ── Figure 2: directional wind shadow ────────────────────────────────
    # Harmonise column names between shadow and wind tables
    wind_for_shadow = wind.rename(columns={"zone": "zone_id"}) if "zone" in wind.columns else wind
    # Convert zone string labels (fru26 → 26) in wind table if needed
    if wind_for_shadow["zone_id"].dtype == object:
        wind_for_shadow = wind_for_shadow.copy()
        wind_for_shadow["zone_id"] = wind_for_shadow["zone_id"].str.extract(r"(\d+)$")[0].astype(int)

    print("Rendering Figure 2 (directional shadow) …")
    _draw_shadow_figure(shadow, wind_for_shadow, out_dir / "shadow_figure.png")

    # ── Figure 3: Panel A comparison — hex5/zone27 vs hex10/zone26 ───────
    COMPARE_CASES = [("05", 27), ("10", 26)]
    print("Rendering Figure 3 (Panel A comparison across hexels) …")
    ncols = len(COMPARE_CASES)
    fig3, axes3 = plt.subplots(1, ncols, figsize=(9.5 * ncols, 8.5), squeeze=False)
    for ax, (hid, zid) in zip(axes3[0], COMPARE_CASES):
        print(f"  Loading rasters for hex{hid} …")
        bp_h, fuel_h, zones_h, tf_h, _crs_h = _load_hex_rasters(hid, args.raw_data_dir)
        nonfuel_h = _nonfuel_mask(hid, args.raw_data_dir, fuel_h)
        zmask_h = zones_h == zid
        crop_h = _find_crop_window(nonfuel_h, zmask_h, half_width_px=crop_half_px)
        # profiles use integer hex_id (5 or 10)
        hid_int = int(hid)
        _draw_spatial_panel(
            ax,
            bp_h,
            nonfuel_h,
            fuel_h,
            zmask_h,
            tf_h,
            crop_h,
            profiles,
            hid_int,
            zid,
            None,
        )
    fig3.suptitle(
        "Burn probability gradient near spread barriers — two independent hexels",
        y=0.98,
        fontsize=13,
        fontweight="bold",
    )
    fig3.savefig(out_dir / "halo_panel_a_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print(f"  Saved Figure 3 → {out_dir / 'halo_panel_a_comparison.png'}")

    # ── Figure 4 & 5: per-hex Panel B (multi-zone) + Panel C ─────────────
    for hid_int in sorted(profiles["hex_id"].unique()):
        print(f"Rendering per-hex B+C figure for hex{hid_int} …")
        fig_bc, (ax_b, ax_c) = plt.subplots(
            1,
            2,
            figsize=(13, 6),
            gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.38},
        )
        _draw_profile_panel_multizones(ax_b, profiles, int(hid_int))
        _draw_zone_contrast_panel(ax_c, contrasts, int(hid_int))
        fig_bc.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13)
        fig_bc.savefig(
            out_dir / f"halo_hex{hid_int}_profiles.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig_bc)
        print(f"  Saved → {out_dir / f'halo_hex{hid_int}_profiles.png'}")

    # ── Figure 6: Ground Truth vs Model Prediction (test hexels) ─────────
    GT_VS_PRED_CASES = [("12", 43), ("39", 52)]
    pred_dir = Path(args.pred_dir)
    if pred_dir.exists():
        print("Rendering GT vs Pred comparison figure …")
        _draw_gt_vs_pred_comparison(
            GT_VS_PRED_CASES,
            args.raw_data_dir,
            pred_dir,
            crop_half_px,
            out_dir / "halo_gt_vs_pred.png",
        )
    else:
        print(f"  Skipping GT vs Pred figure — pred_dir not found: {pred_dir}")


if __name__ == "__main__":
    main()
