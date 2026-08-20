import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.paths import Paths
from data_preparation.spatial.utils import (
    fire_cause_label_mapping,
    fire_cause_mapping,
    load_spatial_raster,
)

logger = logging.getLogger(__name__)

IGNITION_PROBABILITY_MASS_SCALE = 1_000_000.0


def load_ignition_grid(
    root_dir: str,
    hex_id: str,
    cause: int = None,
    season: int = None,
    reference_profile: dict[str, Any] | None = None,
    mask_scope: str = "actual",
) -> np.ma.MaskedArray:
    """Load ignition grids for a specific season/cause or all seasons/causes"""
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ignition_grids_folder_path = all_paths.ignition_prob_dir()

    if season and cause and hex_id:
        file_name = f"hex{hex_id}_ignGrid_{fire_cause_mapping[cause]}_s{season}.tif"
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            mask_path=all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
            reference_profile=reference_profile,
        )

        return ignition_raster

    # else, loop over all ignition grids (across causes/seasons) and output one grid
    ignition_raster_files = [f for f in os.listdir(ignition_grids_folder_path) if f.endswith(".tif")]
    out_ignition_grids = []
    for file_name in ignition_raster_files:
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            mask_path=all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
            reference_profile=reference_profile,
        )
        out_ignition_grids.append(ignition_raster)

    # size (H, W)
    out_ignition_grids = np.ma.stack(out_ignition_grids, axis=0)  # type: ignore
    max_ignition_grid = np.ma.max(out_ignition_grids, axis=0)
    return max_ignition_grid


# IgnitionDistribution.csv cause label -> grid cause letter
_csv_cause_to_letter = {label: letter for letter, label in fire_cause_label_mapping.items()}
# matches e.g. "hex02_ignGrid_H_s3.tif" or "hex100_ignGrid_H_Spring.tif"
# Season token is any non-dot sequence after the cause letter.
_IGN_GRID_PATTERN = re.compile(r"_ignGrid_([A-Z])_([^.]+)\.tif$")


def build_ignition_probability_mass_channels(
    *,
    grids: dict[tuple[str, str], np.ma.MaskedArray],
    distribution_frame: pd.DataFrame,
    firezones_grid: np.ma.MaskedArray,
    zone_name_to_id: dict[str, int],
    scale: float = IGNITION_PROBABILITY_MASS_SCALE,
) -> np.ma.MaskedArray:
    """Build cause-specific location mass for one BurnP3+-sampled ignition."""
    if scale <= 0.0:
        raise ValueError(f"Ignition probability-mass scale must be positive, got {scale}.")
    if not grids:
        raise ValueError("At least one ignition grid is required.")

    frame = distribution_frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    required_columns = {"Season", "Cause", "FireZone", "RelativeLikelihood"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Ignition-distribution table is missing columns: {sorted(missing_columns)}")
    frame["RelativeLikelihood"] = pd.to_numeric(frame["RelativeLikelihood"], errors="raise")
    if (frame["RelativeLikelihood"] < 0.0).any():
        raise ValueError("Ignition-distribution table contains negative relative likelihoods.")

    ref = next(iter(grids.values()))
    shape = ref.shape
    if firezones_grid.shape != shape or any(grid.shape != shape for grid in grids.values()):
        raise ValueError("Ignition grids and fire-zone grid must share one spatial shape.")

    zone_data = np.ma.getdata(firezones_grid)
    landscape_mask = np.ma.getmaskarray(firezones_grid).copy()
    for grid in grids.values():
        landscape_mask |= np.ma.getmaskarray(grid)

    components: list[tuple[float, str, tuple[str, str], int, float]] = []
    for _, row in frame.iterrows():
        likelihood = float(row["RelativeLikelihood"])
        if likelihood <= 0.0:
            continue
        season = str(row["Season"]).strip()
        cause_label = str(row["Cause"]).strip()
        zone_name = str(row["FireZone"]).strip()
        cause_letter = _csv_cause_to_letter.get(cause_label)
        zone_id = zone_name_to_id.get(zone_name)
        if cause_letter is None or not season or zone_id is None:
            logger.warning(
                "Dropping unresolved ignition category season=%r, cause=%r, firezone=%r.",
                season,
                cause_label,
                zone_name,
            )
            continue
        category_grid = grids.get((cause_letter, season))
        if category_grid is None:
            logger.warning("Dropping ignition category with no grid: cause=%s, season=%s.", cause_letter, season)
            continue

        values = np.asarray(np.ma.getdata(category_grid))
        grid_mask = np.ma.getmaskarray(category_grid)
        if np.any(values[~grid_mask] < 0.0):
            raise ValueError(f"Ignition grid {(cause_letter, season)} contains negative values.")
        valid = (~landscape_mask) & (zone_data == zone_id) & np.isfinite(values) & (values > 0.0)
        spatial_total = float(values[valid].sum(dtype=np.float64))
        if spatial_total <= 0.0:
            logger.warning(
                "Dropping ignition category with no positive spatial mass: cause=%s, season=%s, firezone=%s.",
                cause_letter,
                season,
                zone_name,
            )
            continue
        components.append((likelihood, cause_letter, (cause_letter, season), zone_id, spatial_total))

    total_likelihood = sum(likelihood for likelihood, _, _, _, _ in components)
    if total_likelihood <= 0.0:
        raise ValueError("No valid positive-likelihood ignition categories remain after spatial validation.")

    cause_channels = {cause_letter: np.zeros(shape, dtype=np.float64) for cause_letter in fire_cause_mapping.values()}
    for likelihood, cause_letter, grid_key, zone_id, spatial_total in components:
        values = np.asarray(np.ma.getdata(grids[grid_key]))
        valid = (~landscape_mask) & (zone_data == zone_id) & np.isfinite(values) & (values > 0.0)
        cause_channels[cause_letter][valid] += (likelihood / total_likelihood) * values[valid] / spatial_total

    total_mass = sum(float(channel.sum()) for channel in cause_channels.values())
    if not np.isclose(total_mass, 1.0, rtol=1e-6, atol=1e-8):
        raise RuntimeError(f"Per-ignition spatial mass must sum to one, got {total_mass}.")

    stacked = np.stack(
        [cause_channels[cause_letter] * scale for cause_letter in fire_cause_mapping.values()],
        axis=-1,
    ).astype(np.float32)
    return np.ma.array(stacked, mask=np.broadcast_to(landscape_mask[:, :, None], stacked.shape))


def load_ignition_grid_probability_mass(
    root_dir: str,
    hex_id: str,
    firezones_grid: np.ma.MaskedArray,
    reference_profile: dict[str, Any] | None = None,
    mask_scope: str = "actual",
) -> np.ma.MaskedArray:
    """Load Human/Lightning location mass per million sampled ignitions."""
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ign_dir = all_paths.ignition_prob_dir()
    mask_path = all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope)

    grids: dict[tuple[str, str], np.ma.MaskedArray] = {}
    known_causes = set(fire_cause_mapping.values())
    if ign_dir.is_dir():
        for fname in sorted(os.listdir(ign_dir)):
            match = _IGN_GRID_PATTERN.search(fname)
            if match is None or match.group(1) not in known_causes:
                continue
            cause_letter, season = match.group(1), match.group(2)
            raster, _ = load_spatial_raster(
                path=ign_dir / fname,
                mask_path=mask_path,
                reference_profile=reference_profile,
            )
            grids[(cause_letter, season)] = raster
    if not grids:
        raise FileNotFoundError(f"No ignition TIFs found in {ign_dir} for hex{hex_id}")

    firezones_frame = pd.read_csv(all_paths.firezones_table(hex_id=hex_id))
    firezones_frame.columns = [str(column).strip() for column in firezones_frame.columns]
    if not {"Name", "ID"} <= set(firezones_frame.columns):
        raise ValueError(f"Fire-zone table must contain Name and ID columns: {list(firezones_frame.columns)}")
    zone_name_to_id = {str(row["Name"]).strip(): int(row["ID"]) for _, row in firezones_frame.iterrows() if str(row["Name"]).strip()}
    distribution_frame = pd.read_csv(all_paths.ignition_distribution_table(hex_id=hex_id))
    return build_ignition_probability_mass_channels(
        grids=grids,
        distribution_frame=distribution_frame,
        firezones_grid=firezones_grid,
        zone_name_to_id=zone_name_to_id,
    )


def load_ignition_grid_weighted(
    root_dir: str,
    hex_id: str,
    firezones_grid: np.ma.MaskedArray,
    reference_profile: dict[str, Any] | None = None,
    mask_scope: str = "actual",
) -> np.ma.MaskedArray:
    """Load ignition grids as zone-area-weighted Human and Lightning channels.

    For each hex, IgnitionDistribution.csv gives a RelativeLikelihood per
    (Season, Cause, FireZone). We weight every (cause, season) ignition grid
    present for the hex by averaging its RelativeLikelihoods across fire zones
    using their pixel-area fractions, normalise the weights to sum to 1, then
    blend the grids per cause into one channel each:

        human_channel     = sum_s w_H_s * grid_H_s
        lightning_channel = sum_s w_N_s * grid_N_s

    Season tokens are read directly from TIF filenames and the CSV — both use
    the same naming convention (e.g. "s1"/"s2"/"s3" or "Spring"/"Summer-Fall").
    Returns a masked array of shape (H, W, 2): channel 0 = Human,
    channel 1 = Lightning. Falls back to uniform weights if the CSV is missing
    or all weights are zero.
    """
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ign_dir = all_paths.ignition_prob_dir()
    mask_path = all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope)

    # ── 1. Discover the (cause, season) ignition grids present for this hex ───
    # Season count varies per hex, and some hexels lack a whole cause (e.g.
    # hex54 has no lightning grids), so we read whatever TIFs are on disk.
    grids: dict[tuple[str, str], np.ma.MaskedArray] = {}
    known_causes = set(fire_cause_mapping.values())
    if ign_dir.is_dir():
        for fname in sorted(os.listdir(ign_dir)):
            match = _IGN_GRID_PATTERN.search(fname)
            if match is None or match.group(1) not in known_causes:
                continue
            cause_letter, season_str = match.group(1), match.group(2)
            raster, _ = load_spatial_raster(
                path=ign_dir / fname,
                mask_path=mask_path,
                reference_profile=reference_profile,
            )
            grids[(cause_letter, season_str)] = raster

    if not grids:
        raise FileNotFoundError(f"No ignition TIFs found in {ign_dir} for hex{hex_id}")

    # ── 2. Compute area fractions over valid, non-zero zone pixels ─────────────
    zone_data = firezones_grid.data if isinstance(firezones_grid, np.ma.MaskedArray) else firezones_grid
    zone_mask = (
        np.ma.getmaskarray(firezones_grid) if isinstance(firezones_grid, np.ma.MaskedArray) else np.zeros_like(zone_data, dtype=bool)
    )
    valid_zone_pixels = (~zone_mask) & (zone_data > 0)
    zone_ids_valid = zone_data[valid_zone_pixels]

    area_frac: dict[int, float] = {}
    total_valid = zone_ids_valid.size
    if total_valid > 0:
        unique_ids, counts = np.unique(zone_ids_valid, return_counts=True)
        for zid, cnt in zip(unique_ids, counts, strict=True):
            area_frac[int(zid)] = cnt / total_valid

    # ── 3. Load zone name → ID mapping from FireZones.csv ────────────────────
    fz_csv = all_paths.firezones_table(hex_id=hex_id)
    zone_name_to_id: dict[str, int] = {}
    if fz_csv.exists():
        fz_df = pd.read_csv(fz_csv)
        # Expected columns: Name, ID  (may have Description, Color)
        for _, row in fz_df.iterrows():
            name = str(row.get("Name", row.get("name", ""))).strip()
            zone_id = row.get("ID", row.get("id", None))
            if name and zone_id is not None:
                zone_name_to_id[name] = int(zone_id)

    # ── 4. Compute hex-level weights from IgnitionDistribution.csv ───────────
    ign_csv = all_paths.ignition_distribution_table(hex_id=hex_id)
    weights: dict[tuple[str, str], float] = {key: 0.0 for key in grids}

    if ign_csv.exists() and zone_name_to_id and area_frac:
        dist_df = pd.read_csv(ign_csv)
        # Normalise column names
        dist_df.columns = [col.strip() for col in dist_df.columns]
        for _, row in dist_df.iterrows():
            cause_csv = str(row.get("Cause", "")).strip()  # "Human" or "Lightning"
            season_csv = str(row.get("Season", "")).strip()  # "s1", "s2", "s3", ...
            zone_name = str(row.get("FireZone", "")).strip()  # "fru21" etc.
            try:
                rl = float(row.get("RelativeLikelihood", 0) or 0)
            except (ValueError, TypeError):
                continue

            zone_id = zone_name_to_id.get(zone_name)
            if zone_id is None or zone_id not in area_frac:
                continue

            # Map CSV cause/season back to our (cause_letter, season_str) key.
            # Season token in the CSV always matches the token in the TIF filename
            # (e.g. "s1" or "Spring" or "Summer-Fall").
            mapped_cause = _csv_cause_to_letter.get(cause_csv)
            mapped_season = season_csv if season_csv else None
            if mapped_cause is None or mapped_season is None:
                continue

            key = (mapped_cause, mapped_season)
            # Only accumulate weight for (cause, season) grids that exist on disk
            if key in weights:
                weights[key] += area_frac[zone_id] * rl
    else:
        logger.warning(
            "Hex %s: ignition distribution unavailable (table exists=%s, zone map=%s, area fractions=%s); "
            "falling back to uniform ignition weights.",
            hex_id,
            ign_csv.exists(),
            bool(zone_name_to_id),
            bool(area_frac),
        )
        # Fallback: uniform weights across present grids
        for key in grids:
            weights[key] = 1.0

    # ── 5. Normalise the weights over the grids present for this hex ──────────
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: w / total_w for k, w in weights.items()}
    else:
        n_present = len(weights)
        weights = {k: 1.0 / n_present for k in weights}

    # ── 6. Blend the per-season grids into one channel per cause ─────────────
    ref = next(iter(grids.values()))

    def _blend(keys: list[tuple[str, str]]) -> np.ma.MaskedArray:
        result = np.ma.zeros_like(ref)
        for k in keys:
            result = result + weights[k] * grids[k]
        return result

    channels = [_blend([k for k in grids if k[0] == cause_letter]) for cause_letter in fire_cause_mapping.values()]

    H, W = ref.shape
    out = np.ma.stack(channels, axis=-1)  # (H, W, num_causes)
    assert out.shape == (H, W, len(fire_cause_mapping))
    return out
