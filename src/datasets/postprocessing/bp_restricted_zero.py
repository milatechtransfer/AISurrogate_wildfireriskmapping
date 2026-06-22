from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.paths import Paths, normalize_mask_scope
from data_preparation.spatial.utils import load_spatial_raster


def bp_nonfuel_restricted_ids(paths: Paths, hex_id: str) -> tuple[list[int], list[int], list[str]]:
    """Return fuel IDs whose BP should be deterministically zeroed."""
    fuel_table = pd.read_csv(paths.fuel_table(hex_id))
    restrictions = pd.read_csv(paths.tabular_dir / f"hex{hex_id}_IgnitionRestrictions.csv", keep_default_na=False)

    name_to_id = dict(zip(fuel_table["Name"].astype(str), fuel_table["ID"].astype(int), strict=False))
    nonfuel_ids = sorted(fuel_table.loc[fuel_table["Description"].astype(str).eq("Non-fuel"), "ID"].astype(int).tolist())

    restricted_names: list[str] = []
    for value in restrictions["FuelType"].astype(str):
        name = value.strip()
        if not name or name == "All" or name.upper() == "NA":
            continue
        restricted_names.append(name)

    missing_names = sorted({name for name in restricted_names if name not in name_to_id})
    if missing_names:
        raise ValueError(f"hex{hex_id}: restricted fuel names not found in FuelTypes table: {missing_names}")

    restricted_ids = sorted({name_to_id[name] for name in restricted_names})
    return sorted(set(nonfuel_ids) | set(restricted_ids)), nonfuel_ids, restricted_names


def zero_nonfuel_restricted_bp(
    pred_grid: np.ndarray,
    profile: dict[str, Any],
    raw_data_dir: str | Path,
    hex_id: str,
    mask_scope: str = "actual",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Set BP predictions to zero on non-fuel or ignition-restricted fuel support."""
    scope = normalize_mask_scope(mask_scope)
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    rule_ids, nonfuel_ids, restricted_names = bp_nonfuel_restricted_ids(paths, hex_id)

    fuel_grid, _ = load_spatial_raster(
        paths.fuel_grid(hex_id),
        mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=scope),
        reference_profile=profile,
    )
    if fuel_grid.shape != pred_grid.shape:
        raise ValueError(f"hex{hex_id}: fuel grid shape {fuel_grid.shape} does not match prediction shape {pred_grid.shape}")

    fuel_missing = np.ma.getmaskarray(fuel_grid)
    fuel_data = np.asarray(fuel_grid.data).round().astype(np.int32)
    rule_mask = (~fuel_missing) & np.isin(fuel_data, rule_ids)
    finite_pred = np.isfinite(pred_grid)
    zero_mask = finite_pred & rule_mask

    postprocessed = pred_grid.copy()
    removed_values = pred_grid[zero_mask]
    postprocessed[zero_mask] = 0.0

    stats = {
        "hex_id": hex_id,
        "mask_scope": scope,
        "finite_prediction_pixels": int(finite_pred.sum()),
        "nonfuel_or_restricted_pixels": int(rule_mask.sum()),
        "zeroed_prediction_pixels": int(zero_mask.sum()),
        "pct_finite_predictions_zeroed": 100.0 * int(zero_mask.sum()) / int(finite_pred.sum()) if int(finite_pred.sum()) else np.nan,
        "sum_prediction_removed": float(np.nansum(removed_values)) if removed_values.size else 0.0,
        "mean_prediction_before_zeroing": float(np.nanmean(removed_values)) if removed_values.size else np.nan,
        "rule_ids": ";".join(map(str, rule_ids)),
        "nonfuel_ids": ";".join(map(str, nonfuel_ids)),
        "restricted_names": "; ".join(restricted_names),
    }
    return postprocessed, stats
