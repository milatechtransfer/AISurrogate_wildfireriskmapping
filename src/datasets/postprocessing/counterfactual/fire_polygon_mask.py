"""Rasterize BurnP3+ fire perimeters into fuel-edit masks.

BurnP3+ writes *daily cumulative* perimeters, so a fire's final footprint is its
last `BurnDay`. Perimeters are also published in the simulation's own projection
(a hex-local UTM), which is **not** the Canada Lambert Conformal Conic grid the
model patches are cut from. Both facts are handled here: skipping either one
silently yields an empty or misaligned mask rather than a loud failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

FIRE_POLYGON_MASK_CSV_NAME = "fire_polygon_mask_summary.csv"
DEFAULT_LAYER = "daily_burn_perimeters"
DEFAULT_ITERATION_COL = "Iteration"
DEFAULT_FIRE_ID_COL = "FireID"
DEFAULT_BURN_DAY_COL = "BurnDay"
_SELECTION_KEYS = ("iteration", "fire_ids", "top_k_by_area")


@dataclass(frozen=True)
class FirePolygonMaskResult:
    """A rasterized fire-perimeter mask plus the per-fire audit behind it."""

    mask: np.ndarray
    summary: pd.DataFrame
    source_crs: str
    target_crs: str
    buffer_m: float

    @property
    def n_fires(self) -> int:
        return int(len(self.summary))

    @property
    def masked_pixels(self) -> int:
        return int(self.mask.sum())


def fire_polygon_mask_csv_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "fuel_intervention" / f"hex{hex_id}_{FIRE_POLYGON_MASK_CSV_NAME}"


def _require_columns(frame: pd.DataFrame, columns: list[str], *, source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}. Found: {sorted(frame.columns)}.")


def load_final_fire_perimeters(
    path: str | Path,
    *,
    layer: str = DEFAULT_LAYER,
    iteration_col: str = DEFAULT_ITERATION_COL,
    fire_id_col: str = DEFAULT_FIRE_ID_COL,
    burn_day_col: str = DEFAULT_BURN_DAY_COL,
    final_perimeter_only: bool = True,
) -> gpd.GeoDataFrame:
    """Read fire perimeters, reducing each fire to its final (max `BurnDay`) footprint.

    BurnP3+ perimeters are cumulative, so the last day already contains every
    earlier day; taking the max is a reduction, not an approximation. Set
    ``final_perimeter_only=False`` when reading an already-reduced layer such as
    ``final_burn_perimeters``, which need not contain a burn-day column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fire perimeter file not found: {path}.")

    frame = gpd.read_file(path, layer=layer)
    if frame.empty:
        raise ValueError(f"{path} layer {layer!r} contains no features.")
    if frame.crs is None:
        raise ValueError(f"{path} layer {layer!r} has no CRS; cannot reproject onto the model grid.")
    required_columns = [iteration_col, fire_id_col]
    if final_perimeter_only:
        required_columns.append(burn_day_col)
    _require_columns(frame, required_columns, source=path)

    if final_perimeter_only:
        final_index = frame.groupby([iteration_col, fire_id_col])[burn_day_col].idxmax()
        frame = frame.loc[final_index]
    return frame.sort_values([iteration_col, fire_id_col]).reset_index(drop=True)


def select_fire_perimeters(
    frame: gpd.GeoDataFrame,
    *,
    select: dict[str, Any] | None = None,
    iteration_col: str = DEFAULT_ITERATION_COL,
    fire_id_col: str = DEFAULT_FIRE_ID_COL,
) -> gpd.GeoDataFrame:
    """Select a subset of fires.

    Supported keys (at most one): `iteration` restricts to a single simulated
    season, `fire_ids` takes explicit `[iteration, fire_id]` pairs, and
    `top_k_by_area` keeps the k largest final footprints. Omitting `select`
    pools every fire.

    Only `iteration` and `fire_ids` are stable if the perimeter file is
    regenerated; `top_k_by_area` re-resolves against whatever is in the file.
    """
    select = dict(select or {})
    requested = [key for key in _SELECTION_KEYS if key in select]
    unknown = sorted(set(select) - set(_SELECTION_KEYS))
    if unknown:
        raise ValueError(f"Unknown fire polygon selection key(s) {unknown}; expected one of {list(_SELECTION_KEYS)}.")
    if len(requested) > 1:
        raise ValueError(f"Fire polygon selection accepts at most one of {list(_SELECTION_KEYS)}; got {requested}.")
    if not requested:
        return frame.reset_index(drop=True)

    key = requested[0]
    if key == "iteration":
        iteration = int(select["iteration"])
        selected = frame.loc[frame[iteration_col].astype(int) == iteration]
        if selected.empty:
            available = sorted(frame[iteration_col].astype(int).unique())
            raise ValueError(f"No fires found for iteration={iteration}. Available iterations: {available}.")
        return selected.reset_index(drop=True)

    if key == "fire_ids":
        pairs = [(int(iteration), int(fire_id)) for iteration, fire_id in select["fire_ids"]]
        if not pairs:
            raise ValueError("Fire polygon selection 'fire_ids' must not be empty.")
        keys = list(zip(frame[iteration_col].astype(int), frame[fire_id_col].astype(int), strict=True))
        wanted = set(pairs)
        selected = frame.loc[[key_pair in wanted for key_pair in keys]]
        found = set(zip(selected[iteration_col].astype(int), selected[fire_id_col].astype(int), strict=True))
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Fire polygon selection 'fire_ids' references fires absent from the file: {missing}.")
        return selected.reset_index(drop=True)

    top_k = int(select["top_k_by_area"])
    if top_k <= 0:
        raise ValueError(f"Fire polygon selection 'top_k_by_area' must be positive; got {top_k}.")
    if top_k > len(frame):
        raise ValueError(f"Fire polygon selection 'top_k_by_area'={top_k} exceeds the {len(frame)} available fires.")
    # Break area ties deterministically so the same file always yields the same subset.
    ordered = frame.assign(_area=frame.geometry.area).sort_values(["_area", iteration_col, fire_id_col], ascending=[False, True, True])
    return ordered.head(top_k).drop(columns="_area").reset_index(drop=True)


def build_fire_polygon_mask(
    *,
    params: dict[str, Any],
    reference_profile: dict[str, Any],
    hex_id: str,
) -> FirePolygonMaskResult:
    """Build a boolean edit mask on the model grid from configured fire perimeters."""
    params = dict(params)
    raw_path = params.pop("path", None)
    if raw_path is None:
        raise ValueError("Fire polygon masking requires a 'path' to the perimeter file.")
    path = Path(str(raw_path).format(hex_id=hex_id))

    layer = str(params.pop("layer", DEFAULT_LAYER))
    iteration_col = str(params.pop("iteration_col", DEFAULT_ITERATION_COL))
    fire_id_col = str(params.pop("fire_id_col", DEFAULT_FIRE_ID_COL))
    burn_day_col = str(params.pop("burn_day_col", DEFAULT_BURN_DAY_COL))
    final_perimeter_only = bool(params.pop("final_perimeter_only", True))
    buffer_m = float(params.pop("buffer_m", 0.0))
    all_touched = bool(params.pop("all_touched", False))
    select = params.pop("select", None)
    if params:
        raise ValueError(f"Unknown fire polygon parameter(s): {sorted(params)}.")

    perimeters = load_final_fire_perimeters(
        path,
        layer=layer,
        iteration_col=iteration_col,
        fire_id_col=fire_id_col,
        burn_day_col=burn_day_col,
        final_perimeter_only=final_perimeter_only,
    )
    selected = select_fire_perimeters(
        perimeters,
        select=select,
        iteration_col=iteration_col,
        fire_id_col=fire_id_col,
    )

    target_crs = rasterio.crs.CRS.from_user_input(reference_profile["crs"])
    source_crs = selected.crs
    projected = selected.to_crs(target_crs)

    if buffer_m < 0.0:
        raise ValueError(f"Fire polygon 'buffer_m' must be non-negative; got {buffer_m}.")
    geometries = projected.geometry.buffer(buffer_m) if buffer_m else projected.geometry

    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    transform = reference_profile["transform"]
    mask = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype(bool)

    if not mask.any():
        raise ValueError(
            f"Fire polygon mask for hex {hex_id} is empty after reprojecting {len(selected)} fire(s) from "
            f"{source_crs.to_string()} onto the model grid. Check that the perimeter file covers this hexel."
        )

    summary = _build_summary(
        projected=projected,
        geometries=geometries,
        transform=transform,
        height=height,
        width=width,
        all_touched=all_touched,
        iteration_col=iteration_col,
        fire_id_col=fire_id_col,
        hex_id=hex_id,
    )
    return FirePolygonMaskResult(
        mask=mask,
        summary=summary,
        source_crs=source_crs.to_string(),
        target_crs=target_crs.to_string(),
        buffer_m=buffer_m,
    )


def _build_summary(
    *,
    projected: gpd.GeoDataFrame,
    geometries: gpd.GeoSeries,
    transform: Any,
    height: int,
    width: int,
    all_touched: bool,
    iteration_col: str,
    fire_id_col: str,
    hex_id: str,
) -> pd.DataFrame:
    """One audit row per selected fire, so coverage can be checked without re-deriving it."""
    grid = box(*rasterio.transform.array_bounds(height, width, transform))
    rows: list[dict[str, Any]] = []
    for (_, record), geometry in zip(projected.iterrows(), geometries, strict=True):
        rows.append(
            {
                "hex_id": hex_id,
                "iteration": int(record[iteration_col]),
                "fire_id": int(record[fire_id_col]),
                "final_area_ha": float(record.geometry.area) / 10_000.0,
                "buffered_area_ha": float(geometry.area) / 10_000.0,
                "mask_pixels": _count_pixels(geometry, transform, height, width, all_touched=all_touched),
                "fully_within_grid": bool(geometry.within(grid)),
            }
        )
    return pd.DataFrame(rows)


def _count_pixels(geometry: Any, transform: Any, height: int, width: int, *, all_touched: bool) -> int:
    """Pixel count for one geometry, rasterized only within its own bounding window."""
    min_x, min_y, max_x, max_y = geometry.bounds
    rows, cols = rasterio.transform.rowcol(
        transform,
        [min_x, min_x, max_x, max_x],
        [min_y, max_y, min_y, max_y],
        op=float,
    )
    row_start = max(int(np.floor(min(rows))) - 1, 0)
    row_stop = min(int(np.ceil(max(rows))) + 1, height)
    col_start = max(int(np.floor(min(cols))) - 1, 0)
    col_stop = min(int(np.ceil(max(cols))) + 1, width)
    if row_stop <= row_start or col_stop <= col_start:
        return 0

    window = rasterio.windows.Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    patch = rasterize(
        [(geometry, 1)],
        out_shape=(int(window.height), int(window.width)),
        transform=rasterio.windows.transform(window, transform),
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return int(patch.sum())
