from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.errors
import rasterio.windows
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.transform import Affine

from data_preparation.paths import Paths
from data_preparation.spatial import load_fuel_grid, load_ignition_grid, load_spatial_raster
from data_preparation.spatial.utils import FUEL_GROUP_MAP
from data_preparation.utils import MaskType, find_hex_ids, resolve_mask_path

log = logging.getLogger(__name__)

InputGridName = Literal["fuel", "elevation", "ignition", "firezones"]
ALL_INPUT_GRIDS: list[InputGridName] = ["fuel", "elevation", "ignition", "firezones"]


NATIONAL_CRS = "ESRI:102002"


def load_input_grids_per_hexel(
    root_dir: str,
    hex_id: str,
    input_grids: list[InputGridName] | None = None,
    mask_type: MaskType = "actual",
) -> tuple[dict[str, np.ma.MaskedArray], dict[str, Any]]:
    """Load a configurable subset of spatial input grids for a hexel.

    Args:
        root_dir: Root directory containing all hexels.
        hex_id: Hexel ID to load.
        input_grids: List of grid names to load. Valid values are ``"fuel"``,
            ``"elevation"``, ``"ignition"``, and ``"firezones"``.  Defaults to
            all four grids when *None*.
        mask_type: Which hexel boundary to use for clipping — ``"actual"``
            (tight hexel boundary) or ``"buffer"`` (buffered boundary).
            Defaults to ``"actual"``.

    Returns:
        A tuple ``(grids, reference_profile)`` where *grids* maps each
        requested grid name to its loaded masked array of shape ``(H, W)``,
        and *reference_profile* is the shared rasterio profile (CRS, transform,
        size) that all grids in this hexel are aligned to.

    Raises:
        ValueError: If any name in *input_grids* is not a recognised grid.
    """
    if input_grids is None:
        input_grids = list(ALL_INPUT_GRIDS)

    invalid = set(input_grids) - set(ALL_INPUT_GRIDS)
    if invalid:
        raise ValueError(f"Unknown input grid(s): {invalid}. Valid options: {ALL_INPUT_GRIDS}")

    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    mask_path = resolve_mask_path(all_paths, hex_id=hex_id, mask_type=mask_type)

    # Elevation is always loaded first because its profile is used as the
    # reprojection reference for every other grid.
    elevation_grid, reference_profile = load_spatial_raster(
        path=all_paths.elevation_grid(hex_id=hex_id),
        mask_path=mask_path,
    )

    grids: dict[str, np.ma.MaskedArray] = {}

    if "elevation" in input_grids:
        grids["elevation"] = elevation_grid

    if "fuel" in input_grids:
        grids["fuel"] = load_fuel_grid(root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile, mask_path=mask_path)

    if "ignition" in input_grids:
        grids["ignition"] = load_ignition_grid(root_dir=root_dir, hex_id=hex_id, reference_profile=reference_profile, mask_path=mask_path)

    if "firezones" in input_grids:
        firezones_grid, _ = load_spatial_raster(
            path=all_paths.firezones_grid(hex_id=hex_id),
            mask_path=mask_path,
            reference_profile=reference_profile,
        )
        grids["firezones"] = firezones_grid

    return grids, reference_profile


def _read_reference_tif_geometry(
    reference_tif: str | Path,
) -> tuple[int, int, Affine, CRS]:
    """Read width, height, transform, and CRS from an existing GeoTIFF.

    Args:
        reference_tif: Path to the reference GeoTIFF whose grid geometry
            (projection, resolution, extent) should be reused.

    Returns:
        ``(width, height, transform, crs)``
    """
    with rasterio.open(reference_tif) as src:
        return src.width, src.height, src.transform, src.crs


def _sample_resolution(root_dir: str, hex_ids: list[str], n_samples: int = 10) -> float:
    """Estimate pixel resolution by reading elevation raster headers for a few hexels."""
    all_res: list[float] = []
    for hex_id in hex_ids[:n_samples]:
        try:
            path = Paths(hex_id=hex_id, root_dir=root_dir).elevation_grid(hex_id=hex_id)
            with rasterio.open(path) as src:
                all_res.append(abs(src.transform.a))
        except (FileNotFoundError, rasterio.errors.RasterioIOError):
            continue
    if not all_res:
        raise ValueError(f"Could not determine pixel resolution from any hexel in {root_dir!r}")
    return float(np.median(all_res))


def _rasterize_geometry(geom: Any, win_h: int, win_w: int, win_transform: Affine) -> np.ndarray:
    """Return a boolean array (True = inside *geom*) for the given window."""
    if geom is None:
        return np.zeros((win_h, win_w), dtype=bool)
    return geometry_mask(
        [geom.__geo_interface__],
        out_shape=(win_h, win_w),
        transform=win_transform,
        invert=True,
    )


def _load_raw_grids_per_hexel(
    root_dir: str,
    hex_id: str,
    input_grids: list[InputGridName],
    reference_profile: dict[str, Any],
) -> dict[str, np.ma.MaskedArray]:
    """Load hexel grids without any masking, reprojected to *reference_profile*.

    All grids are reprojected directly to the provided reference profile
    (CRS, transform, width, height), so callers receive arrays already aligned
    to the national grid window — no second reproject step is needed.
    """
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)

    elevation_grid, _ = load_spatial_raster(
        path=all_paths.elevation_grid(hex_id=hex_id),
        mask_path=None,
        reference_profile=reference_profile,
    )

    grids: dict[str, np.ma.MaskedArray] = {}

    if "elevation" in input_grids:
        grids["elevation"] = elevation_grid

    if "fuel" in input_grids:
        raw_fuel, _ = load_spatial_raster(
            path=all_paths.fuel_grid(hex_id=hex_id),
            mask_path=None,
            reference_profile=reference_profile,
        )
        data = raw_fuel.data
        fuel_mask = np.ma.getmaskarray(raw_fuel)
        grouped = np.full(data.shape, -1, dtype=np.int16)
        for fuel_id, group_id in FUEL_GROUP_MAP.items():
            grouped[data == fuel_id] = group_id
        grids["fuel"] = np.ma.masked_array(grouped, mask=fuel_mask)

    if "ignition" in input_grids:
        ignition_dir = all_paths.ignition_prob_dir()
        files = [f for f in os.listdir(ignition_dir) if f.endswith(".tif")]
        stacked = []
        for fname in files:
            g, _ = load_spatial_raster(
                path=ignition_dir / fname,
                mask_path=None,
                reference_profile=reference_profile,
            )
            stacked.append(g)
        grids["ignition"] = np.ma.max(np.ma.stack(stacked, axis=0), axis=0)

    if "firezones" in input_grids:
        firezones_grid, _ = load_spatial_raster(
            path=all_paths.firezones_grid(hex_id=hex_id),
            mask_path=None,
            reference_profile=reference_profile,
        )
        grids["firezones"] = firezones_grid

    return grids


def build_national_grid(
    root_dir: str,
    output_dir: str,
    actual_mask_shp: str | Path,
    buffer_mask_shp: str | Path,
    hex_id_field: str = "hexid",
    input_grids: list[InputGridName] | None = None,
    national_crs: str = NATIONAL_CRS,
    reference_tif: str | Path | None = None,
) -> dict[str, Path]:
    """Build a national-level GeoTIFF for each requested input grid.

    Iterates over all hexels under *root_dir*, warps each hexel grid into a
    shared national CRS, and mosaics them using national-level shapefiles to
    define which pixels belong to each hexel's actual (tight) and buffer
    boundaries.  When pixels overlap between adjacent hexels the following
    priority rules apply:

    * **Actual-mask pixels always win over buffer-only pixels.**
    * **Pixels not covered by any actual mask** are filled from the buffer
      mask so that no gaps appear between hexels.
    * **Same-priority conflicts** (actual vs. actual, or buffer-only vs.
      buffer-only) pick a value at random and are flagged in the companion
      ``*_conflict_mask.tif`` outputs.

    Each output GeoTIFF uses LZW compression and NaN as the nodata value.

    Args:
        root_dir: Root directory containing all hexel sub-directories.
        output_dir: Directory where output GeoTIFFs are written.
        actual_mask_shp: Path to the national shapefile containing each
            hexel's actual (tight) boundary polygons.
        buffer_mask_shp: Path to the national shapefile containing each
            hexel's buffered boundary polygons.
        hex_id_field: Attribute field in both shapefiles that matches the
            hexel IDs found under *root_dir*.  Defaults to ``"hexid"``.
        input_grids: Grid names to mosaic (defaults to all four).
        national_crs: Target CRS string used when *reference_tif* is not
            provided (default ESRI:102002).  Ignored when *reference_tif* is
            given.
        reference_tif: Optional path to an existing national GeoTIFF whose
            exact grid geometry (CRS, transform, width, height) will be
            adopted for all outputs.  When supplied, *national_crs* is
            ignored and Phase 1 extent computation is skipped entirely.

    Returns:
        A dict mapping each grid name to the path of the saved GeoTIFF.
    """
    if input_grids is None:
        input_grids = list(ALL_INPUT_GRIDS)

    invalid = set(input_grids) - set(ALL_INPUT_GRIDS)
    if invalid:
        raise ValueError(f"Unknown input grid(s): {invalid}. Valid options: {ALL_INPUT_GRIDS}")

    hex_ids = find_hex_ids(root_dir)
    if not hex_ids:
        raise ValueError(f"No hexels found in {root_dir!r}")

    # --- Phase 1: determine national grid geometry ---
    if reference_tif is not None:
        nat_width, nat_height, nat_transform, nat_crs = _read_reference_tif_geometry(reference_tif)
        log.info(
            "National grid from reference TIF %s: %d × %d px, CRS=%s",
            reference_tif,
            nat_width,
            nat_height,
            nat_crs,
        )
    else:
        tmp_gdf = gpd.read_file(buffer_mask_shp).to_crs(national_crs)
        nat_left, nat_bottom, nat_right, nat_top = tmp_gdf.total_bounds
        del tmp_gdf

        res = _sample_resolution(root_dir, hex_ids)
        nat_left = math.floor(nat_left / res) * res
        nat_bottom = math.floor(nat_bottom / res) * res
        nat_right = math.ceil(nat_right / res) * res
        nat_top = math.ceil(nat_top / res) * res

        nat_width = int(round((nat_right - nat_left) / res))
        nat_height = int(round((nat_top - nat_bottom) / res))
        nat_transform = Affine(res, 0.0, nat_left, 0.0, -res, nat_top)
        nat_crs = CRS.from_user_input(national_crs)
        log.info(
            "National grid (computed): %d × %d px at %.1f m/px, CRS=%s",
            nat_width,
            nat_height,
            res,
            national_crs,
        )

    # Load shapefiles (already in the national CRS per the user's convention;
    # to_crs is a safe no-op if they already match).
    actual_gdf = gpd.read_file(actual_mask_shp).to_crs(nat_crs).set_index(hex_id_field)
    buffer_gdf = gpd.read_file(buffer_mask_shp).to_crs(nat_crs).set_index(hex_id_field)

    # --- Phase 2: mosaic hexels with overlap reporting ---
    # result_arrays: current best value per pixel (NaN = not yet written).
    # source_arrays: which hexel last wrote each pixel (-1 = unwritten).
    # priority_arrays: 1 = written from actual mask, 0 = written from buffer-only, -1 = unwritten.
    # conflict_arrays: 1 where same-priority hexels disagreed on a value.
    # source_dtypes: natural dtype of each grid inferred from the first loaded hexel;
    #   used in Phase 3 to decide whether to write integer or float output.
    result_arrays: dict[str, np.ndarray] = {name: np.full((nat_height, nat_width), np.nan, dtype=np.float64) for name in input_grids}
    source_arrays: dict[str, np.ndarray] = {name: np.full((nat_height, nat_width), -1, dtype=np.int32) for name in input_grids}
    conflict_arrays: dict[str, np.ndarray] = {name: np.zeros((nat_height, nat_width), dtype=np.uint8) for name in input_grids}
    priority_arrays: dict[str, np.ndarray] = {name: np.full((nat_height, nat_width), -1, dtype=np.int8) for name in input_grids}
    source_dtypes: dict[str, np.dtype] = {}
    hex_id_to_idx = {hid: i for i, hid in enumerate(hex_ids)}

    nat_window_full = rasterio.windows.Window(0, 0, nat_width, nat_height)
    skipped: list[str] = []

    for hex_id in hex_ids:
        hex_key = int(hex_id)
        if hex_key not in buffer_gdf.index:
            log.warning("Hex %s not found in buffer shapefile — skipping", hex_id)
            skipped.append(hex_id)
            continue

        buf_geom = buffer_gdf.loc[hex_key, "geometry"]
        act_geom = actual_gdf.loc[hex_key, "geometry"] if hex_key in actual_gdf.index else None
        if act_geom is None:
            log.warning("Hex %s not found in actual shapefile — all pixels treated as buffer-only", hex_id)

        try:
            # Compute the destination window from the buffer geometry bounds
            # BEFORE loading grids, so all grids can be reprojected directly
            # to the national window in one step (no second reproject needed).
            buf_bounds = buf_geom.bounds  # (minx, miny, maxx, maxy)
            dst_window = rasterio.windows.from_bounds(*buf_bounds, transform=nat_transform).round_offsets().round_shape()
            dst_window = dst_window.intersection(nat_window_full)
            if dst_window.width <= 0 or dst_window.height <= 0:
                log.warning("Hex %s projects outside the national extent — skipping", hex_id)
                skipped.append(hex_id)
                continue

            win_h = int(dst_window.height)
            win_w = int(dst_window.width)
            row_off = int(dst_window.row_off)
            col_off = int(dst_window.col_off)
            win_transform = rasterio.windows.transform(dst_window, nat_transform)

            nat_window_profile: dict[str, Any] = {
                "crs": nat_crs,
                "transform": win_transform,
                "width": win_w,
                "height": win_h,
            }
            grids_raw = _load_raw_grids_per_hexel(root_dir, hex_id, input_grids, nat_window_profile)
        except (FileNotFoundError, rasterio.errors.RasterioIOError) as exc:
            log.warning("Skipping hex %s: %s", hex_id, exc)
            skipped.append(hex_id)
            continue

        # Rasterize the hexel's geometries into boolean masks for this window.
        is_buf_mask = _rasterize_geometry(buf_geom, win_h, win_w, win_transform)
        is_act_mask = _rasterize_geometry(act_geom, win_h, win_w, win_transform)

        for grid_name in input_grids:
            raw_array = grids_raw[grid_name]
            # Record the source dtype once so Phase 3 can choose the right output type.
            if grid_name not in source_dtypes:
                source_dtypes[grid_name] = raw_array.dtype
            # Grids are already reprojected to the national window — no second
            # reproject step needed.
            dst_reprojected = np.where(np.ma.getmaskarray(raw_array), np.nan, raw_array.data).astype(np.float64)

            # Apply shapefile masks to the reprojected data.
            dst_buf = np.where(is_buf_mask, dst_reprojected, np.nan)
            dst_act = np.where(is_act_mask, dst_reprojected, np.nan)

            is_buf_valid = np.isfinite(dst_buf)
            is_act_valid = np.isfinite(dst_act)
            new_valid = is_buf_valid | is_act_valid
            new_values = dst_buf.copy()
            new_values[is_act_valid] = dst_act[is_act_valid]
            new_priority = is_act_valid.astype(np.int8)  # 1 = actual, 0 = buffer-only

            result_slice = result_arrays[grid_name][row_off : row_off + win_h, col_off : col_off + win_w]
            source_slice = source_arrays[grid_name][row_off : row_off + win_h, col_off : col_off + win_w]
            priority_slice = priority_arrays[grid_name][row_off : row_off + win_h, col_off : col_off + win_w]
            conflict_slice = conflict_arrays[grid_name][row_off : row_off + win_h, col_off : col_off + win_w]

            existing_valid = np.isfinite(result_slice)
            existing_actual = existing_valid & (priority_slice == 1)
            existing_buf_only = existing_valid & (priority_slice == 0)

            # 1. Unwritten pixels: write freely and record priority.
            new_only = new_valid & ~existing_valid
            result_slice[new_only] = new_values[new_only]
            source_slice[new_only] = hex_id_to_idx[hex_id]
            priority_slice[new_only] = new_priority[new_only]

            # 2. Actual overwrites buffer-only (priority promotion — not a conflict).
            promote_mask = is_act_valid & existing_buf_only
            if promote_mask.any():
                prior_hex_ids = [hex_ids[i] for i in np.unique(source_slice[promote_mask]) if i >= 0]
                log.debug(
                    "[promote] hex=%s promoted over buffer-only data from %s on grid=%s: %d px",
                    hex_id,
                    prior_hex_ids,
                    grid_name,
                    int(promote_mask.sum()),
                )
                result_slice[promote_mask] = new_values[promote_mask]
                source_slice[promote_mask] = hex_id_to_idx[hex_id]
                priority_slice[promote_mask] = 1

            # 3. Same-priority conflicts (actual-actual or buffer-only-buffer-only).
            #    New buffer-only pixels landing on existing actual pixels are silently skipped.
            act_act = is_act_valid & existing_actual
            buf_buf = (~is_act_valid & is_buf_valid) & existing_buf_only
            conflict_candidate = act_act | buf_buf
            if conflict_candidate.any():
                prior_hex_ids = [hex_ids[i] for i in np.unique(source_slice[conflict_candidate]) if i >= 0]
                same_mask = conflict_candidate & np.isclose(result_slice, new_values, rtol=0, atol=1e-6)
                diff_mask = conflict_candidate & ~same_mask
                same_count = int(same_mask.sum())
                diff_count = int(diff_mask.sum())

                print(
                    f"[overlap] hex={hex_id} overlaps with {prior_hex_ids} on grid={grid_name}: {int(conflict_candidate.sum())} overlapping pixels"
                )
                if same_count > 0:
                    print(f"  → {same_count} pixels have the same value — keeping as-is")
                if diff_count > 0:
                    print(f"  → {diff_count} pixels differ — randomly choosing one value")
                    random_pick = np.random.randint(0, 2, size=diff_mask.shape, dtype=bool)
                    choose_new = diff_mask & random_pick
                    result_slice[choose_new] = new_values[choose_new]
                    source_slice[choose_new] = hex_id_to_idx[hex_id]
                    priority_slice[choose_new] = new_priority[choose_new]
                    conflict_slice[diff_mask] = 1

    if skipped:
        log.warning("Skipped %d hexel(s) in total: %s", len(skipped), skipped)

    # --- Phase 3: write one GeoTIFF per grid + one conflict mask per grid ---
    os.makedirs(output_dir, exist_ok=True)
    output_paths: dict[str, Path] = {}

    _INT_NODATA = -1

    base_profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": nat_width,
        "height": nat_height,
        "count": 1,
        "crs": nat_crs,
        "transform": nat_transform,
        "compress": "lzw",
    }
    conflict_profile = {**base_profile, "dtype": "uint8", "nodata": None}

    for grid_name in input_grids:
        src_dtype = source_dtypes.get(grid_name, np.dtype("float32"))
        if np.issubdtype(src_dtype, np.integer):
            out_array = np.where(np.isnan(result_arrays[grid_name]), _INT_NODATA, result_arrays[grid_name]).astype(np.int16)
            profile_to_use = {**base_profile, "dtype": "int16", "nodata": _INT_NODATA}
        else:
            out_array = result_arrays[grid_name].astype(np.float32)
            profile_to_use = {**base_profile, "dtype": "float32", "nodata": float("nan")}

        out_path = Path(output_dir) / f"national_{grid_name}.tif"
        with rasterio.open(out_path, "w", **profile_to_use) as dst:
            dst.write(out_array, 1)
        log.info("Saved %s → %s", grid_name, out_path)
        output_paths[grid_name] = out_path

        conflict_path = Path(output_dir) / f"national_{grid_name}_conflict_mask.tif"
        with rasterio.open(conflict_path, "w", **conflict_profile) as dst:
            dst.write(conflict_arrays[grid_name], 1)
        n_conflict = int(conflict_arrays[grid_name].sum())
        log.info("Saved conflict mask %s → %s (%d conflicting pixels)", grid_name, conflict_path, n_conflict)
        output_paths[f"{grid_name}_conflict_mask"] = conflict_path

    return output_paths


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Mosaic per-hexel spatial grids into national-level GeoTIFFs.",
    )
    parser.add_argument(
        "--root_dir",
        help="Root directory containing all hexel sub-directories (e.g. /data/hexels).",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where national GeoTIFFs will be written.",
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        choices=list(ALL_INPUT_GRIDS),
        default=None,
        metavar="GRID",
        dest="input_grids",
        help=("Grid names to mosaic. Choose from: %(choices)s. Defaults to all four when omitted."),
    )
    parser.add_argument(
        "--actual_mask_shp",
        required=True,
        metavar="PATH",
        help="Path to the national shapefile with each hexel's actual (tight) boundary polygons.",
    )
    parser.add_argument(
        "--buffer_mask_shp",
        required=True,
        metavar="PATH",
        help="Path to the national shapefile with each hexel's buffered boundary polygons.",
    )
    parser.add_argument(
        "--hex-id-field",
        default="hexid",
        metavar="FIELD",
        help="Attribute field in the shapefiles that contains the hexel ID (default: hexid).",
    )
    parser.add_argument(
        "--reference_tif",
        default=None,
        metavar="PATH",
        help=(
            "Path to an existing national GeoTIFF whose CRS, transform, "
            "width and height will be adopted for all outputs. "
            "When omitted the national grid is derived from the buffer shapefile extent."
        ),
    )
    parser.add_argument(
        "--national-crs",
        default=NATIONAL_CRS,
        metavar="CRS",
        help=(f"Target CRS when --reference-tif is not provided (default: {NATIONAL_CRS})."),
    )

    args = parser.parse_args()

    output_paths = build_national_grid(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        actual_mask_shp=args.actual_mask_shp,
        buffer_mask_shp=args.buffer_mask_shp,
        hex_id_field=args.hex_id_field,
        input_grids=args.input_grids,
        national_crs=args.national_crs,
        reference_tif=args.reference_tif,
    )

    print("\nOutput files:")
    for grid_name, path in output_paths.items():
        print(f"  {grid_name}: {path}")


if __name__ == "__main__":
    main()
