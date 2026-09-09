"""Tests for rasterizing BurnP3+ fire perimeters into fuel-edit masks."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from src.datasets.postprocessing.counterfactual.counterfactual_fuel import apply_fuel_edit
from src.datasets.postprocessing.counterfactual.fire_polygon_mask import (
    build_fire_polygon_mask,
    load_final_fire_perimeters,
    select_fire_perimeters,
)

# A 10x10 grid of 1 m pixels with its origin at (0, 100), so pixel (row, col)
# covers x in [col, col+1) and y in (99-row, 100-row].
GRID_CRS = "EPSG:3978"
GRID_TRANSFORM = rasterio.transform.from_origin(0.0, 100.0, 1.0, 1.0)
REFERENCE_PROFILE = {"crs": GRID_CRS, "transform": GRID_TRANSFORM, "height": 10, "width": 10}


def _write_perimeters(path: Path, records: list[dict], crs: str = GRID_CRS) -> Path:
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    frame.to_file(path, layer="daily_burn_perimeters", driver="GPKG")
    return path


def _square(x0: float, y0: float, size: float):
    return box(x0, y0, x0 + size, y0 + size)


def _simple_records() -> list[dict]:
    """Two fires. Fire 1 grows on day 2; fire 2 is single-day and disjoint."""
    return [
        {"Iteration": 1, "FireID": 1, "BurnDay": 1, "geometry": _square(1.0, 97.0, 1.0)},
        {"Iteration": 1, "FireID": 1, "BurnDay": 2, "geometry": _square(1.0, 96.0, 2.0)},
        {"Iteration": 1, "FireID": 2, "BurnDay": 1, "geometry": _square(6.0, 92.0, 2.0)},
    ]


def test_load_final_fire_perimeters_keeps_only_the_last_burn_day(tmp_path: Path) -> None:
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())

    final = load_final_fire_perimeters(path)

    assert len(final) == 2
    assert final["BurnDay"].tolist() == [2, 1]
    # Cumulative perimeters mean the final day is the largest, not a separate ring.
    assert final.loc[final.FireID == 1, "geometry"].iloc[0].area == pytest.approx(4.0)


def test_load_final_fire_perimeters_can_keep_every_daily_step(tmp_path: Path) -> None:
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())

    assert len(load_final_fire_perimeters(path, final_perimeter_only=False)) == 3


def test_load_final_fire_perimeters_accepts_already_final_layer_without_burn_day(tmp_path: Path) -> None:
    records = [
        {"Iteration": 1, "FireID": 1, "geometry": _square(1.0, 96.0, 2.0)},
        {"Iteration": 1, "FireID": 2, "geometry": _square(6.0, 92.0, 2.0)},
    ]
    path = _write_perimeters(tmp_path / "final.gpkg", records)

    final = load_final_fire_perimeters(path, final_perimeter_only=False)

    assert len(final) == 2
    assert list(zip(final["Iteration"], final["FireID"], strict=True)) == [(1, 1), (1, 2)]


def test_build_fire_polygon_mask_rasterizes_expected_pixels(tmp_path: Path) -> None:
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())

    result = build_fire_polygon_mask(params={"path": str(path)}, reference_profile=REFERENCE_PROFILE, hex_id="16")

    expected = np.zeros((10, 10), dtype=bool)
    expected[2:4, 1:3] = True  # fire 1 final footprint: x in [1,3), y in (96,98]
    expected[6:8, 6:8] = True  # fire 2: x in [6,8), y in (92,94]
    np.testing.assert_array_equal(result.mask, expected)
    assert result.masked_pixels == 8
    assert result.n_fires == 2
    assert result.summary["mask_pixels"].tolist() == [4, 4]
    assert result.summary["fully_within_grid"].all()


def test_build_fire_polygon_mask_reprojects_from_a_different_crs(tmp_path: Path) -> None:
    """Perimeters arrive in the simulation's own CRS and must land on the model grid."""
    records = _simple_records()
    native = gpd.GeoDataFrame(records, geometry="geometry", crs=GRID_CRS).to_crs("EPSG:4326")
    path = tmp_path / "perims_wgs84.gpkg"
    native.to_file(path, layer="daily_burn_perimeters", driver="GPKG")

    result = build_fire_polygon_mask(params={"path": str(path)}, reference_profile=REFERENCE_PROFILE, hex_id="16")

    assert "4326" in result.source_crs or "WGS 84" in result.source_crs
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:4, 1:3] = True
    expected[6:8, 6:8] = True
    np.testing.assert_array_equal(result.mask, expected)


def test_build_fire_polygon_mask_rejects_perimeters_that_miss_the_grid(tmp_path: Path) -> None:
    records = [{"Iteration": 1, "FireID": 1, "BurnDay": 1, "geometry": _square(500.0, 500.0, 2.0)}]
    path = _write_perimeters(tmp_path / "far.gpkg", records)

    with pytest.raises(ValueError, match="is empty after reprojecting"):
        build_fire_polygon_mask(params={"path": str(path)}, reference_profile=REFERENCE_PROFILE, hex_id="16")


def test_build_fire_polygon_mask_buffer_grows_the_mask(tmp_path: Path) -> None:
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())

    plain = build_fire_polygon_mask(params={"path": str(path)}, reference_profile=REFERENCE_PROFILE, hex_id="16")
    buffered = build_fire_polygon_mask(params={"path": str(path), "buffer_m": 1.0}, reference_profile=REFERENCE_PROFILE, hex_id="16")

    assert buffered.masked_pixels > plain.masked_pixels
    assert buffered.buffer_m == pytest.approx(1.0)
    # Buffering may only add pixels, never remove them.
    assert bool((plain.mask & ~buffered.mask).sum() == 0)
    assert (buffered.summary["buffered_area_ha"] > buffered.summary["final_area_ha"]).all()


def test_build_fire_polygon_mask_rejects_negative_buffer(tmp_path: Path) -> None:
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())

    with pytest.raises(ValueError, match="buffer_m' must be non-negative"):
        build_fire_polygon_mask(params={"path": str(path), "buffer_m": -5.0}, reference_profile=REFERENCE_PROFILE, hex_id="16")


def test_build_fire_polygon_mask_rejects_unknown_parameters(tmp_path: Path) -> None:
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())

    with pytest.raises(ValueError, match="Unknown fire polygon parameter"):
        build_fire_polygon_mask(params={"path": str(path), "bufer_m": 1.0}, reference_profile=REFERENCE_PROFILE, hex_id="16")


def test_build_fire_polygon_mask_expands_hex_id_in_path(tmp_path: Path) -> None:
    _write_perimeters(tmp_path / "hex16_perims.gpkg", _simple_records())

    result = build_fire_polygon_mask(
        params={"path": str(tmp_path / "hex{hex_id}_perims.gpkg")},
        reference_profile=REFERENCE_PROFILE,
        hex_id="16",
    )

    assert result.n_fires == 2


def test_load_final_fire_perimeters_requires_a_crs(tmp_path: Path) -> None:
    frame = gpd.GeoDataFrame(_simple_records(), geometry="geometry", crs=None)
    path = tmp_path / "nocrs.gpkg"
    frame.to_file(path, layer="daily_burn_perimeters", driver="GPKG")

    with pytest.raises(ValueError, match="has no CRS"):
        load_final_fire_perimeters(path)


def _selection_frame() -> gpd.GeoDataFrame:
    records = [
        {"Iteration": 1, "FireID": 1, "BurnDay": 1, "geometry": _square(0.0, 0.0, 1.0)},
        {"Iteration": 1, "FireID": 2, "BurnDay": 1, "geometry": _square(2.0, 0.0, 3.0)},
        {"Iteration": 2, "FireID": 1, "BurnDay": 1, "geometry": _square(6.0, 0.0, 2.0)},
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs=GRID_CRS)


def test_select_fire_perimeters_defaults_to_pooling_every_fire() -> None:
    assert len(select_fire_perimeters(_selection_frame())) == 3


def test_select_fire_perimeters_by_iteration() -> None:
    selected = select_fire_perimeters(_selection_frame(), select={"iteration": 2})

    assert selected["Iteration"].tolist() == [2]
    assert selected["FireID"].tolist() == [1]


def test_select_fire_perimeters_by_explicit_fire_ids() -> None:
    selected = select_fire_perimeters(_selection_frame(), select={"fire_ids": [[1, 2], [2, 1]]})

    assert sorted(zip(selected["Iteration"], selected["FireID"], strict=True)) == [(1, 2), (2, 1)]


def test_select_fire_perimeters_by_top_k_area() -> None:
    selected = select_fire_perimeters(_selection_frame(), select={"top_k_by_area": 2})

    # Areas are 1, 9 and 4, so the two largest are (1,2) then (2,1).
    assert list(zip(selected["Iteration"], selected["FireID"], strict=True)) == [(1, 2), (2, 1)]


@pytest.mark.parametrize(
    ("select", "match"),
    [
        ({"iteration": 99}, "No fires found for iteration=99"),
        ({"fire_ids": [[9, 9]]}, "references fires absent from the file"),
        ({"fire_ids": []}, "must not be empty"),
        ({"top_k_by_area": 0}, "must be positive"),
        ({"top_k_by_area": 99}, "exceeds the 3 available fires"),
        ({"iteration": 1, "top_k_by_area": 1}, "accepts at most one"),
        ({"itteration": 1}, "Unknown fire polygon selection key"),
    ],
)
def test_select_fire_perimeters_rejects_bad_selections(select: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        select_fire_perimeters(_selection_frame(), select=select)


def test_polygon_mask_restricts_a_fuel_edit_to_the_burned_area(tmp_path: Path) -> None:
    """The mask must confine the conversion, leaving fuel outside the perimeters intact."""
    path = _write_perimeters(tmp_path / "perims.gpkg", _simple_records())
    mask = build_fire_polygon_mask(params={"path": str(path)}, reference_profile=REFERENCE_PROFILE, hex_id="16").mask

    fuel = np.full((10, 10), 2.0, dtype=np.float32)
    fuel[0, :] = 101.0  # a non-fuel row that must never be converted
    fuel[2, 1] = 101.0  # non-fuel inside the burned area

    result = apply_fuel_edit(
        fuel,
        [101],
        mode="burnable_to_burnable_fixed",
        scenario_name="burns_to_aspen",
        params={"replacement_fuel_id": 13, "edit_mask": mask},
    )

    assert result.report.edited_pixels == 7  # 8 masked pixels minus the non-fuel one
    assert (result.fuel[result.edit_mask] == 13).all()
    assert not result.edit_mask[~mask].any()
    assert (result.fuel[0, :] == 101.0).all()
    assert result.fuel[2, 1] == 101.0
    assert result.report.note == "all burnable fuel replaced with fixed burnable fuel"


def test_all_burnable_conversion_requires_omitting_source_fuel_ids_not_an_empty_list() -> None:
    fuel = np.full((4, 4), 2.0, dtype=np.float32)

    with pytest.raises(ValueError, match="must be omitted"):
        apply_fuel_edit(
            fuel,
            [101],
            mode="burnable_to_burnable_fixed",
            scenario_name="scenario",
            params={"replacement_fuel_id": 13, "source_fuel_ids": []},
        )


def test_source_fuel_ids_still_restricts_the_conversion() -> None:
    fuel = np.array([[2.0, 3.0], [2.0, 101.0]], dtype=np.float32)

    result = apply_fuel_edit(
        fuel,
        [101],
        mode="burnable_to_burnable_fixed",
        scenario_name="scenario",
        params={"replacement_fuel_id": 13, "source_fuel_ids": [2]},
    )

    assert result.report.edited_pixels == 2
    assert result.fuel[0, 1] == 3.0
