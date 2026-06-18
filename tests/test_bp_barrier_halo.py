"""Unit tests for diagnose_bp_barrier_halo.py.

Uses synthetic 2-D arrays to test all geometry and statistics helpers without
touching real data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.diagnose_bp_barrier_halo import (
    FUEL_NODATA,
    WIND_DIRECTION_CONVENTION_UNKNOWN,
    ZONE_NODATA,
    FuelBarrierInfo,
    HexelLayers,
    _analytical_halo_ci,
    assign_directional_sectors,
    assign_distance_bins,
    compute_barrier_support,
    compute_boundary_exclusion_mask,
    compute_distance_fields,
    compute_distance_profiles,
    compute_matched_contrasts,
    compute_zone_wind_consistency,
    dist_bin_label,
    dist_bin_labels,
    fuel_to_groups,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layers(
    h: int = 10,
    w: int = 10,
    pixel_m: float = 100.0,
    bp_fill: float = 0.1,
    fuel_fill: int = 1,
    zone_fill: int = 21,
) -> HexelLayers:
    bp = np.full((h, w), bp_fill, dtype=np.float32)
    fuel = np.full((h, w), fuel_fill, dtype=np.int32)
    zones = np.full((h, w), zone_fill, dtype=np.int32)
    elev = np.full((h, w), 200.0, dtype=np.float32)
    return HexelLayers(
        hex_id="01",
        bp=bp,
        fuel=fuel,
        firezones=zones,
        elevation=elev,
        pixel_h_m=pixel_m,
        pixel_w_m=pixel_m,
    )


def _make_fuel_info(
    nonfuel_ids: list[int] | None = None,
    restricted_ids: list[int] | None = None,
) -> FuelBarrierInfo:
    return FuelBarrierInfo(
        hex_id="01",
        nonfuel_ids=nonfuel_ids or [101],
        restricted_ids=restricted_ids or [],
        restricted_names=[],
    )


# ---------------------------------------------------------------------------
# fuel_to_groups
# ---------------------------------------------------------------------------


class TestFuelToGroups:
    def test_known_id_maps_correctly(self) -> None:
        from data_preparation.spatial.utils import FUEL_GROUP_MAP

        fid = next(iter(FUEL_GROUP_MAP))
        fuel = np.array([[fid]], dtype=np.int32)
        result = fuel_to_groups(fuel)
        assert result[0, 0] == FUEL_GROUP_MAP[fid]

    def test_nodata_maps_to_minus_one(self) -> None:
        fuel = np.array([[FUEL_NODATA]], dtype=np.int32)
        result = fuel_to_groups(fuel)
        assert result[0, 0] == -1

    def test_unknown_id_maps_to_minus_one(self) -> None:
        fuel = np.array([[99999]], dtype=np.int32)
        result = fuel_to_groups(fuel)
        assert result[0, 0] == -1

    def test_negative_nodata_no_wrap(self) -> None:
        """FUEL_NODATA = -32768 must not wrap and corrupt a valid index."""
        fuel = np.array([[FUEL_NODATA, 1]], dtype=np.int32)
        result = fuel_to_groups(fuel)
        # -32768 should be -1 (unknown), NOT a valid fuel group via negative wrap.
        assert result[0, 0] == -1


# ---------------------------------------------------------------------------
# Distance fields
# ---------------------------------------------------------------------------


class TestComputeDistanceFields:
    def test_adjacent_pixel_is_one_pixel_width(self) -> None:
        """On a 100 m grid, the pixel immediately to the right of a barrier is 100 m."""
        nonfuel = np.zeros((1, 5), dtype=bool)
        nonfuel[0, 0] = True  # leftmost pixel is barrier
        dist_m, _, _ = compute_distance_fields(nonfuel, 100.0, 100.0, return_nearest_indices=True)
        assert dist_m[0, 0] == pytest.approx(0.0)
        assert dist_m[0, 1] == pytest.approx(100.0)
        assert dist_m[0, 2] == pytest.approx(200.0)

    def test_raises_when_no_barriers(self) -> None:
        nonfuel = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="No non-fuel barrier"):
            compute_distance_fields(nonfuel, 100.0, 100.0)

    def test_nearest_indices_point_to_barrier(self) -> None:
        nonfuel = np.zeros((5, 5), dtype=bool)
        nonfuel[2, 2] = True
        dist_m, nr, nc = compute_distance_fields(nonfuel, 100.0, 100.0, return_nearest_indices=True)
        assert nr is not None and nc is not None
        # All nearest indices should point to the single barrier pixel (2, 2).
        assert np.all(nr == 2)
        assert np.all(nc == 2)

    def test_non_square_pixel_distance(self) -> None:
        """Non-square pixels: use physical dimensions for distance."""
        nonfuel = np.zeros((1, 3), dtype=bool)
        nonfuel[0, 0] = True
        dist_m, _, _ = compute_distance_fields(nonfuel, 100.0, 200.0)
        assert dist_m[0, 1] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Boundary exclusion
# ---------------------------------------------------------------------------


class TestBoundaryExclusionMask:
    def test_no_burnable_nodata_returns_false_everywhere(self) -> None:
        bp = np.ones((5, 5), dtype=np.float32) * 0.1
        nonfuel = np.zeros((5, 5), dtype=bool)
        mask = compute_boundary_exclusion_mask(bp, nonfuel, 100.0, 100.0, 300.0)
        assert not mask.any()

    def test_disabled_by_zero_threshold(self) -> None:
        bp = np.full((5, 5), np.nan, dtype=np.float32)
        bp[2, 2] = 0.1
        nonfuel = np.zeros((5, 5), dtype=bool)
        mask = compute_boundary_exclusion_mask(bp, nonfuel, 100.0, 100.0, 0.0)
        assert not mask.any()

    def test_excludes_pixels_near_boundary_nodata(self) -> None:
        """A column of burnable nodata should trigger exclusion within threshold."""
        bp = np.ones((1, 5), dtype=np.float32) * 0.1
        bp[0, 0] = np.nan  # boundary-artifact nodata in leftmost pixel
        nonfuel = np.zeros((1, 5), dtype=bool)
        # Exclude within 250 m (2.5 pixels) of boundary nodata.
        mask = compute_boundary_exclusion_mask(bp, nonfuel, 100.0, 100.0, 250.0)
        # Pixel 0: nodata itself (dist=0) < 250 → excluded.
        assert mask[0, 0]
        # Pixel 1: dist=100 < 250 → excluded.
        assert mask[0, 1]
        # Pixel 2: dist=200 < 250 → excluded.
        assert mask[0, 2]
        # Pixel 3: dist=300 ≥ 250 → not excluded.
        assert not mask[0, 3]

    def test_nonfuel_nodata_not_treated_as_boundary(self) -> None:
        """BP nodata on non-fuel pixels should NOT cause boundary exclusion."""
        bp = np.ones((1, 5), dtype=np.float32) * 0.1
        bp[0, 0] = np.nan  # nodata on a non-fuel pixel
        nonfuel = np.zeros((1, 5), dtype=bool)
        nonfuel[0, 0] = True  # this is a barrier pixel, not boundary nodata
        mask = compute_boundary_exclusion_mask(bp, nonfuel, 100.0, 100.0, 250.0)
        # Non-fuel nodata excluded from boundary computation → no exclusion.
        assert not mask.any()


# ---------------------------------------------------------------------------
# Bin assignment
# ---------------------------------------------------------------------------


class TestAssignDistanceBins:
    BIN_EDGES = (100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)

    def test_barrier_pixel_in_bin_zero(self) -> None:
        dist = np.array([0.0], dtype=np.float32)
        bins = assign_distance_bins(dist, self.BIN_EDGES)
        assert bins[0] == 0

    def test_exactly_at_edge_goes_to_higher_bin(self) -> None:
        dist = np.array([100.0], dtype=np.float32)
        bins = assign_distance_bins(dist, self.BIN_EDGES)
        # 100.0 is the start of the second bin [100, 250).
        assert bins[0] == 1

    def test_last_bin_captures_very_large_distance(self) -> None:
        dist = np.array([10000.0], dtype=np.float32)
        bins = assign_distance_bins(dist, self.BIN_EDGES)
        assert bins[0] == len(self.BIN_EDGES)

    def test_bin_labels_length(self) -> None:
        labels = dist_bin_labels(self.BIN_EDGES)
        assert len(labels) == len(self.BIN_EDGES) + 1

    def test_bin_label_format(self) -> None:
        label = dist_bin_label(0, self.BIN_EDGES)
        assert "0" in label  # first bin starts at 0
        label_last = dist_bin_label(len(self.BIN_EDGES), self.BIN_EDGES)
        assert label_last.startswith(">")


# ---------------------------------------------------------------------------
# Wind consistency
# ---------------------------------------------------------------------------


class TestWindConsistency:
    def _make_weather_df(
        self,
        directions: list[float],
        speeds: list[float],
        zone: str = "fru21",
    ) -> pd.DataFrame:
        import io
        import tempfile

        rows = [{"WeatherZone": zone, "WindDirection": d, "WindSpeed": s} for d, s in zip(directions, speeds, strict=True)]
        return pd.DataFrame(rows)

    def test_perfectly_aligned_winds_consistency_one(self) -> None:
        """All wind from the same direction → consistency = 1.0."""
        df = self._make_weather_df([45.0] * 5, [5.0] * 5)
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        assert result["consistency"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_opposing_equal_winds_consistency_zero(self) -> None:
        """Two equal-speed opposing winds → vectors cancel → consistency = 0.0."""
        df = self._make_weather_df([0.0, 180.0], [10.0, 10.0])
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        assert result["consistency"].iloc[0] == pytest.approx(0.0, abs=1e-5)

    def test_strong_aligned_with_weak_opposing_high_consistency(self) -> None:
        """Strong aligned + weak opposing → consistency closer to 1 than to 0."""
        df = self._make_weather_df([0.0, 180.0], [10.0, 1.0])
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        c = float(result["consistency"].iloc[0])
        assert 0.5 < c < 1.0

    def test_zero_speed_gives_nan(self) -> None:
        df = self._make_weather_df([45.0], [0.0])
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        assert np.isnan(result["consistency"].iloc[0])

    def test_multiple_zones_separate_rows(self) -> None:
        df1 = self._make_weather_df([0.0] * 3, [5.0] * 3, zone="fru21")
        df2 = self._make_weather_df([90.0] * 3, [3.0] * 3, zone="fru25")
        df = pd.concat([df1, df2], ignore_index=True)
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        assert len(result) == 2

    def test_dominant_direction_matches_input(self) -> None:
        """Uniform 90-degree from-bearing should give physical flow near 270 degrees."""
        df = self._make_weather_df([90.0] * 4, [6.0] * 4)
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        dom = float(result["dominant_direction_deg"].iloc[0])
        from_dom = float(result["dominant_from_direction_deg"].iloc[0])
        assert abs(from_dom - 90.0) < 1.0
        assert abs(dom - 270.0) < 1.0

    def test_warning_string_present(self) -> None:
        df = self._make_weather_df([0.0], [1.0])
        with _mock_csv(df) as path:
            result = compute_zone_wind_consistency(path)
        assert WIND_DIRECTION_CONVENTION_UNKNOWN in result["wind_convention_warning"].iloc[0]


# ---------------------------------------------------------------------------
# Directional sector assignment
# ---------------------------------------------------------------------------


class TestDirectionalSectors:
    def test_pixel_directly_in_dominant_direction(self) -> None:
        """A pixel north of the barrier when physical flow is north → downwind sector."""
        row_idx = np.array([0], dtype=np.int32)
        col_idx = np.array([5], dtype=np.int32)
        nearest_row = np.array([5], dtype=np.int32)
        nearest_col = np.array([5], dtype=np.int32)
        # bearing from barrier to pixel is 0 deg (north); dominant = 0 deg → aligned.
        sectors = assign_directional_sectors(row_idx, col_idx, nearest_row, nearest_col, 0.0, 100.0, 100.0)
        assert sectors[0] == 0  # downwind_of_barrier

    def test_pixel_opposite_dominant_direction(self) -> None:
        """Pixel south of barrier; physical flow north → upwind sector."""
        row_idx = np.array([10], dtype=np.int32)
        col_idx = np.array([5], dtype=np.int32)
        nearest_row = np.array([5], dtype=np.int32)
        nearest_col = np.array([5], dtype=np.int32)
        sectors = assign_directional_sectors(row_idx, col_idx, nearest_row, nearest_col, 0.0, 100.0, 100.0)
        assert sectors[0] == 2  # upwind_of_barrier

    def test_crosswind_right(self) -> None:
        """Pixel east of barrier; wind north → sector 1 (crosswind_right)."""
        row_idx = np.array([5], dtype=np.int32)
        col_idx = np.array([10], dtype=np.int32)
        nearest_row = np.array([5], dtype=np.int32)
        nearest_col = np.array([5], dtype=np.int32)
        sectors = assign_directional_sectors(row_idx, col_idx, nearest_row, nearest_col, 0.0, 100.0, 100.0)
        assert sectors[0] == 1  # crosswind_right

    def test_crosswind_left(self) -> None:
        """Pixel west of barrier; wind north → sector 3 (crosswind_left)."""
        row_idx = np.array([5], dtype=np.int32)
        col_idx = np.array([0], dtype=np.int32)
        nearest_row = np.array([5], dtype=np.int32)
        nearest_col = np.array([5], dtype=np.int32)
        sectors = assign_directional_sectors(row_idx, col_idx, nearest_row, nearest_col, 0.0, 100.0, 100.0)
        assert sectors[0] == 3  # crosswind_left

    def test_from_bearing_example_west_wind_makes_east_pixel_downwind(self) -> None:
        """WindDirection=270 means from west, so physical flow points east."""
        row_idx = np.array([5], dtype=np.int32)
        col_idx = np.array([10], dtype=np.int32)
        nearest_row = np.array([5], dtype=np.int32)
        nearest_col = np.array([5], dtype=np.int32)
        physical_flow_bearing = 90.0
        sectors = assign_directional_sectors(
            row_idx,
            col_idx,
            nearest_row,
            nearest_col,
            physical_flow_bearing,
            100.0,
            100.0,
        )
        assert sectors[0] == 0


# ---------------------------------------------------------------------------
# Analytical CI helper
# ---------------------------------------------------------------------------


class TestAnalyticalHaloCI:
    def test_zero_halo_when_groups_identical(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.uniform(0.1, 0.3, 100)
        halo, lo, hi = _analytical_halo_ci(arr, arr)
        assert halo == pytest.approx(0.0, abs=1e-10)

    def test_positive_halo_when_far_larger(self) -> None:
        near = np.full(50, 0.05)
        far = np.full(50, 0.20)
        halo, lo, hi = _analytical_halo_ci(near, far)
        assert halo == pytest.approx(0.15, abs=1e-8)

    def test_ci_contains_true_halo(self) -> None:
        rng = np.random.default_rng(42)
        near = rng.uniform(0.05, 0.15, 200)
        far = rng.uniform(0.15, 0.30, 200)
        halo, lo, hi = _analytical_halo_ci(near, far)
        assert lo <= halo <= hi

    def test_single_element_groups_no_crash(self) -> None:
        _analytical_halo_ci(np.array([0.1]), np.array([0.2]))


# ---------------------------------------------------------------------------
# Barrier support summary
# ---------------------------------------------------------------------------


class TestBarrierSupport:
    def test_counts_add_up(self) -> None:
        layers = _make_layers(h=5, w=5, bp_fill=0.1, fuel_fill=1, zone_fill=21)
        # Mark top row as non-fuel.
        layers.fuel[0, :] = 101
        layers.bp[0, :] = np.nan  # non-fuel is BP nodata
        fuel_info = _make_fuel_info(nonfuel_ids=[101])
        nonfuel_mask = layers.fuel == 101

        from scipy.ndimage import distance_transform_edt

        dist_m = distance_transform_edt(~nonfuel_mask, sampling=(100.0, 100.0)).astype(np.float32)
        boundary_excl = np.zeros_like(layers.fuel, dtype=bool)
        support = compute_barrier_support(layers, fuel_info, dist_m, boundary_excl, 500.0, 2000.0)
        row = support.iloc[0]
        assert row["nonfuel_barrier_pixels"] == 5  # top row
        assert row["burnable_pixels"] == 20  # remaining 4 rows
        assert row["analysis_pixels"] <= row["burnable_pixels"]


# ---------------------------------------------------------------------------
# Distance profiles (integration)
# ---------------------------------------------------------------------------


class TestDistanceProfiles:
    def test_profiles_are_non_empty(self) -> None:
        layers = _make_layers(h=6, w=6, bp_fill=0.15, fuel_fill=1, zone_fill=21)
        layers.fuel[0, :] = 101  # top row is barrier
        layers.bp[0, :] = np.nan
        fuel_info = _make_fuel_info(nonfuel_ids=[101])
        from scipy.ndimage import distance_transform_edt

        dist_m = distance_transform_edt(~(layers.fuel == 101), sampling=(100.0, 100.0)).astype(np.float32)
        boundary_excl = np.zeros_like(layers.fuel, dtype=bool)
        bin_edges = (100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)
        profiles = compute_distance_profiles(layers, fuel_info, dist_m, boundary_excl, bin_edges)
        assert not profiles.empty
        assert "bp_mean" in profiles.columns


# ---------------------------------------------------------------------------
# Matched contrasts (integration)
# ---------------------------------------------------------------------------


class TestMatchedContrasts:
    def test_near_lower_bp_produces_positive_halo(self) -> None:
        """Near pixels set to low BP, far to high BP → halo_abs > 0."""
        h, w = 1, 60
        bp = np.full((h, w), 0.25, dtype=np.float32)
        fuel = np.full((h, w), 1, dtype=np.int32)
        zones = np.full((h, w), 21, dtype=np.int32)
        elev = np.zeros((h, w), dtype=np.float32)

        # Leftmost pixel is barrier.
        fuel[0, 0] = 101
        bp[0, 0] = np.nan

        # Near: pixels 1-4 (within 500 m), low BP.
        bp[0, 1:5] = 0.05
        # Far: pixels 20-59 (beyond 2000 m), high BP.
        bp[0, 20:] = 0.25

        layers = HexelLayers(
            hex_id="01",
            bp=bp,
            fuel=fuel,
            firezones=zones,
            elevation=elev,
            pixel_h_m=100.0,
            pixel_w_m=100.0,
        )
        fuel_info = _make_fuel_info(nonfuel_ids=[101])
        from scipy.ndimage import distance_transform_edt

        nonfuel_mask = fuel == 101
        dist_m = distance_transform_edt(~nonfuel_mask, sampling=(100.0, 100.0)).astype(np.float32)
        boundary_excl = np.zeros((h, w), dtype=bool)

        contrasts = compute_matched_contrasts(
            layers,
            fuel_info,
            dist_m,
            boundary_excl,
            near_band_m=500.0,
            far_band_m=2000.0,
            min_pixels_per_stratum=1,
            min_bp_for_halo_rel=1e-4,
        )
        assert not contrasts.empty
        assert float(contrasts["halo_abs"].iloc[0]) > 0

    def test_stratum_below_min_support_excluded(self) -> None:
        """Stratum with fewer than min_pixels near/far should be dropped."""
        h, w = 1, 60
        bp = np.full((h, w), 0.10, dtype=np.float32)
        fuel = np.full((h, w), 1, dtype=np.int32)
        zones = np.full((h, w), 21, dtype=np.int32)
        elev = np.zeros((h, w), dtype=np.float32)
        fuel[0, 0] = 101
        bp[0, 0] = np.nan

        layers = HexelLayers(
            hex_id="01",
            bp=bp,
            fuel=fuel,
            firezones=zones,
            elevation=elev,
            pixel_h_m=100.0,
            pixel_w_m=100.0,
        )
        fuel_info = _make_fuel_info(nonfuel_ids=[101])
        from scipy.ndimage import distance_transform_edt

        nonfuel_mask = fuel == 101
        dist_m = distance_transform_edt(~nonfuel_mask, sampling=(100.0, 100.0)).astype(np.float32)
        boundary_excl = np.zeros((h, w), dtype=bool)

        contrasts = compute_matched_contrasts(
            layers,
            fuel_info,
            dist_m,
            boundary_excl,
            near_band_m=500.0,
            far_band_m=2000.0,
            min_pixels_per_stratum=999,  # very high threshold → no strata qualify
            min_bp_for_halo_rel=1e-4,
        )
        assert contrasts.empty


# ---------------------------------------------------------------------------
# Context manager helper for temporary CSV
# ---------------------------------------------------------------------------

import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _mock_csv(df: pd.DataFrame):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
        df.to_csv(fh, index=False)
        path = Path(fh.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
