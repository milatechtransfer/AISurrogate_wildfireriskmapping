"""Unit tests for fuel_barrier_geometry.py.

Uses synthetic 2-D arrays to test the geometry and statistics helpers without
touching real data files.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.fuel_barrier_geometry import (
    WIND_DIRECTION_CONVENTION_UNKNOWN,
    assign_directional_sectors,
    compute_distance_fields,
    compute_zone_wind_consistency,
    dist_bin_label,
)

BIN_EDGES = (100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)


@contextmanager
def _mock_csv(df: pd.DataFrame):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
        df.to_csv(fh, index=False)
        path = Path(fh.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


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


class TestDistBinLabel:
    def test_bin_label_format(self) -> None:
        label = dist_bin_label(0, BIN_EDGES)
        assert "0" in label  # first bin starts at 0
        label_last = dist_bin_label(len(BIN_EDGES), BIN_EDGES)
        assert label_last.startswith(">")


class TestWindConsistency:
    def _make_weather_df(
        self,
        directions: list[float],
        speeds: list[float],
        zone: str = "fru21",
    ) -> pd.DataFrame:
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
