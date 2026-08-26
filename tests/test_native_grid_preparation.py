from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from data_preparation.spatial.utils import assert_raster_grids_match, load_spatial_raster
from src.config import DataPrepConfig


def _write_raster(path: Path, *, transform=None, crs: str = "EPSG:3978") -> None:
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform or from_origin(100.0, 400.0, 100.0, 100.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_native_grid_loading_preserves_original_geometry(tmp_path: Path) -> None:
    path = tmp_path / "native.tif"
    transform = from_origin(100.0, 400.0, 100.0, 100.0)
    _write_raster(path, transform=transform)

    raster, profile = load_spatial_raster(path, reproject_flag=False)

    assert raster.shape == (3, 4)
    assert profile["crs"] == CRS.from_epsg(3978)
    assert profile["transform"] == transform


def test_native_grid_alignment_rejects_transform_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.tif"
    shifted = tmp_path / "shifted.tif"
    _write_raster(reference)
    _write_raster(shifted, transform=from_origin(200.0, 400.0, 100.0, 100.0))

    with pytest.raises(ValueError, match="exact raster alignment"):
        assert_raster_grids_match([reference, shifted])


def test_native_grid_alignment_accepts_matching_rasters(tmp_path: Path) -> None:
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    _write_raster(first)
    _write_raster(second)

    assert_raster_grids_match([first, second])


def test_native_grid_config_defaults_off_and_can_be_enabled() -> None:
    assert not DataPrepConfig().preserve_native_grid
    assert DataPrepConfig(preserve_native_grid=True).preserve_native_grid
