from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform import (
    FuelCounterfactualTransform,
    fuel_intervention_raster_path,
)


def test_fuel_counterfactual_loads_raw_fuel_grid_and_writes_exact_intervention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_fuel = np.array(
        [
            [1, 1, 2, 2],
            [1, 0, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.float32,
    )
    patch_specs = [
        ("left.npy", 0, 0),
        ("right.npy", 0, 1),
    ]
    metadata = pd.DataFrame([{"filename": filename, "hex_id": 16, "row": row, "col": col} for filename, row, col in patch_specs])
    scenario = ScenarioConfig(
        name="remove_barriers",
        kind="fuel",
        description="",
        params={
            "mode": "nonfuel_to_burnable_local_adjacent_modal",
            "nonfuel_ids": [0],
        },
    )
    reference_profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 3, 1, 1),
        "nodata": -9999,
    }
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_spatial_raster",
        lambda **_: (np.ma.masked_array(global_fuel, mask=False), reference_profile),
    )
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_fuel_grid",
        lambda **_: np.ma.masked_array(global_fuel, mask=False),
    )
    prediction_dir = tmp_path / "predictions" / scenario.name / "bp"

    transform = FuelCounterfactualTransform.from_metadata(
        metadata=metadata,
        fuel_channel=0,
        scenario=scenario,
        prediction_dir=prediction_dir,
        raw_data_dir=tmp_path,
    )

    left_patch = np.zeros((3, 3, 2), dtype=np.float32)
    right_patch = np.zeros((3, 3, 2), dtype=np.float32)
    left = transform(left_patch, metadata.iloc[0].to_dict())
    right = transform(right_patch, metadata.iloc[1].to_dict())
    assert left[1, 1, 0] == 1
    assert right[1, 0, 0] == 1
    assert transform.summary["edited_pixels"].tolist() == [1]
    assert transform.components["replacement_fuel_id"].tolist() == [1]

    with rasterio.open(fuel_intervention_raster_path(prediction_dir, "16", "baseline")) as src:
        baseline_fuel = src.read(1, masked=True).filled(np.nan)
    with rasterio.open(fuel_intervention_raster_path(prediction_dir, "16", "scenario")) as src:
        scenario_fuel = src.read(1, masked=True).filled(np.nan)
    assert baseline_fuel.tolist() == global_fuel.tolist()
    assert scenario_fuel[1, 1] == 1


def test_fuel_counterfactual_pads_patches_extending_past_raster_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_fuel = np.array([[1, 1, 2, 2], [1, 0, 2, 2], [1, 1, 2, 2]], dtype=np.float32)
    metadata = pd.DataFrame([{"filename": "edge.npy", "hex_id": 16, "row": 1, "col": 2}])
    scenario = ScenarioConfig(
        name="remove_barriers",
        kind="fuel",
        description="",
        params={"mode": "nonfuel_to_burnable_local_adjacent_modal", "nonfuel_ids": [0]},
    )
    reference_profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 3, 1, 1),
        "nodata": -9999,
    }
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_spatial_raster",
        lambda **_: (np.ma.masked_array(global_fuel, mask=False), reference_profile),
    )
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_fuel_grid",
        lambda **_: np.ma.masked_array(global_fuel, mask=False),
    )
    transform = FuelCounterfactualTransform.from_metadata(
        metadata=metadata,
        fuel_channel=0,
        scenario=scenario,
        raw_data_dir=tmp_path,
    )

    edge_patch = np.zeros((3, 3, 2), dtype=np.float32)
    edited = transform(edge_patch, metadata.iloc[0].to_dict())
    expected_channel = np.full((3, 3), np.nan, dtype=np.float32)
    expected_channel[:2, :2] = global_fuel[1:3, 2:4]
    np.testing.assert_array_equal(edited[:, :, 0], expected_channel)


def test_fuel_counterfactual_preserves_original_nan_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patches encode NODATA as the union of every source grid's own mask, so a pixel
    can be valid in the raw fuel raster yet NODATA in the patch (e.g. masked out via
    elevation/ignition/firezones). The substituted channel must keep matching NaNs.
    """
    global_fuel = np.array([[1, 1, 2, 2], [1, 0, 2, 2], [1, 1, 2, 2]], dtype=np.float32)
    metadata = pd.DataFrame([{"filename": "patch.npy", "hex_id": 16, "row": 0, "col": 0}])
    scenario = ScenarioConfig(
        name="remove_barriers",
        kind="fuel",
        description="",
        params={"mode": "nonfuel_to_burnable_local_adjacent_modal", "nonfuel_ids": [0]},
    )
    reference_profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 3, 1, 1),
        "nodata": -9999,
    }
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_spatial_raster",
        lambda **_: (np.ma.masked_array(global_fuel, mask=False), reference_profile),
    )
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_fuel_grid",
        lambda **_: np.ma.masked_array(global_fuel, mask=False),
    )
    transform = FuelCounterfactualTransform.from_metadata(
        metadata=metadata,
        fuel_channel=0,
        scenario=scenario,
        raw_data_dir=tmp_path,
    )

    # The raw fuel raster has a valid value at (0, 0), but the patch's own fuel
    # channel is NaN there (masked out by another source grid at generation time).
    patch = np.zeros((3, 4, 2), dtype=np.float32)
    patch[0, 0, 0] = np.nan
    edited = transform(patch, metadata.iloc[0].to_dict())
    assert np.isnan(edited[0, 0, 0])


def test_fuel_counterfactual_aligns_context_window_with_target_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_fuel = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.float32)
    metadata = pd.DataFrame(
        [
            {
                "filename": "context.npy",
                "hex_id": 16,
                "row": 0,
                "col": 0,
                "input_win_h": 4,
                "input_win_w": 4,
                "target_crop_h": 2,
                "target_crop_w": 2,
            }
        ]
    )
    scenario = ScenarioConfig(
        name="remove_barriers",
        kind="fuel",
        description="",
        params={"mode": "nonfuel_to_burnable_local_adjacent_modal", "nonfuel_ids": [0]},
    )
    reference_profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 3,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 3, 1, 1),
        "nodata": -9999,
    }
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_spatial_raster",
        lambda **_: (np.ma.masked_array(global_fuel, mask=False), reference_profile),
    )
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.fuel_counterfactual_transform.load_fuel_grid",
        lambda **_: np.ma.masked_array(global_fuel, mask=False),
    )
    transform = FuelCounterfactualTransform.from_metadata(
        metadata=metadata,
        fuel_channel=0,
        scenario=scenario,
        raw_data_dir=tmp_path,
    )

    patch = np.zeros((4, 4, 1), dtype=np.float32)
    patch[0, :, 0] = np.nan
    patch[:, 0, 0] = np.nan
    edited = transform(patch, metadata.iloc[0].to_dict())

    np.testing.assert_array_equal(edited[1:3, 1:3, 0], global_fuel[:2, :2])
