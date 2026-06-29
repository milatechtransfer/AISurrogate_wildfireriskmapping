import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import torchvision.transforms.functional as F
from rasterio.transform import from_origin

from data_preparation.tabular.weather import preprocess_weather_list
from data_preparation.utils import process_fire_size_df
from src.config import (
    DataConfig,
    DataSourceConfig,
    GridParams,
    SpatializedTabularParams,
    TabularParams,
)
from src.datasets.dataset import MultiSourceDataset, build_dataset
from src.datasets.sources import GridSource, SpatializedTabularSource, TabularSource
from src.datasets.transforms import get_transforms, setup_augmentations
from src.datasets.utils import get_dataset_dimensions


@pytest.fixture
def temp_data_dir():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create dummy feature_channel_map
        feature_channel_map = {
            "fuel_grid": [0],
            "elevation_grid": [1],
            "ignition_grid": [2],
            "firezones_grid": [3],
            "bp_out_grid": [4],
            "fi_out_grid": [5],
            "ros_out_grid": [6],
        }
        with open(os.path.join(tmpdir, "feature_channel_map_1.json"), "w") as f:
            json.dump(feature_channel_map, f)

        # Create dummy metadata CSV
        filenames = []
        valid_ratios = []
        for i in range(3):
            fname = f"sample_{i}.npy"
            filenames.append(fname)
            valid_ratios.append(1.0)
        df = pd.DataFrame(
            {
                "filename": filenames,
                "valid_ratio": valid_ratios,
                "hex_id": [1, 1, 2],
                "row": [0, 0, 16],
                "col": [0, 16, 0],
            }
        )
        train_csv = "train.csv"
        df.to_csv(os.path.join(tmpdir, train_csv), index=False)
        val_csv = "val.csv"
        df.to_csv(os.path.join(tmpdir, val_csv), index=False)
        test_csv = "test.csv"
        df.to_csv(os.path.join(tmpdir, test_csv), index=False)

        # Create dummy npy files
        for fname in filenames:
            arr = np.random.rand(32, 32, 7).astype(np.float32)
            # Add some NaNs to input channels
            arr[:, :, 3] = 100.0  # Force fire zone channel to be '100.0' so it matches weather CSV below.
            arr[:, :, 4] = 0.25  # Keep bp_out_grid deterministic.
            arr[:, :, 5] = 0.50  # Keep fi_out_grid deterministic.
            arr[:, :, 6] = 0.75  # Keep ros_out_grid deterministic.
            arr[1, 1, :] = np.nan
            arr[10, 20, :] = np.nan
            np.save(os.path.join(tmpdir, fname), arr)

        # Create dummy weather table csv
        weather_feats = [
            "Temperature",
            "RelativeHumidity",
            "Precipitation",
            "FineFuelMoistureCode",
            "DuffMoistureCode",
            "DroughtCode",
            "InitialSpreadIndex",
            "BuildupIndex",
        ]
        data = {feat: np.random.rand(5) for feat in weather_feats}
        data["WeatherZone"] = [100, 100, 100, 200, 200]  # 3 samples for zone 100
        weather_df = pd.DataFrame(data)
        weather_csv = "weather_table.csv"
        weather_df.to_csv(os.path.join(tmpdir, weather_csv), index=False)

        # Create dummy fire size table csv
        fire_size_feats = ["size"]
        data = {"size": [10.0, 50.0, 100.0, 5.0, 200.0]}  # Varying size values
        data["grid_code"] = [100, 100, 100, 200, 200]  # 3 samples for zone 100
        fire_size_df = pd.DataFrame(data)
        fire_size_csv = "fire_size_table.csv"
        fire_size_df.to_csv(os.path.join(tmpdir, fire_size_csv), index=False)

        yield tmpdir, train_csv, val_csv, test_csv, weather_csv, weather_feats, fire_size_csv, fire_size_feats

    finally:
        shutil.rmtree(tmpdir)


def test_multi_source_integration(temp_data_dir):
    tmpdir, train_csv, val_csv, test_csv, weather_csv, weather_feats, fire_size_csv, fire_size_feats = temp_data_dir

    grid_params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        out_norm="min_max",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
    )

    grid_source = GridSource(
        root_dir=tmpdir,
        params=grid_params,
        modelling_approach="1",
    )

    weather_params = TabularParams(
        csv_name=weather_csv,
        feature_names_list=weather_feats,
        fire_weather_zone_id_col="WeatherZone",
        fire_weather_zone_selection_approach="mode",
        num_samples_per_patch=2,
    )

    fire_size_params = TabularParams(
        csv_name=fire_size_csv,
        feature_names_list=fire_size_feats,
        fire_weather_zone_id_col="grid_code",
        fire_weather_zone_selection_approach="mode",
        num_samples_per_patch=2,
    )

    weather_source = TabularSource(
        root_dir=tmpdir,
        params=weather_params,
        modelling_approach="1",
    )

    fire_size_source = TabularSource(root_dir=tmpdir, params=fire_size_params, modelling_approach="1")

    ds = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={"grid": grid_source, "tabular_weather": weather_source, "tabular_fire_size": fire_size_source},
    )
    sample = ds[0]

    assert "grid" in sample
    assert "tabular_weather" in sample
    assert "tabular_fire_size" in sample
    input_arr, target, mask = sample["grid"]
    assert isinstance(input_arr, torch.Tensor)
    assert input_arr.shape[0] == 3
    mask_np = mask.squeeze(0).numpy()
    assert mask_np.shape == (32, 32)
    assert not mask_np[1, 1]
    assert not mask_np[10, 20]
    assert mask_np[0, 0]
    weather = sample["tabular_weather"]
    assert weather.shape == (2, len(weather_feats))
    fire_size = sample["tabular_fire_size"]
    assert fire_size.shape == (2, len(fire_size_feats))


def test_grid_source_bp_nodata_as_zero_extends_bp_mask(temp_data_dir, monkeypatch):
    tmpdir, *_ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (1.0, 0.0))

    sample_path = os.path.join(tmpdir, "sample_0.npy")
    arr = np.load(sample_path)
    arr[0, 1, 4] = np.nan
    np.save(sample_path, arr)

    params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        target_name="bp",
        out_norm="none",
        fuel_feats_encoding="ordinal",
        bp_nodata_as_zero=True,
    )
    source = GridSource(root_dir=tmpdir, params=params, modelling_approach="1")

    _, target, mask = source.get_sample({"file_path": sample_path})

    assert mask[0, 0, 1]
    assert target[0, 0, 1].item() == pytest.approx(0.0)


def test_grid_source_bp_nodata_as_zero_sets_minmax_range_to_zero(temp_data_dir, monkeypatch):
    tmpdir, *_ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (1.0, 0.0))
    monkeypatch.setattr("src.datasets.sources.grids.get_range_output", lambda *_args, **_kwargs: (0.5, 1.0 / 30000.0))

    params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        target_name="bp",
        out_norm="min_max",
        fuel_feats_encoding="ordinal",
        bp_nodata_as_zero=True,
    )
    source = GridSource(root_dir=tmpdir, params=params, modelling_approach="1")

    assert source.target_ranges["bp"] == pytest.approx((0.5, 0.0))


def test_spatialized_tabular_source_rasterizes_zone_summaries(temp_data_dir):
    tmpdir, _, _, _, weather_csv, weather_feats, _, _ = temp_data_dir

    params = SpatializedTabularParams(
        csv_name=weather_csv,
        feature_names_list=weather_feats[:2],
        fire_weather_zone_id_col="WeatherZone",
        include_missing_firezone_mask=True,
    )
    source = SpatializedTabularSource(root_dir=tmpdir, params=params, modelling_approach="1")

    sample = source.get_sample({"file_path": os.path.join(tmpdir, "sample_0.npy")})
    expected_zone_100 = (
        pd.read_csv(os.path.join(tmpdir, weather_csv))
        .query("WeatherZone == 100")[weather_feats[:2]]
        .mean(axis=0)
        .to_numpy(dtype=np.float32)
    )

    assert sample.shape == (3, 32, 32)
    np.testing.assert_allclose(sample[:2, 0, 0].numpy(), expected_zone_100)
    assert sample[-1, 0, 0].item() == 0.0
    assert sample[-1, 1, 1].item() == 1.0


def test_spatialized_tabular_global_mean_uses_all_rows(temp_data_dir):
    tmpdir, _, _, _, _, _, _, _ = temp_data_dir

    weather_csv = "weather_full_data.csv"
    pd.DataFrame(
        {
            "WeatherZone": [100, 100, 200],
            "Temperature": [1.0, 3.0, 1000.0],
        }
    ).to_csv(os.path.join(tmpdir, weather_csv), index=False)
    missing_patch = np.zeros((4, 4, 7), dtype=np.float32)
    missing_patch[:, :, 3] = np.nan
    missing_patch_path = os.path.join(tmpdir, "missing_zone.npy")
    np.save(missing_patch_path, missing_patch)

    params = SpatializedTabularParams(
        csv_name=weather_csv,
        feature_names_list=["Temperature"],
        fire_weather_zone_id_col="WeatherZone",
        include_missing_firezone_mask=True,
        missing_value_strategy="global_mean",
    )
    source = SpatializedTabularSource(root_dir=tmpdir, params=params, modelling_approach="1")

    sample = source.get_sample({"file_path": missing_patch_path})

    expected_fill = np.float32((1.0 + 3.0 + 1000.0) / 3.0)
    assert sample.shape == (2, 4, 4)
    np.testing.assert_allclose(sample[0].numpy(), np.full((4, 4), expected_fill, dtype=np.float32))
    np.testing.assert_allclose(sample[1].numpy(), np.ones((4, 4), dtype=np.float32))


def test_spatialized_tabular_lut_includes_all_zones(temp_data_dir):
    tmpdir, _, _, _, weather_csv, weather_feats, _, _ = temp_data_dir

    params = SpatializedTabularParams(
        csv_name=weather_csv,
        feature_names_list=weather_feats[:2],
        fire_weather_zone_id_col="WeatherZone",
    )
    source = SpatializedTabularSource(root_dir=tmpdir, params=params, modelling_approach="1")

    # Every zone with weather rows enters the LUT, including zones absent from the training patches.
    assert set(source.lut) == {100, 200}


def test_weather_preprocessing_scalers_fit_train_rows_only():
    weather = pd.DataFrame(
        {
            "Season": [1, 1, 1],
            "WeatherZone": [1, 1, 2],
            "Temperature": [0.0, 10.0, 100.0],
            "RelativeHumidity": [10.0, 20.0, 30.0],
            "WindSpeed": [1.0, 3.0, 5.0],
            "WindDirection": [0.0, 90.0, 180.0],
            "Precipitation": [0.0, 1.0, 3.0],
            "FineFuelMoistureCode": [80.0, 90.0, 100.0],
            "DuffMoistureCode": [1.0, 3.0, 5.0],
            "DroughtCode": [10.0, 20.0, 30.0],
            "InitialSpreadIndex": [1.0, 2.0, 3.0],
            "BuildupIndex": [2.0, 4.0, 6.0],
            "FireWeatherIndex": [3.0, 6.0, 9.0],
        }
    )

    processed = preprocess_weather_list(weather, fit_mask=np.array([True, True, False]))

    assert processed.loc[2, "Temperature"] == pytest.approx(19.0)
    assert processed.loc[2, "RelativeHumidity"] == pytest.approx(2.0)
    assert processed.loc[2, "FineFuelMoistureCode"] == pytest.approx(2.0)


def test_fire_size_processing_normalizes_with_train_gridcodes_only():
    fire_size = pd.DataFrame({"GRIDCODE": [1, 2, 3], "SIZE_HA": [9.0, 99.0, 999.0]})

    processed = process_fire_size_df(fire_size, train_firezone_ids={1, 2})

    assert processed.loc[processed["GRIDCODE"].eq(1), "NORM_LOG_SIZE_HA"].item() == pytest.approx(0.0)
    assert processed.loc[processed["GRIDCODE"].eq(2), "NORM_LOG_SIZE_HA"].item() == pytest.approx(1.0, abs=2e-5)
    assert processed.loc[processed["GRIDCODE"].eq(3), "NORM_LOG_SIZE_HA"].item() > 1.5


def test_build_dataset_appends_spatialized_tabular_channels_to_grid(temp_data_dir, monkeypatch):
    tmpdir, train_csv, _, _, weather_csv, weather_feats, _, _ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_output", lambda *_args, **_kwargs: (1.0, 0.0))
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (1.0, 0.0))

    config = DataConfig(
        root_dir=tmpdir,
        raw_data_dir=tmpdir,
        train_split=train_csv,
        val_split="val.csv",
        test_split="test.csv",
        input_sources=[
            DataSourceConfig(
                name="grid",
                params=GridParams(
                    feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
                    out_norm="min_max",
                    fuel_feats_encoding="ordinal",
                ),
            ),
            DataSourceConfig(
                name="spatialized_weather",
                params=SpatializedTabularParams(
                    csv_name=weather_csv,
                    feature_names_list=weather_feats[:2],
                    fire_weather_zone_id_col="WeatherZone",
                ),
            ),
        ],
    )

    dataset = build_dataset(config=config, csv_name=train_csv, modelling_approach="1")
    spatial_channels, auxiliary_dims = get_dataset_dimensions(dataset)
    sample = dataset[0]

    assert spatial_channels == 5
    assert auxiliary_dims == {}
    assert set(sample) == {"grid"}
    inputs, _, _ = sample["grid"]
    assert inputs.shape == (5, 32, 32)


def test_build_dataset_can_include_patch_metadata(temp_data_dir, monkeypatch):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_output", lambda *_args, **_kwargs: (1.0, 0.0))
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (1.0, 0.0))

    config = DataConfig(
        root_dir=tmpdir,
        raw_data_dir=tmpdir,
        train_split=train_csv,
        val_split="val.csv",
        test_split="test.csv",
        include_patch_metadata=True,
        input_sources=[
            DataSourceConfig(
                name="grid",
                params=GridParams(
                    feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
                    out_norm="min_max",
                    fuel_feats_encoding="ordinal",
                ),
            )
        ],
    )

    dataset = build_dataset(config=config, csv_name=train_csv, modelling_approach="1")
    sample = dataset[0]

    assert "patch_metadata" in sample
    assert sample["patch_metadata"]["hex_id"].item() == 1


def test_build_dataset_passes_raw_data_dir_to_grid_source(temp_data_dir, monkeypatch):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir
    raw_data_dir = "/network/raw/source"
    seen = {}

    def fake_get_range_output(root_dir, output_type, allowed_hex_ids=None):
        seen["output"] = (root_dir, output_type)
        return 1.0, 0.0

    def fake_get_range_elevation(root_dir, allowed_hex_ids=None):
        seen["elevation"] = root_dir
        return 1000.0, 0.0

    monkeypatch.setattr("src.datasets.sources.grids.get_range_output", fake_get_range_output)
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", fake_get_range_elevation)

    config = DataConfig(
        root_dir=tmpdir,
        raw_data_dir=raw_data_dir,
        train_split=train_csv,
        val_split="val.csv",
        test_split="test.csv",
        input_sources=[
            DataSourceConfig(
                name="grid",
                params=GridParams(
                    feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
                    out_norm="min_max",
                    fuel_feats_encoding="ordinal",
                ),
            )
        ],
    )

    ds = build_dataset(config=config, csv_name=train_csv, modelling_approach="1")

    assert ds.sources["grid"].raw_data_dir == raw_data_dir
    assert seen["output"] == (raw_data_dir, "fire_burn_probability")
    assert seen["elevation"] == raw_data_dir


def test_grid_source_rejects_invalid_explicit_raw_data_dir(temp_data_dir, monkeypatch):
    tmpdir, _, _, _, _, _, _, _ = temp_data_dir

    monkeypatch.setattr(
        "src.datasets.sources.grids.get_range_output",
        lambda root_dir, output_type, allowed_hex_ids=None: (float("-inf"), float("inf")),
    )
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda root_dir, allowed_hex_ids=None: (1000.0, 0.0))

    grid_params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        out_norm="min_max",
        fuel_feats_encoding="ordinal",
    )

    with pytest.raises(ValueError, match="Invalid Burn Probability normalization range"):
        GridSource(root_dir=tmpdir, raw_data_dir="/bad/raw", params=grid_params, modelling_approach="1")


def test_grid_one_hot_encoding(temp_data_dir):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir

    grid_params = GridParams(feature_names_list=["fuel_grid"], fuel_feats_encoding="one_hot", normalize_fuel_feats_ordinal=True)

    grid_source = GridSource(
        root_dir=tmpdir,
        params=grid_params,
        modelling_approach="1",
    )

    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source})
    sample = ds[0]
    x, y, mask = sample["grid"]
    # Should have more channels due to one-hot
    assert x.shape[0] > 1


def test_grid_feature_names_list(temp_data_dir):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir

    grid_params = GridParams(
        feature_names_list=["fuel_grid"],
        fuel_feats_encoding="ordinal",
    )

    grid_source = GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source})

    sample = ds[0]
    x, y, mask = sample["grid"]

    # Assert shape is exactly 1
    assert x.shape[0] == 1


def test_grid_source_appends_terrain_derivatives_from_elevation(temp_data_dir, monkeypatch):
    tmpdir, *_ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (3100.0, 0.0))

    sample_path = os.path.join(tmpdir, "sample_0.npy")
    arr = np.zeros((32, 32, 7), dtype=np.float32)
    arr[:, :, 1] = np.broadcast_to(np.arange(32, dtype=np.float32) * 100.0, (32, 32))
    arr[:, :, 4] = 0.25
    np.save(sample_path, arr)

    grid_params = GridParams(
        feature_names_list=["elevation_grid"],
        target_name="bp",
        out_norm="none",
        terrain_derivatives=["slope", "aspect_sin", "aspect_cos"],
        terrain_cell_size_m=100.0,
    )
    grid_source = GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")

    inputs, _, _ = grid_source.get_sample({"file_path": sample_path})

    assert grid_source.input_dim() == 4
    assert inputs.shape == (4, 32, 32)
    torch.testing.assert_close(inputs[1], torch.full((32, 32), 0.5), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(inputs[2], torch.full((32, 32), -1.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(inputs[3], torch.zeros((32, 32)), rtol=1e-5, atol=1e-5)


def test_grid_source_terrain_derivatives_are_computed_after_transforms(temp_data_dir, monkeypatch):
    tmpdir, *_ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (3100.0, 0.0))

    sample_path = os.path.join(tmpdir, "sample_0.npy")
    arr = np.zeros((32, 32, 7), dtype=np.float32)
    arr[:, :, 1] = np.broadcast_to(np.arange(32, dtype=np.float32) * 100.0, (32, 32))
    arr[:, :, 4] = 0.25
    np.save(sample_path, arr)

    def hflip_transform(x, target, mask):
        return F.hflip(x), F.hflip(target), F.hflip(mask)

    grid_params = GridParams(
        feature_names_list=["elevation_grid"],
        target_name="bp",
        out_norm="none",
        terrain_derivatives=["aspect_sin"],
        terrain_cell_size_m=100.0,
    )
    grid_source = GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1", transform=hflip_transform)

    inputs, _, _ = grid_source.get_sample({"file_path": sample_path})

    torch.testing.assert_close(inputs[-1], torch.full((32, 32), 1.0), rtol=1e-5, atol=1e-5)


def test_grid_source_sets_flat_terrain_aspect_to_zero(temp_data_dir, monkeypatch):
    tmpdir, *_ = temp_data_dir
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda *_args, **_kwargs: (1000.0, 0.0))

    sample_path = os.path.join(tmpdir, "sample_0.npy")
    arr = np.zeros((32, 32, 7), dtype=np.float32)
    arr[:, :, 1] = 500.0
    arr[:, :, 4] = 0.25
    np.save(sample_path, arr)

    grid_params = GridParams(
        feature_names_list=["elevation_grid"],
        target_name="bp",
        out_norm="none",
        terrain_derivatives=["slope", "aspect_sin", "aspect_cos"],
        terrain_cell_size_m=100.0,
    )
    grid_source = GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")

    inputs, _, _ = grid_source.get_sample({"file_path": sample_path})

    torch.testing.assert_close(inputs[1:], torch.zeros((3, 32, 32)), rtol=1e-5, atol=1e-5)


def test_grid_source_terrain_derivatives_require_elevation(temp_data_dir):
    tmpdir, *_ = temp_data_dir
    grid_params = GridParams(
        feature_names_list=["ignition_grid"],
        target_name="bp",
        terrain_derivatives=["slope"],
    )

    with pytest.raises(ValueError, match="requires 'elevation_grid'"):
        GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")


def test_grid_source_rejects_unknown_terrain_derivatives(temp_data_dir):
    tmpdir, *_ = temp_data_dir
    grid_params = GridParams(
        feature_names_list=["elevation_grid"],
        target_name="bp",
        terrain_derivatives=["aspect_degrees"],
    )

    with pytest.raises(ValueError, match="Invalid terrain_derivatives"):
        GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")


def test_mask_threshold(temp_data_dir):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir
    # Set threshold above 1.0 so no samples are valid
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, valid_mask_threshold=1.0)

    assert len(ds) == 0


def test_grid_transforms(temp_data_dir):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir

    base_params_dict = {
        "feature_names_list": ["ignition_grid", "fuel_grid", "elevation_grid"],
        "out_norm": "total_iters",
        "fuel_feats_encoding": "ordinal",
        "normalize_fuel_feats_ordinal": True,
    }

    params_orig = GridParams(**base_params_dict)

    # Get Original Data (No Transforms)
    ds_orig = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={
            "grid": GridSource(
                root_dir=tmpdir,
                params=params_orig,
                modelling_approach="1",
                transform=None,  # No transforms here
            )
        },
    )
    x_orig, y_orig, _ = ds_orig[0]["grid"]

    # =============================================
    # Test 1: Mock Config for Augmentation (Flip)
    # =============================================
    params_flip = GridParams(**base_params_dict, transforms_list=["random_flip"], augmentation_prob=1.0)

    flip_config = DataSourceConfig(name="grid", params=params_flip)
    transform_flip = setup_augmentations(flip_config)

    ds_flip = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={
            "grid": GridSource(
                root_dir=tmpdir,
                params=params_flip,
                modelling_approach="1",
                transform=transform_flip,  # Apply Flip Transform
            )
        },
    )
    x_flip, _, _ = ds_flip[0]["grid"]
    # possible augmented versions
    possible_h = F.hflip(x_orig)
    possible_v = F.vflip(x_orig)
    # check if augmented tensor is one of the flips
    assert torch.allclose(x_flip, possible_h, equal_nan=True) or torch.allclose(x_flip, possible_v, equal_nan=True)
    # =============================================
    # Test 2: Mock Config for Augmentation (Rotate)
    # =============================================
    params_rot = GridParams(**base_params_dict, transforms_list=["random_rotate"], augmentation_prob=1.0)
    rot_config = DataSourceConfig(name="grid", params=params_rot)
    transform_rot = setup_augmentations(rot_config)

    ds_rot = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={
            "grid": GridSource(
                root_dir=tmpdir,
                params=params_rot,
                modelling_approach="1",
                transform=transform_rot,  # Apply Rotate Transform
            )
        },
    )
    # here we test with target since transform should apply to it as well
    _, y_rot, _ = ds_rot[0]["grid"]

    # possible rotations expected
    is_90 = torch.equal(y_rot, torch.rot90(y_orig, 1, dims=[1, 2]))
    is_180 = torch.equal(y_rot, torch.rot90(y_orig, 2, dims=[1, 2]))
    is_270 = torch.equal(y_rot, torch.rot90(y_orig, 3, dims=[1, 2]))

    assert is_90 or is_180 or is_270


def test_get_transforms_returns_grid_augmentations():
    config = DataSourceConfig(
        name="grid",
        params=GridParams(
            feature_names_list=["ignition_grid"],
            transforms_list=["random_flip", "random_rotate"],
            augmentation_prob=0.5,
        ),
    )

    assert get_transforms(config) is not None


def test_tabular_weighted_sampling(temp_data_dir):
    tmpdir, train_csv, _, _, _, _, fire_size_csv, fire_size_feats = temp_data_dir

    fire_size_params = TabularParams(
        csv_name=fire_size_csv,
        feature_names_list=fire_size_feats,
        fire_weather_zone_id_col="grid_code",
        fire_weather_zone_selection_approach="weighted",
        num_samples_per_patch=2,
    )

    fire_size_source = TabularSource(root_dir=tmpdir, params=fire_size_params, modelling_approach="1")

    # Build a small patch where the zone channel has 4 occurrences of 100 and 1 of 200
    data = np.full((32, 32, 7), np.nan, dtype=np.float32)
    zone_channel = 3  # matches the fixture's feature_channel_map
    coords = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    for i, (r, c) in enumerate(coords):
        data[r, c, zone_channel] = 100.0 if i < 4 else 200.0

    # Monkeypatch np.random.choice by temporarily replacing it to capture the probability vector `p`
    captured = {}
    original_choice = np.random.choice

    def fake_choice(n, size, replace, p=None):
        captured["p"] = np.array(p, dtype=float) if p is not None else None
        return np.arange(size, dtype=int)

    try:
        np.random.choice = fake_choice

        sample = fire_size_source.get_sample({"data": data})

        # Ensure we captured probabilities and that sample has expected shape
        assert "p" in captured
        assert captured["p"] is not None
        assert sample.shape == (2, len(fire_size_feats))

        # Compute expected raw weights: for each zone, weight per candidate = count_in_patch / len(zone_cands)
        lut100_len = len(fire_size_source.lut[100])
        lut200_len = len(fire_size_source.lut[200])
        raw_weights = np.concatenate([np.full(lut100_len, 4 / lut100_len), np.full(lut200_len, 1 / lut200_len)])
        expected_p = raw_weights / raw_weights.sum()

        np.testing.assert_allclose(captured["p"], expected_p, rtol=1e-8, atol=1e-12)
    finally:
        np.random.choice = original_choice


@pytest.mark.parametrize(
    ("target_name", "expected_value"),
    [
        ("bp", 0.25),
        ("fi", 0.50),
        ("ros", 0.75),
    ],
)
def test_grid_output_channel_from_feature_map(temp_data_dir, target_name, expected_value):
    tmpdir, *_ = temp_data_dir
    grid_params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        target_name=target_name,
        out_norm="none",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
    )
    grid_source = GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")
    _, target, mask = grid_source.get_sample({"file_path": os.path.join(tmpdir, "sample_0.npy")})
    target_np = target.squeeze(0).numpy()
    mask_np = mask.squeeze(0).numpy()
    np.testing.assert_allclose(target_np[mask_np], expected_value, rtol=1e-6, atol=1e-6)


def test_grid_target_nan_excluded_from_mask(temp_data_dir):
    tmpdir, *_ = temp_data_dir
    sample_path = os.path.join(tmpdir, "sample_0.npy")
    arr = np.load(sample_path)
    arr[0, 0, 5] = np.nan
    np.save(sample_path, arr)

    grid_params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        target_name="fi",
        out_norm="none",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
    )
    grid_source = GridSource(root_dir=tmpdir, params=grid_params, modelling_approach="1")
    _, target, mask = grid_source.get_sample({"file_path": sample_path})

    assert not mask.squeeze(0).numpy()[0, 0]
    assert target.squeeze(0).numpy()[0, 0] == 0.0
