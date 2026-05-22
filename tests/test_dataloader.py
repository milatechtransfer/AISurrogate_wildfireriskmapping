import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as F

from data_preparation.spatial import utils as spatial_utils
from src.config import DataConfig, DataSourceConfig, GridParams, TabularParams
from src.datasets.dataset import MultiSourceDataset, build_dataset, get_test_dataloader, get_train_val_dataloader
from src.datasets.sources import GridSource, TabularSource
from src.datasets.transforms import setup_augmentations


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


@pytest.fixture(autouse=True)
def stable_grid_ranges(monkeypatch):
    monkeypatch.setattr("src.datasets.sources.grids.get_range_output", lambda root_dir, output_type: (1.0, 0.0))
    monkeypatch.setattr("src.datasets.sources.grids.get_range_elevation", lambda root_dir: (1000.0, 0.0))


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
        csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source, "weather": weather_source, "fire_size": fire_size_source}
    )
    sample = ds[0]

    assert "grid" in sample
    assert "weather" in sample
    assert "fire_size" in sample
    input_arr, target, mask = sample["grid"]
    assert isinstance(input_arr, torch.Tensor)
    assert input_arr.shape[0] == 3
    mask_np = mask.squeeze(0).numpy()
    assert mask_np.shape == (32, 32)
    assert not mask_np[1, 1]
    assert not mask_np[10, 20]
    assert mask_np[0, 0]
    weather = sample["weather"]
    assert weather.shape == (2, len(weather_feats))
    fire_size = sample["fire_size"]
    assert fire_size.shape == (2, len(fire_size_feats))


def test_build_dataset_passes_raw_data_dir_to_grid_source(temp_data_dir, monkeypatch):
    tmpdir, train_csv, _, _, _, _, _, _ = temp_data_dir
    raw_data_dir = "/network/raw/source"
    seen = {}

    def fake_get_range_output(root_dir, output_type):
        seen["output"] = (root_dir, output_type)
        return 1.0, 0.0

    def fake_get_range_elevation(root_dir):
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


def test_weather_source_filters_candidates_by_patch_season(temp_data_dir):
    tmpdir, _, _, _, weather_csv, _, _, _ = temp_data_dir
    weather_df = pd.DataFrame(
        {
            "WeatherZone": [100, 100, 100, 100],
            "Season": [1, 1, 2, 2],
            "Temperature": [1.0, 2.0, 101.0, 102.0],
        }
    )
    weather_df.to_csv(os.path.join(tmpdir, weather_csv), index=False)
    with open(os.path.join(tmpdir, "feature_channel_map_2.json"), "w") as f:
        json.dump(
            {
                "fuel_grid": [0],
                "elevation_grid": [1],
                "ignition_grid": [2],
                "firezones_grid": [3],
                "bp_out_grid": [4],
                "fi_out_grid": [5],
                "ros_out_grid": [6],
            },
            f,
        )

    params = TabularParams(
        csv_name=weather_csv,
        feature_names_list=["Temperature"],
        fire_weather_zone_id_col="WeatherZone",
        fire_weather_zone_selection_approach="mode",
        num_samples_per_patch=16,
    )
    weather_source = TabularSource(root_dir=tmpdir, params=params, modelling_approach="2")
    patch_data = np.zeros((8, 8, 7), dtype=np.float32)
    patch_data[:, :, 3] = 100.0

    season_two_sample = weather_source.get_sample({"data": patch_data, "season": 2})
    season_one_sample = weather_source.get_sample({"data": patch_data, "season": 1})
    all_season_sample = weather_source.get_sample({"data": patch_data, "season": "all"})

    assert set(np.unique(season_two_sample[:, 0])).issubset({101.0, 102.0})
    assert set(np.unique(season_one_sample[:, 0])).issubset({1.0, 2.0})
    assert set(np.unique(all_season_sample[:, 0])).issubset({1.0, 2.0, 101.0, 102.0})


def test_weather_source_falls_back_to_zone_candidates_when_season_is_missing(temp_data_dir):
    tmpdir, _, _, _, weather_csv, _, _, _ = temp_data_dir
    weather_df = pd.DataFrame(
        {
            "WeatherZone": [100, 100, 100],
            "Season": [1, 1, 1],
            "Temperature": [3.0, 4.0, 5.0],
        }
    )
    weather_df.to_csv(os.path.join(tmpdir, weather_csv), index=False)
    with open(os.path.join(tmpdir, "feature_channel_map_2.json"), "w") as f:
        json.dump(
            {
                "fuel_grid": [0],
                "elevation_grid": [1],
                "ignition_grid": [2],
                "firezones_grid": [3],
                "bp_out_grid": [4],
                "fi_out_grid": [5],
                "ros_out_grid": [6],
            },
            f,
        )

    params = TabularParams(
        csv_name=weather_csv,
        feature_names_list=["Temperature"],
        fire_weather_zone_id_col="WeatherZone",
        fire_weather_zone_selection_approach="mode",
        num_samples_per_patch=8,
    )
    weather_source = TabularSource(root_dir=tmpdir, params=params, modelling_approach="2")
    patch_data = np.zeros((4, 4, 7), dtype=np.float32)
    patch_data[:, :, 3] = 100.0

    sample = weather_source.get_sample({"data": patch_data, "season": 2})

    assert set(np.unique(sample[:, 0])).issubset({3.0, 4.0, 5.0})


def test_weather_source_ignores_patch_season_for_modelling_approach_one(temp_data_dir):
    tmpdir, _, _, _, weather_csv, _, _, _ = temp_data_dir
    weather_df = pd.DataFrame(
        {
            "WeatherZone": [100, 100, 100, 100],
            "Season": [1, 1, 2, 2],
            "Temperature": [1.0, 2.0, 101.0, 102.0],
        }
    )
    weather_df.to_csv(os.path.join(tmpdir, weather_csv), index=False)

    params = TabularParams(
        csv_name=weather_csv,
        feature_names_list=["Temperature"],
        fire_weather_zone_id_col="WeatherZone",
        fire_weather_zone_selection_approach="mode",
        num_samples_per_patch=16,
    )
    weather_source = TabularSource(root_dir=tmpdir, params=params, modelling_approach="1")
    patch_data = np.zeros((8, 8, 7), dtype=np.float32)
    patch_data[:, :, 3] = 100.0

    sample = weather_source.get_sample({"data": patch_data, "season": 2})

    assert set(np.unique(sample[:, 0])).issubset({1.0, 2.0, 101.0, 102.0})


def test_train_val_dataloader_passes_modelling_approach(monkeypatch):
    seen_calls = []

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"grid": (torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool))}

    def fake_build_dataset(config, csv_name, modelling_approach="1"):
        seen_calls.append((csv_name, modelling_approach))
        return DummyDataset()

    monkeypatch.setattr("src.datasets.dataset.build_dataset", fake_build_dataset)

    config = DataConfig(
        root_dir="/tmp/data",
        raw_data_dir="/tmp/raw",
        train_split="train.csv",
        val_split="val.csv",
        test_split="test.csv",
        input_sources=[],
    )

    get_train_val_dataloader(config=config, modelling_approach="2")

    assert seen_calls == [("train.csv", "2"), ("val.csv", "2")]


def test_test_dataloader_passes_modelling_approach(monkeypatch):
    seen_calls = []

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"grid": (torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool))}

    def fake_build_dataset(config, csv_name, modelling_approach="1"):
        seen_calls.append((csv_name, modelling_approach))
        return DummyDataset()

    monkeypatch.setattr("src.datasets.dataset.build_dataset", fake_build_dataset)

    config = DataConfig(
        root_dir="/tmp/data",
        raw_data_dir="/tmp/raw",
        train_split="train.csv",
        val_split="val.csv",
        test_split="test.csv",
        input_sources=[],
    )

    get_test_dataloader(config=config, modelling_approach="2")

    assert seen_calls == [("test.csv", "2")]


def test_grid_source_computes_log_standard_stats_from_raw_data_dir(temp_data_dir, monkeypatch):
    tmpdir, *_ = temp_data_dir
    raw_data_dir = "/network/raw/source"
    seen = {}
    expected_mean = float(np.log1p(0.5))
    expected_std = 2.0

    def fake_get_output_log_stats(root_dir, output_type):
        seen["log_stats"] = (root_dir, output_type)
        return expected_mean, expected_std

    monkeypatch.setattr("src.datasets.sources.grids.get_output_log_stats", fake_get_output_log_stats)

    grid_params = GridParams(
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        target_name="fi",
        out_norm="log_standard",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
    )

    grid_source = GridSource(root_dir=tmpdir, raw_data_dir=raw_data_dir, params=grid_params, modelling_approach="1")
    _, target, mask = grid_source.get_sample({"file_path": os.path.join(tmpdir, "sample_0.npy")})

    assert seen["log_stats"] == (raw_data_dir, "fire_intensity")
    assert grid_source.target_log_mean == expected_mean
    assert grid_source.target_log_std == expected_std
    target_np = target.squeeze(0).numpy()
    mask_np = mask.squeeze(0).numpy()
    np.testing.assert_allclose(target_np[mask_np], 0.0, atol=1e-6)


def test_get_range_output_rejects_invalid_range(monkeypatch):
    monkeypatch.setattr(spatial_utils, "find_hex_ids", lambda root_dir: [])

    with pytest.raises(ValueError, match="Invalid fire_burn_probability normalization range"):
        spatial_utils.get_range_output("/bad/raw", "fire_burn_probability")


def test_get_range_elevation_rejects_invalid_range(monkeypatch):
    monkeypatch.setattr(spatial_utils, "find_hex_ids", lambda root_dir: [])

    with pytest.raises(ValueError, match="Invalid elevation normalization range"):
        spatial_utils.get_range_elevation("/bad/raw")


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
