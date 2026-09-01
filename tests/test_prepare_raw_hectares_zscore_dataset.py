import json

import numpy as np
import pandas as pd
import pytest

from data_preparation.prepare_raw_hectares_zscore_dataset import (
    FIRE_SIZE_TABLE,
    prepare_raw_hectares_zscore_dataset,
    regenerate_fire_size_csv,
)
from data_preparation.utils import RAW_FIRE_SIZE_ZSCORE_FEATURE, RAW_FIRE_SIZE_ZSCORE_PARAMS


def test_prepare_raw_hectares_zscore_dataset_reuses_256_patches(tmp_path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    (source_root / "numpy_files").mkdir(parents=True)
    np.save(source_root / "numpy_files" / "patch.npy", np.zeros((2, 2, 1), dtype=np.float32))
    split = pd.DataFrame({"filename": ["numpy_files/patch.npy"], "hex_id": [3]})
    for split_name in ("train_indices.csv", "val_indices.csv", "test_indices.csv"):
        split.to_csv(source_root / split_name, index=False)
    pd.DataFrame({"hex_id": [3, 3], "GRIDCODE": [1, 2]}).to_csv(source_root / "ignition_count_processed.csv", index=False)
    pd.DataFrame(
        {
            "GRIDCODE": [1, 1, 1, 2, 2, 2, 3],
            "SIZE_HA": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 5_000.0],
        }
    ).to_csv(source_root / FIRE_SIZE_TABLE, index=False)
    (source_root / "feature_channel_map_1.json").write_text('{"firezones_grid": [0, 1]}\n')
    (source_root / "fire_size_norm_params.json").write_text('{"log_size_min": 0, "log_size_max": 6}\n')

    mean, std = prepare_raw_hectares_zscore_dataset(source_root, destination_root)

    assert (destination_root / "numpy_files").is_symlink()
    assert (destination_root / "numpy_files").resolve() == (source_root / "numpy_files").resolve()
    assert not (destination_root / "fire_size_norm_params.json").exists()
    processed = pd.read_csv(destination_root / FIRE_SIZE_TABLE)
    assert set(processed.columns) == {"GRIDCODE", "SIZE_HA", RAW_FIRE_SIZE_ZSCORE_FEATURE}
    assert 36 in set(processed["GRIDCODE"])
    params = json.loads((destination_root / RAW_FIRE_SIZE_ZSCORE_PARAMS).read_text())
    assert params["transform"] == "identity"
    assert params["normalization"] == "z_score"
    assert params["reference_weighting"] == "equal_training_zone_quantiles"
    assert params["size_ha_mean"] == pytest.approx(mean)
    assert params["size_ha_std"] == pytest.approx(std)


def test_prepare_raw_hectares_zscore_dataset_reuses_existing_params(tmp_path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    (source_root / "numpy_files").mkdir(parents=True)
    np.save(source_root / "numpy_files" / "patch.npy", np.zeros((2, 2, 1), dtype=np.float32))
    split = pd.DataFrame({"filename": ["numpy_files/patch.npy"], "hex_id": [3]})
    for split_name in ("train_indices.csv", "val_indices.csv", "test_indices.csv"):
        split.to_csv(source_root / split_name, index=False)
    pd.DataFrame({"hex_id": [3, 3], "GRIDCODE": [1, 2]}).to_csv(source_root / "ignition_count_processed.csv", index=False)
    pd.DataFrame(
        {
            "GRIDCODE": [1, 1, 1, 2, 2, 2, 3],
            "SIZE_HA": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 5_000.0],
        }
    ).to_csv(source_root / FIRE_SIZE_TABLE, index=False)
    (source_root / "feature_channel_map_1.json").write_text('{"firezones_grid": [0, 1]}\n')

    # Pre-existing frozen params at the destination, deliberately different from what would be
    # fit from source_root, so we can confirm they are reused verbatim rather than refit.
    destination_root.mkdir(parents=True)
    frozen_params = {
        "transform": "identity",
        "normalization": "z_score",
        "feature_name": RAW_FIRE_SIZE_ZSCORE_FEATURE,
        "reference_weighting": "equal_training_zone_quantiles",
        "quantiles": [0.1, 0.5, 0.9],
        "size_ha_mean": 12.0,
        "size_ha_std": 3.0,
    }
    (destination_root / RAW_FIRE_SIZE_ZSCORE_PARAMS).write_text(json.dumps(frozen_params))

    mean, std = prepare_raw_hectares_zscore_dataset(source_root, destination_root)

    assert mean == pytest.approx(12.0)
    assert std == pytest.approx(3.0)
    params = json.loads((destination_root / RAW_FIRE_SIZE_ZSCORE_PARAMS).read_text())
    assert params == frozen_params
    processed = pd.read_csv(destination_root / FIRE_SIZE_TABLE)
    assert processed.loc[processed["GRIDCODE"] == 1, RAW_FIRE_SIZE_ZSCORE_FEATURE].tolist() == pytest.approx(
        [(10.0 - 12.0) / 3.0, (20.0 - 12.0) / 3.0, (30.0 - 12.0) / 3.0]
    )


def test_regenerate_fire_size_csv_writes_only_output_file(tmp_path):
    source_df = tmp_path / "input" / "df_fire_fru_raw.csv"
    source_df.parent.mkdir(parents=True)
    pd.DataFrame({"GRIDCODE": [1, 1, 2], "SIZE_HA": [10.0, 20.0, 100.0]}).to_csv(source_df, index=False)

    params_path = tmp_path / "params" / RAW_FIRE_SIZE_ZSCORE_PARAMS
    params_path.parent.mkdir(parents=True)
    frozen_params = {
        "transform": "identity",
        "normalization": "z_score",
        "feature_name": RAW_FIRE_SIZE_ZSCORE_FEATURE,
        "reference_weighting": "equal_training_zone_quantiles",
        "quantiles": [0.1, 0.5, 0.9],
        "size_ha_mean": 10.0,
        "size_ha_std": 5.0,
    }
    params_path.write_text(json.dumps(frozen_params))

    output_df = tmp_path / "output" / "df_fire_fru_processed.csv"

    mean, std = regenerate_fire_size_csv(source_df, output_df, params_path)

    assert mean == pytest.approx(10.0)
    assert std == pytest.approx(5.0)
    # Only the output file was created; the params file is untouched and nothing else appeared.
    assert json.loads(params_path.read_text()) == frozen_params
    assert list((tmp_path / "output").iterdir()) == [output_df]
    processed = pd.read_csv(output_df)
    assert processed.loc[processed["GRIDCODE"].isin([1, 2]), RAW_FIRE_SIZE_ZSCORE_FEATURE].tolist() == pytest.approx([0.0, 2.0, 18.0])
    assert 36 in set(processed["GRIDCODE"])  # zone-36 zero-size row still gets imputed


def test_regenerate_fire_size_csv_requires_existing_params(tmp_path):
    source_df = tmp_path / "df_fire_fru_raw.csv"
    pd.DataFrame({"GRIDCODE": [1], "SIZE_HA": [10.0]}).to_csv(source_df, index=False)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        regenerate_fire_size_csv(source_df, tmp_path / "out.csv", tmp_path / "missing_params.json")
