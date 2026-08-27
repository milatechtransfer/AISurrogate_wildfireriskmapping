import json

import numpy as np
import pandas as pd
import pytest

from data_preparation.prepare_raw_hectares_zscore_dataset import (
    FIRE_SIZE_TABLE,
    prepare_raw_hectares_zscore_dataset,
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
