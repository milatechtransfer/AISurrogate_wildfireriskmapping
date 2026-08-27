import json

import numpy as np
import pandas as pd
import pytest

from data_preparation.prepare_hybrid_v23_dataset import (
    FIRE_SIZE_GLOBAL_FILL_TABLE,
    FIRE_SIZE_STATS,
    FIRE_SIZE_TABLE,
    prepare_v23_dataset,
)


def test_prepare_v23_dataset_reuses_patches_and_builds_training_only_fire_size_reference(tmp_path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    (source_root / "numpy_files").mkdir(parents=True)
    np.save(source_root / "numpy_files" / "patch.npy", np.zeros((2, 2, 1), dtype=np.float32))
    split = pd.DataFrame({"filename": ["numpy_files/patch.npy"], "hex_id": [3]})
    for split_name in ("train_indices.csv", "val_indices.csv", "test_indices.csv"):
        split.to_csv(source_root / split_name, index=False)
    pd.DataFrame(
        {
            "hex_id": [3, 3, 4],
            "GRIDCODE": [1, 36, 2],
            "NORM_LOG1P_IGNITION_COUNT_MEAN": [0.0, 0.0, 0.0],
            "NORM_IGNITION_COUNT_CV": [0.0, 0.0, 0.0],
        }
    ).to_csv(source_root / "ignition_count_processed.csv", index=False)
    pd.DataFrame(
        {
            "GRIDCODE": [1, 1, 1, 2, 2, 36],
            "SIZE_HA": [9.0, 99.0, 999.0, 9_999.0, 99_999.0, 0.0],
        }
    ).to_csv(source_root / FIRE_SIZE_TABLE, index=False)
    (source_root / "feature_channel_map_1.json").write_text('{"firezones_grid": [0, 1]}\n')
    (source_root / "fire_size_norm_params.json").write_text('{"log_size_min": 0, "log_size_max": 6}\n')

    stats = prepare_v23_dataset(source_root, destination_root)

    assert (destination_root / "numpy_files").is_symlink()
    assert (destination_root / "numpy_files").resolve() == (source_root / "numpy_files").resolve()
    assert not (destination_root / "fire_size_norm_params.json").exists()
    processed = pd.read_csv(destination_root / FIRE_SIZE_TABLE)
    global_fill = pd.read_csv(destination_root / FIRE_SIZE_GLOBAL_FILL_TABLE)
    assert set(processed["GRIDCODE"]) == {1, 2}
    assert set(global_fill["GRIDCODE"]) == {1}
    assert "NORM_LOG_SIZE_HA" not in processed
    assert processed["LOG_SIZE_HA"].tolist() == pytest.approx(np.log10(processed["SIZE_HA"] + 1).tolist())

    expected_quantiles = global_fill["LOG_SIZE_HA"].quantile([0.1, 0.5, 0.9]).to_numpy()
    assert stats["neural_mean"] == pytest.approx(float(expected_quantiles.mean()))
    assert stats["neural_std"] == pytest.approx(float(expected_quantiles.std(ddof=0)))
    saved_stats = json.loads((destination_root / FIRE_SIZE_STATS).read_text())
    assert saved_stats["excluded_gridcodes"] == [36]
    assert saved_stats["reference_weighting"] == "equal_training_zone_quantiles"
