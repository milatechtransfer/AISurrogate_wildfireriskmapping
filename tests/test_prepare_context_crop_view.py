from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.prepare_context_crop_view import prepare_context_crop_view


def test_prepare_context_crop_view_reuses_patches_and_recenters_metadata(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "numpy_files").mkdir(parents=True)
    np.save(source / "numpy_files" / "patch.npy", np.zeros((512, 512, 1), dtype=np.float32))
    (source / "feature_channel_map_1.json").write_text("{}")
    row = {
        "filename": "numpy_files/patch.npy",
        "hex_id": 1,
        "row": 10,
        "col": 20,
        "valid_ratio": 1.0,
        "input_win_h": 512,
        "input_win_w": 512,
        "target_crop_h": 256,
        "target_crop_w": 256,
    }
    for split_name in ("train_indices.csv", "val_indices.csv", "test_indices.csv"):
        pd.DataFrame([row]).to_csv(source / split_name, index=False)

    prepare_context_crop_view(source, destination, target_crop_h=128, target_crop_w=128)

    assert (destination / "numpy_files").is_symlink()
    assert (destination / "feature_channel_map_1.json").is_symlink()
    train = pd.read_csv(destination / "train_indices.csv")
    assert train.loc[0, "row"] == 74
    assert train.loc[0, "col"] == 84
    assert train.loc[0, "target_crop_h"] == 128
    assert train.loc[0, "target_crop_w"] == 128
