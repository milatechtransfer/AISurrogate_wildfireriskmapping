from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.process_hexels_into_grids import get_split_hexel_window
from src.config import DataPrepConfig
from src.datasets.context_crop import centered_crop_slices, validate_context_crop_metadata
from src.datasets.postprocessing.utils import get_stitched_windows
from src.utils import _crop_visualization_array_to_prediction


def test_context_crop_config_and_slices() -> None:
    config = DataPrepConfig(win_h=512, win_w=512, target_crop_h=256, target_crop_w=256)

    assert config.resolved_target_crop() == (256, 256)
    assert centered_crop_slices(512, 512, 256, 256) == (slice(128, 384), slice(128, 384))
    with pytest.raises(ValueError, match="both be set"):
        DataPrepConfig(win_h=512, win_w=512, target_crop_h=256)


def test_context_crop_metadata_must_match_config() -> None:
    metadata = pd.DataFrame([{"input_win_h": 512, "input_win_w": 512, "target_crop_h": 256, "target_crop_w": 256}])

    validate_context_crop_metadata(metadata, (256, 256), (512, 512))
    with pytest.raises(ValueError, match="does not match"):
        validate_context_crop_metadata(metadata, (128, 128), (512, 512))
    with pytest.raises(ValueError, match="input window"):
        validate_context_crop_metadata(metadata, (256, 256), (256, 256))


def test_patch_preparation_stores_context_around_prediction_tiles(tmp_path: Path) -> None:
    features = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    mask = np.zeros((1, 4, 4), dtype=bool)

    get_split_hexel_window(
        season_cause_stacked_feats=features,
        season_cause_mask=mask,
        season_cause_mapping=None,
        out_dir=str(tmp_path),
        root_dir=str(tmp_path),
        hex_id="01",
        win_h=4,
        win_w=4,
        target_crop_h=2,
        target_crop_w=2,
        overlap_ratio=0.0,
    )

    metadata = pd.read_csv(tmp_path / "meta_hex_01.csv")
    assert metadata[["row", "col"]].values.tolist() == [[0, 0], [0, 2], [2, 0], [2, 2]]
    assert metadata[["input_win_h", "input_win_w", "target_crop_h", "target_crop_w"]].drop_duplicates().values.tolist() == [[4, 4, 2, 2]]

    top_left_patch = np.load(tmp_path / metadata.iloc[0]["filename"])
    assert top_left_patch.shape == (4, 4, 1)
    assert np.isnan(top_left_patch[0, 0, 0])
    np.testing.assert_array_equal(top_left_patch[1:3, 1:3, 0], features[0, 0:2, 0:2, 0])


def test_stitching_places_center_prediction_at_target_coordinates(tmp_path: Path) -> None:
    patch = np.ones((4, 4, 1), dtype=np.float32)
    np.save(tmp_path / "sample.npy", patch)
    metadata = pd.DataFrame(
        [
            {
                "filename": "sample.npy",
                "row": 1,
                "col": 2,
                "target_crop_h": 2,
                "target_crop_w": 2,
            }
        ]
    )
    predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    stitched = get_stitched_windows(
        base_dir=str(tmp_path),
        df=metadata,
        predictions=predictions,
        start_idx=0,
        gt_shape=(4, 5),
    )

    expected = np.full((4, 5), np.nan, dtype=np.float32)
    expected[1:3, 2:4] = predictions[0]
    np.testing.assert_array_equal(stitched, expected)


def test_prediction_visualization_array_uses_center_prediction_crop() -> None:
    array = np.arange(48, dtype=np.float32).reshape(1, 1, 6, 8)

    cropped = _crop_visualization_array_to_prediction(array, prediction_shape=(2, 4))

    np.testing.assert_array_equal(cropped, array[..., 2:4, 2:6])
