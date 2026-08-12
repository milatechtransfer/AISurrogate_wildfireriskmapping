import json
import shutil
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import (
    Config,
    DataConfig,
    DataPrepConfig,
    DataSourceConfig,
    EvaluationConfig,
    GridParams,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.datasets.postprocessing import utils as post_utils
from src.datasets.postprocessing.hexel_reconstruction import (
    StitchedHexel,
    reconstruct_denormalized_hexels,
)
from src.datasets.postprocessing.utils import (
    denormalize_model_target,
    effective_robust_plot_percentile,
    get_hexel_binary_maps,
    get_stitched_windows,
)
from src.datasets.postprocessing.visualize_predictions import (
    _target_prediction_diff_grids,
    _valid_pair_values,
    get_distribution_axis_limit,
    plot_hexbin_distribution,
    plot_histogram_distribution,
    visualize_hexel_iou,
    visualize_target_grids,
)
from src.datasets.targets import get_target_spec
from src.datasets.utils import output_target_norm


def test_get_stitched_windows_uses_prediction_mask_when_provided(tmp_path):
    patch = np.ones((2, 2, 7), dtype=np.float32)
    patch[:, :, 0] = 1.0
    patch[:, :, 5] = 1.0
    patch[0, 1, 5] = np.nan
    np.save(tmp_path / "sample.npy", patch)

    df = pd.DataFrame([["sample.npy", None, None, None, None, 0, 0]])
    predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    stitched = get_stitched_windows(
        base_dir=str(tmp_path),
        df=df,
        predictions=predictions,
        start_idx=0,
        gt_shape=(2, 2),
        target_channel_index=5,
        prediction_mask_channel_indices=[0],
    )

    assert stitched[0, 0] == 1.0
    assert stitched[0, 1] == 2.0
    assert stitched[1, 0] == 3.0
    assert stitched[1, 1] == 4.0


def test_get_stitched_windows_falls_back_to_target_mask(tmp_path):
    patch = np.ones((2, 2, 7), dtype=np.float32)
    patch[:, :, 5] = 1.0
    patch[0, 1, 5] = np.nan
    np.save(tmp_path / "sample.npy", patch)

    df = pd.DataFrame([["sample.npy", None, None, None, None, 0, 0]])
    predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    stitched = get_stitched_windows(
        base_dir=str(tmp_path),
        df=df,
        predictions=predictions,
        start_idx=0,
        gt_shape=(2, 2),
        target_channel_index=5,
    )

    assert stitched[0, 0] == 1.0
    assert np.isnan(stitched[0, 1])
    assert stitched[1, 0] == 3.0
    assert stitched[1, 1] == 4.0


def test_get_stitched_windows_infers_patch_shape(tmp_path):
    patch = np.ones((2, 3, 7), dtype=np.float32)
    patch[:, :, 0] = 1.0
    np.save(tmp_path / "sample.npy", patch)

    df = pd.DataFrame([["sample.npy", None, None, None, None, 0, 0]])
    predictions = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)

    stitched = get_stitched_windows(
        base_dir=str(tmp_path),
        df=df,
        predictions=predictions,
        start_idx=0,
        gt_shape=(2, 3),
        target_channel_index=0,
    )

    np.testing.assert_array_equal(stitched, predictions[0])


def test_bp_target_nodata_is_zero_inside_prediction_support():
    gt = np.array([[np.nan, 0.2], [np.nan, np.nan]], dtype=np.float32)
    pred = np.array([[0.1, 0.3], [np.nan, 0.4]], dtype=np.float32)

    filled_gt, filled_pred = post_utils.fill_bp_target_nodata_as_zero(
        gt_grid=gt,
        pred_grid=pred,
        target=get_target_spec("bp"),
    )

    np.testing.assert_allclose(filled_pred, pred, equal_nan=True)
    assert filled_gt[0, 0] == pytest.approx(0.0)
    assert filled_gt[0, 1] == pytest.approx(0.2)
    assert np.isnan(filled_gt[1, 0])
    assert filled_gt[1, 1] == pytest.approx(0.0)


def test_non_bp_target_nodata_is_not_zero_filled():
    gt = np.array([[np.nan, 2.0]], dtype=np.float32)
    pred = np.array([[1.0, 3.0]], dtype=np.float32)

    filled_gt, filled_pred = post_utils.fill_bp_target_nodata_as_zero(
        gt_grid=gt,
        pred_grid=pred,
        target=get_target_spec("fi"),
    )

    np.testing.assert_allclose(filled_pred, pred, equal_nan=True)
    np.testing.assert_allclose(filled_gt, gt, equal_nan=True)


def test_load_target_grid_bp_nodata_zero_fill_is_configurable(monkeypatch, tmp_path):
    class FakePaths:
        def output_burn_prob(self):
            return tmp_path / "bp.tif"

        def mask_grid(self, hex_id, mask_scope):
            return tmp_path / f"hex{hex_id}_{mask_scope}.shp"

        def mask_grid_actual(self, hex_id):
            return tmp_path / f"hex{hex_id}_actual.shp"

    monkeypatch.setattr(
        post_utils,
        "load_spatial_raster",
        lambda *args, **kwargs: (np.array([[np.nan, 0.2]], dtype=np.float32), {"dtype": "float32"}),
    )
    monkeypatch.setattr(
        post_utils,
        "apply_mask_scope_to_grids",
        lambda gt_grid, pred_grid, **kwargs: (gt_grid, pred_grid),
    )

    pred = np.array([[0.1, 0.3]], dtype=np.float32)
    no_fill_gt, _ = post_utils.load_target_grid_for_mask_scope(
        paths=FakePaths(),
        target=get_target_spec("bp"),
        pred_grid=pred,
        profile={},
        mask_scope="actual",
        hex_id="01",
        bp_nodata_as_zero=False,
    )
    fill_gt, _ = post_utils.load_target_grid_for_mask_scope(
        paths=FakePaths(),
        target=get_target_spec("bp"),
        pred_grid=pred,
        profile={},
        mask_scope="actual",
        hex_id="01",
        bp_nodata_as_zero=True,
    )

    assert np.isnan(no_fill_gt[0, 0])
    assert fill_gt[0, 0] == pytest.approx(0.0)


def test_prediction_support_policy_can_use_target_mask(tmp_path):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    params = GridParams(feature_names_list=["ignition_grid"], target_name="bp")

    assert (
        post_utils.get_prediction_mask_channel_indices(
            data_dir=str(tmp_path),
            modelling_approach="1",
            grid_params=params,
            prediction_support_policy="target",
        )
        is None
    )
    assert post_utils.get_prediction_mask_channel_indices(
        data_dir=str(tmp_path),
        modelling_approach="1",
        grid_params=params,
        prediction_support_policy="input",
    ) == [0]


def test_evaluate_and_visualize_hexels_hides_support_outline_for_target_policy(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )
    shutil.copyfile(tmp_path / "test_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(
            best_ckpt_metrics=["loss"],
            best_ckpt_metrics_mode=["min"],
            prediction_support_policy="target",
        ),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    visualize_calls = []

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", lambda *args, **kwargs: (np.zeros((2, 2), dtype=np.float32), {}))
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: None)
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: visualize_calls.append(kwargs))
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions=None,
    )

    assert visualize_calls[0]["prediction_support_label"] == "target support"
    assert visualize_calls[0]["show_prediction_support_outline"] is False


def test_reconstruct_denormalized_hexels_yields_single_stitched_hexel(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )
    shutil.copyfile(tmp_path / "test_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (2.0, 0.0))
    monkeypatch.setattr(
        post_utils,
        "load_spatial_raster",
        lambda *args, **kwargs: (np.full((2, 2), 0.3, dtype=np.float32), {"dtype": "float32", "marker": "raster"}),
    )

    hexels = list(
        reconstruct_denormalized_hexels(
            test_predictions=np.full((1, 1, 2, 2), 0.5, dtype=np.float32),
            config=config,
            out_norm="none",
        )
    )

    assert len(hexels) == 1
    hexel = hexels[0]
    assert isinstance(hexel, StitchedHexel)
    assert hexel.hex_id == "01"
    assert hexel.target.name == "bp"
    np.testing.assert_allclose(hexel.pred_grid, np.ones((2, 2), dtype=np.float32))
    np.testing.assert_allclose(hexel.gt_grid, np.full((2, 2), 0.3, dtype=np.float32))
    assert hexel.profile["marker"] == "raster"
    assert hexel.actual_support_mask is None
    assert hexel.buffer_support_mask is None


def test_reconstruct_denormalized_hexels_uses_split_csv_when_given(tmp_path, monkeypatch):
    """`split_csv` should let callers stitch a split other than `config.data.test_split`
    (e.g. the validation split, for end-of-training val-set hexel evaluation)."""
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    # test_indices.csv points at a hex_id that has no corresponding patch metadata
    # row, so using it (instead of split_csv) would yield zero stitched hexels.
    pd.DataFrame(
        [{"filename": "patch.npy", "hex_id": 99, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]
    ).to_csv(tmp_path / "test_indices.csv", index=False)
    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "val_indices.csv", index=False
    )
    shutil.copyfile(tmp_path / "val_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (2.0, 0.0))
    monkeypatch.setattr(
        post_utils,
        "load_spatial_raster",
        lambda *args, **kwargs: (np.full((2, 2), 0.3, dtype=np.float32), {"dtype": "float32"}),
    )

    hexels = list(
        reconstruct_denormalized_hexels(
            test_predictions=np.full((1, 1, 2, 2), 0.5, dtype=np.float32),
            config=config,
            out_norm="none",
            split_csv=config.data.val_split,
        )
    )

    assert len(hexels) == 1
    assert hexels[0].hex_id == "01"


def test_load_filtered_test_metadata_raises_with_split_csv_name(tmp_path):
    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
    )

    from src.datasets.postprocessing.hexel_reconstruction import load_filtered_test_metadata

    with pytest.raises(ValueError, match="val_indices.csv"):
        load_filtered_test_metadata(config=config, mask_scope="actual", split_csv=config.data.val_split)


def test_evaluate_and_visualize_hexels_metrics_only_skips_artifacts(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )
    shutil.copyfile(tmp_path / "test_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    artifact_calls = []

    def fail_artifact_call(*args, **kwargs):
        artifact_calls.append((args, kwargs))

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", fail_artifact_call)
    monkeypatch.setattr(post_utils, "visualize_target_grids", fail_artifact_call)
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", fail_artifact_call)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", fail_artifact_call)
    monkeypatch.setattr(post_utils, "visualize_hexel_iou", fail_artifact_call)

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mae": lambda preds, targets, masks: torch.mean(torch.abs(preds[masks] - targets[masks]))},
        save_artifacts=False,
    )

    assert artifact_calls == []
    assert metrics["hex01/mae"] == pytest.approx(1.0)


def test_validate_patch_metadata_mask_scope_rejects_unmarked_buffer_data():
    with pytest.raises(ValueError, match="matching 'mask_scope'"):
        post_utils.validate_patch_metadata_mask_scope(pd.DataFrame({"hex_id": [1]}), "buffer")

    assert post_utils.validate_patch_metadata_mask_scope(pd.DataFrame({"mask_scope": ["buffer"]}), "buffer_only") == "buffer_only"


def test_evaluate_and_visualize_hexels_uses_buffer_scope_paths_and_outputs(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame(
        [
            {
                "filename": "patch.npy",
                "hex_id": 1,
                "valid_ratio": 1.0,
                "season": "spring",
                "cause": "H",
                "row": 0,
                "col": 0,
                "mask_scope": "buffer",
            }
        ]
    ).to_csv(tmp_path / "test_indices.csv", index=False)
    shutil.copyfile(tmp_path / "test_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    mask_paths = []
    save_dirs = []

    def fake_load_spatial_raster(*args, **kwargs):
        mask_paths.append(str(kwargs["mask_path"]))
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: save_dirs.append(kwargs.get("save_dir", args[3])))
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mae": lambda preds, targets, masks: torch.mean(torch.abs(preds[masks] - targets[masks]))},
        mask_scope="buffer",
    )

    assert all(path.endswith("hex01_buffer.shp") for path in mask_paths)
    assert save_dirs == [str(tmp_path / "out" / "buffer_mask_eval")]


def test_evaluate_and_visualize_hexels_isolates_artifacts_under_save_dir_suffix(tmp_path, monkeypatch):
    """`save_dir_suffix` (e.g. "val") should nest stitched artifacts under a dedicated
    subdirectory so they don't collide with the default test-split outputs."""
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "val_indices.csv", index=False
    )
    shutil.copyfile(tmp_path / "val_indices.csv", tmp_path / "train_indices.csv")
    shutil.copyfile(tmp_path / "val_indices.csv", tmp_path / "test_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    save_dirs = []

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", lambda *args, **kwargs: (np.zeros((2, 2), dtype=np.float32), {}))
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: save_dirs.append(kwargs.get("save_dir", args[3])))
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions=None,
        split_csv=config.data.val_split,
        save_dir_suffix="val",
    )

    assert save_dirs == [str(tmp_path / "out" / "val")]


def test_print_and_log_eval_metrics_uses_split_label_and_metric_prefix(capsys):
    experiment_logger = MagicMock()

    post_utils.print_and_log_eval_metrics(
        test_metrics={"mae": 0.1},
        hexel_metrics={"all/mae": 0.2},
        experiment_logger=experiment_logger,
        split_label="Val",
        metric_prefix="val_hexel",
    )

    captured = capsys.readouterr()
    assert "[Val patch-level metrics]" in captured.out
    assert "[Val per-hexel and aggregated metrics]" in captured.out

    experiment_logger.log_metrics.assert_any_call({"val_patch_mae": 0.1})
    experiment_logger.log_metrics.assert_any_call({"val_hexel/all/mae": 0.2})


def test_print_and_log_eval_metrics_defaults_to_test_split(capsys):
    experiment_logger = MagicMock()

    post_utils.print_and_log_eval_metrics(
        test_metrics={"mae": 0.1},
        hexel_metrics={"all/mae": 0.2},
        experiment_logger=experiment_logger,
    )

    captured = capsys.readouterr()
    assert "[Test patch-level metrics]" in captured.out
    assert "[Test per-hexel and aggregated metrics]" in captured.out

    experiment_logger.log_metrics.assert_any_call({"test_patch_mae": 0.1})
    experiment_logger.log_metrics.assert_any_call({"test_hexel/all/mae": 0.2})


def test_buffer_scope_evaluation_reports_actual_and_buffer_only_splits(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame(
        [
            {
                "filename": "patch.npy",
                "hex_id": 1,
                "valid_ratio": 1.0,
                "season": "spring",
                "cause": "H",
                "row": 0,
                "col": 0,
                "mask_scope": "buffer",
            }
        ]
    ).to_csv(tmp_path / "test_indices.csv", index=False)
    shutil.copyfile(tmp_path / "test_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999, "crs": "EPSG:3978", "transform": "mock"}

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "_actual_area_mask", lambda mask_path, profile, shape: np.array([[True, False], [False, False]]))

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mae": lambda preds, targets, masks: torch.mean(torch.abs(preds[masks] - targets[masks]))},
        mask_scope="buffer",
        save_artifacts=False,
    )

    assert metrics["hex01/buffer_mae"] == pytest.approx(1.0)
    assert metrics["hex01/actual_mae"] == pytest.approx(1.0)
    assert metrics["hex01/buffer_only_mae"] == pytest.approx(1.0)
    assert metrics["all/actual_mae"] == pytest.approx(1.0)
    assert metrics["all/buffer_only_mae"] == pytest.approx(1.0)


def test_apply_mask_scope_to_grids_masks_actual_area_for_buffer_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        post_utils,
        "_actual_area_mask",
        lambda mask_path, profile, shape: np.array([[True, False], [False, False]]),
    )

    gt_grid, pred_grid = post_utils.apply_mask_scope_to_grids(
        gt_grid=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        pred_grid=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        profile={},
        mask_path=tmp_path / "actual.shp",
        mask_scope="buffer_only",
        hex_id="01",
    )

    assert np.isnan(gt_grid[0, 0])
    assert np.isnan(pred_grid[0, 0])
    np.testing.assert_allclose(gt_grid[[0, 1, 1], [1, 0, 1]], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(pred_grid[[0, 1, 1], [1, 0, 1]], np.array([6.0, 7.0, 8.0]))


def test_evaluate_and_visualize_hexels_writes_optional_robust_plot(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )
    shutil.copyfile(tmp_path / "test_indices.csv", tmp_path / "train_indices.csv")

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[
                DataSourceConfig(
                    name="grid",
                    params=GridParams(feature_names_list=["ignition_grid"], target_name="bp", out_norm="none"),
                )
            ],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    visualize_calls = []

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    monkeypatch.setattr(post_utils, "get_range_output_cached", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: None)
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: visualize_calls.append(kwargs))
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions=None,
        robust_plot_percentile=99.0,
    )

    assert len(visualize_calls) == 2
    assert visualize_calls[0]["target_label"] == "Burn Probability"
    assert "value_percentile" not in visualize_calls[0]
    assert visualize_calls[1]["value_percentile"] == 99.0
    assert visualize_calls[1]["diff_percentile"] == 99.0
    assert visualize_calls[1]["filename_suffix"] == "_p99"


def test_distribution_axis_limit_uses_non_probability_fallback():
    empty = np.array([], dtype=np.float32)

    assert get_distribution_axis_limit(empty, empty, probability_scale=True) == 0.15
    assert get_distribution_axis_limit(empty, empty, probability_scale=False) == 1.0


def test_distribution_axis_limit_only_caps_probability_scale():
    gt_vals = np.array([0.0, 2.0], dtype=np.float32)
    pred_vals = np.array([3.0], dtype=np.float32)

    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=True) == 1.0
    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=False) > 3.0


def test_select_prediction_target_channel_supports_multi_target_outputs():
    predictions = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
            np.full((2, 2), 3.0, dtype=np.float32),
        ],
        axis=0,
    )[None]

    selected = post_utils.select_prediction_target_channel(predictions, target_name="fi", target_index=1)

    assert selected.shape == (1, 2, 2)
    np.testing.assert_allclose(selected, 2.0)


def test_distribution_plots_drop_masked_nodata_values(tmp_path):
    gt = np.ma.array(
        [[1.0, -9999.0], [3.0, 4.0]],
        mask=[[False, True], [False, False]],
        dtype=np.float32,
    )
    pred = np.array([[1.5, 9999.0], [2.5, 5.0]], dtype=np.float32)

    gt_vals, pred_vals = _valid_pair_values(gt_grid=gt, pred_grid=pred, hex_id="01")

    np.testing.assert_allclose(gt_vals, np.array([1.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(pred_vals, np.array([1.5, 2.5, 5.0], dtype=np.float32))
    assert get_distribution_axis_limit(gt, pred, probability_scale=False) == pytest.approx(9999.0 * 1.05)
    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=False) == pytest.approx(5.0 * 1.05)

    plot_histogram_distribution(gt_grid=gt, pred_grid=pred, hex_id="01", save_dir=str(tmp_path), probability_scale=False)
    plot_hexbin_distribution(gt_grid=gt, pred_grid=pred, hex_id="01", save_dir=str(tmp_path), probability_scale=False)

    assert (tmp_path / "predicted_hexels_plot" / "hist_hex_01.png").exists()
    assert (tmp_path / "predicted_hexels_plot" / "hexbin_hex_01.png").exists()


def test_target_grid_visualization_handles_masked_nodata_and_constant_difference(tmp_path):
    gt = np.ma.array(
        [[2.0, -9999.0], [2.0, 2.0]],
        mask=[[False, True], [False, False]],
        dtype=np.float32,
    )
    pred = np.array([[2.0, 9999.0], [2.0, 2.0]], dtype=np.float32)

    visualize_target_grids(gt_grid=gt, pred_grid=pred, hex_id="01", save_dir=str(tmp_path), target_label="Fire Intensity")

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted.png").exists()


def test_target_grid_visualization_keeps_prediction_support_outside_target():
    gt = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    pred = np.array([[1.5, 2.5], [3.5, np.nan]], dtype=np.float32)

    gt_plot, pred_plot, diff_plot, pred_mask, overlap_mask = _target_prediction_diff_grids(gt_grid=gt, pred_grid=pred, hex_id="01")

    assert pred_plot[0, 1] == pytest.approx(2.5)
    assert np.isnan(gt_plot[0, 1])
    assert np.isnan(diff_plot[0, 1])
    assert np.isnan(pred_plot[1, 1])
    assert pred_mask.tolist() == [[True, True], [True, False]]
    assert overlap_mask.tolist() == [[True, False], [True, False]]


def test_target_grid_visualization_accepts_actual_boundary_mask(tmp_path):
    gt = np.array([[0.0, 1.0, np.nan], [0.0, 2.0, 3.0], [np.nan, 4.0, 5.0]], dtype=np.float32)
    pred = np.ones((3, 3), dtype=np.float32)
    actual_mask = np.array([[False, True, False], [True, True, True], [False, True, False]])

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Burn Probability",
        actual_support_mask=actual_mask,
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted.png").exists()


def test_target_grid_visualization_accepts_buffer_boundary_mask(tmp_path):
    gt = np.array([[0.0, 1.0, np.nan], [0.0, 2.0, 3.0], [np.nan, 4.0, 5.0]], dtype=np.float32)
    pred = np.ones((3, 3), dtype=np.float32)
    buffer_mask = np.ones((3, 3), dtype=bool)

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Burn Probability",
        buffer_support_mask=buffer_mask,
        show_prediction_support_outline=False,
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted.png").exists()


def test_iou_visualization_accepts_actual_and_buffer_boundary_masks(tmp_path):
    gt = np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5], [np.nan, 0.6, 0.7]], dtype=np.float32)
    pred = np.array([[0.2, 0.1, np.nan], [0.35, 0.45, 0.55], [np.nan, 0.65, 0.75]], dtype=np.float32)
    gt_bin = np.isfinite(gt) & (gt >= 0.5)
    pred_bin = np.isfinite(pred) & (pred >= 0.5)
    actual_mask = np.array([[False, True, False], [True, True, True], [False, True, False]])
    buffer_mask = np.isfinite(gt)

    visualize_hexel_iou(
        gt_grid=gt,
        pred_grid=pred,
        gt_bin=gt_bin,
        pred_bin=pred_bin,
        hex_id="01",
        save_dir=str(tmp_path),
        percentile=0.9,
        actual_support_mask=actual_mask,
        buffer_support_mask=buffer_mask,
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_top_10perc_iou.png").exists()


def test_target_grid_visualization_rejects_mismatched_actual_boundary_mask(tmp_path):
    gt = np.ones((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    actual_mask = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="actual_support_mask shape"):
        visualize_target_grids(
            gt_grid=gt,
            pred_grid=pred,
            hex_id="01",
            save_dir=str(tmp_path),
            actual_support_mask=actual_mask,
        )


def test_target_grid_visualization_rejects_mismatched_buffer_boundary_mask(tmp_path):
    gt = np.ones((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    buffer_mask = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="buffer_support_mask shape"):
        visualize_target_grids(
            gt_grid=gt,
            pred_grid=pred,
            hex_id="01",
            save_dir=str(tmp_path),
            buffer_support_mask=buffer_mask,
        )


def test_target_grid_visualization_writes_robust_percentile_variant(tmp_path):
    gt = np.array([[1.0, 2.0], [3.0, 1000.0]], dtype=np.float32)
    pred = np.array([[1.0, 1.5], [3.5, 10.0]], dtype=np.float32)

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Fire Intensity",
        value_percentile=99.0,
        diff_percentile=99.0,
        filename_suffix="_p99",
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted_p99.png").exists()


def test_fi_ros_default_to_p99_robust_plot_percentile():
    assert effective_robust_plot_percentile(get_target_spec("bp"), None) is None
    assert effective_robust_plot_percentile(get_target_spec("fi"), None) == 99.0
    assert effective_robust_plot_percentile(get_target_spec("ros"), None) == 99.0
    assert effective_robust_plot_percentile(get_target_spec("fi"), 98.0) == 98.0


def test_log_standard_target_transform_roundtrip():
    raw = np.array([[0.0, 1.0, 9.0]], dtype=np.float32)
    mean = 1.25
    std = 0.5

    normalized = output_target_norm(
        output_arr=raw,
        target_max=10.0,
        target_min=0.0,
        out_norm="log_standard",
        target_log_mean=mean,
        target_log_std=std,
    )
    recovered = denormalize_model_target(
        data=normalized,
        min_val=0.0,
        max_val=10.0,
        out_norm="log_standard",
        target_log_mean=mean,
        target_log_std=std,
    )

    np.testing.assert_allclose(recovered, raw, rtol=1e-6, atol=1e-6)


def test_log_standard_target_transform_requires_stats():
    raw = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="target_log_mean"):
        output_target_norm(output_arr=raw, target_max=1.0, target_min=0.0, out_norm="log_standard")


def test_get_hexel_binary_maps_respects_masked_arrays(recwarn):
    pred = np.ma.array(
        [[1.0, 5.0], [100.0, 2.0]],
        mask=[[False, False], [True, False]],
        dtype=np.float32,
    )
    gt = np.ma.array(
        [[1.0, 4.0], [9.0, 3.0]],
        mask=[[False, False], [True, False]],
        dtype=np.float32,
    )

    pred_bin, gt_bin = get_hexel_binary_maps(pred_grid=pred, gt_grid=gt, percentile=0.5)

    assert not recwarn
    assert not pred_bin[1, 0]
    assert pred_bin[0, 1]
    assert pred_bin[1, 1]
    assert not gt_bin[1, 0]
    assert gt_bin[0, 1]
    assert gt_bin[1, 1]
