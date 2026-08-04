import argparse
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from src import evaluate_hexels
from src.config import (
    Config,
    DataConfig,
    DataSourceConfig,
    EvaluationConfig,
    GridParams,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


def _make_config(tmp_path) -> Config:
    return Config(
        save_dir=str(tmp_path / "out"),
        seed=42,
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"], checkpoint_filename="best.pth"),
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


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        config="unused.yaml",
        visualize_predictions=False,
        save_visualizations=False,
        metrics_only=False,
        skip_hexel_plots=False,
        no_save_predictions=True,
        robust_plot_percentile=None,
        stitch_mode="mean",
        mask_scope="actual",
        run_id=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def _mock_heavy_dependencies(monkeypatch, tmp_path):
    """
    Stub out everything in evaluate_hexels.main() except the code path we're
    testing (building/writing test_metrics_row to test_metrics.csv), so this
    test doesn't need a real dataset, checkpoint, or trained model.
    """

    class FakeDataset(evaluate_hexels.MultiSourceDataset):
        def __init__(self):  # deliberately skip heavy I/O setup in the real __init__
            self.metadata = pd.DataFrame()

    fake_dataset = FakeDataset()
    fake_loader = MagicMock()
    fake_loader.dataset = fake_dataset

    monkeypatch.setattr(evaluate_hexels, "get_test_dataloader", lambda **kwargs: fake_loader)
    monkeypatch.setattr(evaluate_hexels, "get_dataset_dimensions", lambda dataset: (1, {}))

    fake_trainer = MagicMock()
    fake_trainer.load_model.return_value = {"epoch": 3, "metric_value": 0.1}
    fake_trainer.test.return_value = ({"mae": 0.2}, torch.zeros(1, 1, 2, 2).numpy())
    fake_trainer.logger = None
    monkeypatch.setattr(evaluate_hexels, "Trainer", lambda *args, **kwargs: fake_trainer)

    monkeypatch.setattr(evaluate_hexels, "evaluate_and_visualize_hexels", lambda **kwargs: {"all/mae": 0.3})
    monkeypatch.setattr(evaluate_hexels, "print_and_log_eval_metrics", lambda **kwargs: None)

    return fake_trainer


def test_main_writes_test_metrics_csv(tmp_path):
    config = _make_config(tmp_path)
    args = _make_args()

    hexel_metrics = evaluate_hexels.main(args=args, config=config)

    assert hexel_metrics == {"all/mae": 0.3}

    csv_path = tmp_path / "out" / "test_metrics.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.loc[0, "seed"] == 42
    assert df.loc[0, "save_dir"] == str(tmp_path / "out")
    assert df.loc[0, "run_id"] == "" or pd.isna(df.loc[0, "run_id"])
    assert df.loc[0, "test_patch_mae"] == pytest.approx(0.2)
    assert df.loc[0, "test_hexel/all/mae"] == pytest.approx(0.3)


def test_main_writes_run_id_into_test_metrics_csv(tmp_path):
    config = _make_config(tmp_path)
    args = _make_args(run_id=1)

    evaluate_hexels.main(args=args, config=config)

    # apply_run_id_overrides nests save_dir under a per-seed subdirectory and
    # overrides the seed to the one derived from run_id=1.
    from src.config import SEEDS

    expected_save_dir = tmp_path / "out" / f"seed_{SEEDS[1]}"
    df = pd.read_csv(expected_save_dir / "test_metrics.csv")
    assert df.loc[0, "run_id"] == 1
    assert df.loc[0, "seed"] == SEEDS[1]
    assert df.loc[0, "save_dir"] == str(expected_save_dir)
