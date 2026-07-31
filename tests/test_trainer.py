from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import (
    Config,
    DataConfig,
    DataSourceConfig,
    EvaluationConfig,
    GridParams,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TargetConfig,
    TargetLossConfig,
    TrainingConfig,
)
from src.losses import MultiTaskLoss
from src.trainer import Trainer

SPATIAL_CHANNELS = 1
WEATHER_FEATS = 5
FIRE_SIZE_FEATS = 3
AUX_SAMPLES = 16  # tabular rows sampled per patch
WIND_CHANNELS = 4
WIND_HEIGHT = 128
WIND_WIDTH = 128


class DummyLoss(torch.nn.Module):
    def forward(self, predictions, targets, masks):
        return torch.tensor(1.0, requires_grad=True)


def dummy_metric(predictions, targets, masks):
    return torch.tensor(0.5)


class GridDataset(torch.utils.data.Dataset):
    """Spatial-only dataset —> yields {'grid': (inputs, targets, masks)}."""

    def __init__(self, size: int = 4, channels: int = 1, height: int = 32, width: int = 32):
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        inputs = torch.rand(self.channels, self.height, self.width)
        targets = inputs * 0.9
        masks = (torch.rand(1, self.height, self.width) > 0.5).float()
        return {"grid": (inputs, targets, masks)}


class MultiTargetGridDataset(GridDataset):
    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        inputs, _, _ = item["grid"]
        targets = torch.cat([inputs * 0.5, inputs * 2.0, inputs * 3.0], dim=0)
        masks = torch.ones_like(targets, dtype=torch.bool)
        item["grid"] = (inputs, targets, masks)
        return item


class WeatherDataset(GridDataset):
    """Spatial + weather tabular dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 1000)
        item["tabular_weather"] = torch.rand(AUX_SAMPLES, WEATHER_FEATS)
        return item


class MultiAuxDataset(GridDataset):
    """Spatial + weather + fire_size tabular dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 1000)
        item["tabular_weather"] = torch.rand(AUX_SAMPLES, WEATHER_FEATS)
        item["tabular_fire_size"] = torch.rand(AUX_SAMPLES, FIRE_SIZE_FEATS)
        return item


class HexMetadataDataset(GridDataset):
    """Spatial dataset with patch metadata for hex-summary losses."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["patch_metadata"] = {"hex_id": torch.tensor(1 if idx < 2 else 2, dtype=torch.long)}
        return item


class WindGridDataset(GridDataset):
    """Spatial + spatial wind grid dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 2000)
        item["wind_grid_mixer"] = torch.rand(WIND_CHANNELS, WIND_HEIGHT, WIND_WIDTH)
        return item


class WindAndWeatherDataset(GridDataset):
    """Spatial + spatial wind grid + weather tabular dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 2000)
        item["wind_grid_mixer"] = torch.rand(WIND_CHANNELS, WIND_HEIGHT, WIND_WIDTH)
        torch.manual_seed(idx + 3000)
        item["tabular_weather"] = torch.rand(AUX_SAMPLES, WEATHER_FEATS)
        return item


def _make_config(
    tmp_path,
    *,
    logger_enabled: bool = False,
    input_branches: list[str] | None = None,
    grid_params: GridParams | None = None,
    num_classes: int = 1,
    optimizer_config: OptimizerConfig | None = None,
    auxiliary_hidden_dims: dict | None = None,
    auxiliary_embed_dims: dict | None = None,
    auxiliary_feature_encoder_poolings: dict | None = None,
    output_head: Literal["shared", "bp_behavior"] = "shared",
) -> Config:
    """Config. factory"""
    if input_branches is None:
        input_branches = ["spatial"]
    resolved_grid_params = grid_params or GridParams(feature_names_list=["dummy_feat"])
    best_checkpoint_metric = "hazard/ccc" if len(resolved_grid_params.resolved_targets()) > 1 else "spearman"

    return Config(
        save_dir=str(tmp_path),
        model=ModelConfig(
            num_classes=num_classes,
            output_head=output_head,
            hidden_features=[8, 16],
            input_branches=input_branches,
            auxiliary_hidden_dims=auxiliary_hidden_dims or {"tabular_weather": [16, 32]},
            auxiliary_embed_dims=auxiliary_embed_dims or {"tabular_weather": 16},
            auxiliary_feature_encoder_poolings=auxiliary_feature_encoder_poolings or {"tabular_weather": "max"},
        ),
        optimizer=optimizer_config or OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(
            best_ckpt_metrics=[best_checkpoint_metric],
            best_ckpt_metrics_mode=["max"],
            checkpoint_filename="best.pth",
        ),
        data=DataConfig(
            root_dir="",
            raw_data_dir="",
            train_split="",
            val_split="",
            test_split="",
            input_sources=[
                DataSourceConfig(
                    name="grid",
                    params=resolved_grid_params,
                )
            ],
        ),
        logger=LoggerConfig(
            enabled=logger_enabled,
            project_name="test",
            workspace="test",
            experiment_name="test",
        ),
        metrics=["mse", "spearman"],
    )


@pytest.fixture
def dummy_config(tmp_path):
    return _make_config(tmp_path)


@pytest.fixture
def dummy_config_with_logger(tmp_path):
    return _make_config(tmp_path, logger_enabled=True)


@pytest.fixture
def auxiliary_config(tmp_path):
    return _make_config(
        tmp_path,
        input_branches=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"tabular_weather": [16, 32]},
        auxiliary_embed_dims={"tabular_weather": 16},
        auxiliary_feature_encoder_poolings={"tabular_weather": "max"},
    )


@pytest.fixture
def multi_aux_config(tmp_path):
    return _make_config(
        tmp_path,
        input_branches=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"tabular_weather": [16, 32], "tabular_fire_size": [16, 32]},
        auxiliary_embed_dims={"tabular_weather": 16, "tabular_fire_size": 16},
        auxiliary_feature_encoder_poolings={"tabular_weather": "max", "tabular_fire_size": "max"},
    )


@pytest.fixture
def wind_grid_config(tmp_path):
    return _make_config(
        tmp_path,
        input_branches=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"wind_grid_mixer": {"mixer": [16], "local": [32, 64, 16], "global": [16]}},
        auxiliary_embed_dims={"wind_grid_mixer": 16},
        auxiliary_feature_encoder_poolings={"wind_grid_mixer": "max"},
    )


@pytest.fixture
def wind_and_weather_config(tmp_path):
    return _make_config(
        tmp_path,
        input_branches=["spatial", "auxiliary"],
        auxiliary_hidden_dims={
            "wind_grid_mixer": {"mixer": [16], "local": [32, 64, 16], "global": [16]},
            "tabular_weather": [16, 32],
        },
        auxiliary_embed_dims={"wind_grid_mixer": 16, "tabular_weather": 16},
        auxiliary_feature_encoder_poolings={"wind_grid_mixer": "max", "tabular_weather": "max"},
    )


@pytest.fixture
def mock_comet_logger(monkeypatch):
    """
    Mock the CometLogger inside src.trainer.
    Apply this fixture explicitly to tests that enable the logger.
    """
    import src.trainer as trainer_module

    mock_logger_cls = MagicMock(name="CometLogger")
    mock_logger_instance = MagicMock(name="logger_instance")
    mock_logger_cls.return_value = mock_logger_instance
    monkeypatch.setattr(trainer_module, "CometLogger", mock_logger_cls, raising=True)
    return mock_logger_instance


@pytest.fixture
def dummy_data():
    ds = GridDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_weather():
    ds = WeatherDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_multi_aux():
    ds = MultiAuxDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_wind_grid():
    ds = WindGridDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_wind_and_weather():
    ds = WindAndWeatherDataset()
    return DataLoader(ds, batch_size=2)


def patch_trainer(trainer: Trainer) -> Trainer:
    """Replace loss and metrics with lightweight stubs for isolated testing."""
    trainer.loss_fn = DummyLoss()
    trainer.metric_functions = {"dummy": dummy_metric, "spearman": dummy_metric}
    return trainer


# Tests — baseline (spatial-only) Trainer
def test_trainer_uses_logger(dummy_config_with_logger, mock_comet_logger):
    Trainer(dummy_config_with_logger, spatial_input_channels=SPATIAL_CHANNELS)
    mock_comet_logger.log_params.assert_called()


def test_trainer_setup(dummy_config):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    assert trainer.spatial_input_channels == SPATIAL_CHANNELS
    assert trainer.model is not None
    assert trainer.loss_fn is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert "mse" in trainer.metric_functions


def test_trainer_step(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    batch = next(iter(dummy_data))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_trainer_step_routes_multi_target_losses(tmp_path):
    config = _make_config(
        tmp_path,
        grid_params=GridParams(
            feature_names_list=["dummy_feat"],
            targets=[
                TargetConfig(name="bp", out_norm="none"),
                TargetConfig(name="fi", out_norm="none"),
                TargetConfig(name="ros", out_norm="none"),
            ],
        ),
        num_classes=3,
        output_head="bp_behavior",
        optimizer_config=OptimizerConfig(
            target_losses={
                "bp": TargetLossConfig(loss="kl", task_weight=0.5),
                "fi": TargetLossConfig(loss="huber", task_weight=0.25),
                "ros": TargetLossConfig(loss="huber", task_weight=0.25),
            }
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    batch = next(iter(DataLoader(MultiTargetGridDataset(size=2), batch_size=2)))

    predictions, loss, loss_parts, targets, masks = trainer._step(batch)

    assert isinstance(trainer.loss_fn, MultiTaskLoss)
    assert predictions.shape == targets.shape == masks.shape
    assert set(loss_parts or {}) == {"bp/total", "fi/total", "ros/total"}
    loss.backward()
    assert trainer.model.multi_output_head.bp_head.weight.grad is not None
    assert trainer.model.multi_output_head.behavior_head.weight.grad is not None


def test_trainer_namespaces_multi_target_metrics_and_adds_hazard(tmp_path):
    config = _make_config(
        tmp_path,
        grid_params=GridParams(
            feature_names_list=["dummy_feat"],
            targets=[
                TargetConfig(name="bp", out_norm="none"),
                TargetConfig(name="fi", out_norm="none"),
                TargetConfig(name="ros", out_norm="none"),
            ],
        ),
        num_classes=3,
        output_head="bp_behavior",
        optimizer_config=OptimizerConfig(
            target_losses={
                "bp": TargetLossConfig(loss="kl", task_weight=0.5),
                "fi": TargetLossConfig(loss="huber", task_weight=0.25),
                "ros": TargetLossConfig(loss="huber", task_weight=0.25),
            }
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    batch = next(iter(DataLoader(MultiTargetGridDataset(size=2), batch_size=2)))
    predictions, _, _, targets, masks = trainer._step(batch)
    metric_predictions, metric_targets = trainer._prepare_metric_tensors(predictions, targets)

    values = trainer._compute_metric_values(metric_predictions, metric_targets, masks)

    assert set(values) == {
        "bp/mse",
        "bp/spearman",
        "fi/mse",
        "fi/spearman",
        "ros/mse",
        "ros/spearman",
        "hazard/ccc",
    }
    assert torch.isfinite(values["hazard/ccc"])


def test_hazard_metric_penalizes_no_burn_false_positives(tmp_path):
    config = _make_config(
        tmp_path,
        grid_params=GridParams(
            feature_names_list=["dummy_feat"],
            targets=[
                TargetConfig(name="bp", out_norm="none"),
                TargetConfig(name="fi", out_norm="none"),
                TargetConfig(name="ros", out_norm="none"),
            ],
        ),
        num_classes=3,
        output_head="bp_behavior",
        optimizer_config=OptimizerConfig(
            target_losses={
                "bp": TargetLossConfig(loss="kl", task_weight=0.5),
                "fi": TargetLossConfig(loss="huber", task_weight=0.25),
                "ros": TargetLossConfig(loss="huber", task_weight=0.25),
            }
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    targets = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [2.0, 4.0]], [[0.0, 0.0], [1.0, 1.0]]]])
    masks = torch.tensor([[[[True, True], [True, True]], [[False, False], [True, True]], [[False, False], [True, True]]]])
    good_predictions = targets.clone()
    bad_predictions = targets.clone()
    bad_predictions[:, 0, 0] = 0.5
    bad_predictions[:, 1, 0] = 10.0

    good_ccc = trainer._compute_metric_values(good_predictions, targets, masks)["hazard/ccc"]
    bad_ccc = trainer._compute_metric_values(bad_predictions, targets, masks)["hazard/ccc"]

    assert good_ccc > bad_ccc


def test_multi_target_model_overfits_one_tiny_batch(tmp_path):
    torch.manual_seed(7)
    config = _make_config(
        tmp_path,
        grid_params=GridParams(
            feature_names_list=["dummy_feat"],
            targets=[
                TargetConfig(name="bp", out_norm="none"),
                TargetConfig(name="fi", out_norm="none"),
                TargetConfig(name="ros", out_norm="none"),
            ],
        ),
        num_classes=3,
        output_head="bp_behavior",
        optimizer_config=OptimizerConfig(
            name="Adam",
            lr=1e-2,
            target_losses={
                "bp": TargetLossConfig(loss="kl", task_weight=0.5),
                "fi": TargetLossConfig(loss="huber", task_weight=0.25),
                "ros": TargetLossConfig(loss="huber", task_weight=0.25),
            },
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    batch = next(iter(DataLoader(MultiTargetGridDataset(size=2, height=8, width=8), batch_size=2)))

    initial_parts = None
    final_parts = None
    for step in range(41):
        _, loss, loss_parts, _, _ = trainer._step(batch)
        if step == 0:
            initial_parts = {name: value.detach().clone() for name, value in (loss_parts or {}).items()}
        if step == 40:
            final_parts = loss_parts
            break
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

    assert initial_parts is not None
    assert final_parts is not None
    for target_name in ("bp", "fi", "ros"):
        key = f"{target_name}/total"
        assert final_parts[key] < initial_parts[key]


def test_metric_tensors_inverse_log_standard(tmp_path):
    mean = 2.0
    std = 0.5
    config = _make_config(
        tmp_path,
        grid_params=GridParams(
            feature_names_list=["dummy_feat"],
            target_name="fi",
            out_norm="log_standard",
            target_log_mean=mean,
            target_log_std=std,
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)

    predictions = torch.tensor([[[[0.0, 1.0]]]])
    targets = torch.tensor([[[[-1.0, 0.5]]]])
    metric_predictions, metric_targets = trainer._prepare_metric_tensors(predictions, targets)

    expected_predictions = torch.expm1(predictions * std + mean).clamp_min(0.0)
    expected_targets = torch.expm1(targets * std + mean).clamp_min(0.0)
    assert torch.allclose(metric_predictions, expected_predictions)
    assert torch.allclose(metric_targets, expected_targets)


def test_train_epoch_runs(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data)
    assert "loss" in results
    assert "dummy" in results


def test_train_epoch_runs_with_hex_summary_loss(tmp_path):
    config = _make_config(
        tmp_path,
        optimizer_config=OptimizerConfig(
            loss=["kl", "ccc", "hex_mean_pearson", "hex_top10_pearson"],
            name="Adam",
            lr=0.001,
            loss_weights={"kl": 0.45, "ccc": 0.45, "hex_mean_pearson": 0.05, "hex_top10_pearson": 0.05},
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    loader = DataLoader(HexMetadataDataset(size=4), batch_size=4)

    results = trainer.train_epoch(loader)

    assert "loss" in results
    assert "loss_hex_mean_pearson" in results
    assert "loss_hex_top10_pearson" in results


def test_validate_runs(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results = trainer.validate(dummy_data)
    assert "loss" in results
    assert "dummy" in results


def test_validate_return_predictions(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4  # dataset size
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range after sigmoid"


def test_save_and_load_model(tmp_path, dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    trainer.train_epoch(dummy_data)
    save_path = trainer.save_model(epoch=1, metric_value=0.5)

    assert tmp_path.joinpath("last.pth").exists()

    checkpoint = trainer.load_model(path=save_path)
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint


def test_run_training(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    trainer.run_training(dummy_data, dummy_data)


def test_checkpoint_carries_resume_state(tmp_path, dummy_data):
    config = _make_config(tmp_path)
    config.training.max_epochs = 1
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    trainer.run_training(dummy_data, dummy_data)

    assert tmp_path.joinpath("last.pth").exists()
    checkpoint = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert "scheduler_state" in checkpoint
    assert "best_metric_list" in checkpoint
    assert "global_step" in checkpoint
    assert "rng_state" in checkpoint


def test_maybe_resume_continues_from_last_checkpoint(tmp_path, dummy_data):
    from src.schedulers import build_lr_scheduler

    config = _make_config(tmp_path)
    config.training.max_epochs = 1
    first_run = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(first_run)
    first_run.run_training(dummy_data, dummy_data)

    # A fresh Trainer over the same save_dir resumes rather than restarting from epoch 1.
    resumed = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(resumed)
    lr_scheduler, _ = build_lr_scheduler(resumed.config, resumed.optimizer, dummy_data)
    assert resumed._maybe_resume(lr_scheduler) == 2


def test_maybe_resume_ignores_incompatible_checkpoint(tmp_path, dummy_data):
    config = _make_config(tmp_path)
    config.training.max_epochs = 1
    first_run = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(first_run)
    first_run.run_training(dummy_data, dummy_data)

    # A model with different layer shapes cannot resume from the saved checkpoint.
    bigger_config = _make_config(tmp_path)
    bigger_config.model.hidden_features = [16, 32]
    other = Trainer(bigger_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(other)
    assert other._maybe_resume() == 1


def test_load_previous_experiment_key_returns_none_without_checkpoint(dummy_config):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    assert trainer._load_previous_experiment_key() is None


def test_checkpoint_persists_comet_experiment_key(tmp_path, dummy_data, mock_comet_logger):
    mock_comet_logger.experiment_key = "abc123"
    config = _make_config(tmp_path, logger_enabled=True)
    config.training.max_epochs = 1
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    trainer.run_training(dummy_data, dummy_data)

    checkpoint = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["comet_experiment_key"] == "abc123"


def test_trainer_resumes_comet_experiment_from_checkpoint(tmp_path, dummy_data, mock_comet_logger):
    mock_comet_logger.experiment_key = "abc123"
    config = _make_config(tmp_path, logger_enabled=True)
    config.training.max_epochs = 1
    first_run = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(first_run)
    first_run.run_training(dummy_data, dummy_data)

    # A fresh Trainer over the same save_dir should read back the stored Comet
    # experiment key and pass it through so CometLogger resumes into it.
    import src.trainer as trainer_module

    mock_logger_cls = trainer_module.CometLogger
    mock_logger_cls.reset_mock()
    mock_logger_cls.return_value = mock_comet_logger

    Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)

    _, kwargs = mock_logger_cls.call_args
    assert kwargs["previous_experiment_key"] == "abc123"


def test_test_method(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results = trainer.test(dummy_data)
    assert "loss" in results
    assert "dummy" in results


# Tests — multi-source Trainer with weather encoder
def test_auxiliary_trainer_setup(auxiliary_config):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_auxiliary_trainer_step(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_weather))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_auxiliary_train_epoch_runs(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_weather)
    assert "loss" in results
    assert "dummy" in results


def test_auxiliary_validate_runs(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.validate(dummy_data_weather)
    assert "loss" in results
    assert "dummy" in results


def test_auxiliary_validate_return_predictions(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data_weather, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range after sigmoid"


def test_auxiliary_run_training(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    trainer.run_training(dummy_data_weather, dummy_data_weather)


# Tests — multi-source Trainer with weather + fire_size encoders
def test_multi_aux_trainer_setup(multi_aux_config):
    trainer = Trainer(
        multi_aux_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS, "tabular_fire_size": FIRE_SIZE_FEATS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None


def test_multi_aux_trainer_step(multi_aux_config, dummy_data_multi_aux):
    trainer = Trainer(
        multi_aux_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS, "tabular_fire_size": FIRE_SIZE_FEATS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_multi_aux))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_multi_aux_train_epoch_runs(multi_aux_config, dummy_data_multi_aux):
    trainer = Trainer(
        multi_aux_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"tabular_weather": WEATHER_FEATS, "tabular_fire_size": FIRE_SIZE_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_multi_aux)
    assert "loss" in results
    assert "dummy" in results


# Tests — multi-source Trainer with WindFeatureEncoder (spatial wind grid)
def test_wind_grid_trainer_setup(wind_grid_config):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_wind_grid_trainer_step(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_wind_grid))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_wind_grid_train_epoch_runs(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_wind_grid)
    assert "loss" in results
    assert "dummy" in results


def test_wind_grid_validate_runs(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    results = trainer.validate(dummy_data_wind_grid)
    assert "loss" in results
    assert "dummy" in results


def test_wind_grid_validate_return_predictions(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data_wind_grid, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4  # dataset size
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range after sigmoid"


def test_wind_grid_run_training(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    trainer.run_training(dummy_data_wind_grid, dummy_data_wind_grid)


# Tests — multi-source Trainer with WindFeatureEncoder + TabularFeatureEncoder (wind_grid + weather)
def test_wind_and_weather_trainer_setup(wind_and_weather_config):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "tabular_weather": WEATHER_FEATS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_wind_and_weather_trainer_step(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_wind_and_weather))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_wind_and_weather_train_epoch_runs(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_wind_and_weather)
    assert "loss" in results
    assert "dummy" in results


def test_wind_and_weather_validate_runs(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.validate(dummy_data_wind_and_weather)
    assert "loss" in results
    assert "dummy" in results


def test_wind_and_weather_run_training(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "tabular_weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    trainer.run_training(dummy_data_wind_and_weather, dummy_data_wind_and_weather)
