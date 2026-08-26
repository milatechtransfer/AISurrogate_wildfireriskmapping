"""Run one finite forward/backward update from a training configuration."""

import argparse

import torch

from src.config_io import load_config
from src.datasets.dataset import get_train_val_dataloader
from src.datasets.utils import get_dataset_dimensions, get_dataset_spatial_feature_names
from src.trainer import Trainer
from src.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="Override config.data.root_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config.data.root_dir = args.data_root
    config.logger.enabled = False
    seed_everything(config.seed, config.deterministic)
    train_loader, _ = get_train_val_dataloader(config.data, config.modelling_approach, seed=config.seed)
    spatial_channels, auxiliary_dims = get_dataset_dimensions(train_loader.dataset)
    trainer = Trainer(
        config,
        spatial_input_channels=spatial_channels,
        auxiliary_input_dims=auxiliary_dims,
        train_dataset=train_loader.dataset,
        spatial_input_names=get_dataset_spatial_feature_names(train_loader.dataset),
    )
    batch = next(iter(train_loader))
    predictions, loss, _, _, _ = trainer._step(batch)
    if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
        raise RuntimeError("Smoke batch produced non-finite predictions or loss.")
    trainer.optimizer.zero_grad()
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters()):
        raise RuntimeError("Smoke batch produced non-finite gradients.")
    trainer.optimizer.step()
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"Smoke batch passed: loss={loss.item():.6f}, predictions={tuple(predictions.shape)}, peak_cuda_gb={peak_memory_gb:.3f}")


if __name__ == "__main__":
    main()
