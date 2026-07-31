import json
import logging
import os
import random
import time
from typing import Any, Literal, cast, overload

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation.spatial.utils import NORM_STATS_JSON, get_output_log_stats_cached, get_range_output_cached, read_split_hex_ids
from src.config import Config, GridParams
from src.datasets.fuel_utils import FUEL_CURVE_ENCODINGS
from src.datasets.targets import activate_target_predictions, get_target_specs
from src.datasets.utils import apply_bp_nodata_zero_range, get_fuel_curve_normalization_stats
from src.logger import CometLogger
from src.losses import MultiTaskLoss, WeightedLoss
from src.models.factory import build_model, resolve_model_architecture
from src.models.utils import get_nbr_model_parameters
from src.schedulers import build_lr_scheduler
from src.utils import AVAILABLE_METRICS, build_single_loss, set_device

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: Config,
        spatial_input_channels: int | None = None,
        auxiliary_input_dims: dict[str, int] | None = None,
        train_dataset=None,
    ):
        self.config = config
        self.spatial_input_channels = spatial_input_channels
        self.auxiliary_input_dims = auxiliary_input_dims if auxiliary_input_dims is not None else {}
        self.train_dataset = train_dataset

        # Set device
        self.device = set_device()
        print(f"\n[Device] Using: {self.device}")

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Comet Logger
        self.logger = None
        # Only initialize logger if not in test-only mode
        if self.config.logger.enabled:
            previous_experiment_key = self._load_previous_experiment_key()
            self.logger = CometLogger(
                project_name=self.config.logger.project_name,
                workspace=self.config.logger.workspace,
                experiment_name=self.config.logger.experiment_name,
                experiment_tags=self.config.logger.tags,
                previous_experiment_key=previous_experiment_key,
            )
            if previous_experiment_key:
                print(f"[Comet] Resuming experiment {previous_experiment_key} instead of creating a new one.")
            self.log_every_n_step = self.config.logger.log_every_n_step
            # log all the params.
            self.logger.log_params(self.config.model_dump())

        self.setup()

        # metrics for best checkpoint saving
        self.best_ckpt_metrics = list(self.config.evaluation.best_ckpt_metrics)
        self.best_ckpt_modes = list(self.config.evaluation.best_ckpt_metrics_mode)
        if len(self.best_ckpt_metrics) != len(self.best_ckpt_modes):
            raise ValueError("Number of best_ckpt_metric and best_ckpt_metric_mode must match!")
        self._best_metric_list: list[float] = []

    def _load_previous_experiment_key(self) -> str | None:
        """
        If a checkpoint from a previous (e.g. preempted/requeued) run of this same
        save_dir exists, return its stored Comet experiment key so the logger can
        continue logging into that same experiment instead of starting a new one.
        """
        last_path = os.path.join(self.save_dir, "last.pth")
        if not os.path.exists(last_path):
            return None
        try:
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - best-effort lookup, never block startup
            print(f"[Comet] Could not read previous checkpoint at {last_path} to resume experiment: {exc}")
            return None
        return checkpoint.get("comet_experiment_key")

    def setup(self):
        """
        Define model, loss function and optimizer.
        """

        self._grid_params = self._get_grid_params()
        target_names = [target.name for target in self._grid_params.resolved_targets()] if self._grid_params is not None else ["bp"]
        self._target_specs = get_target_specs(target_names)
        self._target_names = [target.name for target in self._target_specs]
        if self.config.model.num_classes != len(self._target_specs):
            raise ValueError(
                f"model.num_classes={self.config.model.num_classes} must match number of configured targets "
                f"({len(self._target_specs)}: {self._target_names})."
            )

        # Flag to indicate we are including auxiliary features
        self.auxiliary = "auxiliary" in self.config.model.input_branches

        resolved_architecture = resolve_model_architecture(self.config.model)
        print(f"[Trainer] Model architecture: {self.config.model.architecture} -> {resolved_architecture}")
        print(f"[Trainer] Spatial Channels: {self.spatial_input_channels}, Auxiliary Dim: {self.auxiliary_input_dims}")

        self.model = build_model(
            model_config=self.config.model,
            spatial_input_channels=self.spatial_input_channels,
            auxiliary_input_dims=self.auxiliary_input_dims,
            **self._get_fuel_curve_stats(),
            target_names=self._target_names,
        )

        self.model.to(self.device)

        # Get and log number of model params.
        total_params, trainable_params = get_nbr_model_parameters(self.model)
        print(f"Model Params: Total={total_params:,} | Trainable={trainable_params:,}")
        if self.logger:
            self.logger.log_params({"model_total_params": total_params, "model_trainable_params": trainable_params})

        self.loss_fn = self._build_loss()

        # Setup optimizer
        opt_name = self.config.optimizer.name
        # TODO: add other parameters
        opt_params = {
            "lr": self.config.optimizer.lr,
        }

        OptimizerClass = getattr(optim, opt_name)
        self.optimizer = OptimizerClass(self.model.parameters(), **opt_params)

        self.global_step = 0

        # Validate and load metrics from config.
        self._validate_and_load_metrics()

        self._configure_metric_target_transform()

    def _get_fuel_curve_stats(self) -> dict[str, torch.Tensor | None]:
        """Extract fuel curve normalization stats for model initialisation.

        Priority:
        1. GridSource on the training dataset (training mode).
        2. ``dataset_norm_stats.json`` in ``root_dir`` (eval-only mode, no train dataset).
        3. None → model falls back to zeros/ones (shape mismatch risk if checkpoint differs).
        """
        grid_params = self._get_grid_params()
        if grid_params is None or grid_params.fuel_feats_encoding not in FUEL_CURVE_ENCODINGS:
            return {"fuel_curve_mean": None, "fuel_curve_std": None}

        # Training mode: read from GridSource which already computed/cached the stats.
        if self.train_dataset is not None:
            fuel_curve_mean, fuel_curve_std = get_fuel_curve_normalization_stats(self.train_dataset)
            return {"fuel_curve_mean": fuel_curve_mean, "fuel_curve_std": fuel_curve_std}

        # Eval-only mode: try dataset_norm_stats.json.
        cache_path = os.path.join(self.config.data.root_dir, NORM_STATS_JSON)
        cache_key = f"fuel_curve_{grid_params.fuel_feats_encoding}"
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            entry = cached.get(cache_key, {})
            mean = entry.get("log_mean")
            std = entry.get("log_std")
            if mean is not None and std is not None:
                logger.info("Fuel curve stats for model init loaded from %s: mean=%.4f, std=%.4f", cache_path, mean, std)
                return {
                    "fuel_curve_mean": torch.tensor([float(mean)], dtype=torch.float32),
                    "fuel_curve_std": torch.tensor([float(std)], dtype=torch.float32),
                }
        logger.warning(
            "Fuel curve stats not available (no train dataset and %r not found or missing key %r). "
            "Model will use default zeros/ones — shape may mismatch checkpoint.",
            cache_path,
            cache_key,
        )
        return {"fuel_curve_mean": None, "fuel_curve_std": None}

    def _build_loss(self) -> torch.nn.Module:
        if self.config.optimizer.target_losses:
            configured_targets = set(self.config.optimizer.target_losses)
            expected_targets = set(self._target_names)
            if configured_targets != expected_targets:
                raise ValueError(
                    f"optimizer.target_losses must match configured targets, got {sorted(configured_targets)} "
                    f"and expected {sorted(expected_targets)}."
                )
            task_losses = {}
            task_weights = {}
            for target_name in self._target_names:
                target_config = self.config.optimizer.target_losses[target_name]
                task_losses[target_name] = self._build_loss_group(
                    target_config.loss,
                    target_config.loss_weights,
                    target_config.huber_beta,
                )
                task_weights[target_name] = target_config.task_weight
            return MultiTaskLoss(target_names=self._target_names, losses=task_losses, task_weights=task_weights)

        if self.config.optimizer.loss is None:
            raise RuntimeError("Legacy loss configuration is missing.")
        return self._build_loss_group(
            self.config.optimizer.loss,
            self.config.optimizer.loss_weights,
            self.config.optimizer.huber_beta,
        )

    @staticmethod
    def _build_loss_group(
        loss_config: str | list[str],
        weights: dict[str, float],
        huber_beta: float,
    ) -> torch.nn.Module:
        loss_kwargs = {"huber_beta": huber_beta}
        if isinstance(loss_config, str):
            return build_single_loss(loss_config, **loss_kwargs)
        loss_names = loss_config
        losses = {n: build_single_loss(n, **loss_kwargs) for n in loss_names}
        return WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

    def _get_grid_params(self) -> GridParams | None:
        for source in self.config.data.input_sources:
            if source.name == "grid" and isinstance(source.params, GridParams):
                return source.params
        return None

    def _target_out_norm(self, target_name: str) -> str:
        if self._grid_params is None:
            return "none"
        return self._grid_params.target_config(target_name).out_norm

    def _target_log_stats(self, target_name: str) -> tuple[float | None, float | None]:
        if self._grid_params is None:
            return None, None
        target_config = self._grid_params.target_config(target_name)
        return target_config.log_mean, target_config.log_std

    def _configure_metric_target_transform(self) -> None:
        self._metric_out_norms: list[str] = []
        self._metric_target_mins: list[float] = []
        self._metric_target_maxs: list[float] = []
        self._metric_target_log_means: list[float | None] = []
        self._metric_target_log_stds: list[float | None] = []

        if self._grid_params is None:
            self._metric_out_norms = ["none"] * len(self._target_specs)
            self._metric_target_mins = [0.0] * len(self._target_specs)
            self._metric_target_maxs = [1.0] * len(self._target_specs)
            self._metric_target_log_means = [None] * len(self._target_specs)
            self._metric_target_log_stds = [None] * len(self._target_specs)
            return

        # Normalization stats are train-only: derive the allowed hexes from the train split so
        # held-out hexes never leak into target normalization constants.
        train_hex_ids: set[int] | None = None
        if self.config.data.root_dir and self.config.data.train_split:
            split_path = os.path.join(self.config.data.root_dir, self.config.data.train_split)
            if os.path.isfile(split_path):
                train_hex_ids = read_split_hex_ids(split_path)
            else:
                logger.warning(
                    "Train split file %r not found — normalization stats will use all hexels.",
                    split_path,
                )

        for target in self._target_specs:
            out_norm = self._target_out_norm(target.name)
            target_min = 0.0
            target_max = 1.0
            target_log_mean, target_log_std = self._target_log_stats(target.name)

            if out_norm == "min_max":
                root_dir = self.config.data.root_dir or self.config.data.raw_data_dir
                if root_dir:
                    target_max, target_min = get_range_output_cached(
                        root_dir=root_dir,
                        output_type=target.output_type,
                        allowed_hex_ids=train_hex_ids,
                        raw_data_dir=self.config.data.raw_data_dir,
                    )
                    target_max, target_min = apply_bp_nodata_zero_range(
                        target_name=target.name,
                        max_value=target_max,
                        min_value=target_min,
                        bp_nodata_as_zero=self._grid_params.bp_nodata_as_zero,
                    )
            elif out_norm == "log_standard":
                if (target_log_mean is None or target_log_std is None) and self.config.data.root_dir:
                    target_log_mean, target_log_std = get_output_log_stats_cached(
                        root_dir=self.config.data.root_dir,
                        output_type=target.output_type,
                        allowed_hex_ids=train_hex_ids,
                        raw_data_dir=self.config.data.raw_data_dir,
                    )
                if target_log_mean is None or target_log_std is None:
                    raise ValueError(f"target_log_mean/std are required for target={target.name!r} with out_norm='log_standard'.")
                if target_log_std <= 0.0:
                    raise ValueError(f"target_log_std must be positive for target={target.name!r}, got {target_log_std}.")
            elif out_norm not in {"log", "none", "total_iters", "season_cause_iters"}:
                raise ValueError(f"Unsupported output normalization for target={target.name!r}: {out_norm!r}")

            self._metric_out_norms.append(out_norm)
            self._metric_target_mins.append(target_min)
            self._metric_target_maxs.append(target_max)
            self._metric_target_log_means.append(target_log_mean)
            self._metric_target_log_stds.append(target_log_std)

    def _inverse_model_target_for_metrics(self, data: torch.Tensor) -> torch.Tensor:
        data = data.float()
        transformed_channels = []
        for idx, out_norm in enumerate(self._metric_out_norms):
            channel = data[:, idx : idx + 1]
            if out_norm == "min_max":
                target_min = self._metric_target_mins[idx]
                target_max = self._metric_target_maxs[idx]
                transformed_channels.append(channel * (target_max - target_min) + target_min)
            elif out_norm == "log":
                transformed_channels.append(torch.expm1(channel * float(np.log1p(1000.0))) / 1000.0)
            elif out_norm == "log_standard":
                target_log_mean = self._metric_target_log_means[idx]
                target_log_std = self._metric_target_log_stds[idx]
                if target_log_mean is None or target_log_std is None:
                    raise RuntimeError("log_standard metric transform was not configured.")
                transformed_channels.append(torch.expm1(channel * target_log_std + target_log_mean).clamp_min(0.0))
            else:
                transformed_channels.append(channel)
        return torch.cat(transformed_channels, dim=1)

    def _prepare_metric_tensors(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inverse_model_target_for_metrics(predictions), self._inverse_model_target_for_metrics(targets)

    def _activate_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        return activate_target_predictions(predictions, self._target_specs)

    def _metric_result_keys(self) -> list[str]:
        if len(self._target_specs) == 1:
            return list(self.metric_functions)
        keys = [f"{target.name}/{metric_name}" for target in self._target_specs for metric_name in self.metric_functions]
        if {"bp", "fi"} <= set(self._target_names):
            keys.append("hazard/ccc")
        return keys

    def _compute_metric_values(self, predictions: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        if predictions.device.type == "mps":
            predictions = predictions.cpu()
            targets = targets.cpu()
            masks = masks.cpu()

        if len(self._target_specs) == 1:
            return {name: metric_fn(predictions, targets, masks) for name, metric_fn in self.metric_functions.items()}

        metric_values = {}
        for channel_idx, target in enumerate(self._target_specs):
            channel = slice(channel_idx, channel_idx + 1)
            for metric_name, metric_fn in self.metric_functions.items():
                metric_values[f"{target.name}/{metric_name}"] = metric_fn(
                    predictions[:, channel],
                    targets[:, channel],
                    masks[:, channel],
                )

        if {"bp", "fi"} <= set(self._target_names):
            bp_idx = self._target_names.index("bp")
            fi_idx = self._target_names.index("fi")
            bp_predictions = predictions[:, bp_idx : bp_idx + 1]
            bp_targets = targets[:, bp_idx : bp_idx + 1]
            fi_predictions = predictions[:, fi_idx : fi_idx + 1]
            fi_targets = targets[:, fi_idx : fi_idx + 1]
            bp_mask = masks[:, bp_idx : bp_idx + 1].bool()
            fi_mask = masks[:, fi_idx : fi_idx + 1].bool()
            if self.config.evaluation.hazard_fi_cap is not None:
                fi_cap = self.config.evaluation.hazard_fi_cap
                fi_predictions = fi_predictions.clamp_max(fi_cap)
                fi_targets = fi_targets.clamp_max(fi_cap)
            no_burn = bp_targets <= 0.0
            hazard_mask = bp_mask & (fi_mask | no_burn)
            hazard_targets = bp_targets * torch.where(fi_mask, fi_targets, torch.zeros_like(fi_targets))
            metric_values["hazard/ccc"] = AVAILABLE_METRICS["ccc"](
                bp_predictions * fi_predictions,
                hazard_targets,
                hazard_mask,
            )

        return metric_values

    def _validate_and_load_metrics(self) -> None:
        """Helper to validate and load metrics to be computed."""
        if not set(self.config.metrics).issubset(AVAILABLE_METRICS):
            raise ValueError(f"Invalid metrics found.Available options: {list(AVAILABLE_METRICS)}")
        self.metric_functions = {k: AVAILABLE_METRICS[k] for k in self.config.metrics}

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """
        Default step. Expects batch -> {'grid': (inputs, targets, masks), 'tabular_weather': ...}.
        Returns (predictions, loss, loss_parts, targets_on_device, masks_on_device).
        """
        # Get the spatial grid inputs, targets and masks.
        if "grid" not in batch:
            raise ValueError("Batch is missing required 'grid' data.")
        inputs, targets, masks = [t.to(self.device) for t in batch["grid"]]

        patch_metadata = batch.get("patch_metadata")

        # Unpack all potential auxiliary data
        auxiliary_data = {}
        for key, value in batch.items():
            if key in {"grid", "patch_metadata"}:
                continue
            auxiliary_data[key] = value.to(self.device)

        predictions = self.model(inputs, auxiliary_data)

        if getattr(self.loss_fn, "requires_patch_metadata", False):
            if patch_metadata is None:
                raise ValueError("Configured loss requires patch metadata, but batch does not include 'patch_metadata'.")
            patch_metadata = {key: value.to(self.device) for key, value in patch_metadata.items()}
            loss_out = self.loss_fn(predictions, targets, masks, patch_metadata=patch_metadata)
        else:
            loss_out = self.loss_fn(predictions, targets, masks)

        # Support if it is a single loss or weighted loss
        if isinstance(loss_out, tuple):
            total_loss, loss_parts = loss_out
        else:
            total_loss = cast(torch.Tensor, loss_out)
            loss_parts = None

        metric_predictions = self._activate_predictions(predictions)
        return metric_predictions, total_loss, loss_parts, targets, masks

    @staticmethod
    def _are_metrics_better(curr: list[float], best: list[float], modes: list[str]):
        if not best:
            return True

        improved = False
        for c, b, mode in zip(curr, best, modes, strict=False):
            if mode == "min":
                if c > b:
                    return False  # a metric got worse!
                elif c < b:
                    improved = True
            elif mode == "max":
                if c < b:
                    return False  # a metric got worse!
                elif c > b:
                    improved = True
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return improved  # Only True if at least one metric improved, none worse

    # Supports any LRScheduler object and metric-based ReduceLROnPlateau schedulers
    def train_epoch(
        self, loader: DataLoader, lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None, lr_scheduler_type: str | None = None
    ) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self._metric_result_keys()}
        running_loss_parts: dict[str, float] = {}

        training_loop = tqdm(loader, desc="Training", leave=True)

        for batch in training_loop:
            predictions, loss, loss_parts, targets, masks = self._step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # use scheduler if its type is batch-level
            if lr_scheduler is not None and lr_scheduler_type == "batch":
                lr_scheduler.step()

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            if self.logger and self.global_step % self.log_every_n_step == 0:
                self.logger.log_metrics({"train_step_loss": loss.item()}, step=self.global_step)

                # log LR since it can change with scheduler
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics({"learning_rate": current_lr}, step=self.global_step)

            training_loop.set_description(f"Loss: {running_loss / running_batch_count:.4f}")

            # compute the metrics
            with torch.no_grad():
                metric_predictions, metric_targets = self._prepare_metric_tensors(predictions.detach(), targets)
                for name, value in self._compute_metric_values(metric_predictions, metric_targets, masks).items():
                    running_metrics[name] += value.item() * batch_size
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics({f"train_step_{name}": value.item()}, step=self.global_step)

                if loss_parts is not None:
                    # log each loss part (raw/unweighted)
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics(
                            {f"train_step_loss_{k}": v.item() for k, v in loss_parts.items()},
                            step=self.global_step,
                        )
                    # accumulate epoch averages
                    for k, v in loss_parts.items():
                        running_loss_parts[k] = running_loss_parts.get(k, 0.0) + v.item() * batch_size

            self.global_step += 1

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        if running_loss_parts:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

        # add averaged metrics to results
        for name, total_value in running_metrics.items():
            results[name] = total_value / max(1, running_batch_count)

        return results

    @overload
    def validate(self, loader: DataLoader, return_predictions: Literal[False] = False) -> dict[str, float]: ...

    @overload
    def validate(self, loader: DataLoader, return_predictions: Literal[True]) -> tuple[dict[str, float], np.ndarray]: ...

    @overload
    def validate(self, loader: DataLoader, return_predictions: bool) -> dict[str, float] | tuple[dict[str, float], np.ndarray]: ...

    @torch.no_grad()
    def validate(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self._metric_result_keys()}
        running_loss_parts: dict[str, float] = {}

        peak_mps_driver_allocated_gb = 0.0

        preds_list = []
        validation_loop = tqdm(loader, desc="Evaluating", leave=True)

        for batch_idx, batch in enumerate(validation_loop):
            predictions, loss, loss_parts, targets, masks = self._step(batch)

            if return_predictions:
                preds_list.append(predictions.detach().cpu().numpy())

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            # compute the metrics
            with torch.no_grad():
                metric_predictions, metric_targets = self._prepare_metric_tensors(predictions.detach(), targets)
                for name, value in self._compute_metric_values(metric_predictions, metric_targets, masks).items():
                    running_metrics[name] += value.item() * batch_size

                if loss_parts is not None:
                    for k, v in loss_parts.items():
                        running_loss_parts[k] = running_loss_parts.get(k, 0.0) + v.item() * batch_size

            # NOTE: MPS-only diagnostic -- captured BEFORE empty_cache() flushes it, so
            # this reflects the real per-batch peak, not the post-flush snapshot. The
            # earlier investigation confirmed individual batches genuinely peak near
            # 30GB even though empty_cache() brings end-of-run allocation back down to
            # a couple GB -- this line is what makes that visible.
            if return_predictions and predictions.device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
                batch_peak_gb = torch.mps.driver_allocated_memory() / 1024**3
                peak_mps_driver_allocated_gb = max(peak_mps_driver_allocated_gb, batch_peak_gb)
                if os.getenv("MPS_DIAG") == "1":
                    print(f"[diag] batch {batch_idx + 1} peak driver_allocated: {batch_peak_gb:.3f} GB")

            # NOTE: MPS-only -- the caching allocator retains freed batch memory
            # without releasing it to the OS, so per-batch driver_allocated_memory
            # climbs across the whole loop instead of resetting. empty_cache()
            # after each batch prevents that accumulation. No-op cost on CPU/CUDA,
            # so this is gated rather than applied unconditionally.
            if predictions.device.type == "mps":
                torch.mps.empty_cache()

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        if running_loss_parts:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

        # add averaged metrics to results
        for name, total_value in running_metrics.items():
            results[name] = total_value / max(1, running_batch_count)

        if return_predictions and peak_mps_driver_allocated_gb > 0.0:
            results["_peak_mps_driver_allocated_gb"] = peak_mps_driver_allocated_gb

        if return_predictions:
            return results, np.concatenate(preds_list, axis=0)

        return results

    @overload
    def test(self, loader: DataLoader, return_predictions: Literal[False] = False) -> dict[str, float]: ...

    @overload
    def test(self, loader: DataLoader, return_predictions: Literal[True]) -> tuple[dict[str, float], np.ndarray]: ...

    @overload
    def test(self, loader: DataLoader, return_predictions: bool) -> dict[str, float] | tuple[dict[str, float], np.ndarray]: ...

    @torch.no_grad()
    def test(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
        return self.validate(loader, return_predictions=return_predictions)

    def _maybe_resume(self, lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None) -> int:
        """
        Resume training from ``last.pth`` when it exists so a preempted/requeued SLURM job
        continues instead of restarting from scratch. Returns the next epoch to run
        (``1`` when there is no compatible checkpoint to resume from).
        """
        if not self.save_dir:
            return 1
        last_path = os.path.join(self.save_dir, "last.pth")
        if not os.path.exists(last_path):
            return 1

        # weights_only=False: these checkpoints are produced by this same Trainer and
        # include optimizer/scheduler/RNG state (numpy arrays, etc.) that torch's
        # default `weights_only=True` restricted unpickler will not load.
        checkpoint = torch.load(last_path, map_location=self.device, weights_only=False)
        saved_state = checkpoint.get("model_state", {})
        current_state = self.model.state_dict()
        compatible = saved_state.keys() == current_state.keys() and all(
            saved_state[key].shape == current_state[key].shape for key in current_state
        )
        if not compatible:
            print(f"[Resume] {last_path} does not match the current model; starting from scratch.")
            return 1

        self.model.load_state_dict(saved_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if lr_scheduler is not None and checkpoint.get("scheduler_state") is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("best_metric_list"):
            self._best_metric_list = list(checkpoint["best_metric_list"])
        self.global_step = int(checkpoint.get("global_step", self.global_step))

        # Restore RNG states so a resumed run (e.g. after preemption) reproduces the
        # same data order/augmentations as an uninterrupted one, per standard
        # checkpointing guidelines.
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python_random_state"])
            np.random.set_state(rng_state["numpy_random_state"])
            # The CPU RNG state tensor must stay on CPU even though `map_location`
            # above may have moved other checkpoint tensors to an accelerator device.
            torch.random.set_rng_state(rng_state["torch_random_state"].cpu())
            if torch.cuda.is_available() and rng_state.get("torch_cuda_random_state") is not None:
                # Like the CPU RNG state above, each per-device state tensor must stay on
                # CPU (as a plain torch.ByteTensor) even though `map_location` may have
                # moved it onto an accelerator device; `set_rng_state_all` rejects
                # anything that isn't a CPU ByteTensor.
                cuda_rng_states = [state.cpu() for state in rng_state["torch_cuda_random_state"]]
                torch.cuda.set_rng_state_all(cuda_rng_states)

        last_epoch = int(checkpoint.get("epoch", 0))
        print(f"[Resume] Resuming from {last_path}: completed epoch {last_epoch}, continuing at epoch {last_epoch + 1}.")
        return last_epoch + 1

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        num_epochs = self.config.training.max_epochs
        log_every_n_epoch = self.config.training.log_every_n_epoch

        # get scheduler and its type
        lr_scheduler, lr_scheduler_type = build_lr_scheduler(self.config, self.optimizer, train_loader)

        start_epoch = self._maybe_resume(lr_scheduler)

        for epoch in range(start_epoch, num_epochs + 1):
            start = time.time()
            train_res = self.train_epoch(train_loader, lr_scheduler=lr_scheduler, lr_scheduler_type=lr_scheduler_type)
            elapsed = time.time() - start

            val_result = self.validate(val_loader) if val_loader is not None else None
            if isinstance(val_result, tuple):
                val_result = val_result[0]

            # for epoch level schedulers
            if lr_scheduler is not None:
                if lr_scheduler_type == "epoch":
                    lr_scheduler.step()
                elif lr_scheduler_type == "epoch_metric":
                    # Plateau needs a metric to watch. Default to val_loss, fallback to train_loss
                    watch_metric = val_result["loss"] if val_result else train_res["loss"]
                    lr_scheduler.step(watch_metric)

            # log metrics and loss
            if epoch % log_every_n_epoch == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch}/{num_epochs} - train_loss: {train_res['loss']:.4f}"
                if val_result is not None:
                    msg += f", val_loss: {val_result['loss']:.4f}"
                msg += f", lr: {current_lr:.2e}, time: {elapsed:.1f}s"
                print(msg)

                metrics_to_log = {f"train_{k}": v for k, v in train_res.items()}
                metrics_to_log["epoch_duration"] = elapsed

                if val_result:
                    metrics_to_log.update({f"val_{k}": v for k, v in val_result.items()})

                if self.logger:
                    self.logger.log_metrics(metrics_to_log, epoch=epoch)

            # Save best checkpoint based on multi-metrics
            if val_result:
                curr_metric_list = [val_result[m] for m in self.best_ckpt_metrics]
                if self._are_metrics_better(curr_metric_list, self._best_metric_list, self.best_ckpt_modes):
                    self._best_metric_list = curr_metric_list
                    if self.save_dir:
                        best_path = self.save_model(
                            epoch=epoch,
                            metric_value={m: v for m, v in zip(self.best_ckpt_metrics, curr_metric_list, strict=False)},
                            filename="best.pth",
                        )
                        if self.logger:
                            self.logger.experiment.log_model(name="best", file_or_folder=best_path, overwrite=True)

                # save most recent checkpoint
                self.save_model(
                    epoch=epoch,
                    metric_value={m: v for m, v in zip(self.best_ckpt_metrics, curr_metric_list, strict=False)},
                    lr_scheduler=lr_scheduler,
                )

    def save_model(
        self,
        epoch: int,
        metric_value: float | dict,
        filename: str = "last.pth",
        lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    ):
        if not self.save_dir:
            raise ValueError("save_dir not set")

        path = os.path.join(self.save_dir, filename)

        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric_value": metric_value,
            "config": self.config.model_dump(),
            "global_step": self.global_step,
            "best_metric_list": self._best_metric_list,
            "scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            # RNG states so a resumed run (e.g. after SLURM preemption) can reproduce
            # the same data order/augmentations as an uninterrupted one.
            "rng_state": {
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.random.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            # So a resumed run continues logging into the same Comet experiment
            # instead of starting a new one on each SLURM requeue.
            "comet_experiment_key": self.logger.experiment_key if self.logger else None,
        }
        # Write atomically so a preemption mid-save cannot leave a corrupt checkpoint.
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_model(self, path: str | None = None, filename: str = "last.pth", map_location: str | None = None):
        if path is None:
            path = os.path.join(self.save_dir, filename)

        map_location = map_location or self.device
        # weights_only=False: trusted, self-produced checkpoints that also carry
        # optimizer/scheduler/RNG state beyond plain tensors.
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint
