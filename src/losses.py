# Definitions of loss functions
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary cross entropy loss with optional mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # internally handles sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.bce(logits, targets)  # same shape as logits/targets

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class MSELoss(nn.Module):
    """
    MSELoss with optional mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        probs = torch.sigmoid(logits)
        loss = self.mse(probs, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)

        return loss.sum() / denom


class CCCLoss(nn.Module):
    """
    Concordance correlation coefficient loss, 1 - CCC.

    Use the sigmoid variant for probability targets such as BP.
    """

    def __init__(self, use_sigmoid: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        preds = preds.flatten(1).float()
        targets = targets.flatten(1).float()

        if mask is None:
            mask_flat = torch.ones_like(preds, dtype=torch.bool)
        else:
            mask_flat = mask.bool().flatten(1)
        mask_f = mask_flat.float()

        valid_counts = mask_flat.sum(dim=1)
        safe_counts = valid_counts.clamp_min(1)

        pred_mean = (preds * mask_f).sum(dim=1) / safe_counts
        target_mean = (targets * mask_f).sum(dim=1) / safe_counts

        pred_centered = (preds - pred_mean.unsqueeze(1)) * mask_f
        target_centered = (targets - target_mean.unsqueeze(1)) * mask_f

        covariance = (pred_centered * target_centered).sum(dim=1) / safe_counts
        pred_var = (pred_centered.pow(2)).sum(dim=1) / safe_counts
        target_var = (target_centered.pow(2)).sum(dim=1) / safe_counts
        denominator = pred_var + target_var + (pred_mean - target_mean).pow(2)

        ccc_row = 2.0 * covariance / denominator.clamp_min(self.eps)

        valid_row = valid_counts >= 2
        if not bool(valid_row.any()):
            return logits.new_tensor(0.0)

        ccc_row = torch.where(valid_row, ccc_row, torch.full_like(ccc_row, float("nan")))
        return 1.0 - torch.nanmean(ccc_row)


class PearsonLoss(nn.Module):
    """
    Pearson correlation loss, 1 - r.

    This is a differentiable rank-order proxy. Use the sigmoid variant for BP
    and RegressionPearsonLoss for raw FI/ROS regression outputs.
    """

    def __init__(self, use_sigmoid: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        preds = preds.flatten(1).float()
        targets = targets.flatten(1).float()
        mask_flat = None if mask is None else mask.bool().flatten(1)

        correlations = []
        for idx in range(preds.shape[0]):
            pred_i = preds[idx]
            target_i = targets[idx]
            if mask_flat is not None:
                valid = mask_flat[idx]
                pred_i = pred_i[valid]
                target_i = target_i[valid]
            if pred_i.numel() < 2:
                continue
            pred_centered = pred_i - pred_i.mean()
            target_centered = target_i - target_i.mean()
            denom = pred_centered.norm() * target_centered.norm()
            correlations.append((pred_centered * target_centered).sum() / denom.clamp_min(self.eps))

        if not correlations:
            return logits.new_tensor(0.0)
        return 1.0 - torch.stack(correlations).mean()


class RegressionPearsonLoss(PearsonLoss):
    """Pearson loss on raw model outputs."""

    def __init__(self, eps: float = 1e-8):
        super().__init__(use_sigmoid=False, eps=eps)


class MAELoss(nn.Module):
    """
    MAELoss with optional mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        probs = torch.sigmoid(logits)
        loss = self.mae(probs, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)

        return loss.sum() / denom


class HuberLoss(nn.Module):
    """
    Huber/SmoothL1 loss on raw model outputs with optional mask.

    This is intended for unconstrained regression targets, e.g. standardized log FI/ROS.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"Huber beta must be positive, got {beta}.")
        self.eps = eps
        self.beta = beta
        self.huber = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.huber(logits, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation.

    Expects:
      logits: shape (N, 1, H, W)
      targets: same shape, values in [0,1]
      mask: same shape (bool or 0/1), where 1 means valid pixel
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        probs = torch.sigmoid(logits)
        if mask is None:
            probs = probs.flatten(1)
            targets = targets.flatten(1)
            valid = None
        else:
            mask = mask.to(dtype=probs.dtype)
            valid = mask.flatten(1).sum(dim=1) > 0
            probs = (probs * mask).flatten(1)
            targets = (targets * mask).flatten(1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)

        if valid is None:
            return 1.0 - dice.mean()

        if valid.any():
            return 1.0 - dice[valid].mean()
        else:
            return dice.new_tensor(0.0)


class FocalLoss(nn.Module):
    """
    Binary focal loss

    Expects:
      logits: (N, 1, H, W)
      targets: same shape, values in [0,1]
      mask: same shape (bool or 0/1), where 1 means valid pixel

    Params:
      alpha: class balancing factor. Common: 0.25 for positives (as in RetinaNet).
             If None, no alpha balancing.
      gamma: focusing parameter. Common: 2.0.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # internally handles sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        targets = targets.to(dtype=logits.dtype)

        # per-element BCE with logits
        bce = self.bce(logits, targets)

        # p_t = p if y=1 else (1-p)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_factor = (1.0 - p_t).clamp_min(0.0).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class BernoulliKLLoss(nn.Module):
    """
    Stable KL(p || q) for Bernoulli with soft targets p in [0,1] and q=sigmoid(logits).
    """

    def __init__(self, eps: float = 1e-6, clamp_logits: float | None = 20.0):
        super().__init__()
        self.eps = eps
        self.clamp_logits = clamp_logits

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.clamp_logits is not None:
            logits = logits.clamp(-self.clamp_logits, self.clamp_logits)

        # ensure float
        targets = targets.to(dtype=logits.dtype)

        # clamp targets to avoid log(0) / log(negative)
        p = targets.clamp(self.eps, 1.0 - self.eps)

        # stable log q and log(1-q)
        log_q = F.logsigmoid(logits)  # log(sigmoid(z))
        log_1mq = F.logsigmoid(-logits)  # log(1-sigmoid(z))

        # stable log p and log(1-p)
        log_p = torch.log(p)
        log_1mp = torch.log1p(-p)

        loss = p * (log_p - log_q) + (1.0 - p) * (log_1mp - log_1mq)

        if mask is None:
            return loss.mean()

        m = mask.to(dtype=loss.dtype)
        # TODO: test if really needed. Zero-out masked pixels safely (prevents NaNs in masked regions from propagating)
        loss = torch.where(m > 0, loss, torch.zeros_like(loss))

        denom = m.sum().clamp_min(self.eps)
        return loss.sum() / denom


class HexSummaryLoss(nn.Module):
    """Differentiable cross-hex loss on batch-level BP summaries.

    This is not a stitched-raster loss. It uses patch metadata to group patches
    by hex ID within each batch, summarizes valid BP predictions per patch, then
    compares grouped hex summaries. It is intended as a lightweight proxy for
    the stitched/hex ranking failure mode.
    """

    requires_patch_metadata = True

    def __init__(
        self,
        summary: str = "mean",
        correlation: str = "pearson",
        top_fraction: float = 0.10,
        rank_temperature: float = 1.0,
        min_target_gap: float = 1e-6,
        rank_scale_min: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.summary = summary.lower()
        self.correlation = correlation.lower()
        self.top_fraction = top_fraction
        self.rank_temperature = rank_temperature
        self.min_target_gap = min_target_gap
        self.rank_scale_min = rank_scale_min
        self.eps = eps

        if self.summary not in {"mean", "topk_mean"}:
            raise ValueError(f"Unsupported hex summary={summary!r}. Use 'mean' or 'topk_mean'.")
        if self.correlation not in {"pearson", "ccc", "pairwise_rank"}:
            raise ValueError(f"Unsupported hex summary correlation={correlation!r}. Use 'pearson', 'ccc', or 'pairwise_rank'.")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
        if self.rank_temperature <= 0.0:
            raise ValueError(f"rank_temperature must be positive, got {rank_temperature}.")
        if self.min_target_gap < 0.0:
            raise ValueError(f"min_target_gap must be non-negative, got {min_target_gap}.")
        if self.rank_scale_min <= 0.0:
            raise ValueError(f"rank_scale_min must be positive, got {rank_scale_min}.")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if logits.shape[1] != 1:
            raise ValueError(f"HexSummaryLoss supports single-target BP only, got logits shape {tuple(logits.shape)}.")
        if patch_metadata is None or "hex_id" not in patch_metadata:
            raise ValueError("HexSummaryLoss requires patch_metadata containing a 'hex_id' tensor.")

        hex_ids = patch_metadata["hex_id"].to(device=logits.device)
        if hex_ids.ndim != 1 or hex_ids.shape[0] != logits.shape[0]:
            raise ValueError(f"Expected hex_id shape ({logits.shape[0]},), got {tuple(hex_ids.shape)}.")

        probs = torch.sigmoid(logits)
        patch_pred, patch_target = self._patch_summaries(probs, targets, mask)
        if patch_pred.numel() < 2:
            return logits.new_tensor(0.0)

        hex_pred = []
        hex_target = []
        for hex_id in torch.unique(hex_ids):
            group = hex_ids == hex_id
            if not group.any():
                continue
            hex_pred.append(patch_pred[group].mean())
            hex_target.append(patch_target[group].mean())

        if len(hex_pred) < 2:
            return logits.new_tensor(0.0)

        pred = torch.stack(hex_pred).float()
        target = torch.stack(hex_target).float()
        if self.correlation == "pearson":
            return self._pearson_loss(pred, target)
        if self.correlation == "ccc":
            return self._ccc_loss(pred, target)
        return self._pairwise_rank_loss(pred, target)

    def _patch_summaries(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_bool = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.bool()

        if self.summary == "mean":
            pred_flat = probs[:, 0].flatten(1)
            target_flat = targets[:, 0].flatten(1)
            mask_flat = mask_bool[:, 0].flatten(1).float()

            counts = mask_flat.sum(dim=1)
            safe_counts = counts.clamp_min(1)

            pred_sum = (pred_flat * mask_flat).sum(dim=1)
            target_sum = (target_flat * mask_flat).sum(dim=1)

            has_valid = counts > 0
            pred_values = torch.where(has_valid, pred_sum / safe_counts, torch.zeros_like(pred_sum))
            target_values = torch.where(has_valid, target_sum / safe_counts, torch.zeros_like(target_sum))
            return pred_values, target_values

        top_pred_values: list[torch.Tensor] = []
        top_target_values: list[torch.Tensor] = []
        for idx in range(probs.shape[0]):
            valid = mask_bool[idx, 0]
            pred_i = probs[idx, 0][valid]
            target_i = targets[idx, 0][valid]
            if pred_i.numel() == 0:
                top_pred_values.append(probs.new_tensor(0.0))
                top_target_values.append(targets.new_tensor(0.0))
                continue
            k = max(1, int(torch.ceil(target_i.new_tensor(float(target_i.numel() * self.top_fraction))).item()))
            top_indices = torch.topk(target_i, k=k, largest=True).indices
            top_pred_values.append(pred_i[top_indices].mean())
            top_target_values.append(target_i[top_indices].mean())

        return torch.stack(top_pred_values), torch.stack(top_target_values)

    def _pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        denom = pred_centered.norm() * target_centered.norm()
        return 1.0 - (pred_centered * target_centered).sum() / denom.clamp_min(self.eps)

    def _ccc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()
        denominator = pred.var(correction=0) + target.var(correction=0) + (pred_mean - target_mean).pow(2)
        return 1.0 - 2.0 * covariance / denominator.clamp_min(self.eps)

    def _pairwise_rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[:, None] - pred[None, :]
        target_diff = target[:, None] - target[None, :]
        upper_triangular = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
        valid = upper_triangular & (target_diff.abs() > self.min_target_gap)
        if not valid.any():
            return pred.new_tensor(0.0)

        direction = target_diff[valid].sign()
        pred_scale = pred.std(correction=0).detach().clamp_min(self.rank_scale_min)
        ordered_margin = direction * (pred_diff[valid] / pred_scale) / self.rank_temperature
        return F.softplus(-ordered_margin).mean()


class WeightedLoss(nn.Module):
    """
    Combine multiple loss modules with weights.

    All losses are expected to implement:
        forward(logits: Tensor, targets: Tensor, mask: Tensor|None) -> Tensor
    """

    losses: nn.ModuleDict
    _weights: torch.Tensor

    def __init__(
        self,
        losses: dict[str, nn.Module],
        weights: dict[str, float] | None = None,
        normalize_weights: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not losses:
            raise ValueError("losses must be a non-empty dict of name -> nn.Module")

        self.losses = nn.ModuleDict(losses)
        self.eps = eps
        self.normalize_weights = normalize_weights

        if weights is None:
            weights = {k: 1.0 for k in losses}

        missing = set(losses.keys()) - set(weights.keys())
        extra = set(weights.keys()) - set(losses.keys())
        if missing:
            raise ValueError(f"weights missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"weights has unknown keys: {sorted(extra)}")

        w = torch.tensor([weights[k] for k in losses], dtype=torch.float32)
        if self.normalize_weights:
            w = w / w.sum().clamp_min(self.eps)
        self.register_buffer("_weights", w)
        self.requires_patch_metadata = any(getattr(loss_mod, "requires_patch_metadata", False) for loss_mod in self.losses.values())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self._weights

        total_loss = logits.new_tensor(0.0)
        loss_parts: dict[str, torch.Tensor] = {}

        for i, (name, loss_mod) in enumerate(self.losses.items()):
            loss_mod = cast(nn.Module, loss_mod)

            if getattr(loss_mod, "requires_patch_metadata", False):
                loss_val = cast(torch.Tensor, loss_mod(logits, targets, mask, patch_metadata=patch_metadata))
            else:
                loss_val = cast(torch.Tensor, loss_mod(logits, targets, mask))
            loss_parts[name] = loss_val
            total_loss = total_loss + (w[i].to(dtype=loss_val.dtype) * loss_val)

        return total_loss, loss_parts


class MultiTaskLoss(nn.Module):
    """Route ordered output channels through target-specific loss modules."""

    _task_weights: torch.Tensor

    def __init__(
        self,
        target_names: list[str],
        losses: dict[str, nn.Module],
        task_weights: dict[str, float],
        eps: float = 1e-8,
    ):
        super().__init__()
        if not target_names or len(target_names) != len(set(target_names)):
            raise ValueError(f"target_names must be non-empty and unique, got {target_names}.")
        if set(losses) != set(target_names):
            raise ValueError(f"loss keys must match target_names, got losses={sorted(losses)} and targets={sorted(target_names)}.")
        if set(task_weights) != set(target_names):
            raise ValueError(
                f"task_weights keys must match target_names, got weights={sorted(task_weights)} and targets={sorted(target_names)}."
            )

        weights = torch.tensor([task_weights[name] for name in target_names], dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(f"task_weights must be positive and finite, got {task_weights}.")

        self.target_names = tuple(target_names)
        self.losses = nn.ModuleDict({name: losses[name] for name in target_names})
        self.register_buffer("_task_weights", weights / weights.sum().clamp_min(eps))
        self.requires_patch_metadata = any(getattr(loss, "requires_patch_metadata", False) for loss in self.losses.values())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_channels = len(self.target_names)
        if logits.shape != targets.shape or logits.ndim < 2 or logits.shape[1] != expected_channels:
            raise ValueError(
                f"Expected logits and targets with matching shape and {expected_channels} channels, "
                f"got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}."
            )
        if mask is not None and mask.shape != targets.shape:
            raise ValueError(f"Expected mask shape {tuple(targets.shape)}, got {tuple(mask.shape)}.")

        total_loss = logits.new_tensor(0.0)
        loss_parts: dict[str, torch.Tensor] = {}
        for channel_idx, target_name in enumerate(self.target_names):
            target_loss = cast(nn.Module, self.losses[target_name])
            channel = slice(channel_idx, channel_idx + 1)
            channel_mask = None if mask is None else mask[:, channel]
            if getattr(target_loss, "requires_patch_metadata", False):
                loss_out = target_loss(
                    logits[:, channel],
                    targets[:, channel],
                    channel_mask,
                    patch_metadata=patch_metadata,
                )
            else:
                loss_out = target_loss(logits[:, channel], targets[:, channel], channel_mask)

            if isinstance(loss_out, tuple):
                task_loss, task_parts = loss_out
                loss_parts.update({f"{target_name}/{name}": value for name, value in task_parts.items()})
            else:
                task_loss = cast(torch.Tensor, loss_out)
            loss_parts[f"{target_name}/total"] = task_loss
            total_loss = total_loss + self._task_weights[channel_idx].to(dtype=task_loss.dtype) * task_loss

        return total_loss, loss_parts
