# Definitions of metrics for model evaluation
import math

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure


def _topk_threshold(values: torch.Tensor, percentile: float) -> torch.Tensor:
    """Return a top-k threshold without torch.quantile's large-tensor limit."""
    flat = values.reshape(-1).float()
    if flat.numel() == 0:
        return flat.new_tensor(float("nan"))
    percentile = min(max(float(percentile), 0.0), 1.0)
    top_fraction = 1.0 - percentile
    if top_fraction <= 0.0:
        top_count = 1
    else:
        raw_count = flat.numel() * top_fraction
        nearest_count = round(raw_count)
        top_count = int(nearest_count if math.isclose(raw_count, nearest_count, rel_tol=1e-6, abs_tol=1e-6) else math.ceil(raw_count))
    k = max(1, min(flat.numel(), top_count))
    return torch.topk(flat, k=k, largest=True, sorted=False).values.min()


def _topk_thresholds(values: torch.Tensor, percentiles: torch.Tensor) -> torch.Tensor:
    """Vectorized top-k thresholds for many percentiles using a single sort (no torch.quantile size limit)."""
    flat = values.reshape(-1).float()
    perc = percentiles.reshape(-1).to(flat.device, dtype=torch.float64).clamp(0.0, 1.0)
    if flat.numel() == 0:
        return flat.new_full((perc.numel(),), float("nan"))
    sorted_desc, _ = torch.sort(flat, descending=True)
    n = flat.numel()
    top_fraction = 1.0 - perc
    raw_count = n * top_fraction
    nearest = torch.round(raw_count)
    tol = torch.maximum(torch.maximum(raw_count.abs(), nearest.abs()) * 1e-6, raw_count.new_tensor(1e-6))
    use_nearest = (raw_count - nearest).abs() <= tol
    top_count = torch.where(use_nearest, nearest, torch.ceil(raw_count))
    top_count = torch.where(top_fraction <= 0.0, torch.ones_like(top_count), top_count)
    k = top_count.clamp(1, n).long()
    return sorted_desc[k - 1]


def compute_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Computes Mean Squared Error (MSE), optionally using a mask."""
    if mask is None:
        return F.mse_loss(preds, targets, reduction="mean")

    mask = mask.to(dtype=preds.dtype)
    loss = F.mse_loss(preds, targets, reduction="none")

    # apply mask + normalize by valid count
    loss = loss * mask
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom


def compute_mae(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Computes Mean Absolute Error (MAE), optionally using a mask."""
    if mask is None:
        return F.l1_loss(preds, targets, reduction="mean")

    mask = mask.to(dtype=preds.dtype)
    loss = F.l1_loss(preds, targets, reduction="none")

    # apply mask + normalize by valid count
    loss = loss * mask
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom


def compute_normalized_mae(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Computes MAE normalized by the mean absolute target magnitude over valid pixels."""
    preds = preds.float()
    targets = targets.float()

    if mask is None:
        abs_error = torch.abs(preds - targets).mean()
        target_scale = torch.abs(targets).mean().clamp_min(eps)
        return abs_error / target_scale

    valid = mask.to(dtype=preds.dtype)
    abs_error_sum = (torch.abs(preds - targets) * valid).sum()
    target_scale_sum = (torch.abs(targets) * valid).sum().clamp_min(eps)
    return abs_error_sum / target_scale_sum


def _rank_data_average_ties(data: torch.Tensor) -> torch.Tensor:
    """Return flattened average-tie ranks, avoiding int32 overflow on large hexels."""
    flat = data.reshape(-1)
    n = flat.numel()
    order = flat.argsort()
    ranks = torch.empty(n, dtype=torch.float64, device=flat.device)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64, device=flat.device)
    _, inverse, counts = torch.unique(flat, sorted=True, return_inverse=True, return_counts=True)
    rank_sums = torch.zeros(counts.numel(), dtype=torch.float64, device=flat.device)
    rank_sums.scatter_add_(0, inverse, ranks)
    mean_ranks = rank_sums / counts.to(dtype=torch.float64)
    return mean_ranks[inverse]


def _spearman_corrcoef(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Helper to compute Spearman correlation for a single sample (1D pred and target tensors).
    Returns scalar tensor.
    This mirrors torchmetric.functional.regression.spearman but keeps ranks sums in float64 so large hexels with tied values don't overflow.
    """
    if preds.numel() < 2:
        return torch.tensor(float("nan"), device=preds.device)

    pred_ranks = _rank_data_average_ties(preds)
    target_ranks = _rank_data_average_ties(targets)
    pred_diff = pred_ranks - pred_ranks.mean()
    target_diff = target_ranks - target_ranks.mean()
    denom = torch.linalg.vector_norm(pred_diff) * torch.linalg.vector_norm(target_diff)
    if denom == 0:
        return torch.tensor(float("nan"), device=preds.device)
    return torch.clamp(torch.sum(pred_diff * target_diff) / denom, -1.0, 1.0)


def compute_spearman(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Computes Spearman correlation per sample, then averages. Optionally uses a mask.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    min_valid = 2

    corrs = []
    if mask is None:
        for i in range(batch_size):
            corrs.append(_spearman_corrcoef(flat_preds[i], flat_targets[i]))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)  # True = valid
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]
            if sample_valid_mask.sum() < min_valid:
                corrs.append(torch.tensor(float("nan"), device=preds.device))
                continue
            corrs.append(_spearman_corrcoef(flat_preds[i][sample_valid_mask], flat_targets[i][sample_valid_mask]))

    return torch.nanmean(torch.stack(corrs)).to(dtype=preds.dtype)


def compute_ssim(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Computes SSIM over valid pixels only.
    Assumes preds/targets ∈ [0, 1].
    """
    if mask is None:
        return structural_similarity_index_measure(preds, targets, data_range=1.0)  # type: ignore

    # Ensure mask is boolean and broadcastable
    mask_bool = mask.bool()
    # If mask is missing channel dim, unsqueeze to match preds/targets
    while mask_bool.dim() < preds.dim():
        mask_bool = mask_bool.unsqueeze(1)

    # If all masked, return nan
    if mask_bool.sum() == 0:
        return torch.tensor(float("nan"), device=preds.device)

    # Set masked (invalid) pixels to 0 (or another constant)
    preds_masked = preds.clone().masked_fill(~mask_bool, 0.0)
    targets_masked = targets.clone().masked_fill(~mask_bool, 0.0)

    return structural_similarity_index_measure(preds_masked, targets_masked, data_range=1.0)  # type: ignore


def compute_bias(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    preds, targets, mask: same shape
    returns scalar bias (mean(preds-target) over valid pixels)
    """
    preds = preds.float()
    targets = targets.float()
    if mask is not None:
        valid = (mask > 0).float()
    else:
        valid = torch.ones_like(preds)

    denom = valid.sum().clamp_min(1.0)  # avoid divide-by-zero
    return ((preds - targets) * valid).sum() / denom


def compute_normalized_bias(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8
) -> torch.Tensor:
    """Computes bias normalized by the mean target value over valid pixels."""
    preds = preds.float()
    targets = targets.float()
    valid = mask.to(dtype=preds.dtype) if mask is not None else torch.ones_like(preds)

    error_sum = ((preds - targets) * valid).sum()
    target_sum = (targets * valid).sum().clamp_min(eps)
    return error_sum / target_sum


def compute_topK_iou(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, percentile: float = 0.90, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) on binarized top K percentile maps.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    ious = []

    if mask is None:
        for i in range(batch_size):
            p = flat_preds[i]
            t = flat_targets[i]

            p_thresh = _topk_threshold(p, percentile)
            t_thresh = _topk_threshold(t, percentile)

            # binarization
            p_bin = p >= p_thresh
            t_bin = t >= t_thresh

            intersection = (p_bin & t_bin).sum().float()
            union = (p_bin | t_bin).sum().float()

            ious.append(intersection / (union + eps))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]

            if sample_valid_mask.sum() == 0:
                ious.append(torch.tensor(float("nan"), device=preds.device))
                continue

            p_valid = flat_preds[i][sample_valid_mask]
            t_valid = flat_targets[i][sample_valid_mask]

            p_thresh = _topk_threshold(p_valid, percentile)
            t_thresh = _topk_threshold(t_valid, percentile)

            p_bin = p_valid >= p_thresh
            t_bin = t_valid >= t_thresh

            intersection = (p_bin & t_bin).sum().float()
            union = (p_bin | t_bin).sum().float()

            ious.append(intersection / (union + eps))

    return torch.nanmean(torch.stack(ious))


def compute_auc_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    k_values: tuple[float, float] = (0.01, 0.99),
    steps: int = 99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the Area Under the Curve (AUC) for IoU for a specified continuous TopK perc. range.

    Args:
        k_values (tuple[float, float]): Tuple (min_k, max_k) defining the continuous range for eval.
        steps (int): Number of points to evaluate within the continuous range.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    # validations of inputs
    if not (isinstance(k_values, tuple) and len(k_values) == 2):
        raise ValueError("k_values must be a tuple of (min_k, max_k).")
    if not all(isinstance(k, (int, float)) for k in k_values):
        raise ValueError("k_values must contain numeric values (int or float).")
    min_k, max_k = float(k_values[0]), float(k_values[1])
    if not (0.0 < min_k <= 1.0 and 0.0 < max_k <= 1.0):
        raise ValueError("Each value in k_values must be within the open-closed interval (0, 1].")
    if not min_k < max_k:
        raise ValueError("k_values must satisfy min_k < max_k.")
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer greater than or equal to 2.")

    k_tensor = torch.linspace(min_k, max_k, steps=steps, device=preds.device)
    percentiles = 1.0 - k_tensor

    aucs = []

    if mask is None:
        for i in range(batch_size):
            p = flat_preds[i]
            t = flat_targets[i]

            p_thresh = _topk_thresholds(p, percentiles)
            t_thresh = _topk_thresholds(t, percentiles)

            p_bin = p.unsqueeze(0) >= p_thresh.unsqueeze(1)
            t_bin = t.unsqueeze(0) >= t_thresh.unsqueeze(1)

            intersection = (p_bin & t_bin).sum(dim=1).float()
            union = (p_bin | t_bin).sum(dim=1).float()

            ious = intersection / (union + eps)

            auc = torch.trapz(ious, k_tensor)
            max_area = k_tensor[-1] - k_tensor[0]

            aucs.append(auc / max_area if max_area > 0 else torch.tensor(float("nan"), device=preds.device))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]

            if sample_valid_mask.sum() == 0:
                aucs.append(torch.tensor(float("nan"), device=preds.device))
                continue

            p_valid = flat_preds[i][sample_valid_mask]
            t_valid = flat_targets[i][sample_valid_mask]

            p_thresh = _topk_thresholds(p_valid, percentiles)
            t_thresh = _topk_thresholds(t_valid, percentiles)

            p_bin = p_valid.unsqueeze(0) >= p_thresh.unsqueeze(1)
            t_bin = t_valid.unsqueeze(0) >= t_thresh.unsqueeze(1)

            intersection = (p_bin & t_bin).sum(dim=1).float()
            union = (p_bin | t_bin).sum(dim=1).float()

            ious = intersection / (union + eps)

            auc = torch.trapz(ious, k_tensor)
            max_area = k_tensor[-1] - k_tensor[0]

            aucs.append(auc / max_area if max_area > 0 else torch.tensor(float("nan"), device=preds.device))

    # return mean of auc values (scalar)
    return torch.nanmean(torch.stack(aucs))


def compute_ccc(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the Concordance Correlation Coefficient (CCC), optionally using a mask.

    CCC = 2 * cov(preds, targets) / (var(preds) + var(targets) + (mean(preds) - mean(targets))^2)

    Returns a scalar tensor averaging CCC across the batch.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1).float()
    flat_targets = targets.reshape(batch_size, -1).float()

    min_valid = 2  # need at least 2 values for meaningful variance/covariance
    cccs = []

    if mask is None:
        for i in range(batch_size):
            p = flat_preds[i]
            t = flat_targets[i]
            mean_p = p.mean()
            mean_t = t.mean()
            var_p = p.var(correction=0)
            var_t = t.var(correction=0)
            cov_pt = ((p - mean_p) * (t - mean_t)).mean()
            denom = var_p + var_t + (mean_p - mean_t) ** 2
            cccs.append(2.0 * cov_pt / denom.clamp_min(eps))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)
        for i in range(batch_size):
            m = valid_mask[i]
            if m.sum() < min_valid:
                cccs.append(torch.tensor(float("nan"), device=preds.device))
                continue
            p = flat_preds[i][m]
            t = flat_targets[i][m]
            mean_p = p.mean()
            mean_t = t.mean()
            var_p = p.var(correction=0)
            var_t = t.var(correction=0)
            cov_pt = ((p - mean_p) * (t - mean_t)).mean()
            denom = var_p + var_t + (mean_p - mean_t) ** 2
            cccs.append(2.0 * cov_pt / denom.clamp_min(eps))

    return torch.nanmean(torch.stack(cccs))


def compute_kl_divergence(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes the Kullback-Leibler (KL) Divergence, optionally using a mask.
    The tensors are normalized per-sample to form a valid probability distribution.

    KL(P||Q) = sum(P * log(P / Q))
    where P is the target distribution and Q is the predicted distribution.

    Returns a scalar tensor averaging KL across the batch.
    """
    batch_size = preds.size(0)
    # Ensure non-negative and add epsilon to avoid log(0) or div by 0
    flat_preds = preds.reshape(batch_size, -1).float().clamp_min(eps)
    flat_targets = targets.reshape(batch_size, -1).float().clamp_min(eps)

    kl_divs = []

    for i in range(batch_size):
        p_raw = flat_targets[i]
        q_raw = flat_preds[i]

        if mask is not None:
            valid = mask.bool().reshape(batch_size, -1)[i]
            if valid.sum() == 0:
                kl_divs.append(torch.tensor(float("nan"), device=preds.device))
                continue
            p_raw = p_raw[valid]
            q_raw = q_raw[valid]

        # Normalize to create a probability distribution (sum to 1)
        p = p_raw / p_raw.sum().clamp_min(eps)
        q = q_raw / q_raw.sum().clamp_min(eps)

        # Compute KL(P || Q)
        kl = torch.sum(p * torch.log(p / q))
        kl_divs.append(kl)

    return torch.nanmean(torch.stack(kl_divs))


def compute_topK_mae(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, percentile: float = 0.90
) -> torch.Tensor:
    """
    Computes the MAE specifically over the top-K target pixels.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    errors = []

    if mask is None:
        for i in range(batch_size):
            p = flat_preds[i]
            t = flat_targets[i]

            p_thresh = _topk_threshold(p, percentile)
            t_thresh = _topk_threshold(t, percentile)

            topK_mask = (p >= p_thresh) | (t >= t_thresh)

            if topK_mask.sum() == 0:
                errors.append(torch.tensor(float("nan"), device=preds.device, dtype=preds.dtype))
                continue

            p_topK = p[topK_mask]
            t_topK = t[topK_mask]

            mae = compute_mae(preds=p_topK, targets=t_topK, mask=None)
            errors.append(mae)
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]

            if sample_valid_mask.sum() == 0:
                errors.append(torch.tensor(float("nan"), device=preds.device, dtype=preds.dtype))
                continue

            p_valid = flat_preds[i][sample_valid_mask]
            t_valid = flat_targets[i][sample_valid_mask]

            p_thresh = _topk_threshold(p_valid, percentile)
            t_thresh = _topk_threshold(t_valid, percentile)

            topK_mask = (p_valid >= p_thresh) | (t_valid >= t_thresh)

            if topK_mask.sum() == 0:
                errors.append(torch.tensor(float("nan"), device=preds.device, dtype=preds.dtype))
                continue

            p_topK = p_valid[topK_mask]
            t_topK = t_valid[topK_mask]

            mae = compute_mae(preds=p_topK, targets=t_topK, mask=None)
            errors.append(mae)

    return torch.nanmean(torch.stack(errors))
