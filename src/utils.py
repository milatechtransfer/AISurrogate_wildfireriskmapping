import os
import random
from collections.abc import Callable
from contextlib import suppress
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.losses import (
    BCELoss,
    BernoulliKLLoss,
    CCCLoss,
    DiceLoss,
    FocalLoss,
    HexSummaryLoss,
    HuberLoss,
    MAELoss,
    MSELoss,
    RegressionPearsonLoss,
)
from src.metrics import (
    compute_auc_iou,
    compute_bias,
    compute_ccc,
    compute_kl_divergence,
    compute_mae,
    compute_mse,
    compute_normalized_bias,
    compute_normalized_mae,
    compute_spearman,
    compute_ssim,
    compute_topK_iou,
    compute_topK_mae,
)

AVAILABLE_METRICS: dict[str, Callable[..., torch.Tensor]] = {
    "mse": compute_mse,
    "mae": compute_mae,
    "normalized_mae": compute_normalized_mae,
    "nmae": compute_normalized_mae,
    "normalized_bias": compute_normalized_bias,
    "nbias": compute_normalized_bias,
    "spearman": compute_spearman,
    "ssim": compute_ssim,
    "bias": compute_bias,
    "ccc": compute_ccc,
    "kl_div": compute_kl_divergence,
    "iou_top10": partial(compute_topK_iou, percentile=0.90),
    "iou_top05": partial(compute_topK_iou, percentile=0.95),
    "iou_top02": partial(compute_topK_iou, percentile=0.98),
    "iou_top01": partial(compute_topK_iou, percentile=0.99),
    "iou_top005": partial(compute_topK_iou, percentile=0.995),
    "auc_iou_full": partial(compute_auc_iou, k_values=(0.01, 0.99), steps=99),  # AUC on full range of K (granularity 1%)
    "auc_iou_top10": partial(compute_auc_iou, k_values=(0.01, 0.10), steps=10),  # AUC for top 10% (granularity 1%)
    "mae_top10": partial(compute_topK_mae, percentile=0.90),
    "mae_top05": partial(compute_topK_mae, percentile=0.95),
    "mae_top02": partial(compute_topK_mae, percentile=0.98),
    "mae_top01": partial(compute_topK_mae, percentile=0.99),
}


AVAILABLE_LR_SCHEDULERS = ["onecycle", "cosine_warmup", "plateau", "multistep"]


def build_single_loss(name: str, huber_beta: float = 1.0) -> torch.nn.Module:
    name = str(name).lower()
    if name in ["bce", "bceloss"]:
        return BCELoss()
    if name in ["mse", "mseloss"]:
        return MSELoss()
    if name in ["mae", "maeloss"]:
        return MAELoss()
    if name in ["ccc", "cccloss"]:
        return CCCLoss()
    if name in ["huber", "huberloss", "smoothl1", "smooth_l1", "smoothl1loss"]:
        return HuberLoss(beta=huber_beta)
    if name in ["raw_pearson", "regression_pearson", "regressionpearsonloss", "raw_corr"]:
        return RegressionPearsonLoss()
    if name in ["focal", "focalloss"]:
        return FocalLoss()
    if name in ["dice", "diceloss"]:
        return DiceLoss()
    if name in ["klloss", "kl", "bernoullikl", "bernoulliklloss"]:
        return BernoulliKLLoss()
    if name in ["hex_mean_pearson", "hex_summary_mean_pearson"]:
        return HexSummaryLoss(summary="mean", correlation="pearson")
    if name in ["hex_top10_pearson", "hex_summary_top10_pearson"]:
        return HexSummaryLoss(summary="topk_mean", correlation="pearson", top_fraction=0.10)
    if name in ["hex_mean_ccc", "hex_summary_mean_ccc"]:
        return HexSummaryLoss(summary="mean", correlation="ccc")
    if name in ["hex_top10_ccc", "hex_summary_top10_ccc"]:
        return HexSummaryLoss(summary="topk_mean", correlation="ccc", top_fraction=0.10)
    if name in ["hex_mean_pairwise_rank", "hex_summary_mean_pairwise_rank"]:
        return HexSummaryLoss(summary="mean", correlation="pairwise_rank")
    if name in ["hex_top10_pairwise_rank", "hex_summary_top10_pairwise_rank"]:
        return HexSummaryLoss(summary="topk_mean", correlation="pairwise_rank", top_fraction=0.10)
    raise ValueError(f"Unknown loss type: {name}")


def visualize_model_predictions(
    test_loader: DataLoader,
    test_predictions: np.ndarray,
    n_samples: int = 4,
    seed: int = 42,
    save_path: str = None,
    channel_map: dict = None,
    feature_names_list: list = None,
) -> None:
    """
    Visualize model predictions versus targets for a selection of random samples.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader providing test batches as (inputs, targets, masks).
    test_predictions : numpy.ndarray
        Array containing model predictions corresponding to all samples in
        ``test_loader``, with shape ``(N, ...)`` or ``(N, 1, ...)``.
    n_samples : int, optional
        Number of random samples to visualize. Defaults to 4.
    seed: int, optional
        Random seed to get same patch IDs across different inference runs.
    save_path: str, optional
        Save path for the visualization figure (not saved if None).
    channel_map: dict, optional
        Dict mapping channel IDs to input feature names for plotting.
    feature_names_list: list, optional
        The list of used input features from the config. file.

    Returns
    -------
    None
        This function creates matplotlib figures and displays/saves them.
    """
    all_inputs, all_targets, all_masks = [], [], []

    # If more than one data source, we only need the grid for the viz.
    for batch in test_loader:
        if isinstance(batch, dict) and "grid" in batch:
            inputs, targets, masks = batch["grid"]
        else:
            inputs, targets, masks = batch

        all_inputs.append(inputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_masks.append(masks.detach().cpu().numpy())

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    preds = test_predictions.squeeze(1) if test_predictions.ndim == 4 else test_predictions
    targets = all_targets.squeeze(1) if all_targets.ndim == 4 else all_targets
    masks = all_masks.squeeze(1) if all_masks.ndim == 4 else all_masks

    selected_indices = []
    idx_to_label = {}

    # use channel names map and config features list if we provide it
    if channel_map and feature_names_list:
        current_tensor_idx = 0
        for feat_name in feature_names_list:
            if feat_name not in channel_map:
                continue

            # see how many channels this feature originally had in the map
            orig_indices = channel_map[feat_name]
            num_channels_for_feat = len(orig_indices)

            # make new relative indices for the current data
            relative_indices = list(range(current_tensor_idx, current_tensor_idx + num_channels_for_feat))
            # for multichannel input feats, only show first and last as examples
            if num_channels_for_feat > 2:
                subset = [relative_indices[0], relative_indices[-1]]
                for i, rel_idx in enumerate(subset):
                    selected_indices.append(rel_idx)
                    suffix = "first" if i == 0 else "last"
                    idx_to_label[rel_idx] = f"{feat_name}\n({suffix})"
            else:
                for rel_idx in relative_indices:
                    selected_indices.append(rel_idx)
                    idx_to_label[rel_idx] = feat_name

            current_tensor_idx += num_channels_for_feat
    else:
        # if no map, we just print channel indices for the fig
        selected_indices = list(range(all_inputs.shape[1]))
        idx_to_label = {i: f"Ch {i}" for i in selected_indices}

    # check we aren't out of bounds after new mapping
    selected_indices = [idx for idx in selected_indices if idx < all_inputs.shape[1]]

    n_cols = len(selected_indices) + 2
    rng = np.random.RandomState(seed)  # fix seed to get same patch ids between inferences
    indices = rng.choice(preds.shape[0], n_samples, replace=False)

    _, axes = plt.subplots(n_samples, n_cols, figsize=(4.2 * n_cols, 4 * n_samples), dpi=300)
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample_idx in enumerate(indices):
        axes[i, 0].annotate(
            f"Patch ID: {sample_idx}",
            xy=(-0.5, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
            rotation=90,
        )

        # show the input channels
        for col_idx, channel_idx in enumerate(selected_indices):
            ax = axes[i, col_idx]
            data = all_inputs[sample_idx, channel_idx]

            im = ax.imshow(data, cmap="viridis", vmin=data.min(), vmax=data.max())
            ax.set_title(idx_to_label[channel_idx], fontsize=10, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        masked_pred = preds[sample_idx] * masks[sample_idx]
        target_data = targets[sample_idx]

        # Get min/max for consistent color scale for targets and preds
        v_min = min(masked_pred.min(), target_data.min())
        v_max = max(masked_pred.max(), target_data.max())

        # show targets
        ax_t = axes[i, n_cols - 2]
        im_t = ax_t.imshow(target_data, cmap="viridis", vmin=v_min, vmax=v_max)
        ax_t.set_title("Target", fontsize=10, fontweight="bold")
        ax_t.axis("off")
        plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

        # show preds
        ax_p = axes[i, n_cols - 1]
        im_p = ax_p.imshow(masked_pred, cmap="viridis", vmin=v_min, vmax=v_max)
        ax_p.set_title("Prediction", fontsize=10, fontweight="bold")
        ax_p.axis("off")
        plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=(0.05, 0, 1, 1))

    # save the viz fig for easier usage if set to True
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    plt.tight_layout()
    plt.show()


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Seed all RNG sources for determinism.

    Parameters
    ----------
    seed: int
        Seed value
    determinstic: bool
        Ensures strict determinism but might slow down training

    Returns
    -------
    None
    """

    # (CPU) Python, OS, NumPy, Torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # (GPU, if available)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi GPU in case
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # MacOS / MPS specific
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # For PyTorch >= 1.8
    # Outside 'if cuda' because PyTorch has deterministic CPU algorithms too.
    if deterministic:
        with suppress(Exception):
            torch.use_deterministic_algorithms(True)

    print(f"[Info] Seed set to: {seed}")


def seed_worker(worker_id: int):
    """
    Helper function to set the seed for each worker based on the global seed.
    This ensures numpy and random in subprocesses are deterministic.
    """
    base_seed = torch.initial_seed()
    worker_seed = (base_seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_device() -> str:
    """Utils. to set up device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    return device
