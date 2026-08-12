import pytest
import torch
import torch.nn.functional as F

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
    MultiTaskLoss,
    RegressionPearsonLoss,
    WeightedLoss,
)
from src.utils import build_single_loss


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    targets = torch.rand(4, 1, 32, 32)
    logits = targets * 0.9
    masks = torch.rand(4, 1, 32, 32) > 0.5
    return logits, targets, masks


# -------------------------
# BCE
# -------------------------


def test_bce_loss_no_mask(dummy_data):
    loss_fn = BCELoss()
    logits, targets, _ = dummy_data
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bce_loss_with_mask(dummy_data):
    loss_fn = BCELoss()
    logits, targets, masks = dummy_data
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked_loss = (bce * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, masked_loss, atol=1e-6)


def test_bce_loss_all_masked(dummy_data):
    loss_fn = BCELoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# MSE
# -------------------------


def test_mse_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = MSELoss()
    expected = ((torch.sigmoid(logits) - targets) ** 2).mean()
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = MSELoss()
    loss = (torch.sigmoid(logits) - targets) ** 2
    expected = (loss * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_all_masked(dummy_data):
    loss_fn = MSELoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


def test_ccc_loss_is_zero_for_perfect_probability_predictions():
    targets = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    logits = torch.logit(targets)
    loss = CCCLoss()(logits, targets, torch.ones_like(targets, dtype=torch.bool))
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_regression_pearson_loss_is_zero_for_perfect_predictions():
    targets = torch.tensor([[[[-1.0, 0.0], [1.0, 2.0]]]])
    loss = RegressionPearsonLoss()(targets, targets, torch.ones_like(targets, dtype=torch.bool))
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_build_single_loss_supports_campaign_losses():
    assert isinstance(build_single_loss("ccc"), CCCLoss)
    assert isinstance(build_single_loss("raw_pearson"), RegressionPearsonLoss)
    assert isinstance(build_single_loss("hex_mean_pairwise_rank"), HexSummaryLoss)
    assert isinstance(build_single_loss("hex_top10_pairwise_rank"), HexSummaryLoss)


# -------------------------
# MAE
# -------------------------


def test_mae_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = MAELoss()
    expected = (torch.abs(torch.sigmoid(logits) - targets)).mean()
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mae_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = MAELoss()
    loss = torch.abs(torch.sigmoid(logits) - targets)
    expected = (loss * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mae_loss_all_masked(dummy_data):
    loss_fn = MAELoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# Huber
# -------------------------


def test_huber_loss_no_mask_uses_raw_outputs(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = HuberLoss(beta=1.0)
    expected = F.smooth_l1_loss(logits, targets, beta=1.0, reduction="mean")
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_huber_loss_with_mask_uses_raw_outputs(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = HuberLoss(beta=1.0)
    loss = F.smooth_l1_loss(logits, targets, beta=1.0, reduction="none")
    expected = (loss * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_huber_loss_all_masked_is_finite(dummy_data):
    loss_fn = HuberLoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


def test_huber_loss_rejects_non_positive_beta():
    with pytest.raises(ValueError, match="Huber beta must be positive"):
        HuberLoss(beta=0.0)


def test_build_single_loss_passes_huber_beta():
    loss_fn = build_single_loss("huber", huber_beta=0.25)
    assert isinstance(loss_fn, HuberLoss)
    assert loss_fn.beta == 0.25


# -------------------------
# Dice
# -------------------------


def _dice_reference(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None, eps: float):
    probs = torch.sigmoid(logits)
    if mask is None:
        probs_f = probs.flatten(1)
        targets_f = targets.flatten(1)
    else:
        m = mask.to(dtype=probs.dtype)
        probs_f = (probs * m).flatten(1)
        targets_f = (targets * m).flatten(1)

    intersection = (probs_f * targets_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + targets_f.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def test_dice_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = DiceLoss(eps=1e-8)
    expected = _dice_reference(logits, targets, None, eps=1e-8)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_dice_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = DiceLoss(eps=1e-8)
    expected = _dice_reference(logits, targets, masks, eps=1e-8)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_dice_loss_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = DiceLoss(eps=1e-8)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# Focal
# -------------------------


def _focal_reference(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    gamma: float,
    alpha: float | None,
    eps: float,
):
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - p_t).clamp_min(0.0).pow(gamma)

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * focal_factor * bce
    else:
        loss = focal_factor * bce

    if mask is None:
        return loss.mean()

    m = mask.to(dtype=loss.dtype)
    loss = loss * m
    denom = m.sum().clamp_min(eps)
    return loss.sum() / denom


@pytest.mark.parametrize("alpha", [0.25, None])
def test_focal_loss_no_mask_matches_reference(dummy_data, alpha):
    logits, targets, _ = dummy_data
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha, eps=1e-8)
    expected = _focal_reference(logits, targets, None, gamma=2.0, alpha=alpha, eps=1e-8)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.25, None])
def test_focal_loss_with_mask_matches_reference(dummy_data, alpha):
    logits, targets, masks = dummy_data
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha, eps=1e-8)
    expected = _focal_reference(logits, targets, masks, gamma=2.0, alpha=alpha, eps=1e-8)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_focal_loss_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, eps=1e-8)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# Bernoulli KL
# -------------------------


def _bernoulli_kl_reference(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float,
    clamp_logits: float | None,
):
    if clamp_logits is not None:
        logits = logits.clamp(-clamp_logits, clamp_logits)

    targets = targets.to(dtype=logits.dtype)
    p = targets.clamp(eps, 1.0 - eps)

    log_q = F.logsigmoid(logits)
    log_1mq = F.logsigmoid(-logits)
    log_p = torch.log(p)
    log_1mp = torch.log1p(-p)

    loss = p * (log_p - log_q) + (1.0 - p) * (log_1mp - log_1mq)

    if mask is None:
        return loss.mean()

    m = mask.to(dtype=loss.dtype)
    loss = torch.where(m > 0, loss, torch.zeros_like(loss))
    denom = m.sum().clamp_min(eps)
    return loss.sum() / denom


def test_bernoulli_kl_no_mask_matches_reference(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = BernoulliKLLoss(eps=1e-6, clamp_logits=20.0)
    expected = _bernoulli_kl_reference(logits, targets, None, eps=1e-6, clamp_logits=20.0)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bernoulli_kl_with_mask_matches_reference(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = BernoulliKLLoss(eps=1e-6, clamp_logits=20.0)
    expected = _bernoulli_kl_reference(logits, targets, masks, eps=1e-6, clamp_logits=20.0)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bernoulli_kl_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = BernoulliKLLoss(eps=1e-6, clamp_logits=20.0)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# WeightedLoss
# -------------------------


def test_weighted_loss_returns_total_and_parts(dummy_data):
    logits, targets, masks = dummy_data
    losses = {"bce": BCELoss(), "mse": MSELoss()}
    weights = {"bce": 0.7, "mse": 0.3}
    loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

    total, parts = loss_fn(logits, targets, masks)

    assert isinstance(total, torch.Tensor)
    assert isinstance(parts, dict)
    assert set(parts.keys()) == {"bce", "mse"}


def test_weighted_loss_total_matches_manual(dummy_data):
    logits, targets, masks = dummy_data
    losses = {"bce": BCELoss(), "mse": MSELoss()}
    weights = {"bce": 0.7, "mse": 0.3}

    loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)
    total, parts = loss_fn(logits, targets, masks)

    # manual total using normalized weights
    w = torch.tensor([weights["bce"], weights["mse"]], dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-8)

    expected_total = w[0].to(dtype=parts["bce"].dtype) * parts["bce"] + w[1].to(dtype=parts["mse"].dtype) * parts["mse"]
    assert torch.allclose(total, expected_total, atol=1e-6)


def test_weighted_loss_buffer_moves_device(dummy_data):
    logits, targets, masks = dummy_data
    losses = {"bce": BCELoss(), "mse": MSELoss()}
    weights = {"bce": 1.0, "mse": 1.0}
    loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

    # Move to same device as logits (this is CPU in unit tests typically)
    loss_fn = loss_fn.to(logits.device)
    assert loss_fn._weights.device == logits.device


def test_hex_summary_loss_is_zero_for_perfect_hex_mean_rank():
    targets = torch.tensor(
        [
            [[[[0.1, 0.2], [0.1, 0.2]]]],
            [[[[0.2, 0.3], [0.2, 0.3]]]],
            [[[[0.7, 0.8], [0.7, 0.8]]]],
            [[[[0.8, 0.9], [0.8, 0.9]]]],
        ],
        dtype=torch.float32,
    ).squeeze(1)
    logits = torch.logit(targets.clamp(1e-4, 1 - 1e-4))
    masks = torch.ones_like(targets, dtype=torch.bool)
    patch_metadata = {"hex_id": torch.tensor([1, 1, 2, 2])}

    loss = HexSummaryLoss(summary="mean", correlation="pearson")(logits, targets, masks, patch_metadata=patch_metadata)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_hex_pairwise_rank_loss_prefers_correct_hex_order():
    targets = torch.tensor(
        [
            [[[[0.05, 0.10], [0.05, 0.10]]]],
            [[[[0.10, 0.15], [0.10, 0.15]]]],
            [[[[0.40, 0.45], [0.40, 0.45]]]],
            [[[[0.45, 0.50], [0.45, 0.50]]]],
            [[[[0.80, 0.85], [0.80, 0.85]]]],
            [[[[0.85, 0.90], [0.85, 0.90]]]],
        ],
        dtype=torch.float32,
    ).squeeze(1)
    correct_logits = torch.logit(targets.clamp(1e-4, 1 - 1e-4))
    reversed_probs = 1.0 - targets
    reversed_logits = torch.logit(reversed_probs.clamp(1e-4, 1 - 1e-4))
    masks = torch.ones_like(targets, dtype=torch.bool)
    patch_metadata = {"hex_id": torch.tensor([1, 1, 2, 2, 3, 3])}
    loss_fn = HexSummaryLoss(summary="mean", correlation="pairwise_rank")

    correct_loss = loss_fn(correct_logits, targets, masks, patch_metadata=patch_metadata)
    reversed_loss = loss_fn(reversed_logits, targets, masks, patch_metadata=patch_metadata)

    assert torch.isfinite(correct_loss)
    assert torch.isfinite(reversed_loss)
    assert correct_loss < reversed_loss


def test_hex_pairwise_rank_loss_returns_zero_without_meaningful_pairs():
    targets = torch.full((4, 1, 2, 2), 0.2, dtype=torch.float32)
    logits = torch.zeros_like(targets)
    masks = torch.ones_like(targets, dtype=torch.bool)
    patch_metadata = {"hex_id": torch.tensor([1, 1, 2, 2])}

    loss = HexSummaryLoss(summary="mean", correlation="pairwise_rank")(logits, targets, masks, patch_metadata=patch_metadata)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_weighted_loss_passes_patch_metadata_to_hex_summary_loss():
    targets = torch.tensor(
        [
            [[[[0.1, 0.2], [0.1, 0.2]]]],
            [[[[0.2, 0.3], [0.2, 0.3]]]],
            [[[[0.7, 0.8], [0.7, 0.8]]]],
            [[[[0.8, 0.9], [0.8, 0.9]]]],
        ],
        dtype=torch.float32,
    ).squeeze(1)
    logits = torch.logit(targets.clamp(1e-4, 1 - 1e-4))
    masks = torch.ones_like(targets, dtype=torch.bool)
    patch_metadata = {"hex_id": torch.tensor([1, 1, 2, 2])}
    loss_fn = WeightedLoss(
        losses={"kl": BernoulliKLLoss(), "hex_mean_pearson": HexSummaryLoss(summary="mean", correlation="pearson")},
        weights={"kl": 0.9, "hex_mean_pearson": 0.1},
    )

    total, parts = loss_fn(logits, targets, masks, patch_metadata=patch_metadata)

    assert loss_fn.requires_patch_metadata is True
    assert set(parts) == {"kl", "hex_mean_pearson"}
    assert torch.isfinite(total)


def test_multi_task_loss_routes_channels_masks_and_metadata():
    targets = torch.tensor(
        [
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.5, 1.0], [1.5, 2.0]],
            ],
            [
                [[0.6, 0.7], [0.8, 0.9]],
                [[2.0, 3.0], [4.0, 5.0]],
                [[1.0, 1.5], [2.0, 2.5]],
            ],
        ],
        dtype=torch.float32,
    )
    logits = targets.clone().requires_grad_(True)
    masks = torch.ones_like(targets, dtype=torch.bool)
    masks[:, 1:, 0, 0] = False
    patch_metadata = {"hex_id": torch.tensor([1, 2])}

    loss_fn = MultiTaskLoss(
        target_names=["bp", "fi", "ros"],
        losses={
            "bp": WeightedLoss(
                losses={"kl": BernoulliKLLoss(), "hex": HexSummaryLoss(summary="mean", correlation="pearson")},
                weights={"kl": 0.9, "hex": 0.1},
            ),
            "fi": HuberLoss(),
            "ros": HuberLoss(),
        },
        task_weights={"bp": 0.5, "fi": 0.25, "ros": 0.25},
    )

    total, parts = loss_fn(logits, targets, masks, patch_metadata=patch_metadata)
    total.backward()

    assert loss_fn.requires_patch_metadata is True
    assert set(parts) == {"bp/kl", "bp/hex", "bp/total", "fi/total", "ros/total"}
    assert torch.isfinite(total)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]) > 0
