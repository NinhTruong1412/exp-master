"""Loss functions for multi-task sequential recommendation.

Centralized loss computation for all model variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_item_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss at masked positions.

    Args:
        logits: (B, L, V) vocabulary logits
        labels: (B, L) target item IDs (0 = no prediction)
    """
    valid_mask = labels > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits_flat = logits[valid_mask]
    labels_flat = labels[valid_mask]
    return F.cross_entropy(logits_flat, labels_flat, ignore_index=0)


def huber_loss_masked(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.5,
) -> torch.Tensor:
    """Huber loss at valid positions."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    return F.huber_loss(predictions[mask], targets[mask], delta=delta)


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    diff = pos_scores - neg_scores
    if weights is not None:
        diff = diff * weights
    return -F.logsigmoid(diff).mean()


def infonce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    sim = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(B, device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
) -> torch.Tensor:
    """Pinball (quantile) loss.

    Args:
        predictions: (..., num_taus)
        targets: (..., 1) or same shape
        taus: (num_taus,) quantile levels
    """
    if targets.dim() < predictions.dim():
        targets = targets.unsqueeze(-1)
    errors = targets - predictions
    loss = torch.where(errors >= 0, taus * errors, (taus - 1) * errors)
    return loss.mean()


def watch_ratio_to_ordinal_class(
    watch_ratio: torch.Tensor,
    num_classes: int = 5,
) -> torch.Tensor:
    """Convert watch ratio [0, 1+] to ordinal class.

    Classes: 0=skip (<0.05), 1=glance (0.05-0.2), 2=partial (0.2-0.5),
             3=deep (0.5-0.9), 4=complete (>0.9)
    """
    classes = torch.zeros_like(watch_ratio, dtype=torch.long)
    classes[watch_ratio >= 0.05] = 1
    classes[watch_ratio >= 0.2] = 2
    classes[watch_ratio >= 0.5] = 3
    classes[watch_ratio >= 0.9] = 4
    return classes.clamp(0, num_classes - 1)


def duration_stratified_quantile_target(
    watch_ratios: torch.Tensor,
    duration_bucket_ids: torch.Tensor,
    num_quantile_bins: int = 10,
) -> torch.Tensor:
    """Compute within-duration-bucket quantile targets (D2Q-style).

    For each position, replace the raw watch ratio with its
    quantile rank within the same duration bucket.
    Returns values in [0, 1].
    """
    result = torch.zeros_like(watch_ratios)
    for bucket_id in duration_bucket_ids.unique():
        mask = duration_bucket_ids == bucket_id
        if mask.sum() == 0:
            continue
        vals = watch_ratios[mask]
        # Rank-based quantile
        sorted_indices = vals.argsort()
        ranks = torch.zeros_like(sorted_indices, dtype=torch.float)
        ranks[sorted_indices] = torch.arange(len(vals), device=vals.device, dtype=torch.float)
        quantiles = ranks / max(len(vals) - 1, 1)
        result[mask] = quantiles
    return result
