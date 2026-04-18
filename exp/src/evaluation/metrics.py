"""Evaluation metrics for sequential recommendation.

Supports HR@K, NDCG@K, MRR, and watch-time-specific metrics.
All metrics operate on per-user score vectors with full-sort evaluation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def hit_at_k(scores: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Hit Rate @ K: 1 if target item is in top-K, else 0.

    Args:
        scores: (num_users, num_items) predicted scores
        target: (num_users,) target item IDs (1-indexed)
        k: cutoff
    Returns:
        (num_users,) binary hit indicators
    """
    topk_items = np.argpartition(-scores, k, axis=1)[:, :k]
    # Convert 1-indexed targets to 0-indexed
    target_idx = target - 1
    hits = np.array([t in topk_items[i] for i, t in enumerate(target_idx)], dtype=np.float32)
    return hits


def ndcg_at_k(scores: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """NDCG @ K for single-target next-item prediction.

    For single relevant item, NDCG = 1/log2(rank+1) if rank <= k, else 0.
    """
    num_users = scores.shape[0]
    target_idx = target - 1
    target_scores = scores[np.arange(num_users), target_idx]

    # Count how many items have score >= target score (1-indexed rank)
    ranks = (scores >= target_scores[:, None]).sum(axis=1)
    ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    return ndcg.astype(np.float32)


def mrr(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean Reciprocal Rank (no cutoff)."""
    num_users = scores.shape[0]
    target_idx = target - 1
    target_scores = scores[np.arange(num_users), target_idx]
    ranks = (scores >= target_scores[:, None]).sum(axis=1)
    return (1.0 / ranks).astype(np.float32)


def compute_all_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    ks: list[int] = (5, 10, 20),
) -> dict[str, float]:
    """Compute all ranking metrics.

    Returns dict with keys like 'hr@5', 'ndcg@10', 'mrr'.
    Values are mean across users.
    """
    metrics = {}
    for k in ks:
        metrics[f"hr@{k}"] = float(hit_at_k(scores, targets, k).mean())
        metrics[f"ndcg@{k}"] = float(ndcg_at_k(scores, targets, k).mean())
    metrics["mrr"] = float(mrr(scores, targets).mean())
    return metrics


def compute_per_user_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    ks: list[int] = (5, 10, 20),
) -> dict[str, np.ndarray]:
    """Compute per-user metrics (for significance testing).

    Returns dict mapping metric name to (num_users,) arrays.
    """
    metrics = {}
    for k in ks:
        metrics[f"hr@{k}"] = hit_at_k(scores, targets, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(scores, targets, k)
    metrics["mrr"] = mrr(scores, targets)
    return metrics


def watch_time_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute watch-time prediction metrics (MAE, RMSE).

    Args:
        predicted: predicted watch ratios
        actual: actual watch ratios
        mask: boolean mask for valid positions
    """
    if mask is not None:
        predicted = predicted[mask]
        actual = actual[mask]

    if len(predicted) == 0:
        return {"watch_mae": 0.0, "watch_rmse": 0.0}

    mae = float(np.abs(predicted - actual).mean())
    rmse = float(np.sqrt(((predicted - actual) ** 2).mean()))
    return {"watch_mae": mae, "watch_rmse": rmse}
