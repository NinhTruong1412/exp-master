"""Full-sort evaluator with stratified metrics.

Evaluates models by scoring all items for each user, with optional
duration-stratified and user-activity-stratified breakdowns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_all_metrics, compute_per_user_metrics

logger = logging.getLogger("experiment")


class Evaluator:
    """Full-sort evaluator for sequential recommendation models."""

    def __init__(
        self,
        num_items: int,
        ks: list[int] = (5, 10, 20),
        device: torch.device = None,
    ):
        self.num_items = num_items
        self.ks = ks
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        return_per_user: bool = False,
    ) -> dict:
        """Evaluate model on dataloader with full-sort ranking.

        Args:
            model: model with predict_scores(batch) -> (B, num_items) method
            dataloader: eval dataloader returning dicts with 'target_item'
            return_per_user: if True, include per-user metric arrays

        Returns:
            dict with mean metrics and optionally per-user arrays
        """
        model.eval()
        all_scores = []
        all_targets = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            scores = model.predict_scores(batch)  # (B, num_items)

            # Zero out scores for padding item (index 0)
            if scores.shape[1] > self.num_items:
                scores = scores[:, 1 : self.num_items + 1]

            # Exclude items in user history from ranking
            item_ids = batch["item_ids"]
            for i in range(scores.shape[0]):
                history = item_ids[i][item_ids[i] > 0].cpu().numpy() - 1  # 0-index
                history = history[history < scores.shape[1]]
                scores[i, history] = -float("inf")

            all_scores.append(scores.cpu().numpy())
            all_targets.append(batch["target_item"].cpu().numpy())

        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_all_metrics(all_scores, all_targets, self.ks)

        if return_per_user:
            per_user = compute_per_user_metrics(all_scores, all_targets, self.ks)
            metrics["per_user"] = per_user

        logger.info(
            f"Eval results: "
            + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if not isinstance(v, dict))
        )
        return metrics

    @torch.no_grad()
    def evaluate_stratified(
        self,
        model,
        dataloader: DataLoader,
        duration_bucket_map: Optional[dict] = None,
    ) -> dict:
        """Evaluate with duration-stratified metrics.

        Returns both aggregate and per-duration-bucket metrics.
        """
        model.eval()
        all_scores = []
        all_targets = []
        all_dur_buckets = []

        for batch in tqdm(dataloader, desc="Stratified eval", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            scores = model.predict_scores(batch)

            if scores.shape[1] > self.num_items:
                scores = scores[:, 1 : self.num_items + 1]

            item_ids = batch["item_ids"]
            for i in range(scores.shape[0]):
                history = item_ids[i][item_ids[i] > 0].cpu().numpy() - 1
                history = history[history < scores.shape[1]]
                scores[i, history] = -float("inf")

            all_scores.append(scores.cpu().numpy())
            all_targets.append(batch["target_item"].cpu().numpy())
            # Last position's duration bucket
            dur = batch["duration_bucket_ids"][:, -1].cpu().numpy()
            all_dur_buckets.append(dur)

        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_dur_buckets = np.concatenate(all_dur_buckets, axis=0)

        result = {"aggregate": compute_all_metrics(all_scores, all_targets, self.ks)}

        # Per-bucket metrics
        bucket_names = {0: "unknown", 1: "<5m", 2: "5-15m", 3: "15-30m", 4: "30-60m", 5: "60-120m", 6: "120m+"}
        result["by_duration"] = {}
        for bucket_id, bucket_name in bucket_names.items():
            mask = all_dur_buckets == bucket_id
            if mask.sum() > 0:
                result["by_duration"][bucket_name] = compute_all_metrics(
                    all_scores[mask], all_targets[mask], self.ks
                )

        return result
