"""Popularity baseline model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BaseSequentialModel


class PopModel(BaseSequentialModel):
    """Popularity-based baseline: rank items by training frequency.

    Not a learnable model — just counts item occurrences.
    """

    def __init__(self, num_items: int, **kwargs):
        super().__init__(num_items=num_items, hidden_size=0, **kwargs)
        self.register_buffer("item_scores", torch.zeros(num_items + 1))

    def fit_popularity(self, train_loader):
        """Count item frequencies from training data."""
        counts = torch.zeros(self.num_items + 1)
        for batch in train_loader:
            items = batch["item_ids"].flatten()
            items = items[items > 0]
            for item_id in items.tolist():
                if item_id <= self.num_items:
                    counts[item_id] += 1
        # Normalize to [0, 1]
        max_count = counts.max()
        if max_count > 0:
            counts = counts / max_count
        self.item_scores.copy_(counts)

    def forward(self, batch: dict) -> dict:
        return {"loss": torch.tensor(0.0)}

    def predict_scores(self, batch: dict) -> torch.Tensor:
        B = batch["item_ids"].shape[0]
        # Return same popularity scores for every user
        scores = self.item_scores.unsqueeze(0).expand(B, -1)
        return scores
