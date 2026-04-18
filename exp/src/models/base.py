"""Abstract base model for sequential recommendation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSequentialModel(ABC, nn.Module):
    """Base class for all sequential recommendation models.

    Subclasses must implement:
      - forward(batch) -> dict of losses
      - predict_scores(batch) -> (B, num_items) score tensor
    """

    def __init__(self, num_items: int, hidden_size: int = 128, max_seq_len: int = 50, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    @abstractmethod
    def forward(self, batch: dict) -> dict:
        """Compute losses for training.

        Args:
            batch: dict from DataLoader with item_ids, labels, etc.

        Returns:
            dict with 'loss' (total scalar), and optionally individual
            loss components like 'loss_mask', 'loss_ratio', etc.
        """

    @abstractmethod
    def predict_scores(self, batch: dict) -> torch.Tensor:
        """Predict item scores for evaluation.

        Args:
            batch: dict from eval DataLoader with item_ids, target_item, etc.

        Returns:
            (B, num_items) tensor of item scores for ranking.
        """

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
