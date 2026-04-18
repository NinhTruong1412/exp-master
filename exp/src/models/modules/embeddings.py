"""Embedding modules for watch-time-enriched sequential models.

Provides item, position, time-gap, duration, and watch-signal embeddings.
Each historical interaction token is represented as:
  x_t = e(item) + e(pos) + e(gap_bucket) + W_d·φ(log(1+d)) + e(dur_bucket)
        + W_r·φ(watch_ratio) + e(watch_bucket) + e(flags)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EnrichedEmbedding(nn.Module):
    """Combines item ID, position, temporal, duration, and watch-signal embeddings."""

    def __init__(
        self,
        num_items: int,
        hidden_size: int,
        max_seq_len: int = 50,
        num_duration_buckets: int = 16,
        num_watch_buckets: int = 32,
        num_time_gap_buckets: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mask_token_id = num_items + 1

        # Core embeddings
        self.item_embedding = nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Temporal embedding: time-gap bucket
        self.time_gap_embedding = nn.Embedding(num_time_gap_buckets + 1, hidden_size, padding_idx=0)

        # Duration embeddings: continuous projection + bucket embedding
        self.duration_proj = nn.Linear(1, hidden_size)
        self.duration_bucket_embedding = nn.Embedding(num_duration_buckets + 1, hidden_size, padding_idx=0)

        # Watch-signal embeddings: continuous ratio projection + bucket embedding
        self.watch_ratio_proj = nn.Linear(1, hidden_size)
        self.watch_bucket_embedding = nn.Embedding(num_watch_buckets + 1, hidden_size, padding_idx=0)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.time_gap_embedding.weight, std=0.02)
        nn.init.normal_(self.duration_bucket_embedding.weight, std=0.02)
        nn.init.normal_(self.watch_bucket_embedding.weight, std=0.02)
        nn.init.zeros_(self.item_embedding.weight[0])
        nn.init.zeros_(self.time_gap_embedding.weight[0])
        nn.init.zeros_(self.duration_bucket_embedding.weight[0])
        nn.init.zeros_(self.watch_bucket_embedding.weight[0])

    def forward(
        self,
        item_ids: torch.Tensor,
        positions: torch.Tensor,
        watch_ratios: torch.Tensor,
        watch_bucket_ids: torch.Tensor,
        duration_bucket_ids: torch.Tensor,
        time_gap_bucket_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: (B, L) item IDs
            positions: (B, L) position indices
            watch_ratios: (B, L) watch ratio values [0, 1]
            watch_bucket_ids: (B, L) watch bucket indices
            duration_bucket_ids: (B, L) duration bucket indices
            time_gap_bucket_ids: (B, L) time gap bucket indices, optional

        Returns:
            (B, L, H) fused token embeddings
        """
        x = self.item_embedding(item_ids)
        x = x + self.position_embedding(positions)

        # Duration: continuous log-transform + bucket
        dur_continuous = self.duration_proj(
            torch.log1p(duration_bucket_ids.float()).unsqueeze(-1)
        )
        x = x + dur_continuous + self.duration_bucket_embedding(duration_bucket_ids)

        # Watch signal: continuous ratio + bucket
        wr_continuous = self.watch_ratio_proj(watch_ratios.unsqueeze(-1))
        x = x + wr_continuous + self.watch_bucket_embedding(watch_bucket_ids)

        # Time gap (if available)
        if time_gap_bucket_ids is not None:
            x = x + self.time_gap_embedding(time_gap_bucket_ids)

        x = self.layer_norm(self.dropout(x))
        return x
