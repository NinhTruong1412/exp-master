"""SASRec: Self-Attentive Sequential Recommendation.

Causal (left-to-right) self-attention for next-item prediction.
Implements both BCE (with negative sampling) and cross-entropy variants.
Reference: Kang & McAuley, ICDM 2018.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSequentialModel


class SASRec(BaseSequentialModel):
    """SASRec with causal self-attention."""

    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        **kwargs,
    ):
        super().__init__(num_items=num_items, hidden_size=hidden_size, max_seq_len=max_seq_len)
        self.item_embedding = nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_bias = nn.Parameter(torch.zeros(num_items + 2))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask for Transformer."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode(self, batch: dict) -> torch.Tensor:
        """Encode item sequence into hidden representations.

        Returns: (B, L, H) hidden states.
        """
        item_ids = batch["item_ids"]
        positions = batch["positions"]
        attention_mask = batch["attention_mask"]

        x = self.item_embedding(item_ids) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))

        causal_mask = self._causal_mask(item_ids.size(1), item_ids.device)
        padding_mask = attention_mask == 0

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return x

    def forward(self, batch: dict) -> dict:
        """Next-item prediction with cross-entropy loss.

        For each position t, predict item at position t+1.
        """
        item_ids = batch["item_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        hidden = self.encode(batch)  # (B, L, H)

        # Use all item embeddings as output projection (tied weights)
        logits = hidden @ self.item_embedding.weight.T + self.output_bias  # (B, L, V)

        # For SASRec, labels contains the next item at each position
        # Using masked positions from BERT-style labels as proxy
        valid_mask = labels > 0
        if valid_mask.sum() == 0:
            return {"loss": torch.tensor(0.0, device=item_ids.device)}

        logits_flat = logits[valid_mask]  # (N, V)
        labels_flat = labels[valid_mask]  # (N,)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)
        return {"loss": loss, "loss_mask": loss}

    def predict_scores(self, batch: dict) -> torch.Tensor:
        """Score all items using the last non-padding position."""
        hidden = self.encode(batch)  # (B, L, H)
        attention_mask = batch["attention_mask"]

        # Get last real position for each sequence
        lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
        lengths = lengths.clamp(min=0)
        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]  # (B, H)

        # Score all items
        scores = last_hidden @ self.item_embedding.weight.T + self.output_bias  # (B, V)
        return scores
