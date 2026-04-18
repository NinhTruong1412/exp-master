"""BERT4Rec: Sequential Recommendation with Bidirectional Encoder.

Bidirectional Transformer with masked item prediction (Cloze task).
Reference: Sun et al., CIKM 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSequentialModel


class BERT4Rec(BaseSequentialModel):
    """Vanilla BERT4Rec with masked-item prediction."""

    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        mask_ratio: float = 0.15,
        **kwargs,
    ):
        super().__init__(num_items=num_items, hidden_size=hidden_size, max_seq_len=max_seq_len)
        self.mask_ratio = mask_ratio
        # +2: 0=pad, num_items+1=mask
        self.mask_token_id = num_items + 1
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

    def encode(self, batch: dict) -> torch.Tensor:
        """Encode sequence with bidirectional attention.

        Returns: (B, L, H) hidden states.
        """
        item_ids = batch["item_ids"]
        positions = batch["positions"]
        attention_mask = batch["attention_mask"]

        x = self.item_embedding(item_ids) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))

        # Bidirectional — no causal mask
        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x

    def forward(self, batch: dict) -> dict:
        """Compute masked item prediction loss."""
        labels = batch["labels"]

        hidden = self.encode(batch)  # (B, L, H)

        # Project to item vocabulary
        logits = hidden @ self.item_embedding.weight.T + self.output_bias  # (B, L, V)

        # Only compute loss at masked positions
        valid_mask = labels > 0
        if valid_mask.sum() == 0:
            return {"loss": torch.tensor(0.0, device=hidden.device)}

        logits_flat = logits[valid_mask]  # (N, V)
        labels_flat = labels[valid_mask]  # (N,)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)
        return {"loss": loss, "loss_mask": loss}

    def predict_scores(self, batch: dict) -> torch.Tensor:
        """Score all items for evaluation.

        Append a [MASK] token at the last position and use its
        hidden state to predict the next item.
        """
        item_ids = batch["item_ids"].clone()
        attention_mask = batch["attention_mask"].clone()

        # Find last real position and place mask token after it
        lengths = attention_mask.sum(dim=1)  # number of real tokens
        B, L = item_ids.shape

        for i in range(B):
            pos = min(int(lengths[i]), L - 1)
            item_ids[i, pos] = self.mask_token_id
            attention_mask[i, pos] = 1

        modified_batch = {**batch, "item_ids": item_ids, "attention_mask": attention_mask}
        hidden = self.encode(modified_batch)  # (B, L, H)

        # Get hidden state at mask position
        mask_positions = []
        for i in range(B):
            pos = min(int(lengths[i]), L - 1)
            mask_positions.append(pos)
        mask_positions = torch.tensor(mask_positions, device=hidden.device)

        mask_hidden = hidden[torch.arange(B, device=hidden.device), mask_positions]  # (B, H)
        scores = mask_hidden @ self.item_embedding.weight.T + self.output_bias  # (B, V)
        return scores
