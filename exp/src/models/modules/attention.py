"""Duration-biased multi-head self-attention.

Extends standard self-attention with:
  a_ij = (Q_i · K_j^T) / √h + b_gap(|τ_i - τ_j|) + b_dur(d_j)

The time-gap bias captures irregular temporal spacing.
The duration bias allows the model to weigh attention differently
based on content duration (e.g., skipping a 10s clip vs. a 4min video).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationBiasedAttention(nn.Module):
    """Multi-head attention with learned time-gap and duration biases."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_time_gap_buckets: int = 32,
        num_duration_buckets: int = 16,
        dropout: float = 0.1,
        use_time_gap_bias: bool = True,
        use_duration_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.use_time_gap_bias = use_time_gap_bias
        self.use_duration_bias = use_duration_bias

        # Per-head learned biases
        if use_time_gap_bias:
            self.gap_bias = nn.Embedding(num_time_gap_buckets + 1, num_heads, padding_idx=0)
            nn.init.zeros_(self.gap_bias.weight)

        if use_duration_bias:
            self.dur_bias = nn.Embedding(num_duration_buckets + 1, num_heads, padding_idx=0)
            nn.init.zeros_(self.dur_bias.weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        time_gap_bucket_ids: torch.Tensor = None,
        duration_bucket_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, H) input hidden states
            attention_mask: (B, L) with 1=real, 0=pad
            time_gap_bucket_ids: (B, L) for relative gap bias
            duration_bucket_ids: (B, L) for per-key duration bias

        Returns:
            (B, L, H) attention output
        """
        B, L, H = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, num_heads, L, L)

        # Add time-gap relative bias
        if self.use_time_gap_bias and time_gap_bucket_ids is not None:
            # Compute pairwise relative gap buckets
            # Simple approach: |bucket_i - bucket_j| clamped to max bucket
            gap_i = time_gap_bucket_ids.unsqueeze(2)  # (B, L, 1)
            gap_j = time_gap_bucket_ids.unsqueeze(1)  # (B, 1, L)
            relative_gap = (gap_i - gap_j).abs()
            max_bucket = self.gap_bias.num_embeddings - 1
            relative_gap = relative_gap.clamp(0, max_bucket)

            gap_bias_values = self.gap_bias(relative_gap)  # (B, L, L, num_heads)
            gap_bias_values = gap_bias_values.permute(0, 3, 1, 2)  # (B, num_heads, L, L)
            attn_scores = attn_scores + gap_bias_values

        # Add per-key duration bias
        if self.use_duration_bias and duration_bucket_ids is not None:
            dur_bias_values = self.dur_bias(duration_bucket_ids)  # (B, L, num_heads)
            dur_bias_values = dur_bias_values.permute(0, 2, 1).unsqueeze(2)  # (B, num_heads, 1, L)
            attn_scores = attn_scores + dur_bias_values

        # Apply attention mask
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.out_proj(out)
        return out


class DurationBiasedTransformerLayer(nn.Module):
    """Transformer encoder layer with duration-biased attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_time_gap_buckets: int = 32,
        num_duration_buckets: int = 16,
        dropout: float = 0.1,
        use_time_gap_bias: bool = True,
        use_duration_bias: bool = True,
    ):
        super().__init__()
        self.attn = DurationBiasedAttention(
            hidden_size, num_heads, num_time_gap_buckets,
            num_duration_buckets, dropout, use_time_gap_bias, use_duration_bias,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        time_gap_bucket_ids: torch.Tensor = None,
        duration_bucket_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attention_mask, time_gap_bucket_ids, duration_bucket_ids)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class DurationBiasedEncoder(nn.Module):
    """Stack of duration-biased transformer layers."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_time_gap_buckets: int = 32,
        num_duration_buckets: int = 16,
        dropout: float = 0.1,
        use_time_gap_bias: bool = True,
        use_duration_bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DurationBiasedTransformerLayer(
                hidden_size, num_heads, num_time_gap_buckets,
                num_duration_buckets, dropout, use_time_gap_bias, use_duration_bias,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        time_gap_bucket_ids: torch.Tensor = None,
        duration_bucket_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask, time_gap_bucket_ids, duration_bucket_ids)
        return self.final_norm(x)
