"""Variant A — Duration-Aware Multi-Task BERT4Rec.

Extends RecBole's BERT4Rec with three watch-time-aware auxiliary losses
on top of the standard masked-item cross-entropy:

  1. **Watch-ratio regression** (Linear -> GELU -> Linear -> Sigmoid, Huber loss)
     predicts the continuous watch ratio at each masked position.
  2. **BPR pair-wise ranking** pushes scores of high-engagement items
     above low-engagement items, weighted by watch ratio.
  3. **Contrastive InfoNCE** separates high- vs low-engagement
     representations via supervised contrastive learning.

Total loss::

    L = CE + lambda_ratio * Huber + lambda_pair * BPR + lambda_contrastive * InfoNCE

Extra config keys (defaults)::

    lambda_ratio           0.3
    lambda_pair            0.2
    lambda_contrastive     0.1
    contrastive_temp       0.1
    watch_ratio_threshold  0.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.bert4rec import BERT4Rec


# ---------------------------------------------------------------------------
# Config helper -- RecBole Config.__getitem__ returns None for missing keys
# ---------------------------------------------------------------------------

def _cfg(config, key, default):
    val = config[key]
    return val if val is not None else default


# ===================================================================
# Variant A
# ===================================================================

class VariantA(BERT4Rec):
    """Duration-Aware Multi-Task BERT4Rec (Variant A).

    Inherits the full BERT4Rec backbone (item / position embeddings,
    TransformerEncoder, FFN prediction head).  Only ``calculate_loss``
    is overridden; ``predict`` and ``full_sort_predict`` remain
    untouched so evaluation uses the same next-item scoring as vanilla
    BERT4Rec.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # --- hyper-parameters ------------------------------------------------
        self.lambda_ratio = _cfg(config, "lambda_ratio", 0.3)
        self.lambda_pair = _cfg(config, "lambda_pair", 0.2)
        self.lambda_contrastive = _cfg(config, "lambda_contrastive", 0.1)
        self.contrastive_temp = _cfg(config, "contrastive_temp", 0.1)
        self.watch_ratio_threshold = _cfg(config, "watch_ratio_threshold", 0.3)

        # --- watch-ratio prediction: Linear -> GELU -> Linear -> Sigmoid -----
        self.ratio_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

        # --- contrastive projection head -------------------------------------
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # re-initialise (parent already called once; new modules need it too)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def calculate_loss(self, interaction):
        """Compute base CE + three watch-time-aware auxiliary losses."""
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]       # [B, L]
        pos_items = interaction[self.POS_ITEMS]                 # [B, M]
        neg_items = interaction[self.NEG_ITEMS]                 # [B, M]
        masked_index = interaction[self.MASK_INDEX]             # [B, M]
        watch_ratio_list = interaction["watch_ratio_list"]      # [B, L]

        # ---- encoder forward ------------------------------------------------
        seq_output = self.forward(masked_item_seq)              # [B, L, H]

        # ---- gather masked-position hidden states ---------------------------
        B, M = masked_index.shape
        L = masked_item_seq.size(-1)
        pred_index_map = self.multi_hot_embed(masked_index, L)  # [B*M, L]
        pred_index_map = pred_index_map.view(B, M, L)           # [B, M, L]
        seq_masked = torch.bmm(pred_index_map, seq_output)      # [B, M, H]

        targets = (masked_index > 0).float()                    # [B, M]
        n_valid = targets.sum().clamp(min=1.0)

        # ---- 1. base CE loss (same logic as parent) -------------------------
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        test_item_emb = self.item_embedding.weight[: self.n_items]  # [V, H]
        logits = (
            torch.matmul(seq_masked, test_item_emb.transpose(0, 1))
            + self.output_bias
        )                                                       # [B, M, V]
        ce_loss = (
            torch.sum(
                loss_fct(
                    logits.view(-1, test_item_emb.size(0)),
                    pos_items.view(-1),
                )
                * targets.view(-1)
            )
            / n_valid
        )

        # ---- extract watch ratios at masked positions -----------------------
        wr_masked = torch.bmm(
            pred_index_map,
            watch_ratio_list.unsqueeze(-1),
        ).squeeze(-1)                                           # [B, M]

        # ---- 2. watch-ratio Huber loss --------------------------------------
        ratio_loss = self._ratio_loss(seq_masked, wr_masked, targets)

        # ---- 3. BPR pair-wise loss ------------------------------------------
        bpr_loss = self._bpr_loss(
            seq_masked, pos_items, neg_items, wr_masked, targets,
        )

        # ---- 4. contrastive InfoNCE ----------------------------------------
        cl_loss = self._contrastive_loss(seq_masked, wr_masked, targets)

        return (
            ce_loss
            + self.lambda_ratio * ratio_loss
            + self.lambda_pair * bpr_loss
            + self.lambda_contrastive * cl_loss
        )

    # ------------------------------------------------------------------ #
    #  Auxiliary losses                                                    #
    # ------------------------------------------------------------------ #

    def _ratio_loss(self, seq_masked, wr_masked, targets):
        """Huber loss between predicted and actual watch ratio."""
        valid = targets.bool()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq_masked.device)
        pred = self.ratio_head(seq_masked).squeeze(-1)          # [B, M]
        return F.huber_loss(pred[valid], wr_masked[valid], delta=0.5)

    def _bpr_loss(self, seq_masked, pos_items, neg_items,
                  wr_masked, targets):
        """Engagement-weighted BPR pairwise ranking loss."""
        if targets.sum() == 0:
            return torch.tensor(0.0, device=seq_masked.device)

        pos_emb = self.item_embedding(pos_items)                # [B, M, H]
        neg_emb = self.item_embedding(neg_items)                # [B, M, H]

        pos_score = (
            torch.sum(seq_masked * pos_emb, dim=-1)
            + self.output_bias[pos_items]
        )                                                       # [B, M]
        neg_score = (
            torch.sum(seq_masked * neg_emb, dim=-1)
            + self.output_bias[neg_items]
        )                                                       # [B, M]

        # higher watch ratio -> stronger ranking signal
        weights = 1.0 + wr_masked.detach()
        diff = (pos_score - neg_score) * weights

        loss = -torch.sum(
            torch.log(1e-14 + torch.sigmoid(diff)) * targets
        ) / targets.sum().clamp(min=1.0)
        return loss

    def _contrastive_loss(self, seq_masked, wr_masked, targets):
        """Supervised contrastive (InfoNCE) separating engagement levels.

        Per-sequence mean-pooled representations are computed for the
        *high-engagement* and *low-engagement* masked positions.  Same
        engagement class -> positive pair; different class -> negative.
        """
        device = seq_masked.device
        B, M, H = seq_masked.shape
        valid = targets.bool()

        high = (wr_masked > self.watch_ratio_threshold) & valid
        low = (wr_masked <= self.watch_ratio_threshold) & valid

        has_both = high.any(dim=1) & low.any(dim=1)
        if has_both.sum() < 2:
            return torch.tensor(0.0, device=device)

        # mean-pool each engagement group per sequence
        high_f = high.float().unsqueeze(-1)                     # [B, M, 1]
        low_f = low.float().unsqueeze(-1)
        z_high = (seq_masked * high_f).sum(1) / high_f.sum(1).clamp(min=1)
        z_low = (seq_masked * low_f).sum(1) / low_f.sum(1).clamp(min=1)

        z_high = z_high[has_both]                               # [N, H]
        z_low = z_low[has_both]
        N = z_high.size(0)

        z_high = F.normalize(self.contrastive_proj(z_high), dim=-1)
        z_low = F.normalize(self.contrastive_proj(z_low), dim=-1)

        # 2N representations: first N = high, last N = low
        all_z = torch.cat([z_high, z_low], dim=0)               # [2N, H]
        sim = all_z @ all_z.T / self.contrastive_temp            # [2N, 2N]

        # mask self-similarity (use large negative, not -inf, to avoid nan)
        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(diag_mask, -1e9)

        # positive mask: same engagement class, excluding self
        eng_labels = torch.cat([
            torch.zeros(N, device=device),
            torch.ones(N, device=device),
        ]).long()
        pos_mask = (eng_labels.unsqueeze(0) == eng_labels.unsqueeze(1)) & ~diag_mask
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
        return loss.mean()
