"""Variant B — Counterfactual Duration-Debiased BERT4Rec.

Extends RecBole's BERT4Rec with four watch-time-aware auxiliary losses
designed to remove duration bias while preserving engagement signals:

  1. **Ordinal engagement head** — classifies watch_ratio into N ordinal
     levels using cumulative logits with BCE (P(Y >= k) for k=1..K-1).
  2. **Duration-to-Query cross-attention (D2Q)** — a duration embedding
     table (7 buckets) followed by cross-attention that enriches sequence
     representations with duration context; trained via Huber loss on
     watch-ratio prediction from the enriched representation.
  3. **Gradient Reversal Layer + duration discriminator** — adversarial
     debiasing that discourages the main encoder from encoding pure
     duration shortcuts.
  4. **Contrastive InfoNCE** — same supervised contrastive loss as
     Variant A, separating high- vs low-engagement representations.

Total loss::

    L = CE
        + lambda_ordinal      * ordinal_BCE
        + lambda_d2q          * Huber(enriched_pred, watch_ratio)
        + lambda_adversarial  * CE(discriminator, duration_bucket)
        + lambda_contrastive  * InfoNCE

Extra config keys (defaults)::

    lambda_ordinal       0.3
    lambda_d2q           0.2
    lambda_adversarial   0.02
    lambda_contrastive   0.1
    contrastive_temp     0.1
    n_ordinal_levels     5
    grl_lambda           0.1
    watch_ratio_threshold 0.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from recbole.model.sequential_recommender.bert4rec import BERT4Rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(config, key, default):
    val = config[key]
    return val if val is not None else default


class _GradientReversal(Function):
    """Reverses gradients during backward pass (identity in forward)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def _grad_reverse(x, lam=1.0):
    return _GradientReversal.apply(x, lam)


def _watch_ratio_to_ordinal(watch_ratio, n_levels=5):
    """Map continuous watch ratio to ordinal class 0 .. n_levels-1.

    Default boundaries (5 levels):
        0: skip     (< 0.05)
        1: glance   (0.05 – 0.2)
        2: partial  (0.2  – 0.5)
        3: deep     (0.5  – 0.9)
        4: complete (>= 0.9)
    """
    cls = torch.zeros_like(watch_ratio, dtype=torch.long)
    cls[watch_ratio >= 0.05] = 1
    cls[watch_ratio >= 0.20] = 2
    cls[watch_ratio >= 0.50] = 3
    cls[watch_ratio >= 0.90] = 4
    return cls.clamp(max=n_levels - 1)


# ===================================================================
# Variant B
# ===================================================================

class VariantB(BERT4Rec):
    """Counterfactual Duration-Debiased BERT4Rec (Variant B).

    Inherits the full BERT4Rec backbone.  Only ``calculate_loss`` is
    overridden; ``predict`` and ``full_sort_predict`` use the parent's
    implementation unchanged.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # --- hyper-parameters ------------------------------------------------
        self.lambda_ordinal = _cfg(config, "lambda_ordinal", 0.3)
        self.lambda_d2q = _cfg(config, "lambda_d2q", 0.2)
        self.lambda_adversarial = _cfg(config, "lambda_adversarial", 0.02)
        self.lambda_contrastive = _cfg(config, "lambda_contrastive", 0.1)
        self.contrastive_temp = _cfg(config, "contrastive_temp", 0.1)
        self.n_ordinal_levels = int(_cfg(config, "n_ordinal_levels", 5))
        self.grl_lambda = _cfg(config, "grl_lambda", 0.1)
        self.watch_ratio_threshold = _cfg(config, "watch_ratio_threshold", 0.3)

        H = self.hidden_size
        K = self.n_ordinal_levels

        # --- ordinal engagement head (cumulative logits) ---------------------
        self.ordinal_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, K - 1),      # K-1 cumulative logits
        )

        # --- duration embedding + cross-attention (D2Q) ----------------------
        # 7 duration buckets + 1 padding (idx 0)
        self.duration_embedding = nn.Embedding(8, H, padding_idx=0)
        self.d2q_attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=max(1, self.n_heads),
            batch_first=True,
            dropout=self.hidden_dropout_prob,
        )
        self.d2q_norm = nn.LayerNorm(H, eps=self.layer_norm_eps)

        # D2Q watch-ratio prediction (from enriched representation)
        self.d2q_ratio_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, 1),
            nn.Sigmoid(),
        )

        # --- adversarial duration discriminator (behind GRL) -----------------
        self.duration_disc = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 8),      # predict duration bucket (0-7)
        )

        # --- contrastive projection head ------------------------------------
        self.contrastive_proj = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        # re-initialise all weights for newly added modules
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def calculate_loss(self, interaction):
        """Compute base CE + four debiasing / engagement auxiliary losses."""
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]           # [B, L]
        pos_items = interaction[self.POS_ITEMS]                     # [B, M]
        masked_index = interaction[self.MASK_INDEX]                 # [B, M]
        watch_ratio_list = interaction["watch_ratio_list"]          # [B, L]
        duration_bucket_list = interaction["duration_bucket_list"]  # [B, L]

        # ---- encoder forward ------------------------------------------------
        seq_output = self.forward(masked_item_seq)                  # [B, L, H]

        # ---- gather masked-position hidden states ---------------------------
        B, M = masked_index.shape
        L = masked_item_seq.size(-1)
        pred_index_map = self.multi_hot_embed(masked_index, L)      # [B*M, L]
        pred_index_map = pred_index_map.view(B, M, L)               # [B, M, L]
        seq_masked = torch.bmm(pred_index_map, seq_output)          # [B, M, H]

        targets = (masked_index > 0).float()                        # [B, M]
        n_valid = targets.sum().clamp(min=1.0)
        valid = targets.bool()

        # ---- extract auxiliary features at masked positions -----------------
        wr_masked = torch.bmm(
            pred_index_map,
            watch_ratio_list.unsqueeze(-1),
        ).squeeze(-1)                                               # [B, M]

        dur_masked = torch.bmm(
            pred_index_map,
            duration_bucket_list.unsqueeze(-1).float(),
        ).squeeze(-1).long()                                        # [B, M]

        # ---- 1. base CE loss ------------------------------------------------
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        test_item_emb = self.item_embedding.weight[: self.n_items]  # [V, H]
        logits = (
            torch.matmul(seq_masked, test_item_emb.transpose(0, 1))
            + self.output_bias
        )                                                           # [B, M, V]
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

        # ---- 2. ordinal engagement loss -------------------------------------
        ordinal_loss = self._ordinal_loss(seq_masked, wr_masked, valid)

        # ---- 3. D2Q cross-attention + watch-ratio prediction ----------------
        d2q_loss = self._d2q_loss(
            seq_output, duration_bucket_list,
            pred_index_map, wr_masked, valid,
        )

        # ---- 4. adversarial duration discriminator --------------------------
        adv_loss = self._adversarial_loss(seq_masked, dur_masked, valid)

        # ---- 5. contrastive InfoNCE ----------------------------------------
        cl_loss = self._contrastive_loss(seq_masked, wr_masked, targets)

        return (
            ce_loss
            + self.lambda_ordinal * ordinal_loss
            + self.lambda_d2q * d2q_loss
            + self.lambda_adversarial * adv_loss
            + self.lambda_contrastive * cl_loss
        )

    # ------------------------------------------------------------------ #
    #  Auxiliary losses                                                    #
    # ------------------------------------------------------------------ #

    def _ordinal_loss(self, seq_masked, wr_masked, valid):
        """Ordinal engagement classification via cumulative BCE.

        Predicts P(Y >= k) for k = 1 .. K-1 using sigmoid on
        cumulative logits, trained with binary cross-entropy.
        """
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq_masked.device)

        K = self.n_ordinal_levels
        cum_logits = self.ordinal_head(seq_masked)              # [B, M, K-1]

        # ordinal class targets
        ord_cls = _watch_ratio_to_ordinal(wr_masked, K)         # [B, M]

        # binary targets: y_k = 1 if class >= k  for k in 1..K-1
        thresholds = torch.arange(
            1, K, device=seq_masked.device,
        ).float().unsqueeze(0)                                  # [1, K-1]
        bin_targets = (
            ord_cls[valid].unsqueeze(-1) >= thresholds
        ).float()                                               # [N, K-1]

        cum_probs = torch.sigmoid(cum_logits[valid])            # [N, K-1]
        return F.binary_cross_entropy(cum_probs, bin_targets)

    def _d2q_loss(self, seq_output, duration_bucket_list,
                  pred_index_map, wr_masked, valid):
        """Duration-to-Query cross-attention enrichment + Huber loss.

        Embeds duration buckets, performs cross-attention (query = sequence
        hidden states, key/value = duration embeddings), adds a residual
        connection, and predicts watch ratio from the enriched hidden states
        at masked positions.
        """
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq_output.device)

        dur_emb = self.duration_embedding(duration_bucket_list) # [B, L, H]

        # cross-attention: sequence queries attend to duration keys/values
        attn_out, _ = self.d2q_attn(
            query=seq_output,
            key=dur_emb,
            value=dur_emb,
        )                                                       # [B, L, H]
        enriched = self.d2q_norm(seq_output + attn_out)         # residual

        # gather enriched hidden states at masked positions
        enriched_masked = torch.bmm(pred_index_map, enriched)   # [B, M, H]

        pred = self.d2q_ratio_head(enriched_masked).squeeze(-1) # [B, M]
        return F.huber_loss(pred[valid], wr_masked[valid], delta=0.5)

    def _adversarial_loss(self, seq_masked, dur_masked, valid):
        """Gradient-reversed duration discrimination.

        The discriminator tries to predict the duration bucket from the
        encoder representation.  GRL reverses the gradient so that the
        encoder learns to *remove* duration information.
        """
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq_masked.device)

        reversed_h = _grad_reverse(seq_masked, self.grl_lambda)
        disc_logits = self.duration_disc(reversed_h)            # [B, M, 8]
        return F.cross_entropy(
            disc_logits[valid],
            dur_masked[valid],
        )

    def _contrastive_loss(self, seq_masked, wr_masked, targets):
        """Supervised contrastive (InfoNCE) separating engagement levels.

        Identical mechanism to Variant A: per-sequence mean-pooled
        high / low engagement representations with same-class positives.
        """
        device = seq_masked.device
        B, M, H = seq_masked.shape
        valid = targets.bool()

        high = (wr_masked > self.watch_ratio_threshold) & valid
        low = (wr_masked <= self.watch_ratio_threshold) & valid

        has_both = high.any(dim=1) & low.any(dim=1)
        if has_both.sum() < 2:
            return torch.tensor(0.0, device=device)

        high_f = high.float().unsqueeze(-1)
        low_f = low.float().unsqueeze(-1)
        z_high = (seq_masked * high_f).sum(1) / high_f.sum(1).clamp(min=1)
        z_low = (seq_masked * low_f).sum(1) / low_f.sum(1).clamp(min=1)

        z_high = z_high[has_both]
        z_low = z_low[has_both]
        N = z_high.size(0)

        z_high = F.normalize(self.contrastive_proj(z_high), dim=-1)
        z_low = F.normalize(self.contrastive_proj(z_low), dim=-1)

        all_z = torch.cat([z_high, z_low], dim=0)
        sim = all_z @ all_z.T / self.contrastive_temp

        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(diag_mask, -1e9)

        eng_labels = torch.cat([
            torch.zeros(N, device=device),
            torch.ones(N, device=device),
        ]).long()
        pos_mask = (eng_labels.unsqueeze(0) == eng_labels.unsqueeze(1)) & ~diag_mask
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
        return loss.mean()
