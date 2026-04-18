"""Variant C — Calibrated Distributional BERT4Rec.

Extends RecBole's BERT4Rec with three watch-time-aware auxiliary losses
that provide calibrated distributional predictions of engagement:

  1. **Quantile head** — predicts multiple quantiles (0.1, 0.25, 0.5,
     0.75, 0.9) of the watch-ratio distribution via a shared MLP.
  2. **Pinball loss** — proper quantile-regression loss ensuring each
     predicted quantile converges to the correct conditional quantile.
  3. **Calibration loss** — soft empirical-coverage penalty that
     encourages the predicted tau-th quantile to have roughly tau
     fraction of observations below it.
  4. **Contrastive InfoNCE** — same supervised contrastive loss as
     Variant A, separating high- vs low-engagement representations.

Total loss::

    L = CE
        + lambda_pinball      * pinball
        + lambda_calibration  * calibration
        + lambda_contrastive  * InfoNCE

Extra config keys (defaults)::

    lambda_pinball        0.4
    lambda_calibration    0.1
    lambda_contrastive    0.1
    contrastive_temp      0.1
    quantile_levels       [0.1, 0.25, 0.5, 0.75, 0.9]
    watch_ratio_threshold 0.3
    calibration_sigma     0.1     (soft indicator temperature)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.bert4rec import BERT4Rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(config, key, default):
    val = config[key]
    return val if val is not None else default


# ===================================================================
# Variant C
# ===================================================================

class VariantC(BERT4Rec):
    """Calibrated Distributional BERT4Rec (Variant C).

    Inherits the full BERT4Rec backbone.  Only ``calculate_loss`` is
    overridden; ``predict`` and ``full_sort_predict`` use the parent's
    implementation unchanged.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # --- hyper-parameters ------------------------------------------------
        self.lambda_pinball = _cfg(config, "lambda_pinball", 0.4)
        self.lambda_calibration = _cfg(config, "lambda_calibration", 0.1)
        self.lambda_contrastive = _cfg(config, "lambda_contrastive", 0.1)
        self.contrastive_temp = _cfg(config, "contrastive_temp", 0.1)
        self.watch_ratio_threshold = _cfg(config, "watch_ratio_threshold", 0.3)
        self.calibration_sigma = _cfg(config, "calibration_sigma", 0.1)

        raw_taus = _cfg(config, "quantile_levels", [0.1, 0.25, 0.5, 0.75, 0.9])
        if isinstance(raw_taus, str):
            import json
            raw_taus = json.loads(raw_taus)
        self.quantile_levels = list(raw_taus)
        n_q = len(self.quantile_levels)

        H = self.hidden_size

        # --- quantile prediction head: Linear -> GELU -> Linear -> Sigmoid ---
        self.quantile_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, n_q),
            nn.Sigmoid(),           # watch ratios in [0, 1]
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
        """Compute base CE + three distributional / calibration losses."""
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]       # [B, L]
        pos_items = interaction[self.POS_ITEMS]                 # [B, M]
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
        valid = targets.bool()

        # ---- extract watch ratios at masked positions -----------------------
        wr_masked = torch.bmm(
            pred_index_map,
            watch_ratio_list.unsqueeze(-1),
        ).squeeze(-1)                                           # [B, M]

        # ---- 1. base CE loss ------------------------------------------------
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

        # ---- 2. pinball (quantile) loss ------------------------------------
        pinball_loss = self._pinball_loss(seq_masked, wr_masked, valid)

        # ---- 3. calibration loss -------------------------------------------
        cal_loss = self._calibration_loss(seq_masked, wr_masked, valid)

        # ---- 4. contrastive InfoNCE ----------------------------------------
        cl_loss = self._contrastive_loss(seq_masked, wr_masked, targets)

        return (
            ce_loss
            + self.lambda_pinball * pinball_loss
            + self.lambda_calibration * cal_loss
            + self.lambda_contrastive * cl_loss
        )

    # ------------------------------------------------------------------ #
    #  Auxiliary losses                                                    #
    # ------------------------------------------------------------------ #

    def _pinball_loss(self, seq_masked, wr_masked, valid):
        """Pinball (quantile regression) loss across all quantile levels.

        L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
        averaged over quantiles and valid positions.
        """
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq_masked.device)

        preds = self.quantile_head(seq_masked)                  # [B, M, Q]
        preds_valid = preds[valid]                              # [N, Q]
        targets_valid = wr_masked[valid].unsqueeze(-1)          # [N, 1]

        taus = torch.tensor(
            self.quantile_levels,
            device=seq_masked.device,
            dtype=torch.float32,
        ).unsqueeze(0)                                          # [1, Q]

        errors = targets_valid - preds_valid                    # [N, Q]
        loss = torch.where(
            errors >= 0,
            taus * errors,
            (taus - 1.0) * errors,
        )
        return loss.mean()

    def _calibration_loss(self, seq_masked, wr_masked, valid):
        """Soft calibration loss: predicted quantiles should match empirical coverage.

        For each quantile level tau, the empirical coverage is the fraction
        of actual watch ratios that fall below the predicted tau-th quantile.
        A soft sigmoid indicator is used to keep the loss differentiable.
        The loss is MSE between empirical coverage and the nominal tau.
        """
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq_masked.device)

        preds = self.quantile_head(seq_masked)                  # [B, M, Q]
        preds_valid = preds[valid]                              # [N, Q]
        targets_valid = wr_masked[valid]                        # [N]

        taus = torch.tensor(
            self.quantile_levels,
            device=seq_masked.device,
            dtype=torch.float32,
        )                                                       # [Q]

        # soft indicator: sigma(pred_q - actual) / sigma  ~  1{actual < pred_q}
        diff = preds_valid - targets_valid.unsqueeze(-1)        # [N, Q]
        soft_indicator = torch.sigmoid(diff / self.calibration_sigma)

        # empirical coverage per quantile
        empirical_cov = soft_indicator.mean(dim=0)              # [Q]

        # MSE between empirical coverage and nominal quantile levels
        return F.mse_loss(empirical_cov, taus)

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
