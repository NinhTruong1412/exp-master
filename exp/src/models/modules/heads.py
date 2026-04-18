"""Auxiliary prediction heads for multi-task training.

Includes:
  - WatchRatioHead: regression head for watch-ratio prediction (Huber loss)
  - PairwiseRankingHead: BPR loss weighted by engagement difference
  - ContrastiveHead: InfoNCE on sequence-level representations
  - OrdinalHead: ordinal classification for watch depth
  - AdversarialDurationHead: gradient reversal for duration debiasing
  - QuantileHead: multi-quantile prediction with pinball loss
  - PrototypeCalibrationHead: prototype-based calibrated prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# --- Gradient Reversal Layer ---

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# --- Watch-Ratio Regression Head ---

class WatchRatioHead(nn.Module):
    """Predict watch ratio at masked positions using Huber loss."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict watch ratio in [0, 1]."""
        return self.net(hidden).squeeze(-1)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Huber loss at valid masked positions.

        Args:
            hidden: (B, L, H)
            targets: (B, L) watch ratio values
            mask: (B, L) boolean mask for positions to predict
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        pred = self.forward(hidden)  # (B, L)
        pred_valid = pred[mask]
        target_valid = targets[mask]
        return F.huber_loss(pred_valid, target_valid, delta=0.5)


# --- Pairwise Ranking Head ---

class PairwiseRankingHead(nn.Module):
    """BPR-style pairwise ranking loss weighted by engagement difference."""

    def __init__(self):
        super().__init__()

    def compute_loss(
        self,
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute BPR loss between positive and negative item scores.

        Args:
            scores: (B, L, V) item logits
            positive_mask: (B, L) indices of positive items
            negative_mask: (B, L) indices of negative (randomly sampled) items
            weights: (B, L) optional engagement-difference weights
        """
        B, L, V = scores.shape

        pos_scores = scores.gather(2, positive_mask.unsqueeze(2)).squeeze(2)
        neg_scores = scores.gather(2, negative_mask.unsqueeze(2)).squeeze(2)

        valid = (positive_mask > 0) & (negative_mask > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=scores.device)

        diff = pos_scores - neg_scores
        if weights is not None:
            diff = diff * weights

        loss = -F.logsigmoid(diff[valid]).mean()
        return loss


# --- Contrastive Head ---

class ContrastiveHead(nn.Module):
    """InfoNCE contrastive loss on sequence representations."""

    def __init__(self, hidden_size: int, temperature: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss between two views.

        Args:
            z1, z2: (B, H) sequence-level representations from two augmented views
        """
        z1 = F.normalize(self.projector(z1), dim=-1)
        z2 = F.normalize(self.projector(z2), dim=-1)

        B = z1.size(0)
        sim = torch.matmul(z1, z2.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=z1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


# --- Ordinal Classification Head ---

class OrdinalHead(nn.Module):
    """Ordinal classification for watch depth.

    Classes: skip(0), glance(1), partial(2), deep(3), complete(4)
    Uses cumulative link model for ordinal regression.
    """

    def __init__(self, hidden_size: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, ..., num_classes)."""
        return self.net(hidden)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ordinal cross-entropy loss.

        Args:
            hidden: (B, L, H)
            targets: (B, L) ordinal class indices (0 to num_classes-1)
            mask: (B, L) boolean for valid positions
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        logits = self.forward(hidden)  # (B, L, C)
        logits_valid = logits[mask]
        targets_valid = targets[mask]
        return F.cross_entropy(logits_valid, targets_valid)


# --- Adversarial Duration Discriminator ---

class AdversarialDurationHead(nn.Module):
    """Duration bucket discriminator with gradient reversal.

    Discourages encoder from encoding pure duration shortcuts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_duration_buckets: int = 16,
        adversarial_hidden: int = 64,
        lambda_adv: float = 0.02,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_adv)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, adversarial_hidden),
            nn.GELU(),
            nn.Linear(adversarial_hidden, num_duration_buckets),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict duration bucket from encoder output (with gradient reversal)."""
        return self.net(self.grl(hidden))

    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        logits = self.forward(hidden)
        logits_valid = logits[mask]
        targets_valid = targets[mask]
        return F.cross_entropy(logits_valid, targets_valid)


# --- Multi-Quantile Head ---

class QuantileHead(nn.Module):
    """Multi-quantile prediction head with pinball loss."""

    def __init__(
        self,
        hidden_size: int,
        taus: list[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.taus = taus
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(taus)),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns quantile predictions (B, ..., num_taus)."""
        return self.net(hidden)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pinball loss across all quantiles.

        Args:
            hidden: (B, L, H)
            targets: (B, L) actual watch ratio values
            mask: (B, L) boolean for valid positions
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        preds = self.forward(hidden)  # (B, L, num_taus)
        preds_valid = preds[mask]  # (N, num_taus)
        targets_valid = targets[mask].unsqueeze(-1)  # (N, 1)

        errors = targets_valid - preds_valid  # (N, num_taus)
        taus = torch.tensor(self.taus, device=hidden.device).unsqueeze(0)  # (1, num_taus)
        loss = torch.where(errors >= 0, taus * errors, (taus - 1) * errors)
        return loss.mean()


# --- Prototype Calibration Head ---

class PrototypeCalibrationHead(nn.Module):
    """Prototype-based calibration for watch-time prediction.

    Learns K prototypes per duration bucket with soft assignment
    and a compactness loss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_prototypes: int = 8,
        num_duration_buckets: int = 16,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        # Shared prototypes (could be per-bucket, but shared is simpler)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_size) * 0.02)
        self.proto_predictor = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prototype assignments and predictions.

        Args:
            hidden: (N, H) hidden states at valid positions

        Returns:
            predictions: (N,) calibrated watch-ratio predictions
            assignments: (N, K) soft assignment weights
        """
        # Soft assignment: cosine similarity / temperature
        hidden_norm = F.normalize(hidden, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(hidden_norm, proto_norm.T) / self.temperature  # (N, K)
        assignments = F.softmax(sim, dim=-1)  # (N, K)

        # Weighted prototype combination
        weighted = torch.matmul(assignments, self.prototypes)  # (N, H)
        predictions = torch.sigmoid(self.proto_predictor(weighted)).squeeze(-1)  # (N,)

        return predictions, assignments

    def compute_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute calibration loss = MSE + compactness.

        Returns: (calibration_loss, compactness_loss)
        """
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero

        hidden_valid = hidden[mask]
        targets_valid = targets[mask]

        preds, assignments = self.forward(hidden_valid)

        # Prediction loss (MSE for calibration)
        pred_loss = F.mse_loss(preds, targets_valid)

        # Compactness: encourage prototype diversity
        proto_norm = F.normalize(self.prototypes, dim=-1)
        proto_sim = torch.matmul(proto_norm, proto_norm.T)
        # Penalize high similarity between different prototypes
        eye = torch.eye(self.num_prototypes, device=hidden.device)
        compact_loss = ((proto_sim - eye) ** 2).mean()

        return pred_loss, compact_loss
