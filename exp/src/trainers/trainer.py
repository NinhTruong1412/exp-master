"""Unified training loop for all sequential recommendation models.

Supports:
  - Mixed-precision training (FP16)
  - Learning rate warmup + cosine decay
  - Early stopping on validation metric
  - TensorBoard + JSON logging
  - Checkpoint save/load
  - Reproducible seeding
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast as torch_autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("experiment")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine decay with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Unified trainer for sequential recommendation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        config,
        experiment_logger=None,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.exp_logger = experiment_logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Training params
        train_cfg = config.training if hasattr(config, "training") else config
        self.epochs = getattr(train_cfg, "epochs", 100)
        self.lr = getattr(train_cfg, "lr", 3e-4)
        self.weight_decay = getattr(train_cfg, "weight_decay", 0.01)
        self.warmup_ratio = getattr(train_cfg, "warmup_ratio", 0.1)
        self.patience = getattr(train_cfg, "early_stopping_patience", 10)
        self.es_metric = getattr(train_cfg, "early_stopping_metric", "ndcg@10")
        self.fp16 = getattr(train_cfg, "fp16", True) and torch.cuda.is_available()
        self.grad_accum = getattr(train_cfg, "gradient_accumulation_steps", 1)
        self.log_every = 50
        if hasattr(config, "logging"):
            self.log_every = getattr(config.logging, "log_every_n_steps", 50)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        total_steps = len(train_loader) * self.epochs // self.grad_accum
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Mixed precision — only enable for CUDA
        self.fp16 = getattr(train_cfg, "fp16", True) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.fp16)

        # Tracking
        self.best_metric = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0

        # Checkpoint dir
        checkpoint_dir = "exp/outputs/checkpoints"
        if hasattr(config, "logging"):
            checkpoint_dir = getattr(config.logging, "checkpoint_dir", checkpoint_dir)
        model_name = config.model.name if hasattr(config, "model") else "model"
        seed = getattr(train_cfg, "seed", 42)
        self.output_dir = Path(checkpoint_dir) / f"{model_name}_seed{seed}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluator
        from ..evaluation.evaluator import Evaluator
        num_items = model.num_items if hasattr(model, "num_items") else 0
        self.evaluator = Evaluator(num_items=num_items, device=self.device)

    def train(self) -> dict:
        """Full training loop with validation and early stopping.

        Returns: dict with best metrics and training history.
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Device: {self.device}, FP16: {self.fp16}")
        logger.info(f"Early stopping: metric={self.es_metric}, patience={self.patience}")

        history = {"train_losses": [], "valid_metrics": []}

        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self._train_epoch(epoch)
            history["train_losses"].append(train_metrics)

            # Validate
            valid_metrics = self.evaluator.evaluate(self.model, self.valid_loader)
            history["valid_metrics"].append(valid_metrics)

            # Log
            current_metric = valid_metrics.get(self.es_metric, 0.0)
            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"{self.es_metric}: {current_metric:.4f} | "
                f"Best: {self.best_metric:.4f} (ep {self.best_epoch})"
            )

            if self.exp_logger:
                self.exp_logger.log_scalars(train_metrics, epoch, prefix="train")
                self.exp_logger.log_scalars(valid_metrics, epoch, prefix="valid")
                self.exp_logger.log_epoch(epoch, train_metrics, valid_metrics)

            # Early stopping
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, valid_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            self._save_checkpoint("latest_model.pt", epoch, valid_metrics)

        # Final evaluation on test set with best model
        self._load_checkpoint("best_model.pt")
        test_metrics = self.evaluator.evaluate(
            self.model, self.test_loader, return_per_user=True
        )

        per_user = test_metrics.pop("per_user", None)

        logger.info("=" * 60)
        logger.info("TEST RESULTS (best checkpoint):")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)

        if self.exp_logger:
            config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
            self.exp_logger.log_final(test_metrics, config_dict)

        return {
            "test_metrics": test_metrics,
            "per_user_metrics": per_user,
            "best_epoch": self.best_epoch,
            "best_valid_metric": self.best_metric,
            "history": history,
        }

    def _train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch_autocast(device_type="cuda", enabled=self.fp16):
                outputs = self.model(batch)
                loss = outputs["loss"] / self.grad_accum

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += outputs["loss"].item()
            num_batches += 1

            # Track individual loss components
            for key, val in outputs.items():
                if key.startswith("loss_") and isinstance(val, torch.Tensor):
                    loss_components.setdefault(key, 0.0)
                    loss_components[key] += val.item()

            # Progress bar
            pbar.set_postfix(
                loss=f"{outputs['loss'].item():.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

            # TensorBoard step logging
            if self.exp_logger and self.global_step % self.log_every == 0:
                self.exp_logger.log_scalar("train/loss_step", outputs["loss"].item(), self.global_step)
                self.exp_logger.log_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"loss": avg_loss, "epoch_time_sec": elapsed}
        for key in loss_components:
            metrics[key] = loss_components[key] / max(num_batches, 1)

        logger.info(
            f"Epoch {epoch} training: loss={avg_loss:.4f}, "
            f"time={elapsed:.1f}s, "
            f"throughput={len(self.train_loader.dataset) / elapsed:.0f} samples/s"
        )
        return metrics

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_metric": self.best_metric,
                "metrics": metrics,
            },
            path,
        )

    def _load_checkpoint(self, filename: str):
        path = self.output_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        else:
            logger.warning(f"Checkpoint not found: {path}")
