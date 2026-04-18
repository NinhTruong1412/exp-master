"""Dataset and DataLoader for sequential recommendation experiments.

Supports BERT4Rec-style masked-item prediction and SASRec-style next-item prediction.
Loads preprocessed parquet data with aligned auxiliary features (watch_ratio, duration buckets).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("experiment")

PAD_TOKEN = 0
MASK_TOKEN_OFFSET = 1  # mask_token_id = num_items + 1


class SequentialRecDataset(Dataset):
    """Dataset for sequential recommendation with auxiliary watch-time features.

    Each sample contains:
      - item_ids: padded/truncated item ID sequence
      - watch_ratios: aligned clipped watch ratios (0-1)
      - duration_bucket_ids: integer-encoded duration buckets
      - watch_bucket_ids: integer-encoded watch-ratio buckets
      - positions: absolute position indices
      - attention_mask: 1 for real tokens, 0 for padding
      - labels: target items for masked positions (BERT4Rec) or next item (SASRec)
    """

    WATCH_BUCKET_MAP = {
        "0": 0, "(0,0.1]": 1, "(0.1,0.3]": 2,
        "(0.3,0.7]": 3, "(0.7,1.0]": 4, ">1": 5,
    }
    DURATION_BUCKET_MAP = {
        "unknown": 0, "<5m": 1, "5-15m": 2, "15-30m": 3,
        "30-60m": 4, "60-120m": 5, "120m+": 6,
    }

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 50,
        mode: str = "train",
        mask_ratio: float = 0.15,
        num_items: int = 0,
        engagement_weighted_masking: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.engagement_weighted_masking = engagement_weighted_masking

        df = pd.read_parquet(data_path)
        self.num_samples = len(df)

        if mode == "train":
            self.item_seqs = df["item_sequence"].tolist()
            self.watch_ratio_seqs = df["watch_ratio_sequence"].tolist()
            self.watch_bucket_seqs = df["watch_bucket_sequence"].tolist()
            self.duration_bucket_seqs = df["duration_bucket_sequence"].tolist()
        else:
            self.item_seqs = df["input_sequence"].tolist()
            self.watch_ratio_seqs = df["watch_ratio_sequence"].tolist()
            self.watch_bucket_seqs = df["watch_bucket_sequence"].tolist()
            self.duration_bucket_seqs = df["duration_bucket_sequence"].tolist()
            self.target_items = df["target_item"].tolist()

        # Determine num_items from data if not provided
        if num_items > 0:
            self.num_items = num_items
        else:
            all_items = set()
            for seq in self.item_seqs:
                all_items.update(seq)
            if mode != "train":
                all_items.update(self.target_items)
            self.num_items = max(all_items) if all_items else 0

        self.mask_token_id = self.num_items + MASK_TOKEN_OFFSET

    def __len__(self):
        return self.num_samples

    def _encode_bucket_seq(self, bucket_seq: list, bucket_map: dict) -> list[int]:
        return [bucket_map.get(str(b), 0) for b in bucket_seq]

    def _truncate_and_pad(self, seq: list, pad_value, max_len: int):
        """Right-truncate (keep most recent) and left-pad."""
        seq = seq[-max_len:]
        pad_len = max_len - len(seq)
        return [pad_value] * pad_len + seq, pad_len

    def __getitem__(self, idx):
        item_seq = list(self.item_seqs[idx])
        watch_ratios = [float(r) for r in self.watch_ratio_seqs[idx]]
        watch_buckets = self._encode_bucket_seq(self.watch_bucket_seqs[idx], self.WATCH_BUCKET_MAP)
        duration_buckets = self._encode_bucket_seq(self.duration_bucket_seqs[idx], self.DURATION_BUCKET_MAP)

        seq_len = len(item_seq)

        if self.mode == "train":
            return self._prepare_train(item_seq, watch_ratios, watch_buckets, duration_buckets, seq_len)
        else:
            return self._prepare_eval(item_seq, watch_ratios, watch_buckets, duration_buckets, seq_len, idx)

    def _prepare_train(self, item_seq, watch_ratios, watch_buckets, duration_buckets, seq_len):
        """BERT4Rec-style: mask random items, predict them."""
        masked_seq = list(item_seq)
        labels = [PAD_TOKEN] * seq_len  # 0 means no prediction needed

        for i in range(seq_len):
            prob = self.mask_ratio
            if self.engagement_weighted_masking and watch_ratios[i] > 0:
                # Bias masking toward high-engagement positions
                prob = self.mask_ratio * (1.0 + watch_ratios[i])
                prob = min(prob, 0.5)

            if np.random.random() < prob:
                labels[i] = item_seq[i]
                rand = np.random.random()
                if rand < 0.8:
                    masked_seq[i] = self.mask_token_id
                elif rand < 0.9:
                    masked_seq[i] = np.random.randint(1, self.num_items + 1)
                # else: keep original (10%)

        # Truncate and pad
        masked_seq, pad_len = self._truncate_and_pad(masked_seq, PAD_TOKEN, self.max_seq_len)
        labels, _ = self._truncate_and_pad(labels, PAD_TOKEN, self.max_seq_len)
        watch_ratios, _ = self._truncate_and_pad(watch_ratios, 0.0, self.max_seq_len)
        watch_buckets, _ = self._truncate_and_pad(watch_buckets, 0, self.max_seq_len)
        duration_buckets, _ = self._truncate_and_pad(duration_buckets, 0, self.max_seq_len)

        attention_mask = [0] * pad_len + [1] * min(seq_len, self.max_seq_len)
        positions = list(range(self.max_seq_len))

        return {
            "item_ids": torch.tensor(masked_seq, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "watch_ratios": torch.tensor(watch_ratios, dtype=torch.float),
            "watch_bucket_ids": torch.tensor(watch_buckets, dtype=torch.long),
            "duration_bucket_ids": torch.tensor(duration_buckets, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "original_items": torch.tensor(
                self._truncate_and_pad(list(item_seq), PAD_TOKEN, self.max_seq_len)[0],
                dtype=torch.long,
            ),
        }

    def _prepare_eval(self, item_seq, watch_ratios, watch_buckets, duration_buckets, seq_len, idx):
        """Eval: provide full input sequence and target item."""
        item_seq, pad_len = self._truncate_and_pad(item_seq, PAD_TOKEN, self.max_seq_len)
        watch_ratios, _ = self._truncate_and_pad(watch_ratios, 0.0, self.max_seq_len)
        watch_buckets, _ = self._truncate_and_pad(watch_buckets, 0, self.max_seq_len)
        duration_buckets, _ = self._truncate_and_pad(duration_buckets, 0, self.max_seq_len)

        attention_mask = [0] * pad_len + [1] * min(seq_len, self.max_seq_len)
        positions = list(range(self.max_seq_len))

        return {
            "item_ids": torch.tensor(item_seq, dtype=torch.long),
            "target_item": torch.tensor(self.target_items[idx], dtype=torch.long),
            "watch_ratios": torch.tensor(watch_ratios, dtype=torch.float),
            "watch_bucket_ids": torch.tensor(watch_buckets, dtype=torch.long),
            "duration_bucket_ids": torch.tensor(duration_buckets, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def load_item_vocab(data_dir: str) -> dict:
    """Load item vocabulary and return mapping info."""
    vocab = pd.read_parquet(Path(data_dir) / "item_vocab.parquet")
    num_items = int(vocab["item_id"].max())
    content_to_item = dict(zip(vocab["content_id"], vocab["item_id"]))
    return {"num_items": num_items, "content_to_item": content_to_item, "vocab_df": vocab}


def get_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    max_seq_len: int = 50,
    mask_ratio: float = 0.15,
    num_workers: int = 4,
    num_items: int = 0,
    engagement_weighted_masking: bool = False,
) -> dict:
    """Create train/valid/test DataLoaders."""
    data_dir = Path(data_dir)

    if num_items == 0:
        vocab_info = load_item_vocab(str(data_dir))
        num_items = vocab_info["num_items"]

    logger.info(f"Loading data from {data_dir}, num_items={num_items}, max_seq_len={max_seq_len}")

    train_ds = SequentialRecDataset(
        str(data_dir / "subset_train.parquet"),
        max_seq_len=max_seq_len,
        mode="train",
        mask_ratio=mask_ratio,
        num_items=num_items,
        engagement_weighted_masking=engagement_weighted_masking,
    )
    valid_ds = SequentialRecDataset(
        str(data_dir / "subset_valid.parquet"),
        max_seq_len=max_seq_len,
        mode="eval",
        num_items=num_items,
    )
    test_ds = SequentialRecDataset(
        str(data_dir / "subset_test.parquet"),
        max_seq_len=max_seq_len,
        mode="eval",
        num_items=num_items,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    logger.info(f"Valid: {len(valid_ds)} samples, {len(valid_loader)} batches")
    logger.info(f"Test: {len(test_ds)} samples, {len(test_loader)} batches")

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
        "num_items": num_items,
        "mask_token_id": num_items + MASK_TOKEN_OFFSET,
    }
