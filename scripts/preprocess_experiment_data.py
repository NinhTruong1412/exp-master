#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PERFORMANCE_COLUMNS = [
    "views_30d",
    "views_14d",
    "views_7d",
    "views_3d",
    "views_1d",
    "weekend_views",
    "unique_viewers_30d",
    "returning_viewers_30d",
    "completed_viewers",
    "completion_rate",
    "avg_watchtime",
    "avg_watch_ratio",
    "end_date_simulcast",
]

METADATA_COLUMNS = [
    "content_id",
    "content_title",
    "runtime_mins",
    "content_type",
    "release_year",
    "country",
    "provider",
    "content_genre",
    "is_simulcast",
    "geo_check",
    "short_description",
    "long_description",
]

EVENT_COLUMNS = [
    "user_id",
    "content_id",
    "event_ts",
    "watch_time_sec",
    "runtime_sec_raw",
    "runtime_sec_meta",
    "runtime_sec_final",
    "watch_ratio",
    "watch_ratio_clipped",
    "watch_bucket",
    "is_zero_watch",
    "is_short_watch",
    "is_over_runtime",
    "is_extreme_over_runtime",
    "content_duration_bucket",
    "event_date",
    "content_title",
    "content_type",
    "release_year",
    "country",
    "provider",
    "content_genre",
    "is_simulcast",
    "geo_check",
    "short_description",
    "long_description",
]


@dataclass
class ShardStats:
    shard_name: str
    raw_user_rows: int = 0
    exploded_events: int = 0
    missing_user_id: int = 0
    missing_content_id: int = 0
    missing_event_ts: int = 0
    missing_metadata_join: int = 0
    invalid_runtime: int = 0
    exact_duplicates_removed: int = 0
    extreme_over_runtime_removed: int = 0
    kept_after_cleaning: int = 0
    users_after_cleaning: int = 0
    zero_watch_kept: int = 0
    short_watch_kept: int = 0
    over_runtime_kept: int = 0


def clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def watch_bucket(value: float) -> str:
    if pd.isna(value) or value <= 0:
        return "0"
    if value <= 0.10:
        return "(0,0.1]"
    if value <= 0.30:
        return "(0.1,0.3]"
    if value <= 0.70:
        return "(0.3,0.7]"
    if value <= 1.00:
        return "(0.7,1.0]"
    return ">1"


def duration_bucket(seconds: float) -> str:
    if pd.isna(seconds) or seconds <= 0:
        return "unknown"
    minutes = seconds / 60.0
    if minutes < 5:
        return "<5m"
    if minutes < 15:
        return "5-15m"
    if minutes < 30:
        return "15-30m"
    if minutes < 60:
        return "30-60m"
    if minutes < 120:
        return "60-120m"
    return "120m+"


def load_metadata(content_profile_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(content_profile_path)
    raw_columns = list(raw.columns)
    metadata = raw[METADATA_COLUMNS].copy()
    metadata["content_id"] = metadata["content_id"].astype("string")
    metadata["content_title"] = metadata["content_title"].apply(clean_text)
    metadata["country"] = metadata["country"].apply(clean_text).fillna("unknown")
    metadata["provider"] = metadata["provider"].apply(clean_text).fillna("unknown")
    metadata["content_genre"] = metadata["content_genre"].apply(clean_text).fillna("unknown")
    metadata["content_type"] = metadata["content_type"].apply(clean_text).fillna("unknown")
    metadata["short_description"] = metadata["short_description"].apply(clean_text)
    metadata["long_description"] = metadata["long_description"].apply(clean_text)
    metadata["release_year"] = pd.to_numeric(metadata["release_year"], errors="coerce").astype("Int64")
    metadata["runtime_mins"] = pd.to_numeric(metadata["runtime_mins"], errors="coerce")
    metadata["runtime_sec_meta"] = metadata["runtime_mins"] * 60.0
    metadata["is_simulcast"] = pd.to_numeric(metadata["is_simulcast"], errors="coerce").fillna(0).astype("int8")
    metadata["geo_check"] = pd.to_numeric(metadata["geo_check"], errors="coerce").fillna(0).astype("int8")

    metadata_stats = {
        "raw_rows": int(len(raw)),
        "raw_unique_content_id": int(raw["content_id"].nunique()),
        "columns_before": raw_columns,
        "columns_after": list(metadata.columns),
        "dropped_performance_columns": PERFORMANCE_COLUMNS,
        "null_rate_after_cleaning": {
            col: float(metadata[col].isna().mean())
            for col in [
                "content_title",
                "runtime_mins",
                "content_type",
                "release_year",
                "country",
                "provider",
                "content_genre",
                "short_description",
                "long_description",
            ]
        },
        "content_type_distribution": {
            str(key): int(value)
            for key, value in metadata["content_type"].value_counts(dropna=False).to_dict().items()
        },
    }
    return metadata, metadata_stats


def flatten_contents(contents: object) -> list[dict[str, object]]:
    if contents is None or (isinstance(contents, float) and math.isnan(contents)):
        return []
    if isinstance(contents, np.ndarray):
        return [item for item in contents.tolist() if item is not None]
    if isinstance(contents, list):
        return [item for item in contents if item is not None]
    return []


def explode_raw_shard(shard_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(shard_path, columns=["user_id", "contents"])
    rows: list[dict[str, object]] = []
    for user_id, contents in zip(raw["user_id"], raw["contents"], strict=False):
        for item in flatten_contents(contents):
            rows.append(
                {
                    "user_id": user_id,
                    "content_id": item.get("content_id"),
                    "event_ts": item.get("min_wt_timestamp"),
                    "watch_time_sec": item.get("watch_time"),
                    "runtime_sec_raw": item.get("runtime"),
                }
            )
    df = pd.DataFrame.from_records(rows)
    return raw, df


def clean_and_enrich_events(
    shard_path: Path,
    metadata: pd.DataFrame,
    tmp_dir: Path,
) -> tuple[ShardStats, Counter]:
    raw, events = explode_raw_shard(shard_path)
    stats = ShardStats(shard_name=shard_path.name, raw_user_rows=int(len(raw)), exploded_events=int(len(events)))

    if events.empty:
        out_path = tmp_dir / f"{shard_path.stem}-clean.parquet"
        pd.DataFrame(columns=EVENT_COLUMNS + ["user_event_count"]).to_parquet(out_path, index=False)
        return stats, Counter()

    events["user_id"] = events["user_id"].astype("string")
    events["content_id"] = events["content_id"].astype("string")
    events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
    events["watch_time_sec"] = pd.to_numeric(events["watch_time_sec"], errors="coerce")
    events["runtime_sec_raw"] = pd.to_numeric(events["runtime_sec_raw"], errors="coerce")

    stats.missing_user_id = int(events["user_id"].isna().sum())
    stats.missing_content_id = int(events["content_id"].isna().sum())
    stats.missing_event_ts = int(events["event_ts"].isna().sum())

    events = events.merge(metadata, on="content_id", how="left", validate="many_to_one", indicator=True)
    stats.missing_metadata_join = int((events["_merge"] == "left_only").sum())

    events["runtime_sec_final"] = np.where(
        events["runtime_sec_raw"].fillna(0) > 0,
        events["runtime_sec_raw"],
        events["runtime_sec_meta"],
    )
    events["is_zero_watch"] = events["watch_time_sec"].fillna(0) <= 0
    events["is_short_watch"] = events["watch_time_sec"].fillna(0).between(1, 30, inclusive="both")
    events["is_over_runtime"] = (
        events["watch_time_sec"].fillna(-1) > events["runtime_sec_final"].fillna(np.inf)
    )
    events["is_extreme_over_runtime"] = (
        (events["watch_time_sec"].fillna(0) > events["runtime_sec_final"].fillna(np.inf) * 5)
        & ((events["watch_time_sec"].fillna(0) - events["runtime_sec_final"].fillna(0)) > 3600)
    )

    valid_mask = (
        events["user_id"].notna()
        & events["content_id"].notna()
        & events["event_ts"].notna()
        & events["runtime_sec_final"].notna()
        & (events["runtime_sec_final"] > 0)
        & (events["_merge"] == "both")
    )
    stats.invalid_runtime = int((events["runtime_sec_final"].isna() | (events["runtime_sec_final"] <= 0)).sum())
    stats.extreme_over_runtime_removed = int((valid_mask & events["is_extreme_over_runtime"]).sum())
    events = events.loc[valid_mask & ~events["is_extreme_over_runtime"]].copy()
    events = events.drop(columns=["_merge"])

    before_dedup = len(events)
    events = events.drop_duplicates(subset=["user_id", "content_id", "event_ts", "watch_time_sec"], keep="first")
    stats.exact_duplicates_removed = int(before_dedup - len(events))

    events = events.sort_values(["user_id", "event_ts", "content_id"], kind="stable").reset_index(drop=True)
    events["watch_ratio"] = np.where(
        events["runtime_sec_final"] > 0,
        events["watch_time_sec"].fillna(0) / events["runtime_sec_final"],
        np.nan,
    )
    events["watch_ratio_clipped"] = events["watch_ratio"].clip(lower=0, upper=1)
    events["watch_bucket"] = events["watch_ratio"].map(watch_bucket)
    events["content_duration_bucket"] = events["runtime_sec_final"].map(duration_bucket)
    events["event_date"] = events["event_ts"].dt.date.astype("string")
    events["user_event_count"] = events.groupby("user_id")["content_id"].transform("size")

    stats.kept_after_cleaning = int(len(events))
    stats.users_after_cleaning = int(events["user_id"].nunique())
    stats.zero_watch_kept = int(events["is_zero_watch"].sum())
    stats.short_watch_kept = int(events["is_short_watch"].sum())
    stats.over_runtime_kept = int(events["is_over_runtime"].sum())

    out_path = tmp_dir / f"{shard_path.stem}-clean.parquet"
    events[EVENT_COLUMNS + ["user_event_count"]].to_parquet(out_path, index=False)

    item_counts = Counter(events["content_id"].astype(str).tolist())
    return stats, item_counts


def build_item_vocab(content_ids: Iterable[str]) -> pd.DataFrame:
    ordered = sorted(set(content_ids))
    return pd.DataFrame({"content_id": ordered, "item_id": range(1, len(ordered) + 1)})


def apply_k_core_filter(events: pd.DataFrame, min_user_events: int, min_item_events: int) -> pd.DataFrame:
    filtered = events.copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user_id")["content_id"].transform("size")
        filtered = filtered.loc[user_counts >= min_user_events].copy()
        item_counts = filtered.groupby("content_id")["user_id"].transform("size")
        filtered = filtered.loc[item_counts >= min_item_events].copy()
        if len(filtered) == before:
            break
    return filtered


def sequence_to_split_rows(group: pd.DataFrame) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    group = group.sort_values(["event_ts", "content_id"], kind="stable")
    items = group["item_id"].astype(int).tolist()
    event_ts = group["event_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    watch_ratio = group["watch_ratio_clipped"].astype(float).round(6).tolist()
    watch_bucket_seq = group["watch_bucket"].astype(str).tolist()
    duration_bucket_seq = group["content_duration_bucket"].astype(str).tolist()

    common = {
        "user_id": group["user_id"].iloc[0],
        "item_sequence": items,
        "event_ts_sequence": event_ts,
        "watch_ratio_sequence": watch_ratio,
        "watch_bucket_sequence": watch_bucket_seq,
        "duration_bucket_sequence": duration_bucket_seq,
        "seq_len": len(items),
    }
    train = {
        "user_id": common["user_id"],
        "item_sequence": items[:-2],
        "watch_ratio_sequence": watch_ratio[:-2],
        "watch_bucket_sequence": watch_bucket_seq[:-2],
        "duration_bucket_sequence": duration_bucket_seq[:-2],
        "seq_len": len(items[:-2]),
    }
    valid = {
        "user_id": common["user_id"],
        "input_sequence": items[:-2],
        "watch_ratio_sequence": watch_ratio[:-2],
        "watch_bucket_sequence": watch_bucket_seq[:-2],
        "duration_bucket_sequence": duration_bucket_seq[:-2],
        "target_item": items[-2],
        "target_event_ts": event_ts[-2],
        "seq_len": len(items[:-2]),
    }
    test = {
        "user_id": common["user_id"],
        "input_sequence": items[:-1],
        "watch_ratio_sequence": watch_ratio[:-1],
        "watch_bucket_sequence": watch_bucket_seq[:-1],
        "duration_bucket_sequence": duration_bucket_seq[:-1],
        "target_item": items[-1],
        "target_event_ts": event_ts[-1],
        "seq_len": len(items[:-1]),
    }
    return common, train, valid, test


def generate_report(
    output_dir: Path,
    summary: dict[str, object],
    metadata_stats: dict[str, object],
    sequence_stats: dict[str, object],
) -> None:
    output_prefix = output_dir.name
    report = f"""# Data Processing Report

## Scope
This preprocessing pipeline prepares user-content interaction data for sequential recommendation experiments with metadata enrichment beyond pure item order. The main emphasis is on `watch_time` and content duration, with content metadata retained for auxiliary sequence features and analysis.

## Inputs
- `data/raw_data/*.parquet`: 16 parquet shards containing one row per user and an ordered `contents` array of interactions.
- `data/content_profile.csv`: full content catalog. Only metadata fields were retained for the processed outputs; performance metrics were excluded from the modeling artifacts.

## Raw Data Status
### Content catalog
- Rows: {metadata_stats['raw_rows']:,}
- Unique `content_id`: {metadata_stats['raw_unique_content_id']:,}
- Columns before cleaning: {len(metadata_stats['columns_before'])}
- Columns after metadata selection: {len(metadata_stats['columns_after'])}
- Content type distribution: {json.dumps(metadata_stats['content_type_distribution'], ensure_ascii=False)}

### Interaction data
- Raw user rows: {summary['raw_user_rows']:,}
- Exploded events: {summary['exploded_events']:,}
- Unique users after structural cleaning: {summary['users_after_cleaning']:,}
- Unique content IDs after structural cleaning: {summary['items_after_cleaning']:,}
- Content join coverage after exploding: {summary['content_join_coverage']:.4%}

## What Was Done
1. Exploded each user's `contents` array into one event per row.
2. Preserved user order and re-sorted by timestamp as a safety check.
3. Cleaned metadata text fields by stripping HTML and collapsing whitespace.
4. Removed catalog performance metrics from the experiment-facing outputs.
5. Joined interactions to metadata on `content_id`.
6. Built `runtime_sec_final` using raw runtime when valid and catalog runtime as fallback.
7. Dropped events with missing user/item/timestamp, failed metadata join, or non-positive runtime.
8. Removed exact duplicate interaction rows.
9. Kept zero and short watch events, but added explicit quality flags.
10. Kept most over-runtime events, clipped the normalized ratio for modeling, and dropped only extreme over-runtime anomalies.
11. Filtered to users with at least {summary['min_user_events']} cleaned events and items with at least {summary['min_item_events']} cleaned events.
12. Built item vocabulary and leave-last-two sequence splits for BERT4Rec-style evaluation.

## Data Quality Before Support Filtering
- Missing `user_id`: {summary['missing_user_id']:,}
- Missing `content_id`: {summary['missing_content_id']:,}
- Missing timestamp: {summary['missing_event_ts']:,}
- Failed content join: {summary['missing_metadata_join']:,}
- Invalid runtime: {summary['invalid_runtime']:,}
- Exact duplicates removed: {summary['exact_duplicates_removed']:,}
- Extreme over-runtime removed: {summary['extreme_over_runtime_removed']:,}
- Zero-watch events kept: {summary['zero_watch_kept']:,}
- Short-watch events kept: {summary['short_watch_kept']:,}
- Over-runtime events kept: {summary['over_runtime_kept']:,}

## Final Output Status
- Retained interactions: {summary['final_events']:,}
- Retained users: {summary['final_users']:,}
- Retained items: {summary['final_items']:,}
- Vocabulary size: {summary['final_items']:,}
- Users in sequence dataset: {sequence_stats['users']:,}
- Average sequence length: {sequence_stats['seq_len_mean']:.2f}
- Median sequence length: {sequence_stats['seq_len_p50']:.2f}
- 95th percentile sequence length: {sequence_stats['seq_len_p95']:.2f}
- Watch ratio mean: {sequence_stats['watch_ratio_mean']:.4f}
- Watch ratio median: {sequence_stats['watch_ratio_p50']:.4f}
- Watch ratio 95th percentile: {sequence_stats['watch_ratio_p95']:.4f}

## Output Files
- `{output_prefix}/content_metadata_clean.parquet`: selected and cleaned metadata only.
- `{output_prefix}/interactions_enriched.parquet`: cleaned flat event table with derived watch and duration features.
- `{output_prefix}/item_vocab.parquet`: `content_id` to integer `item_id` mapping.
- `{output_prefix}/user_sequences.parquet`: full per-user ordered sequences with aligned auxiliary features.
- `{output_prefix}/train.parquet`: training prefixes using sequence[:-2].
- `{output_prefix}/valid.parquet`: validation rows using sequence[:-2] to predict the second-to-last item.
- `{output_prefix}/test.parquet`: test rows using sequence[:-1] to predict the last item.
- `{output_prefix}/feature_config.json`: schema and feature usage notes for experiments.
- `{output_prefix}/processing_summary.json`: machine-readable processing statistics.

## Final Schema
### `interactions_enriched.parquet`
{json.dumps(EVENT_COLUMNS + ['item_id'], ensure_ascii=False)}

### `content_metadata_clean.parquet`
{json.dumps(metadata_stats['columns_after'], ensure_ascii=False)}

## Notes for Experiments
- `watch_ratio_clipped` is the safest normalized engagement feature to feed alongside item tokens.
- `watch_bucket` is useful when the base sequential model expects categorical side information.
- `content_duration_bucket` helps separate short-form and long-form consumption patterns.
- Text metadata was retained in the catalog output for optional later encoding, but not forced into the sequence tables.
"""
    (output_dir / "DATA_PROCESSING_REPORT.md").write_text(report, encoding="utf-8")


def write_partitioned_parquet(frames: list[pd.DataFrame], output_path: Path) -> None:
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(output_path, index=False)
    else:
        pd.DataFrame().to_parquet(output_path, index=False)


def default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("prep_%Y%m%dT%H%M%SZ")


def build_output_paths(root: Path, prepared_root_arg: str, run_name: str, output_dir_arg: str | None) -> tuple[Path, Path, Path]:
    prepared_root = root / prepared_root_arg
    manifest_dir = prepared_root / "manifests"
    if output_dir_arg:
        output_dir = root / output_dir_arg
    else:
        output_dir = prepared_root / "runs" / run_name
    return prepared_root, manifest_dir, output_dir


def file_entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
    }


def write_manifest(
    manifest_path: Path,
    run_manifest_path: Path,
    repo_root: Path,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    summary: dict[str, object],
    metadata_stats: dict[str, object],
    sequence_stats: dict[str, object],
) -> None:
    tracked_files = [
        output_dir / "content_metadata_clean.parquet",
        output_dir / "interactions_enriched.parquet",
        output_dir / "user_sequences.parquet",
        output_dir / "train.parquet",
        output_dir / "valid.parquet",
        output_dir / "test.parquet",
        output_dir / "item_vocab.parquet",
        output_dir / "feature_config.json",
        output_dir / "processing_summary.json",
        output_dir / "DATA_PROCESSING_REPORT.md",
    ]

    manifest = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_data_dir": str(output_dir.relative_to(repo_root)),
        "manifest_paths": {
            "run_manifest": str(run_manifest_path.relative_to(repo_root)),
            "tracked_manifest": str(manifest_path.relative_to(repo_root)),
        },
        "inputs": {
            "data_dir": args.data_dir,
            "clean_shards_dir": args.clean_shards_dir,
        },
        "filters": {
            "min_user_events": args.min_user_events,
            "min_item_events": args.min_item_events,
        },
        "artifacts": [file_entry(path, repo_root) for path in tracked_files if path.exists()],
        "summary": {
            "final_events": summary["final_events"],
            "final_users": summary["final_users"],
            "final_items": summary["final_items"],
            "content_join_coverage": summary["content_join_coverage"],
            "sequence_users": sequence_stats["users"],
            "sequence_length_p50": sequence_stats["seq_len_p50"],
            "sequence_length_p95": sequence_stats["seq_len_p95"],
        },
        "metadata": {
            "rows_after_cleaning": metadata_stats["rows_after_cleaning"],
            "content_type_distribution": metadata_stats["content_type_distribution"],
        },
    }
    encoded = json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    run_manifest_path.write_text(encoded, encoding="utf-8")
    manifest_path.write_text(encoded, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare enriched experiment data from raw interactions and content metadata.")
    parser.add_argument("--data-dir", default="data", help="Input data directory")
    parser.add_argument("--prepared-root", default="prepared_data", help="Prepared-data root directory")
    parser.add_argument("--run-name", default=None, help="Name for this prepared-data run")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory for this run")
    parser.add_argument(
        "--clean-shards-dir",
        default=None,
        help="Optional directory of pre-cleaned shard parquet files to reuse",
    )
    parser.add_argument("--min-user-events", type=int, default=3, help="Minimum events per user after cleaning")
    parser.add_argument("--min-item-events", type=int, default=5, help="Minimum events per item after cleaning")
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = root / args.data_dir
    run_name = args.run_name or default_run_name()
    _, manifest_dir, output_dir = build_output_paths(root, args.prepared_root, run_name, args.output_dir)
    tmp_dir = output_dir / "tmp_interactions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    metadata, metadata_stats = load_metadata(data_dir / "content_profile.csv")
    metadata.to_parquet(output_dir / "content_metadata_clean.parquet", index=False)

    item_counts: Counter = Counter()
    shard_stats: list[dict[str, object]] = []
    raw_user_rows = 0
    exploded_events = 0

    if args.clean_shards_dir:
        clean_dir = root / args.clean_shards_dir
        clean_files = sorted(clean_dir.glob("*-clean.parquet"))
        raw_files = sorted((data_dir / "raw_data").glob("*.parquet"))
        raw_user_rows = 0
        for raw_file in raw_files:
            raw_user_rows += int(len(pd.read_parquet(raw_file, columns=["user_id"])))
        exploded_events = 0
        for clean_file in clean_files:
            df = pd.read_parquet(clean_file, columns=["user_id", "content_id"])
            exploded_events += int(len(df))
            item_counts.update(df["content_id"].astype(str).tolist())
        shard_stats = []
    else:
        raw_files = sorted((data_dir / "raw_data").glob("*.parquet"))
        for shard_path in raw_files:
            print(f"Cleaning shard: {shard_path.name}", flush=True)
            stats, shard_counter = clean_and_enrich_events(shard_path, metadata, tmp_dir)
            raw_user_rows += stats.raw_user_rows
            exploded_events += stats.exploded_events
            item_counts.update(shard_counter)
            shard_stats.append(asdict(stats))
        clean_dir = tmp_dir
        clean_files = sorted(clean_dir.glob("*-clean.parquet"))

    current_items = {item for item, count in item_counts.items() if count >= args.min_item_events}
    while True:
        next_item_counts: Counter = Counter()
        for clean_file in clean_files:
            df = pd.read_parquet(clean_file)
            df = df.loc[df["content_id"].astype(str).isin(current_items)].copy()
            if df.empty:
                continue
            user_counts = df.groupby("user_id")["content_id"].transform("size")
            df = df.loc[user_counts >= args.min_user_events].copy()
            if df.empty:
                continue
            next_item_counts.update(df["content_id"].astype(str).tolist())
        next_items = {item for item, count in next_item_counts.items() if count >= args.min_item_events}
        if next_items == current_items:
            break
        current_items = next_items

    final_interaction_parts: list[pd.DataFrame] = []
    user_sequence_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    vocab = build_item_vocab(current_items)
    vocab.to_parquet(output_dir / "item_vocab.parquet", index=False)

    for clean_file in clean_files:
        print(f"Building final outputs from: {clean_file.name}", flush=True)
        df = pd.read_parquet(clean_file)
        df = df.loc[df["content_id"].astype(str).isin(current_items)].copy()
        if df.empty:
            continue
        user_counts = df.groupby("user_id")["content_id"].transform("size")
        df = df.loc[user_counts >= args.min_user_events].copy()
        if df.empty:
            continue

        df = df.sort_values(["user_id", "event_ts", "content_id"], kind="stable").reset_index(drop=True)
        df = df.merge(vocab, on="content_id", how="left", validate="many_to_one")
        final_interaction_parts.append(df.drop(columns=["user_event_count"], errors="ignore"))

        sequence_rows = []
        train_rows = []
        valid_rows = []
        test_rows = []
        for _, group in df.groupby("user_id", sort=False):
            if len(group) < args.min_user_events:
                continue
            full_row, train_row, valid_row, test_row = sequence_to_split_rows(group)
            sequence_rows.append(full_row)
            train_rows.append(train_row)
            valid_rows.append(valid_row)
            test_rows.append(test_row)

        if sequence_rows:
            user_sequence_parts.append(pd.DataFrame(sequence_rows))
            train_parts.append(pd.DataFrame(train_rows))
            valid_parts.append(pd.DataFrame(valid_rows))
            test_parts.append(pd.DataFrame(test_rows))

    filtered = pd.concat(final_interaction_parts, ignore_index=True) if final_interaction_parts else pd.DataFrame()
    user_sequences = pd.concat(user_sequence_parts, ignore_index=True) if user_sequence_parts else pd.DataFrame()
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    valid_df = pd.concat(valid_parts, ignore_index=True) if valid_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    filtered.to_parquet(output_dir / "interactions_enriched.parquet", index=False)
    user_sequences.to_parquet(output_dir / "user_sequences.parquet", index=False)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    valid_df.to_parquet(output_dir / "valid.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    feature_config = {
        "sequence_item_field": "item_sequence",
        "item_id_field": "item_id",
        "id_mapping_file": "item_vocab.parquet",
        "aligned_sequence_features": [
            "watch_ratio_sequence",
            "watch_bucket_sequence",
            "duration_bucket_sequence",
        ],
        "event_feature_fields": [
            "watch_ratio_clipped",
            "watch_bucket",
            "content_duration_bucket",
            "content_type",
            "provider",
            "country",
        ],
        "splitting_strategy": {
            "type": "leave_last_two",
            "train_input": "sequence[:-2]",
            "valid_input": "sequence[:-2]",
            "valid_target": "sequence[-2]",
            "test_input": "sequence[:-1]",
            "test_target": "sequence[-1]",
        },
        "filtering": {
            "min_user_events": args.min_user_events,
            "min_item_events": args.min_item_events,
            "moderate_filtering": {
                "drop_missing_join": True,
                "drop_invalid_runtime": True,
                "keep_zero_short_watch": True,
                "drop_extreme_over_runtime_only": True,
            },
        },
    }
    (output_dir / "feature_config.json").write_text(json.dumps(feature_config, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "raw_user_rows": raw_user_rows,
        "exploded_events": exploded_events,
        "users_after_cleaning": int(sum(s["users_after_cleaning"] for s in shard_stats)),
        "items_after_cleaning": int(len(item_counts)),
        "content_join_coverage": float(1 - (sum(s["missing_metadata_join"] for s in shard_stats) / exploded_events if exploded_events else 0)),
        "missing_user_id": int(sum(s["missing_user_id"] for s in shard_stats)),
        "missing_content_id": int(sum(s["missing_content_id"] for s in shard_stats)),
        "missing_event_ts": int(sum(s["missing_event_ts"] for s in shard_stats)),
        "missing_metadata_join": int(sum(s["missing_metadata_join"] for s in shard_stats)),
        "invalid_runtime": int(sum(s["invalid_runtime"] for s in shard_stats)),
        "exact_duplicates_removed": int(sum(s["exact_duplicates_removed"] for s in shard_stats)),
        "extreme_over_runtime_removed": int(sum(s["extreme_over_runtime_removed"] for s in shard_stats)),
        "zero_watch_kept": int(sum(s["zero_watch_kept"] for s in shard_stats)),
        "short_watch_kept": int(sum(s["short_watch_kept"] for s in shard_stats)),
        "over_runtime_kept": int(sum(s["over_runtime_kept"] for s in shard_stats)),
        "min_user_events": args.min_user_events,
        "min_item_events": args.min_item_events,
        "final_events": int(len(filtered)),
        "final_users": int(filtered["user_id"].nunique()),
        "final_items": int(filtered["content_id"].nunique()),
        "shards": shard_stats,
    }

    if user_sequences.empty:
        sequence_stats = {
            "users": 0,
            "seq_len_mean": 0.0,
            "seq_len_p50": 0.0,
            "seq_len_p95": 0.0,
            "watch_ratio_mean": 0.0,
            "watch_ratio_p50": 0.0,
            "watch_ratio_p95": 0.0,
        }
    else:
        seq_lengths = user_sequences["seq_len"].astype(float)
        watch_values = filtered["watch_ratio_clipped"].astype(float)
        sequence_stats = {
            "users": int(len(user_sequences)),
            "seq_len_mean": float(seq_lengths.mean()),
            "seq_len_p50": float(seq_lengths.quantile(0.50)),
            "seq_len_p95": float(seq_lengths.quantile(0.95)),
            "watch_ratio_mean": float(watch_values.mean()),
            "watch_ratio_p50": float(watch_values.quantile(0.50)),
            "watch_ratio_p95": float(watch_values.quantile(0.95)),
        }

    (output_dir / "processing_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "metadata_stats": metadata_stats,
                "sequence_stats": sequence_stats,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    generate_report(output_dir, summary, metadata_stats, sequence_stats)
    write_manifest(
        manifest_path=manifest_dir / f"{run_name}.json",
        run_manifest_path=output_dir / "manifest.json",
        repo_root=root,
        run_name=run_name,
        output_dir=output_dir,
        args=args,
        summary=summary,
        metadata_stats=metadata_stats,
        sequence_stats=sequence_stats,
    )


if __name__ == "__main__":
    main()
