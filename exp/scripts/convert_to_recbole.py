"""Convert preprocessed parquet data to RecBole atomic file format.

Creates dataset/vod_data/vod_data.inter with columns:
  user_id:token  item_id:token  timestamp:float  watch_ratio:float
  watch_bucket:token  duration_bucket:token

Optionally subsets to N users for fast experimentation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_interactions(data_dir: str) -> pd.DataFrame:
    """Load raw interactions from the data_prep pipeline."""
    path = Path(data_dir)
    df = pd.read_parquet(path / "interactions_enriched.parquet")
    return df


def subset_users(df: pd.DataFrame, n_users: int, min_interactions: int = 5,
                 seed: int = 42) -> pd.DataFrame:
    """Stratified sampling of users by activity level."""
    user_counts = df.groupby("user_id").size()
    eligible = user_counts[user_counts >= min_interactions]
    print(f"Eligible users (>={min_interactions} interactions): {len(eligible)}")

    if n_users >= len(eligible):
        print(f"Requested {n_users} users, but only {len(eligible)} eligible. Using all.")
        return df[df["user_id"].isin(eligible.index)]

    rng = np.random.RandomState(seed)
    bins = pd.qcut(eligible, q=5, labels=False, duplicates="drop")
    sampled_users = []
    per_bin = n_users // bins.nunique()
    for bin_label in sorted(bins.unique()):
        bin_users = bins[bins == bin_label].index.tolist()
        n_sample = min(per_bin, len(bin_users))
        sampled_users.extend(rng.choice(bin_users, n_sample, replace=False).tolist())

    remaining = n_users - len(sampled_users)
    if remaining > 0:
        all_eligible = set(eligible.index.tolist())
        already = set(sampled_users)
        pool = list(all_eligible - already)
        sampled_users.extend(rng.choice(pool, remaining, replace=False).tolist())

    sampled_users = sampled_users[:n_users]
    return df[df["user_id"].isin(sampled_users)]


def convert_to_recbole(df: pd.DataFrame, output_dir: str):
    """Convert interactions DataFrame to RecBole .inter format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Select and rename columns
    rec_df = pd.DataFrame()
    rec_df["user_id:token"] = df["user_id"].values
    rec_df["item_id:token"] = df["item_id"].values

    # Convert timestamp to epoch seconds (datetime64[us] → int64 gives microseconds)
    ts = pd.to_datetime(df["event_ts"])
    rec_df["timestamp:float"] = (ts.astype(np.int64) // 1_000_000).values

    # Watch ratio (clipped to [0, 1])
    rec_df["watch_ratio:float"] = df["watch_ratio_clipped"].clip(0.0, 1.0).values

    # Categorical features
    rec_df["watch_bucket:token"] = df["watch_bucket"].fillna("unknown").values
    rec_df["duration_bucket:token"] = df["content_duration_bucket"].fillna("unknown").values

    # Sort by user and timestamp
    rec_df = rec_df.sort_values(["user_id:token", "timestamp:float"]).reset_index(drop=True)

    # Write .inter file (tab-separated)
    inter_path = out_path / "vod_data.inter"
    rec_df.to_csv(inter_path, sep="\t", index=False)
    print(f"Written {len(rec_df)} interactions to {inter_path}")

    # Stats
    n_users = rec_df["user_id:token"].nunique()
    n_items = rec_df["item_id:token"].nunique()
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(rec_df)}")
    print(f"Density: {len(rec_df) / (n_users * n_items):.6f}")

    # Save metadata
    meta = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(len(rec_df)),
        "density": float(len(rec_df) / (n_users * n_items)),
    }
    with open(out_path / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data_prep/legacy/processed_final",
                        help="Path to preprocessed data directory")
    parser.add_argument("--output-dir", default="dataset/vod_data",
                        help="Output directory for RecBole dataset")
    parser.add_argument("--n-users", type=int, default=20000,
                        help="Number of users to sample (0 = all)")
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading interactions...")
    df = load_interactions(args.data_dir)
    print(f"Total: {len(df)} interactions, {df['user_id'].nunique()} users")

    if args.n_users > 0:
        print(f"\nSubsetting to {args.n_users} users...")
        df = subset_users(df, args.n_users, args.min_interactions, args.seed)
        print(f"After subset: {len(df)} interactions, {df['user_id'].nunique()} users")

    print("\nConverting to RecBole format...")
    meta = convert_to_recbole(df, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
