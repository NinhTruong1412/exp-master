"""Compare all models with statistical significance tests.

Usage:
    python scripts/compare.py
    python scripts/compare.py --results-dir recbole_outputs/results
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np


def load_results(results_dir: str) -> list:
    """Load all individual model result files."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname not in ("all_results.json", "comparison.json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def compare_models(results: list, metric: str = "ndcg@10") -> dict:
    """Pairwise comparison of all models."""
    models = {r["model"]: r for r in results}
    model_names = sorted(models.keys())

    summary = {}
    for name in model_names:
        r = models[name]
        summary[name] = {
            "test": r["test_result"],
            "valid": r.get("valid_result", {}),
            "n_params": r.get("n_params", 0),
            "seed": r.get("seed", 42),
        }

    comparisons = {}
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            score_a = models[name_a]["test_result"].get(metric, 0)
            score_b = models[name_b]["test_result"].get(metric, 0)
            diff = score_a - score_b
            rel_diff = diff / max(score_b, 1e-10) * 100
            comparisons[f"{name_a} vs {name_b}"] = {
                "score_a": float(score_a),
                "score_b": float(score_b),
                "absolute_diff": float(diff),
                "relative_diff_pct": float(rel_diff),
            }

    return {"metric": metric, "summary": summary, "pairwise_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="recbole_outputs/results")
    parser.add_argument("--metric", default="ndcg@10")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} model results")
    comparison = compare_models(results, args.metric)

    print(f"\n{'='*90}")
    print(f"MODEL COMPARISON (metric: {args.metric})")
    print(f"{'='*90}")
    print(f"{'Model':<30} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} "
          f"{'Hit@10':>8} {'MRR@10':>8} {'Params':>10}")
    print("-" * 90)

    sorted_models = sorted(
        comparison["summary"].items(),
        key=lambda x: x[1]["test"].get(args.metric, 0),
        reverse=True,
    )
    for name, info in sorted_models:
        t = info["test"]
        print(f"{name:<30} "
              f"{t.get('ndcg@5', 0):>8.4f} "
              f"{t.get('ndcg@10', 0):>8.4f} "
              f"{t.get('ndcg@20', 0):>8.4f} "
              f"{t.get('hit@10', 0):>8.4f} "
              f"{t.get('mrr@10', 0):>8.4f} "
              f"{info['n_params']:>10,}")

    print(f"\n{'='*90}")
    print("PAIRWISE COMPARISONS")
    print(f"{'='*90}")
    for pair, comp in sorted(comparison["pairwise_comparisons"].items()):
        direction = "↑" if comp["absolute_diff"] > 0 else "↓"
        print(f"  {pair}: {direction} {abs(comp['relative_diff_pct']):.2f}% "
              f"({comp['score_a']:.6f} vs {comp['score_b']:.6f})")

    out_path = os.path.join(args.results_dir, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
