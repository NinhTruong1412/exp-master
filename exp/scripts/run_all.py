"""Train and evaluate all models sequentially.

Usage:
    python scripts/run_all.py --seeds 42
    python scripts/run_all.py --seeds 42,123,456 --models Pop SASRec BERT4Rec
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_single import run

MODEL_CONFIGS = {
    "Pop": ["configs/base.yaml", "configs/pop.yaml"],
    "SASRec": ["configs/base.yaml", "configs/sasrec.yaml"],
    "BERT4Rec": ["configs/base.yaml", "configs/bert4rec.yaml"],
    "DurationAwareBERT4Rec": ["configs/base.yaml", "configs/variant_a.yaml"],
    "CounterfactualBERT4Rec": ["configs/base.yaml", "configs/variant_b.yaml"],
    "CalibratedBERT4Rec": ["configs/base.yaml", "configs/variant_c.yaml"],
}


def main():
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        help="Models to train")
    parser.add_argument("--dataset", default="vod_data")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []
    total_start = time.time()

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Training {model_name} (seed={seed})")
            print(f"{'#'*60}")
            start = time.time()
            try:
                result = run(model_name, MODEL_CONFIGS[model_name], seed, args.dataset)
                result["training_time"] = time.time() - start
                all_results.append(result)
                print(f"Done in {result['training_time']:.1f}s — "
                      f"Test NDCG@10: {result['test_result'].get('ndcg@10', 'N/A'):.6f}")
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    "model": model_name, "seed": seed, "error": str(e),
                    "training_time": time.time() - start,
                })

    total_time = time.time() - total_start

    # Save all results
    out_dir = "recbole_outputs/results"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "total_time": total_time}, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY (total time: {total_time:.1f}s)")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Seed':>4} {'NDCG@10':>10} {'Hit@10':>10} {'MRR@10':>10} {'Time':>8}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<30} {r['seed']:>4} {'ERROR':>10} {r.get('error','')[:30]}")
        else:
            tr = r["test_result"]
            print(f"{r['model']:<30} {r['seed']:>4} "
                  f"{tr.get('ndcg@10', 0):>10.6f} "
                  f"{tr.get('hit@10', 0):>10.6f} "
                  f"{tr.get('mrr@10', 0):>10.6f} "
                  f"{r.get('training_time', 0):>7.1f}s")

    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
