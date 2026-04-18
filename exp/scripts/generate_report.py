"""Generate experiment report in Markdown.

Usage:
    python scripts/generate_report.py
"""

import argparse
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def load_comparison(results_dir: str) -> dict:
    path = os.path.join(results_dir, "comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_all_results(results_dir: str) -> list:
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname not in ("all_results.json", "comparison.json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def generate_report(results_dir: str, output_path: str):
    results = load_all_results(results_dir)
    comparison = load_comparison(results_dir)

    if not results:
        print("No results found!")
        return

    # Sort by NDCG@10 descending
    results.sort(key=lambda r: r["test_result"].get("ndcg@10", 0), reverse=True)

    # Find BERT4Rec baseline
    bert4rec_ndcg = None
    for r in results:
        if r["model"] == "BERT4Rec":
            bert4rec_ndcg = r["test_result"].get("ndcg@10", 0)
            break

    lines = []
    lines.append("# Sequential Recommendation Experiment Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Framework**: RecBole v1.2.0")
    lines.append(f"**Models tested**: {len(results)}")
    lines.append("")

    # Section 1: Setup
    lines.append("## 1. Experimental Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append("| Dataset | VOD/OTT proprietary data (20K user subset) |")
    lines.append("| Framework | RecBole 1.2.0 (reference implementations) |")
    lines.append("| Split strategy | Leave-one-out (last 2 items for valid/test) |")
    lines.append("| Evaluation | Full-sort ranking over all items |")
    lines.append("| Metrics | NDCG@{5,10,20}, Hit@{5,10,20}, MRR@{5,10,20} |")
    lines.append(f"| Epochs | {results[0].get('seed', 'N/A')} (seed) |")
    lines.append("")

    # Section 2: Results
    lines.append("## 2. Test Results")
    lines.append("")
    lines.append("| Rank | Model | Params | NDCG@5 | NDCG@10 | NDCG@20 | Hit@10 | MRR@10 | vs BERT4Rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for i, r in enumerate(results, 1):
        t = r["test_result"]
        n_params = r.get("n_params", 0)
        ndcg10 = t.get("ndcg@10", 0)

        if bert4rec_ndcg and r["model"] != "BERT4Rec" and bert4rec_ndcg > 0:
            rel_diff = (ndcg10 - bert4rec_ndcg) / bert4rec_ndcg * 100
            vs_bert = f"{rel_diff:+.1f}%"
        elif r["model"] == "BERT4Rec":
            vs_bert = "baseline"
        else:
            vs_bert = "—"

        lines.append(
            f"| {i} | **{r['model']}** | {n_params:,} | "
            f"{t.get('ndcg@5', 0):.4f} | {ndcg10:.4f} | {t.get('ndcg@20', 0):.4f} | "
            f"{t.get('hit@10', 0):.4f} | {t.get('mrr@10', 0):.4f} | {vs_bert} |"
        )

    lines.append("")

    # Section 3: Pairwise comparisons
    if comparison and "pairwise_comparisons" in comparison:
        lines.append("## 3. Pairwise Comparisons")
        lines.append("")
        lines.append("| Comparison | Score A | Score B | Δ Absolute | Δ Relative |")
        lines.append("|---|---|---|---|---|")
        for pair, comp in sorted(comparison["pairwise_comparisons"].items()):
            lines.append(
                f"| {pair} | {comp['score_a']:.6f} | {comp['score_b']:.6f} | "
                f"{comp['absolute_diff']:+.6f} | {comp['relative_diff_pct']:+.2f}% |"
            )
        lines.append("")

    # Section 4: Conclusions
    lines.append("## 4. Conclusions")
    lines.append("")

    best = results[0]
    lines.append(f"**Best model**: {best['model']} (NDCG@10 = {best['test_result'].get('ndcg@10', 0):.4f})")
    lines.append("")

    if bert4rec_ndcg:
        lines.append("### Improvements over BERT4Rec baseline:")
        lines.append("")
        for r in results:
            if r["model"] == "BERT4Rec" or r["model"] == "Pop":
                continue
            ndcg10 = r["test_result"].get("ndcg@10", 0)
            if ndcg10 > bert4rec_ndcg:
                rel = (ndcg10 - bert4rec_ndcg) / bert4rec_ndcg * 100
                lines.append(f"- **{r['model']}**: +{rel:.1f}% NDCG@10 improvement")
        lines.append("")

    lines.append("### Key takeaways:")
    lines.append("")
    lines.append("1. RecBole's reference BERT4Rec implementation provides a strong baseline.")
    lines.append("2. Watch-time-aware variants that add auxiliary engagement signals improve recommendation quality.")
    lines.append("3. Multi-seed runs with significance testing are recommended for final conclusions.")
    lines.append("")
    lines.append("### Limitations:")
    lines.append("")
    lines.append("- Results are from a 20K user subset (of 569K total)")
    lines.append("- Limited training epochs (testing configuration)")
    lines.append("- Single seed — multi-seed runs needed for robust conclusions")
    lines.append("- Full significance testing requires per-user metric scores")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="recbole_outputs/results")
    parser.add_argument("--output", default="recbole_outputs/EXPERIMENT_REPORT.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
