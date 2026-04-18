"""Statistical significance testing for model comparison.

Implements paired bootstrap test and Wilcoxon signed-rank test
with Bonferroni correction for multiple comparisons.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy import stats


def paired_bootstrap_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap significance test.

    Tests whether model_b is significantly better than model_a.

    Args:
        metric_a: per-user metric values for model A
        metric_b: per-user metric values for model B
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        dict with p_value, mean_diff, ci_lower, ci_upper (95% CI)
    """
    rng = np.random.RandomState(seed)
    n = len(metric_a)
    assert len(metric_b) == n

    diffs = metric_b - metric_a
    observed_diff = diffs.mean()

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bootstrap_diffs[i] = diffs[indices].mean()

    # Two-sided p-value
    p_value = float((np.abs(bootstrap_diffs - observed_diff) >= np.abs(observed_diff)).mean())
    # Alternatively: proportion of bootstrap diffs on wrong side
    if observed_diff >= 0:
        p_value = float((bootstrap_diffs <= 0).mean())
    else:
        p_value = float((bootstrap_diffs >= 0).mean())

    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "p_value": p_value,
        "mean_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_0.05": p_value < 0.05,
    }


def wilcoxon_test(metric_a: np.ndarray, metric_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric paired test).

    Args:
        metric_a: per-user metrics for model A
        metric_b: per-user metrics for model B

    Returns:
        dict with statistic and p_value
    """
    diffs = metric_b - metric_a
    # Remove zero differences (ties)
    nonzero = diffs != 0
    if nonzero.sum() < 10:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_0.05": False}

    stat, p_value = stats.wilcoxon(metric_a[nonzero], metric_b[nonzero], alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
    }


def compare_all_models(
    per_user_metrics: dict[str, dict[str, np.ndarray]],
    metric_name: str = "ndcg@10",
    n_bootstrap: int = 10000,
) -> dict:
    """Run pairwise significance tests across all models.

    Args:
        per_user_metrics: {model_name: {metric_name: per_user_array}}
        metric_name: which metric to test
        n_bootstrap: bootstrap samples

    Returns:
        dict with pairwise test results and Bonferroni-corrected p-values
    """
    model_names = sorted(per_user_metrics.keys())
    pairs = list(itertools.combinations(model_names, 2))
    n_comparisons = len(pairs)

    results = {}
    for model_a, model_b in pairs:
        arr_a = per_user_metrics[model_a][metric_name]
        arr_b = per_user_metrics[model_b][metric_name]

        bootstrap = paired_bootstrap_test(arr_a, arr_b, n_bootstrap=n_bootstrap)
        wilcoxon = wilcoxon_test(arr_a, arr_b)

        # Bonferroni correction
        corrected_p_bootstrap = min(bootstrap["p_value"] * n_comparisons, 1.0)
        corrected_p_wilcoxon = min(wilcoxon["p_value"] * n_comparisons, 1.0)

        key = f"{model_a}_vs_{model_b}"
        results[key] = {
            "model_a": model_a,
            "model_b": model_b,
            "mean_a": float(arr_a.mean()),
            "mean_b": float(arr_b.mean()),
            "bootstrap": bootstrap,
            "wilcoxon": wilcoxon,
            "bonferroni_p_bootstrap": corrected_p_bootstrap,
            "bonferroni_p_wilcoxon": corrected_p_wilcoxon,
            "bonferroni_significant_at_0.05": corrected_p_bootstrap < 0.05,
        }

    return {"metric": metric_name, "n_comparisons": n_comparisons, "pairwise": results}


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and confidence interval via bootstrap.

    Returns: (mean, ci_lower, ci_upper)
    """
    n = len(values)
    mean = float(values.mean())
    rng = np.random.RandomState(42)
    boot_means = np.array([values[rng.randint(0, n, n)].mean() for _ in range(10000)])
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return mean, ci_lower, ci_upper
