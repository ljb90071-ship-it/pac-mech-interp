"""
utils.py
--------
Utility functions for evaluation metrics and statistical significance testing.

Implements the statistical analysis described in Section 5.3 of the paper:
  - Paired t-test (scipy.stats.ttest_rel)
  - Wilcoxon signed-rank test (scipy.stats.wilcoxon)
  - Cohen's d effect size
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_metrics(per_permutation_accuracies: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for per-permutation accuracies.

    Args:
        per_permutation_accuracies: List of accuracy values (one per permutation).

    Returns:
        Dict with keys: 'mean', 'worst_case', 'std_dev', 'best_case'.
    """
    arr = np.array(per_permutation_accuracies)
    return {
        "mean": float(arr.mean()),
        "worst_case": float(arr.min()),
        "best_case": float(arr.max()),
        "std_dev": float(arr.std()),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two paired samples.

    Cohen's d = mean(diff) / std(diff)

    Args:
        group1: Array of values for condition 1 (e.g., PAC).
        group2: Array of values for condition 2 (e.g., baseline).

    Returns:
        Cohen's d value.
    """
    diff = group1 - group2
    return float(diff.mean() / (diff.std() + 1e-10))


def run_statistical_tests(
    pac_accuracies: List[float],
    baseline_accuracies: List[float],
) -> Dict[str, float]:
    """
    Run paired t-test and Wilcoxon signed-rank test comparing PAC to baseline.

    Both tests are one-sided (testing that PAC > baseline).

    Args:
        pac_accuracies: Per-permutation accuracies for PAC.
        baseline_accuracies: Per-permutation accuracies for baseline (λ=0).

    Returns:
        Dict with keys:
          - 'ttest_pvalue': p-value from paired t-test.
          - 'wilcoxon_pvalue': p-value from Wilcoxon signed-rank test.
          - 'cohens_d': Cohen's d effect size.
          - 'mean_improvement': Mean absolute accuracy improvement.
    """
    pac = np.array(pac_accuracies)
    baseline = np.array(baseline_accuracies)

    # Paired t-test (one-sided: PAC > baseline)
    t_stat, t_pval = stats.ttest_rel(pac, baseline, alternative="greater")

    # Wilcoxon signed-rank test (one-sided)
    try:
        w_stat, w_pval = stats.wilcoxon(pac, baseline, alternative="greater")
    except ValueError:
        # If all differences are zero
        w_pval = 1.0

    d = cohens_d(pac, baseline)
    mean_improvement = float((pac - baseline).mean())

    return {
        "ttest_pvalue": float(t_pval),
        "wilcoxon_pvalue": float(w_pval),
        "cohens_d": d,
        "mean_improvement": mean_improvement,
    }


def run_statistical_tests_sweep(
    sweep_results: List[Dict],
) -> List[Dict]:
    """
    Run statistical tests for each λ value in a sweep, comparing to baseline (λ=0).

    Args:
        sweep_results: List of result dicts from PACInference.sweep_lambda().

    Returns:
        List of statistical test result dicts (one per λ value, excluding baseline).
    """
    # Find baseline (λ = 0)
    baseline_result = next(r for r in sweep_results if r["lam"] == 0.0)
    baseline_accs = np.array(baseline_result["per_permutation_accuracy"])

    stat_results = []
    for result in sweep_results:
        lam = result["lam"]
        pac_accs = np.array(result["per_permutation_accuracy"])
        tests = run_statistical_tests(pac_accs.tolist(), baseline_accs.tolist())
        tests["lam"] = lam
        stat_results.append(tests)

    return stat_results


def print_results_table(sweep_results: List[Dict]) -> None:
    """
    Print a formatted table of results across λ values.

    Args:
        sweep_results: List of result dicts from PACInference.sweep_lambda().
    """
    print("\n" + "=" * 70)
    print(f"{'λ':>6} | {'Mean Acc':>10} | {'Worst-Case':>10} | {'Std Dev':>8}")
    print("-" * 70)
    for r in sweep_results:
        lam_str = f"{r['lam']:.1f}"
        if r["lam"] == 1.0:
            lam_str += " (CC)"
        print(
            f"{lam_str:>6} | "
            f"{r['mean_accuracy']:>10.3f} | "
            f"{r['worst_case_accuracy']:>10.3f} | "
            f"{r['std_dev']:>8.3f}"
        )
    print("=" * 70 + "\n")
