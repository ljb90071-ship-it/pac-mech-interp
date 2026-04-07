"""
run_pac_experiments.py
-----------------------
Reproduces Figures 2 and 3 from the paper:
  "Mechanistic Evidence of Layer-Dependent Positional Bias in In-Context Learning"

This script:
  1. Evaluates PAC across all K!=24 permutations and 10 test queries (240 inferences)
     for λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}.
  2. Generates:
     - Figure 2(a): Box plot of per-permutation accuracy distributions.
     - Figure 2(b): Line plot of mean accuracy, worst-case accuracy, and std dev vs λ.
     - Figure 3(a): Bar chart of p-values (t-test and Wilcoxon) vs λ.
     - Figure 3(b): Bar chart of Cohen's d effect sizes vs λ.

Expected results (from paper):
  - Baseline (λ=0): Mean=58.3%, Worst-case=50.0%, Std=0.131
  - PAC (λ=0.7):    Mean=83.3%, p<0.001, Cohen's d=1.58
  - PAC (λ=1.0/CC): Mean=82.9%, Worst-case=60.0%, Std=0.121, Cohen's d=1.95

Usage:
    python experiments/run_pac_experiments.py [--output_dir figures/] [--device cpu]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import TEST_QUERIES
from src.pac import PACInference
from src.utils import print_results_table, run_statistical_tests_sweep


LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PAC experiments.")
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures",
        help="Directory to save output figures (default: figures/)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to save raw results JSON (default: results/)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)"
    )
    parser.add_argument(
        "--load_cached", action="store_true",
        help="Load cached results from results_dir instead of re-running inference."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_accuracy_distribution(sweep_results, output_path):
    """Figure 2(a): Box plot of per-permutation accuracy distributions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    data = [r["per_permutation_accuracy"] for r in sweep_results]
    lam_labels = [
        f"Baseline\n(λ=0)" if r["lam"] == 0.0
        else (f"λ={r['lam']:.1f}\n(=CC)" if r["lam"] == 1.0 else f"λ={r['lam']:.1f}")
        for r in sweep_results
    ]
    colors = ["#FF6B6B", "#FFA07A", "#FFD700", "#90EE90", "#4169E1", "#9370DB"]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Mean diamonds
    means = [r["mean_accuracy"] for r in sweep_results]
    ax.scatter(range(1, len(means) + 1), means, marker="D", color="black",
               zorder=5, s=60, label="Mean accuracy")

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5, label="Random chance")
    ax.set_xticklabels(lam_labels, fontsize=9)
    ax.set_ylabel("Accuracy across Permutations", fontsize=11)
    # Title and significance annotation placed separately to avoid overlap
    ax.set_title("(a) PAC Accuracy Distribution", fontsize=12, fontweight="bold")
    # Place *** as a separate text annotation to the right of the title, clear of text
    ax.annotate(
        "***",
        xy=(1.0, 1.02),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
        color="red",
        ha="right",
        va="bottom",
    )
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2(a) to: {output_path}")


def plot_accuracy_variance_tradeoff(sweep_results, output_path):
    """Figure 2(b): Line plot of mean accuracy, worst-case accuracy, and std dev vs λ."""
    lam_vals = [r["lam"] for r in sweep_results]
    means = [r["mean_accuracy"] for r in sweep_results]
    worsts = [r["worst_case_accuracy"] for r in sweep_results]
    stds = [r["std_dev"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(lam_vals, means, "b-o", linewidth=2, markersize=6, label="Mean Accuracy")
    ax1.plot(lam_vals, worsts, "r-s", linewidth=2, markersize=6, label="Worst-case")
    ax2.plot(lam_vals, stds, "y--^", linewidth=2, markersize=6, label="Std. Dev.")

    ax1.set_xlabel("Calibration Strength λ", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylabel("Std. Dev.", fontsize=11, color="goldenrod")
    ax2.tick_params(axis="y", labelcolor="goldenrod")

    ax1.set_ylim(0.4, 1.05)
    ax2.set_ylim(0.0, 0.40)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Place legend in upper-left to avoid the Std. Dev. line (which rises then falls
    # on the right side) from passing through the legend box
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=9,
        loc="upper left",
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    ax1.set_title("(b) Accuracy-Variance Trade-off", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2(b) to: {output_path}")


def plot_pvalues(stat_results, output_path):
    """Figure 3(a): Bar chart of p-values (t-test and Wilcoxon) vs λ."""
    lam_vals = [r["lam"] for r in stat_results if r["lam"] > 0]
    ttest_pvals = [r["ttest_pvalue"] for r in stat_results if r["lam"] > 0]
    wilcoxon_pvals = [r["wilcoxon_pvalue"] for r in stat_results if r["lam"] > 0]

    x = np.arange(len(lam_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, ttest_pvals, width, label="Paired t-test",
           color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, wilcoxon_pvals, width, label="Wilcoxon",
           color="#FF5722", alpha=0.8, linestyle="--")

    ax.axhline(y=0.05, color="black", linestyle="--", linewidth=1.5,
               label="p = 0.05")
    ax.set_yscale("log")
    ax.set_xlabel("Calibration Strength λ", fontsize=11)
    ax.set_ylabel("P-value (log scale)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in lam_vals])
    ax.set_title("(a) Statistical Significance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3(a) to: {output_path}")


def plot_effect_sizes(stat_results, output_path):
    """Figure 3(b): Bar chart of Cohen's d effect sizes vs λ."""
    lam_vals = [r["lam"] for r in stat_results if r["lam"] > 0]
    cohens_ds = [r["cohens_d"] for r in stat_results if r["lam"] > 0]

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#4CAF50" if d > 0.8 else "#FF9800" for d in cohens_ds]
    bars = ax.bar(
        [f"{v:.1f}" for v in lam_vals], cohens_ds, color=colors, alpha=0.85
    )

    for bar, d in zip(bars, cohens_ds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{d:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1.5,
               label="Large effect (d=0.8)")
    ax.set_xlabel("Calibration Strength λ", fontsize=11)
    ax.set_ylabel("Cohen's d", fontsize=11)
    ax.set_title("(b) Effect Size", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(cohens_ds) * 1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3(b) to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    results_cache_path = os.path.join(args.results_dir, "sweep_results.json")

    # --- Run or load inference ---
    if args.load_cached and os.path.exists(results_cache_path):
        print(f"Loading cached results from: {results_cache_path}")
        with open(results_cache_path, "r") as f:
            sweep_results = json.load(f)
    else:
        pac = PACInference(model_name=args.model_name, device=args.device)
        print(f"\nRunning PAC sweep over λ = {LAMBDA_VALUES}")
        print(f"Test queries: {len(TEST_QUERIES)}, Permutations: 24, Total inferences: {len(TEST_QUERIES) * 24 * len(LAMBDA_VALUES)}\n")
        sweep_results = pac.sweep_lambda(TEST_QUERIES, lambda_values=LAMBDA_VALUES)

        # Save results
        with open(results_cache_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nResults saved to: {results_cache_path}")

    # --- Print summary table ---
    print_results_table(sweep_results)

    # --- Statistical tests ---
    stat_results = run_statistical_tests_sweep(sweep_results)
    print("\nStatistical test results:")
    print(f"{'λ':>6} | {'t-test p':>12} | {'Wilcoxon p':>12} | {'Cohen d':>8}")
    print("-" * 50)
    for r in stat_results:
        if r["lam"] > 0:
            print(
                f"{r['lam']:>6.1f} | "
                f"{r['ttest_pvalue']:>12.2e} | "
                f"{r['wilcoxon_pvalue']:>12.2e} | "
                f"{r['cohens_d']:>8.2f}"
            )

    # --- Generate figures ---
    plot_accuracy_distribution(
        sweep_results,
        os.path.join(args.output_dir, "fig2a_accuracy_distribution.png"),
    )
    plot_accuracy_variance_tradeoff(
        sweep_results,
        os.path.join(args.output_dir, "fig2b_accuracy_variance_tradeoff.png"),
    )
    plot_pvalues(
        stat_results,
        os.path.join(args.output_dir, "fig3a_pvalues.png"),
    )
    plot_effect_sizes(
        stat_results,
        os.path.join(args.output_dir, "fig3b_effect_sizes.png"),
    )

    print("\nAll figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
