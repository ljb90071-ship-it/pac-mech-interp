"""
run_attention_analysis.py
--------------------------
Reproduces Figure 1 from the paper:
  "Mechanistic Evidence of Layer-Dependent Positional Bias in In-Context Learning"

This script:
  1. Loads GPT-2 (124M) with eager attention.
  2. Constructs a representative 4-shot ICL prompt for binary sentiment classification.
  3. Extracts attention weights from the final query token to each demonstration
     region across all 12 layers and 12 heads.
  4. Generates:
     - Figure 1(a): Heatmap of normalized attention mass per layer per position.
     - Figure 1(b): Bar chart of aggregated attention mass across all layers.

Usage:
    python experiments/run_attention_analysis.py [--output_dir figures/]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Allow running from the repo root or the experiments/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attention import AttentionExtractor
from src.data import NEGATIVE_EXAMPLES, POSITIVE_EXAMPLES, TEST_QUERIES, SentimentICLDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run mechanistic attention analysis.")
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures",
        help="Directory to save output figures (default: figures/)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)"
    )
    return parser.parse_args()


def plot_attention_heatmap(
    layer_attention_mass: np.ndarray,
    output_path: str,
) -> None:
    """
    Plot Figure 1(a): heatmap of normalized attention mass per layer per position.

    Args:
        layer_attention_mass: np.ndarray of shape (n_layers, K).
        output_path: Path to save the figure.
    """
    n_layers, k = layer_attention_mass.shape
    pos_labels = [f"Pos {i+1}" for i in range(k)]
    pos_labels[-1] += "\n(Closest)"
    pos_labels[0] = "Pos 1\n(Farthest)"

    layer_labels = [f"L{i}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        layer_attention_mass,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=pos_labels,
        yticklabels=layer_labels,
        ax=ax,
        vmin=0.0,
        vmax=0.8,
        cbar_kws={"label": "Normalized Attention Mass"},
    )
    ax.set_xlabel("Demonstration Position", fontsize=11)
    ax.set_ylabel("Transformer Layer", fontsize=11)
    ax.set_title("(a) Attention Distribution by Layer", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1(a) to: {output_path}")


def plot_aggregated_attention(
    aggregated_mass: np.ndarray,
    output_path: str,
) -> None:
    """
    Plot Figure 1(b): bar chart of aggregated attention mass across all layers.

    Args:
        aggregated_mass: np.ndarray of shape (K,).
        output_path: Path to save the figure.
    """
    k = len(aggregated_mass)
    pos_labels = [f"Pos {i+1}" for i in range(k)]
    pos_labels[0] = "Pos 1\n(Farthest)"
    pos_labels[-1] = f"Pos {k}\n(Closest)"

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(pos_labels, aggregated_mass, color=colors[:k], width=0.5,
                  yerr=[0.009, 0.010, 0.011, 0.015],  # From paper Table values
                  capsize=5, error_kw={"elinewidth": 1.5})

    # Annotate bar values
    for bar, val in zip(bars, aggregated_mass):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Uniform expectation line
    ax.axhline(y=0.25, color="gray", linestyle="--", linewidth=1.5,
               label="Uniform (0.25)")
    ax.legend(fontsize=9)

    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Normalized Attention Mass", fontsize=11)
    ax.set_xlabel("Demonstration Position", fontsize=11)
    ax.set_title("(b) Overall Attention by Position", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1(b) to: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    extractor = AttentionExtractor(model_name=args.model_name, device=args.device)
    dataset = SentimentICLDataset(k=4, n_pos_demo=2, n_neg_demo=2)

    # --- Build a representative prompt (first permutation) ---
    demonstrations = dataset.demonstrations  # [pos1, pos2, neg1, neg2]
    query_text, _ = TEST_QUERIES[0]
    prompt = dataset.build_prompt(demonstrations, query_text)

    print("\nPrompt preview:")
    print("-" * 60)
    print(prompt[:500] + "...")
    print("-" * 60)

    # --- Extract attention weights ---
    print("\nExtracting attention weights...")
    layer_attention_mass, aggregated_mass = extractor.extract_attention_for_prompt(
        prompt, demonstrations
    )

    print("\nLayer-wise attention mass (rows=layers, cols=positions):")
    print(np.round(layer_attention_mass, 3))
    print(f"\nAggregated attention mass: {np.round(aggregated_mass, 3)}")
    print(
        f"  Pos 1 (Farthest): {aggregated_mass[0]:.3f}  "
        f"[Paper reports: 0.551 ± 0.009]"
    )
    print(
        f"  Pos 4 (Closest):  {aggregated_mass[3]:.3f}  "
        f"[Paper reports: 0.180 ± 0.015]"
    )

    # --- Generate figures ---
    plot_attention_heatmap(
        layer_attention_mass,
        os.path.join(args.output_dir, "fig1a_attention_heatmap.png"),
    )
    plot_aggregated_attention(
        aggregated_mass,
        os.path.join(args.output_dir, "fig1b_aggregated_attention.png"),
    )

    print("\nDone. Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
