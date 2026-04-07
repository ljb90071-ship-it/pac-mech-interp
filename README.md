# PAC-Mech-Interp: Mechanistic Evidence of Layer-Dependent Positional Bias in In-Context Learning

[![Paper](https://img.shields.io/badge/Paper-ICML%202026%20Submission-blue)](https://anonymous.4open.science/r/pac-mech-interp)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

This repository contains the official code for the paper:

> **Mechanistic Evidence of Layer-Dependent Positional Bias in In-Context Learning**
> Anonymous Authors. *Under review at ICML 2026.*

## Overview

Large Language Models (LLMs) exhibit remarkable in-context learning (ICL) capabilities, but are notoriously sensitive to the permutation of demonstrations. This repository provides:

1. **Mechanistic Analysis**: Code for extracting and visualizing layer-wise attention weights from GPT-2 during ICL, revealing a complex, layer-dependent positional bias.
2. **PAC Intervention**: Implementation of Position-Aware Calibration (PAC), a lightweight, training-free inference intervention that corrects for this structural bias.
3. **Experiments**: Scripts to reproduce all results and figures from the paper.

### Key Findings

- **Contrary to the common assumption of uniform recency bias**, our layer-wise analysis reveals that:
  - Early layers (e.g., L0) exhibit **recency bias** (Position 4 receives ~37% of attention mass).
  - Middle and late layers (L2–L11) are dominated by **strong primacy bias** (Position 1 receives 52–75% of attention mass).
- **Net effect**: The farthest demonstration (Position 1) receives a disproportionate **55.1%** of total attention mass, far exceeding the uniform expectation of 25%.
- **PAC** significantly improves mean accuracy from **58.3% to 83.3%** (p < 0.001, Cohen's d = 1.58).

## Repository Structure

```
pac-mech-interp/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py              # Synthetic sentiment dataset construction
│   ├── attention.py         # Attention weight extraction from GPT-2
│   ├── pac.py               # Position-Aware Calibration (PAC) implementation
│   └── utils.py             # Utility functions (metrics, statistical tests)
├── experiments/
│   ├── run_attention_analysis.py   # Reproduce Figure 1 (mechanistic analysis)
│   ├── run_pac_experiments.py      # Reproduce Figures 2 & 3 (PAC effectiveness)
│   └── run_all.sh                  # Run all experiments end-to-end
├── notebooks/
│   └── demo.ipynb           # Interactive demo notebook
└── figures/                 # Output directory for generated figures
```

## Installation

```bash
git clone https://anonymous.4open.science/r/pac-mech-interp
cd pac-mech-interp
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- Transformers >= 4.30 (Hugging Face)
- See `requirements.txt` for the full list.

## Quick Start

### 1. Mechanistic Analysis (Figure 1)

Extract and visualize layer-wise attention weights from GPT-2 during ICL:

```bash
python experiments/run_attention_analysis.py
```

This script will:
- Load GPT-2 (124M) with `eager` attention implementation to access raw attention weights.
- Construct a 4-shot ICL prompt for binary sentiment classification.
- Extract attention weights from the final query token to each demonstration region across all 12 layers and 12 heads.
- Generate the attention heatmap (Figure 1a) and the aggregated bar chart (Figure 1b).

### 2. PAC Experiments (Figures 2 & 3)

Reproduce the main PAC effectiveness results:

```bash
python experiments/run_pac_experiments.py
```

This script will:
- Construct K=4 demonstrations (2 positive, 2 negative) and evaluate all K!=24 permutations across 10 test queries (240 inferences per λ setting).
- Sweep λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} and measure mean accuracy, worst-case accuracy, and standard deviation.
- Run paired t-tests and Wilcoxon signed-rank tests against the baseline (λ=0).
- Generate Figures 2 and 3.

### 3. Run Everything

```bash
bash experiments/run_all.sh
```

## Method: Position-Aware Calibration (PAC)

Standard ICL predicts the label `y` that maximizes `P(y | x_1, y_1, ..., x_K, y_K, x_q)`. Due to layer-dependent positional biases, the positions of demonstrations heavily influence the output logits.

PAC introduces a **content-free query** `x_cf` (e.g., the string `"N/A"`) to isolate the positional bias. The calibrated logit is:

```
L_PAC(y) = L(y) - λ · L_cf(y)
```

where:
- `L(y)` is the original logit for label `y` given the real query.
- `L_cf(y)` is the logit for the content-free query, capturing the model's inherent positional bias.
- `λ ∈ [0, 1]` is a tunable calibration strength parameter.

**Relationship to Contextual Calibration (CC):** Standard CC (Zhao et al., 2021) is mathematically equivalent to PAC at `λ = 1.0`. PAC generalizes CC by introducing `λ` as a tunable parameter, enabling a principled trade-off between peak accuracy (optimal at `λ ≈ 0.7`) and worst-case robustness (optimal at `λ = 1.0`).

## Reproducing Paper Results

| Method | Mean Acc. | Worst-Case Acc. | Std. Dev. |
|--------|-----------|-----------------|-----------|
| Baseline ICL (λ=0) | 58.3% | 50.0% | 0.131 |
| PAC (λ=0.1) | ~63% | ~50% | ~0.128 |
| PAC (λ=0.3) | ~72% | ~55% | ~0.126 |
| PAC (λ=0.5) | ~79% | ~57% | ~0.124 |
| PAC (λ=0.7) | **83.3%** | 58.3% | 0.124 |
| PAC / CC (λ=1.0) | 82.9% | **60.0%** | **0.121** |

*All improvements over baseline are statistically significant (p < 0.001).*

## Citation

```bibtex
@inproceedings{anonymous2026pac,
  title     = {Mechanistic Evidence of Layer-Dependent Positional Bias in In-Context Learning},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Under review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

We thank the authors of [Contextual Calibration](https://arxiv.org/abs/2102.09690) (Zhao et al., 2021) and [PEARL](https://arxiv.org/abs/2311.09558) (Chen et al., 2025) for their foundational work on ICL calibration.
