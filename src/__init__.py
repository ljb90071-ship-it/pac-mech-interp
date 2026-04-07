"""
PAC-Mech-Interp: Mechanistic Evidence of Layer-Dependent Positional Bias in ICL.

This package provides tools for:
  - Constructing synthetic ICL datasets (data.py)
  - Extracting attention weights from GPT-2 (attention.py)
  - Position-Aware Calibration inference (pac.py)
  - Evaluation metrics and statistical tests (utils.py)
"""

from .data import SentimentICLDataset
from .attention import AttentionExtractor
from .pac import PACInference
from .utils import compute_metrics, run_statistical_tests

__all__ = [
    "SentimentICLDataset",
    "AttentionExtractor",
    "PACInference",
    "compute_metrics",
    "run_statistical_tests",
]
