"""
pac.py
------
Position-Aware Calibration (PAC) for In-Context Learning.

Implements the PAC inference method described in Section 4 of the paper.

PAC calibrates the output logits by subtracting a position-dependent bias
estimated from a content-free query:

    L_PAC(y) = L(y) - λ · L_cf(y)

where:
  - L(y)    = logit for label y given the real query
  - L_cf(y) = logit for label y given the content-free query ("N/A")
  - λ ∈ [0, 1] = calibration strength (tunable hyperparameter)

At λ = 0, PAC reduces to standard ICL (no calibration).
At λ = 1.0, PAC is mathematically equivalent to Contextual Calibration (CC)
from Zhao et al. (2021).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .data import CONTENT_FREE_INPUT, LABEL_SPACE, SentimentICLDataset


class PACInference:
    """
    Position-Aware Calibration (PAC) inference engine.

    This class wraps a GPT-2 model and provides methods for:
      1. Standard ICL inference (λ = 0).
      2. PAC inference with a tunable calibration strength λ.
      3. Batch evaluation across all permutations and multiple test queries.

    Args:
        model_name (str): HuggingFace model identifier. Default: 'gpt2'.
        device (str): Device to run inference on. Default: 'cpu'.
        label_space (List[str]): The set of possible labels. Default: ['Positive', 'Negative'].
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        label_space: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.label_space = label_space or LABEL_SPACE

        print(f"Loading {model_name} tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading {model_name} model (eager attention)...")
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        self.model.to(device)
        self.model.eval()

        self.dataset = SentimentICLDataset()

    def _get_label_token_ids(self) -> Dict[str, int]:
        """
        Map each label string to the token ID of its first subword token.

        Following the convention of Zhao et al. (2021), we prepend a space
        before each label to match GPT-2's tokenization of mid-sentence words.
        """
        return {
            label: self.tokenizer.encode(
                " " + label, add_special_tokens=False
            )[0]
            for label in self.label_space
        }

    def get_logits(self, prompt: str) -> Dict[str, float]:
        """
        Run a forward pass and return the logit for each label.

        Args:
            prompt: A full ICL prompt string ending with 'Sentiment:'.

        Returns:
            Dict mapping label string to its raw logit value.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        label_token_ids = self._get_label_token_ids()
        return {label: last_logits[tid].item() for label, tid in label_token_ids.items()}

    def predict_standard(
        self,
        demonstrations: List[Tuple[str, str]],
        query_text: str,
    ) -> str:
        """
        Standard ICL prediction (no calibration, λ = 0).

        Args:
            demonstrations: Ordered list of (text, label) pairs.
            query_text: The test input text.

        Returns:
            The predicted label string.
        """
        prompt = self.dataset.build_prompt(demonstrations, query_text)
        logits = self.get_logits(prompt)
        return max(logits, key=logits.get)

    def predict_pac(
        self,
        demonstrations: List[Tuple[str, str]],
        query_text: str,
        lam: float,
    ) -> str:
        """
        PAC prediction with calibration strength λ.

        The calibrated logit is:
            L_PAC(y) = L(y) - λ · L_cf(y)

        Args:
            demonstrations: Ordered list of (text, label) pairs.
            query_text: The test input text.
            lam: Calibration strength λ ∈ [0, 1].

        Returns:
            The predicted label string.
        """
        assert 0.0 <= lam <= 1.0, f"λ must be in [0, 1], got {lam}"

        # Real query logits
        prompt = self.dataset.build_prompt(demonstrations, query_text)
        logits = self.get_logits(prompt)

        if lam == 0.0:
            return max(logits, key=logits.get)

        # Content-free query logits (positional bias estimate)
        cf_prompt = self.dataset.build_content_free_prompt(demonstrations)
        cf_logits = self.get_logits(cf_prompt)

        # PAC calibration: L_PAC(y) = L(y) - λ · L_cf(y)
        pac_logits = {
            label: logits[label] - lam * cf_logits[label]
            for label in self.label_space
        }

        return max(pac_logits, key=pac_logits.get)

    def evaluate_all_permutations(
        self,
        test_queries: List[Tuple[str, str]],
        lam: float,
    ) -> Dict:
        """
        Evaluate PAC across all K! permutations of the demonstration pool
        and all test queries.

        This reproduces the experimental setup in Section 5.1:
          - K = 4 demonstrations (2 positive, 2 negative)
          - All 24 permutations
          - 10 test queries
          - 240 total inferences per λ setting

        Args:
            test_queries: List of (text, true_label) pairs.
            lam: Calibration strength λ.

        Returns:
            A dict with keys:
              - 'per_permutation_accuracy': List of per-permutation mean accuracies.
              - 'mean_accuracy': Mean accuracy across all permutations and queries.
              - 'worst_case_accuracy': Minimum per-permutation accuracy.
              - 'std_dev': Standard deviation of per-permutation accuracies.
              - 'all_predictions': Nested list [permutation][query] of predicted labels.
        """
        all_permutations = self.dataset.get_all_permutations()
        per_perm_accuracies = []
        all_predictions = []

        for perm in all_permutations:
            perm_correct = 0
            perm_preds = []
            for query_text, true_label in test_queries:
                pred = self.predict_pac(perm, query_text, lam)
                perm_preds.append(pred)
                if pred == true_label:
                    perm_correct += 1
            perm_acc = perm_correct / len(test_queries)
            per_perm_accuracies.append(perm_acc)
            all_predictions.append(perm_preds)

        per_perm_accuracies = np.array(per_perm_accuracies)

        return {
            "per_permutation_accuracy": per_perm_accuracies.tolist(),
            "mean_accuracy": float(per_perm_accuracies.mean()),
            "worst_case_accuracy": float(per_perm_accuracies.min()),
            "std_dev": float(per_perm_accuracies.std()),
            "all_predictions": all_predictions,
            "lam": lam,
        }

    def sweep_lambda(
        self,
        test_queries: List[Tuple[str, str]],
        lambda_values: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Sweep over a range of λ values and collect results for each.

        Args:
            test_queries: List of (text, true_label) pairs.
            lambda_values: List of λ values to evaluate. Defaults to
                           [0.0, 0.1, 0.3, 0.5, 0.7, 1.0].

        Returns:
            List of result dicts (one per λ value), as returned by
            `evaluate_all_permutations`.
        """
        if lambda_values is None:
            lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

        results = []
        for lam in lambda_values:
            print(f"Evaluating λ = {lam:.1f} ...")
            result = self.evaluate_all_permutations(test_queries, lam)
            results.append(result)
            print(
                f"  Mean Acc: {result['mean_accuracy']:.3f} | "
                f"Worst-Case: {result['worst_case_accuracy']:.3f} | "
                f"Std Dev: {result['std_dev']:.3f}"
            )

        return results
