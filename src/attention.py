"""
attention.py
------------
Attention weight extraction from GPT-2 during ICL.

This module implements the mechanistic analysis described in Section 3 of the
paper. We use GPT-2 (124M) with the 'eager' attention implementation to access
raw per-head attention weights, then aggregate them to compute the normalized
attention mass from the final query token to each demonstration region.

Key findings reproduced:
  - L0 exhibits recency bias (Position 4 receives ~37% of attention mass)
  - L2–L11 exhibit strong primacy bias (Position 1 receives 52–75%)
  - Net aggregated attention mass: Pos1=0.551, Pos2=0.127, Pos3=0.142, Pos4=0.180
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class AttentionExtractor:
    """
    Extracts and aggregates attention weights from GPT-2 during ICL.

    The model is loaded with `attn_implementation='eager'` to ensure that
    raw attention weight tensors are returned in the model outputs.

    Args:
        model_name (str): HuggingFace model identifier. Default: 'gpt2'.
        device (str): Device to run inference on ('cpu' or 'cuda'). Default: 'cpu'.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name} tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading {model_name} model (eager attention)...")
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            output_attentions=True,
        )
        self.model.to(device)
        self.model.eval()

        self.n_layers = self.model.config.n_layer   # 12 for gpt2
        self.n_heads = self.model.config.n_head     # 12 for gpt2

        print(
            f"Model loaded: {n_layers} layers, {n_heads} heads."
            .replace("n_layers", str(self.n_layers))
            .replace("n_heads", str(self.n_heads))
        )

    def tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize a prompt and return input tensors on the correct device."""
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def forward_with_attentions(
        self, prompt: str
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a forward pass and return logits and per-layer attention weights.

        Args:
            prompt: The full ICL prompt string.

        Returns:
            logits: Tensor of shape (vocab_size,) for the last token.
            attentions: List of L tensors, each of shape (n_heads, seq_len, seq_len).
        """
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (batch=1, n_heads, seq_len, seq_len)
        attentions = [attn[0] for attn in outputs.attentions]  # remove batch dim
        logits = outputs.logits[0, -1, :]  # logits for the last (query) token
        return logits, attentions

    def get_demo_token_spans(
        self,
        demonstrations: List[Tuple[str, str]],
        tokenizer,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Compute token index spans for each demonstration in the prompt.

        Demonstrations are numbered 1 (farthest/first in prompt) to K (closest).

        Args:
            demonstrations: Ordered list of (text, label) pairs as they appear
                            in the prompt.
            tokenizer: The GPT-2 tokenizer.

        Returns:
            Dict mapping position index (1=farthest) to (start, end) token spans.
        """
        spans = {}
        sep = "\n\n"

        for i in range(len(demonstrations)):
            text, label = demonstrations[i]
            demo_str = f"Review: {text}\nSentiment: {label}"

            # Build prefix up to and including this demonstration
            prefix_parts = []
            for j in range(i + 1):
                t, l = demonstrations[j]
                prefix_parts.append(f"Review: {t}\nSentiment: {l}")
            prefix = sep.join(prefix_parts)

            end_idx = len(tokenizer(prefix, return_tensors="pt")["input_ids"][0])

            if i == 0:
                start_idx = 0
            else:
                prev_parts = []
                for j in range(i):
                    t, l = demonstrations[j]
                    prev_parts.append(f"Review: {t}\nSentiment: {l}")
                prev_prefix = sep.join(prev_parts)
                start_idx = len(
                    tokenizer(prev_prefix, return_tensors="pt")["input_ids"][0]
                )

            spans[i + 1] = (start_idx, end_idx)

        return spans

    def compute_normalized_attention_mass(
        self,
        attentions: List[torch.Tensor],
        demo_spans: Dict[int, Tuple[int, int]],
        seq_len: int,
    ) -> np.ndarray:
        """
        Compute normalized attention mass from the final query token to each
        demonstration region, for each layer.

        The attention mass for demonstration at position p in layer l is:
            mass[l, p] = mean over heads of sum of attention weights
                         from the last token to all tokens in the demo span.

        The masses are then normalized across positions so they sum to 1.

        Args:
            attentions: List of L tensors of shape (n_heads, seq_len, seq_len).
            demo_spans: Dict mapping position (1-indexed) to (start, end) spans.
            seq_len: Total sequence length.

        Returns:
            attention_mass: np.ndarray of shape (n_layers, K) where K = len(demo_spans).
                            Values are normalized to sum to 1 across positions per layer.
        """
        n_layers = len(attentions)
        k = len(demo_spans)
        attention_mass = np.zeros((n_layers, k))

        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (n_heads, seq_len, seq_len)
            # Attention from the last token (query position) to all others
            query_attn = layer_attn[:, -1, :]  # (n_heads, seq_len)

            for pos_idx, (pos, (start, end)) in enumerate(sorted(demo_spans.items())):
                # Sum attention over tokens in this demonstration's span
                demo_attn = query_attn[:, start:end].sum(dim=-1)  # (n_heads,)
                # Average over heads
                attention_mass[layer_idx, pos_idx] = demo_attn.mean().item()

            # Normalize across positions so they sum to 1
            total = attention_mass[layer_idx].sum()
            if total > 0:
                attention_mass[layer_idx] /= total

        return attention_mass

    def extract_attention_for_prompt(
        self,
        prompt: str,
        demonstrations: List[Tuple[str, str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline: tokenize, forward pass, extract and aggregate attention.

        Args:
            prompt: The full ICL prompt string.
            demonstrations: Ordered list of (text, label) pairs.

        Returns:
            layer_attention_mass: np.ndarray of shape (n_layers, K).
                                  Normalized attention mass per layer per position.
            aggregated_mass: np.ndarray of shape (K,).
                             Mean normalized attention mass across all layers and heads.
        """
        inputs = self.tokenize(prompt)
        seq_len = inputs["input_ids"].shape[1]

        logits, attentions = self.forward_with_attentions(prompt)

        demo_spans = self.get_demo_token_spans(demonstrations, self.tokenizer)

        layer_attention_mass = self.compute_normalized_attention_mass(
            attentions, demo_spans, seq_len
        )

        # Aggregate across layers: mean over all layers
        aggregated_mass = layer_attention_mass.mean(axis=0)

        return layer_attention_mass, aggregated_mass

    def get_label_logits(
        self,
        prompt: str,
        label_space: List[str],
    ) -> Dict[str, float]:
        """
        Get the model's output logits for each label token.

        For each label in label_space, we take the logit of the first token
        of the label string (following the convention in Zhao et al., 2021).

        Args:
            prompt: The full ICL prompt string (ending with 'Sentiment:').
            label_space: List of label strings, e.g. ['Positive', 'Negative'].

        Returns:
            Dict mapping label string to its logit value.
        """
        logits, _ = self.forward_with_attentions(prompt)

        label_logits = {}
        for label in label_space:
            # Tokenize the label and take the first token's ID
            label_token_ids = self.tokenizer.encode(" " + label, add_special_tokens=False)
            label_token_id = label_token_ids[0]
            label_logits[label] = logits[label_token_id].item()

        return label_logits
