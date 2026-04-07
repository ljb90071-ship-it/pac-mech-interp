"""
data.py
-------
Synthetic binary sentiment classification dataset for ICL experiments.

Following the experimental setup in Section 5.1 of the paper:
  - Binary labels: "Positive" / "Negative"
  - K = 4 demonstrations per prompt (2 positive, 2 negative)
  - All K! = 24 permutations are evaluated
  - 10 test queries are used, yielding 240 inferences per λ setting
"""

import itertools
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Sentiment examples (positive and negative)
# ---------------------------------------------------------------------------

POSITIVE_EXAMPLES: List[Tuple[str, str]] = [
    ("The movie was absolutely fantastic and I loved every moment.", "Positive"),
    ("What a wonderful experience, I highly recommend it to everyone.", "Positive"),
    ("The acting was superb and the storyline was deeply moving.", "Positive"),
    ("I was thoroughly impressed by the quality of the performance.", "Positive"),
    ("This is one of the best films I have seen in years.", "Positive"),
    ("The characters were compelling and the dialogue was sharp.", "Positive"),
]

NEGATIVE_EXAMPLES: List[Tuple[str, str]] = [
    ("The film was a complete waste of time and utterly boring.", "Negative"),
    ("I was deeply disappointed by the poor writing and direction.", "Negative"),
    ("The plot made no sense and the acting was unconvincing.", "Negative"),
    ("I found the whole experience tedious and forgettable.", "Negative"),
    ("This movie was a disaster from start to finish.", "Negative"),
    ("The characters were flat and the story was painfully predictable.", "Negative"),
]

# Test queries (ground truth labels are used only for evaluation)
TEST_QUERIES: List[Tuple[str, str]] = [
    ("The cinematography was breathtaking and the score was beautiful.", "Positive"),
    ("I regret spending money on this dreadful production.", "Negative"),
    ("An uplifting story that left me smiling for hours.", "Positive"),
    ("The pacing was dreadful and I nearly fell asleep.", "Negative"),
    ("Brilliant performances all around, a true cinematic gem.", "Positive"),
    ("Poorly executed and devoid of any emotional depth.", "Negative"),
    ("A heartwarming tale that exceeded all my expectations.", "Positive"),
    ("Completely uninspired and a chore to sit through.", "Negative"),
    ("The director's vision was clear and the result was stunning.", "Positive"),
    ("A forgettable mess that fails on every level.", "Negative"),
]

LABEL_SPACE: List[str] = ["Positive", "Negative"]
CONTENT_FREE_INPUT: str = "N/A"


class SentimentICLDataset:
    """
    Constructs ICL prompts for binary sentiment classification.

    Each prompt consists of K demonstrations followed by a test query.
    The demonstrations are drawn from a fixed pool of positive and negative
    examples (2 positive + 2 negative for K=4), and all K! permutations
    are enumerated for evaluation.

    Args:
        k (int): Number of demonstrations per prompt. Must be even. Default: 4.
        n_pos_demo (int): Number of positive demonstrations. Default: 2.
        n_neg_demo (int): Number of negative demonstrations. Default: 2.
        seed (int): Random seed for reproducibility. Default: 42.
    """

    def __init__(
        self,
        k: int = 4,
        n_pos_demo: int = 2,
        n_neg_demo: int = 2,
        seed: int = 42,
    ):
        assert k == n_pos_demo + n_neg_demo, (
            f"k ({k}) must equal n_pos_demo ({n_pos_demo}) + n_neg_demo ({n_neg_demo})"
        )
        self.k = k
        self.n_pos_demo = n_pos_demo
        self.n_neg_demo = n_neg_demo
        self.seed = seed

        # Fixed demonstration pool (first n_pos + n_neg examples)
        self.demonstrations: List[Tuple[str, str]] = (
            POSITIVE_EXAMPLES[:n_pos_demo] + NEGATIVE_EXAMPLES[:n_neg_demo]
        )
        self.test_queries: List[Tuple[str, str]] = TEST_QUERIES

    def format_demonstration(self, text: str, label: str) -> str:
        """Format a single demonstration as 'Review: <text>\nSentiment: <label>'."""
        return f"Review: {text}\nSentiment: {label}"

    def format_query(self, text: str) -> str:
        """Format a test query (label omitted)."""
        return f"Review: {text}\nSentiment:"

    def build_prompt(
        self,
        demonstrations: List[Tuple[str, str]],
        query_text: str,
    ) -> str:
        """
        Build a full ICL prompt from an ordered list of demonstrations and a query.

        Args:
            demonstrations: Ordered list of (text, label) pairs.
            query_text: The test input text.

        Returns:
            A formatted prompt string.
        """
        demo_strs = [
            self.format_demonstration(text, label)
            for text, label in demonstrations
        ]
        query_str = self.format_query(query_text)
        return "\n\n".join(demo_strs) + "\n\n" + query_str

    def build_content_free_prompt(
        self,
        demonstrations: List[Tuple[str, str]],
    ) -> str:
        """
        Build a content-free ICL prompt using 'N/A' as the query.

        This is used by PAC to estimate the positional bias.

        Args:
            demonstrations: Ordered list of (text, label) pairs.

        Returns:
            A formatted content-free prompt string.
        """
        return self.build_prompt(demonstrations, CONTENT_FREE_INPUT)

    def get_all_permutations(self) -> List[List[Tuple[str, str]]]:
        """
        Enumerate all K! permutations of the demonstration pool.

        Returns:
            A list of K! orderings, each being a list of (text, label) tuples.
        """
        return [list(perm) for perm in itertools.permutations(self.demonstrations)]

    def get_demo_region_boundaries(
        self,
        prompt: str,
        tokenizer,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Identify the token index ranges corresponding to each demonstration
        in the tokenized prompt. Used for attention weight aggregation.

        Args:
            prompt: The full prompt string.
            tokenizer: A HuggingFace tokenizer.

        Returns:
            A dict mapping demonstration position (1-indexed, 1=farthest) to
            (start_token_idx, end_token_idx) tuples (exclusive end).
        """
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        full_text = tokenizer.decode(input_ids)

        boundaries = {}
        demo_strs = [
            self.format_demonstration(text, label)
            for text, label in self.demonstrations
        ]

        # Tokenize each prefix up to the end of each demonstration
        current_pos = 0
        for i, demo_str in enumerate(demo_strs):
            # Find the token span for this demonstration
            prefix_up_to_demo = "\n\n".join(demo_strs[: i + 1])
            prefix_tokens = tokenizer(prefix_up_to_demo, return_tensors="pt")[
                "input_ids"
            ][0]
            end_idx = len(prefix_tokens)

            if i == 0:
                start_idx = 0
            else:
                prefix_up_to_prev = "\n\n".join(demo_strs[:i])
                prev_tokens = tokenizer(prefix_up_to_prev, return_tensors="pt")[
                    "input_ids"
                ][0]
                start_idx = len(prev_tokens)

            # Position 1 = farthest (first in prompt), Position K = closest
            boundaries[i + 1] = (start_idx, end_idx)
            current_pos = end_idx

        return boundaries
