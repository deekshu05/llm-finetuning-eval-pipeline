"""Evaluates a fine-tuned model's generations against reference completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.dataset import Example
from src.metrics import exact_match, token_overlap_f1

Generator = Callable[[str], str]


@dataclass
class EvalResult:
    exact_match_rate: float
    avg_token_overlap_f1: float
    num_examples: int


def evaluate(examples: list[Example], generate: Generator) -> EvalResult:
    """Run `generate` over each example's prompt and score against its completion."""
    if not examples:
        return EvalResult(exact_match_rate=0.0, avg_token_overlap_f1=0.0, num_examples=0)

    exact_matches = 0
    overlap_scores: list[float] = []
    for example in examples:
        prediction = generate(example.prompt)
        if exact_match(prediction, example.completion):
            exact_matches += 1
        overlap_scores.append(token_overlap_f1(prediction, example.completion))

    return EvalResult(
        exact_match_rate=round(exact_matches / len(examples), 4),
        avg_token_overlap_f1=round(sum(overlap_scores) / len(overlap_scores), 4),
        num_examples=len(examples),
    )
