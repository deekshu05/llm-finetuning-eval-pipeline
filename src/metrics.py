"""Lightweight, dependency-free evaluation metrics for generated text."""

from __future__ import annotations


def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip() == reference.strip()


def token_overlap_f1(prediction: str, reference: str) -> float:
    """Token-level F1 overlap between prediction and reference (ROUGE-1-style)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens))
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)
