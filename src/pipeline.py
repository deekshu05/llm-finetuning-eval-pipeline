"""End-to-end CLI: load dataset -> fine-tune -> evaluate -> track.

Usage:
    python -m src.pipeline --dataset data.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import sys

from src.config import FineTuneConfig
from src.dataset import load_jsonl_dataset, split_dataset
from src.evaluator import evaluate
from src.trainer import MockTrainingBackend, run_fine_tuning_job
from src.tracking import log_eval_result, log_training_result, mlflow_run


def _placeholder_generate(prompt: str) -> str:
    """Stand-in generator; swap for real model inference in production."""
    return "mock generation"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM fine-tuning and evaluation pipeline")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset of {prompt, completion} pairs")
    parser.add_argument("--dry-run", action="store_true", help="Use the mock training backend instead of a real model")
    args = parser.parse_args()

    config = FineTuneConfig()
    examples = load_jsonl_dataset(args.dataset)

    backend = MockTrainingBackend() if args.dry_run else None
    with mlflow_run(config):
        training_result = run_fine_tuning_job(config, examples, backend=backend)
        log_training_result(training_result)

        _, val_examples = split_dataset(examples, val_ratio=config.val_ratio)
        eval_result = evaluate(val_examples, _placeholder_generate)
        log_eval_result(eval_result)

    print(
        f"Trained on {training_result.num_train_examples} examples, "
        f"evaluated on {eval_result.num_examples} -- "
        f"exact_match={eval_result.exact_match_rate}, f1={eval_result.avg_token_overlap_f1}"
    )


if __name__ == "__main__":
    sys.exit(main())
