"""MLflow experiment tracking for fine-tuning runs and evaluations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mlflow

from src.config import FineTuneConfig
from src.evaluator import EvalResult
from src.trainer import TrainingResult


@contextmanager
def mlflow_run(config: FineTuneConfig) -> Iterator[None]:
    mlflow.set_experiment(config.mlflow_experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": config.base_model,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
            }
        )
        yield


def log_training_result(result: TrainingResult) -> None:
    metrics = {
        "num_train_examples": result.num_train_examples,
        "num_val_examples": result.num_val_examples,
    }
    if result.final_train_loss is not None:
        metrics["final_train_loss"] = result.final_train_loss
    mlflow.log_metrics(metrics)


def log_eval_result(result: EvalResult) -> None:
    mlflow.log_metrics(
        {
            "exact_match_rate": result.exact_match_rate,
            "avg_token_overlap_f1": result.avg_token_overlap_f1,
            "num_eval_examples": result.num_examples,
        }
    )
