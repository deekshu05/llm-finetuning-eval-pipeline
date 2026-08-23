"""Configuration for the LLM fine-tuning and evaluation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class FineTuneConfig:
    base_model: str = os.getenv("BASE_MODEL", "meta-llama/Llama-2-7b-hf")
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    epochs: int = int(os.getenv("EPOCHS", "3"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    val_ratio: float = float(os.getenv("VAL_RATIO", "0.1"))
    output_dir: str = os.getenv("OUTPUT_DIR", "./checkpoints")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "llm-fine-tuning")
