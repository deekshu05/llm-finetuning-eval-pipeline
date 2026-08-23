"""Fine-tuning job orchestration with a pluggable training backend."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import FineTuneConfig
from src.dataset import Example, split_dataset


@dataclass
class TrainingResult:
    base_model: str
    epochs: int
    num_train_examples: int
    num_val_examples: int
    final_train_loss: float | None = None


class TrainingBackend:
    """Minimal interface any fine-tuning backend must implement."""

    def train(
        self, config: FineTuneConfig, train_examples: list[Example], val_examples: list[Example]
    ) -> TrainingResult:
        raise NotImplementedError


class MockTrainingBackend(TrainingBackend):
    """Deterministic backend for local testing without GPUs or model downloads."""

    def train(
        self, config: FineTuneConfig, train_examples: list[Example], val_examples: list[Example]
    ) -> TrainingResult:
        return TrainingResult(
            base_model=config.base_model,
            epochs=config.epochs,
            num_train_examples=len(train_examples),
            num_val_examples=len(val_examples),
            final_train_loss=0.42,
        )


class HuggingFaceTrainingBackend(TrainingBackend):
    """Fine-tunes a Hugging Face causal LM using `transformers.Trainer`."""

    def train(
        self, config: FineTuneConfig, train_examples: list[Example], val_examples: list[Example]
    ) -> TrainingResult:
        import torch  # imported lazily -- heavy, optional dependency
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.base_model)

        class _TextDataset(Dataset):
            def __init__(self, examples: list[Example]):
                self.examples = examples

            def __len__(self) -> int:
                return len(self.examples)

            def __getitem__(self, idx: int) -> dict:
                encoding = tokenizer(
                    self.examples[idx].as_text(), truncation=True, padding="max_length", max_length=512
                )
                encoding["labels"] = encoding["input_ids"].copy()
                return {k: torch.tensor(v) for k, v in encoding.items()}

        args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=_TextDataset(train_examples),
            eval_dataset=_TextDataset(val_examples) if val_examples else None,
        )
        train_output = trainer.train()

        return TrainingResult(
            base_model=config.base_model,
            epochs=config.epochs,
            num_train_examples=len(train_examples),
            num_val_examples=len(val_examples),
            final_train_loss=train_output.training_loss,
        )


def run_fine_tuning_job(
    config: FineTuneConfig,
    examples: list[Example],
    backend: TrainingBackend | None = None,
) -> TrainingResult:
    """Split examples into train/val and run the given (or default mock) backend."""
    backend = backend or MockTrainingBackend()
    train_examples, val_examples = split_dataset(examples, val_ratio=config.val_ratio)
    return backend.train(config, train_examples, val_examples)
