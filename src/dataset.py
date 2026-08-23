"""Dataset loading, formatting, and splitting for supervised fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    prompt: str
    completion: str

    def as_text(self) -> str:
        """Render as a single training string with a clear prompt/completion boundary."""
        return f"### Prompt:\n{self.prompt}\n\n### Completion:\n{self.completion}"


def load_jsonl_dataset(path: str | Path) -> list[Example]:
    """Load a JSONL file of {"prompt": ..., "completion": ...} records."""
    examples: list[Example] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record or "completion" not in record:
                raise ValueError(f"line {line_no}: record missing 'prompt' or 'completion'")
            examples.append(Example(prompt=record["prompt"], completion=record["completion"]))
    return examples


def split_dataset(
    examples: list[Example], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Deterministically split examples into (train, val) sets."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")

    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    return train, val
