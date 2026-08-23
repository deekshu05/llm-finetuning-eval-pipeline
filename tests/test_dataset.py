import json
from pathlib import Path

import pytest

from src.dataset import Example, load_jsonl_dataset, split_dataset


def test_load_jsonl_dataset(tmp_path: Path):
    file_path = tmp_path / "data.jsonl"
    records = [
        {"prompt": "2+2=", "completion": "4"},
        {"prompt": "3+3=", "completion": "6"},
    ]
    file_path.write_text("\n".join(json.dumps(r) for r in records))

    examples = load_jsonl_dataset(file_path)

    assert len(examples) == 2
    assert examples[0].prompt == "2+2="
    assert examples[0].completion == "4"


def test_load_jsonl_dataset_raises_on_missing_field(tmp_path: Path):
    file_path = tmp_path / "bad.jsonl"
    file_path.write_text(json.dumps({"prompt": "only prompt"}))

    with pytest.raises(ValueError):
        load_jsonl_dataset(file_path)


def test_split_dataset_is_deterministic():
    examples = [Example(prompt=str(i), completion=str(i)) for i in range(20)]

    train1, val1 = split_dataset(examples, val_ratio=0.2, seed=1)
    train2, val2 = split_dataset(examples, val_ratio=0.2, seed=1)

    assert [e.prompt for e in train1] == [e.prompt for e in train2]
    assert len(val1) == 4
    assert len(train1) == 16


def test_example_as_text_format():
    example = Example(prompt="Hello", completion="World")
    text = example.as_text()
    assert "Hello" in text and "World" in text
