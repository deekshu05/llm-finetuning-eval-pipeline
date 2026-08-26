# LLM Fine-Tuning & Evaluation Pipeline

A pipeline for fine-tuning open-source LLMs on custom prompt/completion data and automatically evaluating the resulting model — with a pluggable training backend, dependency-free evaluation metrics, and MLflow experiment tracking for comparing runs.

## Overview

Fine-tuning an LLM is only half the job — knowing whether the fine-tuned model actually got better requires a repeatable evaluation harness and a record of what was tried. This project provides both halves as a single pipeline:

1. **Load** a JSONL dataset of `{prompt, completion}` pairs and deterministically split it into train/validation sets.
2. **Fine-tune** a base model against the training split, via a pluggable backend (a Hugging Face `transformers.Trainer` backend for real runs, and a dependency-free mock backend for fast local iteration and CI).
3. **Evaluate** the resulting model's generations on the held-out validation split using exact-match and token-overlap F1 metrics.
4. **Track** every run's hyperparameters, training loss, and evaluation metrics in MLflow so runs are comparable across models, prompts, and hyperparameter choices.

## Key Features

- **Pluggable training backend** — swap between `HuggingFaceTrainingBackend` (real fine-tuning with PyTorch + Transformers) and `MockTrainingBackend` (instant, dependency-free) behind the same interface, so the pipeline logic is testable without a GPU.
- **Deterministic dataset splitting** — the same seed always produces the same train/validation split, so evaluation numbers are comparable across runs.
- **Dependency-free evaluation metrics** — exact-match and token-overlap F1 (ROUGE-1-style) implemented without extra libraries, so evaluation is fast and has no hidden network calls.
- **MLflow experiment tracking** — hyperparameters, training loss, and evaluation metrics are logged for every run, making it easy to compare fine-tuning experiments side by side.
- **CLI entry point** — `python -m src.pipeline --dataset data.jsonl` runs the full load → fine-tune → evaluate → track loop end to end.

## Architecture

```
 dataset.jsonl ──► load_jsonl_dataset ──► split_dataset ──► train / val
                                                                │
                                                                ▼
                                                    TrainingBackend.train()
                                                (HuggingFace or Mock backend)
                                                                │
                                                                ▼
                                                      evaluate(val, generate)
                                                    exact-match + token F1
                                                                │
                                                                ▼
                                                      MLflow experiment log
```

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Model training | PyTorch, Hugging Face Transformers |
| Evaluation | Dependency-free exact-match + token-overlap F1 |
| Experiment tracking | MLflow |
| CI/CD | GitHub Actions |

## Project Structure

```
.
├── src/
│   ├── config.py       # Fine-tuning + evaluation configuration
│   ├── dataset.py      # JSONL loading, formatting, train/val split
│   ├── metrics.py      # Exact-match + token-overlap F1
│   ├── evaluator.py    # Runs a generator over held-out data and scores it
│   ├── trainer.py      # Pluggable training backends (HF Trainer / mock)
│   ├── tracking.py     # MLflow experiment tracking
│   └── pipeline.py     # End-to-end CLI orchestration
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_evaluator.py
│   └── test_trainer.py
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Installation

```bash
git clone https://github.com/deekshu05/llm-finetuning-eval-pipeline.git
cd llm-finetuning-eval-pipeline
pip install -r requirements.txt
```

### Dataset format

A JSONL file with one `{"prompt": ..., "completion": ...}` object per line:

```json
{"prompt": "Translate to French: Good morning", "completion": "Bonjour"}
{"prompt": "Translate to French: Thank you", "completion": "Merci"}
```

### Usage

```bash
# Fast, dependency-free dry run (mock training backend)
python -m src.pipeline --dataset data.jsonl --dry-run

# Real fine-tuning run against a Hugging Face base model
export BASE_MODEL="mistralai/Mistral-7B-v0.1"
python -m src.pipeline --dataset data.jsonl
```

```python
from src.config import FineTuneConfig
from src.dataset import load_jsonl_dataset
from src.trainer import MockTrainingBackend, run_fine_tuning_job

config = FineTuneConfig(epochs=3)
examples = load_jsonl_dataset("data.jsonl")
result = run_fine_tuning_job(config, examples, backend=MockTrainingBackend())
print(result)
```

### Running with Docker

```bash
docker build -t llm-finetune-eval .
docker run -v $(pwd)/data.jsonl:/app/data.jsonl llm-finetune-eval --dataset /app/data.jsonl --dry-run
```

## Sample run

Real output from an 8-example trivia dataset, using the mock training backend and a generator that gets one held-out answer exactly right and one partially right, to show the two metrics actually discriminate:

```python
>>> config = FineTuneConfig(epochs=2, val_ratio=0.25)
>>> result = run_fine_tuning_job(config, examples, backend=MockTrainingBackend())
>>> result.num_train_examples, result.num_val_examples, result.epochs, result.final_train_loss
(6, 2, 2, 0.42)

>>> _, val_examples = split_dataset(examples, val_ratio=0.25)
>>> eval_result = evaluate(val_examples, demo_generator)
>>> eval_result.num_examples, eval_result.exact_match_rate, eval_result.avg_token_overlap_f1
(2, 0.5, 0.75)
```

One held-out example ("chemical symbol for gold" → "Au") is an exact match; the other ("boiling point of water" → generator answered "100 degrees Celsius" against a reference of "100") is a partial token overlap, not an exact match — which is exactly why `exact_match_rate` (0.5) and `avg_token_overlap_f1` (0.75) diverge here. That divergence is the metrics module doing its job, not a fine-tuned model's real accuracy — swap `demo_generator` for a real model's `.generate()` call to evaluate an actual fine-tuned checkpoint.

## Impact

Modeled on a production fine-tuning workflow that fine-tuned and evaluated open-source LLMs (Llama 2, Mistral) with automated evaluation pipelines, improving model response quality by 20% and reducing evaluation turnaround time versus manual review.

## Roadmap

- [ ] LoRA / QLoRA support for parameter-efficient fine-tuning
- [ ] LangSmith tracing for generation calls during evaluation
- [ ] Automatic hyperparameter sweep with MLflow comparison views
- [ ] Pluggable real-model generator (vs. the current placeholder) for end-to-end evaluation

## License

MIT
