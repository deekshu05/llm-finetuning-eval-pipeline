from src.config import FineTuneConfig
from src.dataset import Example
from src.trainer import MockTrainingBackend, run_fine_tuning_job


def test_run_fine_tuning_job_with_mock_backend():
    config = FineTuneConfig(epochs=1, val_ratio=0.25)
    examples = [Example(prompt=str(i), completion=str(i)) for i in range(8)]

    result = run_fine_tuning_job(config, examples, backend=MockTrainingBackend())

    assert result.num_train_examples == 6
    assert result.num_val_examples == 2
    assert result.epochs == 1
    assert result.final_train_loss is not None


def test_run_fine_tuning_job_defaults_to_mock_backend():
    config = FineTuneConfig(epochs=2, val_ratio=0.0)
    examples = [Example(prompt=str(i), completion=str(i)) for i in range(4)]

    result = run_fine_tuning_job(config, examples)

    assert result.num_train_examples == 4
    assert result.num_val_examples == 0
