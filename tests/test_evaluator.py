from src.dataset import Example
from src.evaluator import evaluate


def test_evaluate_perfect_generator():
    examples = [Example(prompt="2+2=", completion="4"), Example(prompt="3+3=", completion="6")]

    def perfect_generate(prompt: str) -> str:
        return {"2+2=": "4", "3+3=": "6"}[prompt]

    result = evaluate(examples, perfect_generate)

    assert result.exact_match_rate == 1.0
    assert result.num_examples == 2


def test_evaluate_empty_examples():
    result = evaluate([], lambda p: "anything")
    assert result.num_examples == 0
    assert result.exact_match_rate == 0.0


def test_evaluate_partial_correctness():
    examples = [Example(prompt="2+2=", completion="4"), Example(prompt="3+3=", completion="6")]

    def half_right_generate(prompt: str) -> str:
        return "4" if prompt == "2+2=" else "wrong"

    result = evaluate(examples, half_right_generate)

    assert result.exact_match_rate == 0.5
