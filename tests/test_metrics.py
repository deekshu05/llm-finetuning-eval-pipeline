from src.metrics import exact_match, token_overlap_f1


def test_exact_match_true():
    assert exact_match("hello world", "hello world")


def test_exact_match_ignores_surrounding_whitespace():
    assert exact_match("  hello world  ", "hello world")


def test_exact_match_false():
    assert not exact_match("hello", "world")


def test_token_overlap_f1_perfect_match():
    assert token_overlap_f1("the cat sat", "the cat sat") == 1.0


def test_token_overlap_f1_no_overlap():
    assert token_overlap_f1("apple banana", "car truck") == 0.0


def test_token_overlap_f1_partial_overlap():
    score = token_overlap_f1("the cat sat on the mat", "the cat sat")
    assert 0 < score < 1.0
