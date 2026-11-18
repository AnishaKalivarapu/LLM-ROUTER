import pytest

from app.classifiers.heuristic import HeuristicClassifier
from app.schemas import Complexity

h = HeuristicClassifier()


@pytest.mark.parametrize("query", ["Hello!", "thanks", "What is the capital of Japan?", "Translate 'good morning' to Spanish"])
def test_simple(query):
    assert h.classify_sync(query).level == Complexity.SIMPLE


@pytest.mark.parametrize(
    "query",
    [
        "Design a distributed rate limiter across data centers and analyze the trade-offs between consistency and latency",
        "Implement an LRU cache from scratch with O(1) operations and explain the time complexity",
        "Debug this:\n```python\ndef f(x):\n    return x/0\n```",
    ],
)
def test_complex(query):
    assert h.classify_sync(query).level == Complexity.COMPLEX


def test_moderate():
    assert h.classify_sync("Summarize the pros and cons of remote work").level == Complexity.MODERATE


def test_confident_cases_have_high_confidence():
    assert h.classify_sync("Hello!").confidence >= 0.9


def test_borderline_has_low_confidence():
    # one medium keyword, short query -> lands near a threshold
    r = h.classify_sync("Explain how photosynthesis works")
    assert r.confidence < 0.7


def test_signals_are_reported():
    r = h.classify_sync("Hi")
    assert "greeting" in r.signals
