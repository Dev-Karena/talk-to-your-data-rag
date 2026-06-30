import pytest
from app.eval.tool_metrics import (
    calculate_intent_accuracy,
    calculate_tool_success_rate,
    calculate_end_to_end_accuracy,
    calculate_confusion_matrix,
    calculate_latency_stats,
)

def test_calculate_intent_accuracy():
    assert calculate_intent_accuracy(["RAG", "CALCULATOR"], ["RAG", "CALCULATOR"]) == 1.0
    assert calculate_intent_accuracy(["RAG", "CALCULATOR"], ["RAG", "DATETIME"]) == 0.5
    assert calculate_intent_accuracy([], []) == 0.0

def test_calculate_tool_success_rate():
    assert calculate_tool_success_rate([True, True]) == 1.0
    assert calculate_tool_success_rate([True, False]) == 0.5
    assert calculate_tool_success_rate([]) == 1.0

def test_calculate_end_to_end_accuracy():
    assert calculate_end_to_end_accuracy(["Answer is 425", "June 30"], ["425", "June"]) == 1.0
    assert calculate_end_to_end_accuracy(["Answer is 425", "June 30"], ["425", "July"]) == 0.5

def test_calculate_confusion_matrix():
    expected = ["rag", "calculator", "calculator"]
    actual = ["rag", "calculator", "datetime"]
    matrix = calculate_confusion_matrix(expected, actual)
    assert matrix["rag"]["rag"] == 1
    assert matrix["calculator"]["calculator"] == 1
    assert matrix["calculator"]["datetime"] == 1

def test_calculate_latency_stats():
    stats = calculate_latency_stats([10.0, 20.0, 30.0])
    assert stats["avg"] == 20.0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
