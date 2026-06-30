"""Metrics module for tool calling evaluation."""

from typing import Dict, List, Sequence

def calculate_intent_accuracy(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Calculate the accuracy of intent routing."""
    if not expected:
        return 0.0
    matches = sum(1 for exp, act in zip(expected, actual) if exp.lower() == act.lower())
    return matches / len(expected)

def calculate_tool_success_rate(successes: Sequence[bool]) -> float:
    """Calculate the execution success rate of tool executions."""
    if not successes:
        return 1.0 # Vacuously successful if no tools are executed
    return sum(1 for s in successes if s) / len(successes)

def calculate_end_to_end_accuracy(answers: Sequence[str], expected_substrings: Sequence[str]) -> float:
    """Calculate accuracy based on whether generated answers contain expected substrings."""
    if not answers:
        return 0.0
    correct = 0
    for ans, exp in zip(answers, expected_substrings):
        if not exp:
            correct += 1
        elif exp.lower() in ans.lower():
            correct += 1
    return correct / len(answers)

def calculate_confusion_matrix(expected: Sequence[str], actual: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Compute a confusion matrix dictionary for intent classification."""
    matrix: Dict[str, Dict[str, int]] = {}
    all_categories = sorted(list(set(list(expected) + list(actual))))
    
    for exp_cat in all_categories:
        matrix[exp_cat] = {act_cat: 0 for act_cat in all_categories}
        
    for exp, act in zip(expected, actual):
        matrix[exp][act] += 1
        
    return matrix

def calculate_latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    """Compute statistics for execution latency."""
    if not latencies:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": sum(latencies) / len(latencies),
        "min": min(latencies),
        "max": max(latencies)
    }
