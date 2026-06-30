"""Metrics for evaluating citation quality (Precision, Recall, F1)."""

from typing import List, Dict

def calculate_citation_metrics(cited_sources: List[str], expected_sources: List[str]) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1-score for cited documents against expected ground truth.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    """
    cleaned_cited = set(c.strip().lower() for c in cited_sources if c.strip())
    cleaned_expected = set(e.strip().lower() for e in expected_sources if e.strip())

    if not cleaned_expected:
        # If no sources are expected (negative queries), any citation is a false positive.
        if cleaned_cited:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not cleaned_cited:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(cleaned_cited & cleaned_expected)
    fp = len(cleaned_cited - cleaned_expected)
    fn = len(cleaned_expected - cleaned_cited)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
