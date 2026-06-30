import pytest
from app.eval.citation_metrics import calculate_citation_metrics

def test_citation_metrics_perfect():
    res = calculate_citation_metrics(
        cited_sources=["OS.pdf", "DBMS.pdf"],
        expected_sources=["OS.pdf", "DBMS.pdf"]
    )
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0

def test_citation_metrics_partial():
    # TP = 1 (OS.pdf)
    # FP = 1 (ML.pdf)
    # FN = 1 (DBMS.pdf)
    res = calculate_citation_metrics(
        cited_sources=["OS.pdf", "ML.pdf"],
        expected_sources=["OS.pdf", "DBMS.pdf"]
    )
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5

def test_citation_metrics_negative_match():
    # Correctly cited nothing for negative query
    res = calculate_citation_metrics(
        cited_sources=[],
        expected_sources=[]
    )
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0

def test_citation_metrics_negative_hallucinated():
    # Cited document when nothing was expected
    res = calculate_citation_metrics(
        cited_sources=["OS.pdf"],
        expected_sources=[]
    )
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0
