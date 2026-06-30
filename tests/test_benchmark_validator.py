import pytest
from unittest.mock import MagicMock
from app.eval.dataset import Case, Benchmark
from app.eval.benchmark_validator import extract_keywords, verify_case_answerability, validate_benchmark

def test_extract_keywords():
    q = "How does a B-tree index speed up queries in SQL databases?"
    kw = extract_keywords(q)
    assert "b-tree" in kw
    assert "index" in kw
    assert "speed" in kw
    assert "queries" in kw
    assert "sql" in kw
    assert "databases" in kw
    assert "how" not in kw
    assert "does" not in kw
    assert "in" not in kw

def test_verify_case_answerability_negative():
    case = Case(id="neg-01", query="Out of bounds query", type="negative")
    store = MagicMock()
    verdict = verify_case_answerability(case, store)
    assert verdict["impossible"] is False
    assert verdict["status"] == "PASS"

def test_verify_case_answerability_missing_doc():
    case = Case(
        id="case-1",
        query="What is supervised learning?",
        type="single",
        expected_doc_hashes=["hash1"]
    )
    store = MagicMock()
    # Mock Chroma collection .get returning empty
    store._collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    
    verdict = verify_case_answerability(case, store)
    assert verdict["impossible"] is True
    assert verdict["status"] == "FAIL_MISSING_DOC"
    assert "hash1" in verdict["missing_docs"]

def test_verify_case_answerability_missing_keywords():
    case = Case(
        id="case-2",
        query="What is backpropagation?",
        type="single",
        expected_doc_hashes=["hash1"]
    )
    store = MagicMock()
    # Mock doc contains text without backpropagation
    store._collection.get.return_value = {
        "ids": ["c1"],
        "documents": ["This chunk discusses regression and classification models."],
        "metadatas": [{"doc_hash": "hash1", "page_number": 1, "chunk_index": 0, "source": "ML.pdf"}]
    }
    
    verdict = verify_case_answerability(case, store, threshold=0.5)
    assert verdict["impossible"] is True
    assert verdict["status"] == "FAIL_MISSING_KEYWORDS"
    assert "backpropagation" in verdict["missing_critical_terms"]

def test_verify_case_answerability_success():
    case = Case(
        id="case-3",
        query="What is supervised learning?",
        type="single",
        expected_doc_hashes=["hash1"]
    )
    store = MagicMock()
    store._collection.get.return_value = {
        "ids": ["c1"],
        "documents": ["Supervised learning uses labeled training datasets to train models."],
        "metadatas": [{"doc_hash": "hash1", "page_number": 1, "chunk_index": 0, "source": "ML.pdf"}]
    }
    
    verdict = verify_case_answerability(case, store, threshold=0.25)
    assert verdict["impossible"] is False
    assert verdict["status"] == "PASS"
    assert verdict["evidence_score"] == 1.0 # supervised, learning
