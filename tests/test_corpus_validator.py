import pytest
from unittest.mock import MagicMock
from app.eval.corpus_validator import audit_corpus

def test_audit_corpus_empty():
    store = MagicMock()
    store._collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    report = audit_corpus(store)
    assert report["total_chunks"] == 0
    assert report["is_healthy"] is True

def test_audit_corpus_anomalies():
    store = MagicMock()
    # Mock return with duplicates, short text, and missing metadata keys
    store._collection.get.return_value = {
        "ids": ["c1", "c2", "c3"],
        "documents": [
            "This is a normal chunk of text explaining transaction locking.",
            "Short",  # <50 characters
            "This is a normal chunk of text explaining transaction locking." # Duplicate of c1
        ],
        "metadatas": [
            {"chunk_id": "c1", "page_number": 1, "chunk_index": 0, "doc_hash": "h1", "source": "DBMS.pdf"},
            {"chunk_id": "c2", "page_number": 1, "chunk_index": 1, "doc_hash": "h1", "source": "DBMS.pdf"},
            {"chunk_id": "c3"}  # Missing source, doc_hash, page_number, chunk_index
        ]
    }
    
    report = audit_corpus(store)
    assert report["total_chunks"] == 3
    assert report["is_healthy"] is False
    
    anom = report["anomalies"]
    assert anom["short_chunks_count"] == 1
    assert anom["short_chunks"][0]["chunk_id"] == "c2"
    
    assert anom["duplicates_count"] == 1
    assert anom["duplicates"][0]["chunk_id_2"] == "c3"
    
    assert anom["missing_metadata_count"] == 1
    assert anom["missing_metadata"][0]["chunk_id"] == "c3"
    assert "source" in anom["missing_metadata"][0]["missing_keys"]
