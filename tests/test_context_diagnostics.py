import pytest
from unittest.mock import MagicMock, patch
from app.rag.vector_store import RetrievedChunk
from scripts.context_diagnostics import tokenize, calculate_jaccard, run_diagnostics

def test_tokenize():
    text = "Virtual memory can affect database performance."
    tokens = tokenize(text)
    assert "virtual" in tokens
    assert "memory" in tokens
    assert "affect" in tokens
    assert "database" in tokens
    assert "performance" in tokens
    assert "does" not in tokens # Stopword

def test_calculate_jaccard():
    text1 = "This is a normal query for transaction execution."
    text2 = "This is a normal query for locking databases."
    sim = calculate_jaccard(text1, text2)
    assert 0.0 < sim < 1.0
    
    sim_ident = calculate_jaccard(text1, text1)
    assert sim_ident == 1.0

@patch("scripts.context_diagnostics.retrieve")
@patch("scripts.context_diagnostics.get_vector_store")
def test_run_diagnostics(mock_get_store, mock_retrieve):
    # Mock retrieved chunks
    c1 = RetrievedChunk(
        chunk_id="c1", doc_hash="h1", source="OS.pdf", page_number=86, chunk_index=0,
        text="Virtual memory is a feature of the operating system that allows execution. This system provides a mapping between virtual addresses used by programs and physical addresses in hardware memory.", score=0.89
    )
    c2 = RetrievedChunk(
        chunk_id="c2", doc_hash="h2", source="DBMS.pdf", page_number=94, chunk_index=0,
        text="Databases rely heavily on index page caching for query performance. Buffering blocks of database records in main memory prevents slow disk accesses and speeds up lock transactions.", score=0.85
    )
    c3 = RetrievedChunk(
        chunk_id="c3", doc_hash="h1", source="OS.pdf", page_number=87, chunk_index=1,
        text="allows execution of processes that are not completely in memory. This paging mechanism splits virtual address spaces into pages and transfers them to swap storage as needed.", score=0.83
    )
    c4 = RetrievedChunk(
        chunk_id="c4", doc_hash="h2", source="DBMS.pdf", page_number=95, chunk_index=1,
        text="performance is affected if virtual memory causes thrashing. High page fault rates lead to constant disk queue waits, halting CPU execution and query processing cycles.", score=0.81
    )
    mock_retrieve.return_value = [c1, c2, c3, c4]
    
    # Mock vector store
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store
    mock_store._collection.get.return_value = {
        "ids": ["c1", "c2", "c3", "c4"],
        "metadatas": [
            {"doc_hash": "h1", "page_number": 86, "chunk_index": 0, "source": "OS.pdf"},
            {"doc_hash": "h2", "page_number": 94, "chunk_index": 0, "source": "DBMS.pdf"},
            {"doc_hash": "h1", "page_number": 87, "chunk_index": 1, "source": "OS.pdf"},
            {"doc_hash": "h2", "page_number": 95, "chunk_index": 1, "source": "DBMS.pdf"}
        ]
    }
    
    diag = run_diagnostics("How does virtual memory affect database performance?")
    
    assert diag["query"] == "How does virtual memory affect database performance?"
    assert len(diag["chunks"]) == 4
    assert diag["unique_docs"] == 2
    assert diag["consecutive"] == 2
    assert diag["duplicates"] == 0
    assert diag["low_info"] == 0
    assert len(diag["opportunities"]) == 2
    assert "Merge OS p86+p87" in diag["opportunities"][0]
    assert "Merge DBMS p94+p95" in diag["opportunities"][1]
    assert diag["reduction_pct"] > 0
