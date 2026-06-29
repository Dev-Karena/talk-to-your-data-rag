"""Unit tests for the BM25 store and tokenization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.rag.bm25_store import BM25Store, tokenize


def test_tokenize() -> None:
    assert tokenize("Hello World!") == ["hello", "world"]
    assert tokenize("RAG pipelines; fast & offline.") == ["rag", "pipelines", "fast", "offline"]
    assert tokenize("") == []
    assert tokenize(None) == []


@patch("app.rag.bm25_store.get_vector_store")
def test_bm25_store_sync_empty(mock_get_vector_store) -> None:
    mock_store = MagicMock()
    mock_store.list_sources.return_value = {}
    mock_store.get_all_chunks.return_value = ([], [])
    mock_get_vector_store.return_value = mock_store

    bm25_store = BM25Store()
    bm25_store.sync()
    assert bm25_store._bm25 is None
    assert bm25_store.search("test", 5) == []


@patch("app.rag.bm25_store.get_vector_store")
def test_bm25_store_indexing_and_search(mock_get_vector_store) -> None:
    mock_store = MagicMock()
    mock_store.list_sources.return_value = {"hash1": "source1"}
    mock_store.get_all_chunks.return_value = (
        [
            "B-tree index speeds up database queries.",
            "CPU scheduling and scheduling processes.",
            "Database indexing and SQL queries."
        ],
        ["chunk1", "chunk2", "chunk3"],
    )
    mock_get_vector_store.return_value = mock_store

    bm25_store = BM25Store()
    bm25_store.sync()
    assert bm25_store._bm25 is not None
    assert len(bm25_store._chunk_ids) == 3


    # Query for exact keywords
    results = bm25_store.search("B-tree queries", 5)
    assert len(results) > 0
    assert results[0][0] == "chunk1"
    assert results[0][1] > 0.0

    results_process = bm25_store.search("processes", 5)
    assert len(results_process) > 0
    assert results_process[0][0] == "chunk2"
