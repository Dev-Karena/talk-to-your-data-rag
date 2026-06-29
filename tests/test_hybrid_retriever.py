"""Unit tests for the hybrid retriever and RRF score fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.config.settings import Settings
from app.rag.vector_store import RetrievedChunk
from app.services.hybrid_retriever import hybrid_retrieve


def test_hybrid_retrieve_cosine_mode() -> None:
    settings = Settings()
    settings.hybrid_enabled = True
    settings.hybrid_relevance_mode = "cosine"
    settings.use_mmr = True

    chunk1 = RetrievedChunk(
        chunk_id="chunk1",
        text="B-tree index speeds up queries.",
        source="DBMS.pdf",
        page_number=1,
        chunk_index=0,
        doc_hash="hash1",
        score=0.8,
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk2",
        text="Operating system schedulers.",
        source="OS.pdf",
        page_number=1,
        chunk_index=0,
        doc_hash="hash2",
        score=0.4,
    )

    with patch("app.services.hybrid_retriever.get_settings", return_value=settings), \
         patch("app.services.hybrid_retriever.get_vector_store") as mock_get_vector_store, \
         patch("app.services.hybrid_retriever.get_bm25_store") as mock_get_bm25_store, \
         patch("app.services.hybrid_retriever.get_embedder") as mock_get_embedder:

        mock_store = MagicMock()
        mock_store.get_chunks_by_ids.return_value = [(chunk1, [1.0, 0.0]), (chunk2, [0.0, 1.0])]
        mock_store.query_candidates.return_value = [(chunk1, [1.0, 0.0])]
        mock_get_vector_store.return_value = mock_store

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [("chunk1", 10.0), ("chunk2", 5.0)]
        mock_get_bm25_store.return_value = mock_bm25

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [1.0, 0.0]
        mock_get_embedder.return_value = mock_emb

        results = hybrid_retrieve("B-tree", top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"


def test_hybrid_retrieve_fused_mode() -> None:
    settings = Settings()
    settings.hybrid_enabled = True
    settings.hybrid_relevance_mode = "fused"
    settings.use_mmr = True

    chunk1 = RetrievedChunk(
        chunk_id="chunk1",
        text="Text chunk 1",
        source="DBMS.pdf",
        page_number=1,
        chunk_index=0,
        doc_hash="hash1",
        score=0.8,
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk2",
        text="Text chunk 2",
        source="OS.pdf",
        page_number=1,
        chunk_index=0,
        doc_hash="hash2",
        score=0.4,
    )

    with patch("app.services.hybrid_retriever.get_settings", return_value=settings), \
         patch("app.services.hybrid_retriever.get_vector_store") as mock_get_vector_store, \
         patch("app.services.hybrid_retriever.get_bm25_store") as mock_get_bm25_store, \
         patch("app.services.hybrid_retriever.get_embedder") as mock_get_embedder:

        mock_store = MagicMock()
        mock_store.get_chunks_by_ids.return_value = [(chunk1, [1.0, 0.0]), (chunk2, [0.0, 1.0])]
        mock_store.query_candidates.return_value = [(chunk2, [0.0, 1.0])]
        mock_get_vector_store.return_value = mock_store

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [("chunk1", 20.0), ("chunk2", 1.0)]
        mock_get_bm25_store.return_value = mock_bm25

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [1.0, 0.0]
        mock_get_embedder.return_value = mock_emb

        results = hybrid_retrieve("some query", top_k=2)
        assert len(results) == 2
