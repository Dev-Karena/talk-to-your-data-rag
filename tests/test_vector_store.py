"""Unit tests for app.rag.vector_store.

Exercises the ChromaDB wrapper against a temporary, on-disk store using
deterministic stub embeddings (no model download, no network). Covers upsert
idempotency, similarity search + score conversion, deduplication, listing, and
clearing.
"""

from __future__ import annotations

from typing import List

import pytest

from app.config.settings import Settings
from app.rag.chunker import Chunk


def _make_chunk(chunk_id: str, text: str, doc_hash: str, page: int = 1) -> Chunk:
    """Build a Chunk with the given identity for storage tests."""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=f"{doc_hash}.pdf",
        page_number=page,
        doc_hash=doc_hash,
        chunk_index=0,
    )


@pytest.fixture
def store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Return a fresh VectorStore backed by a temporary directory.

    Patches settings so the store persists under ``tmp_path`` with a unique
    collection name, isolating each test from the real ``chroma_db/``.
    """
    test_settings = Settings(
        GROQ_API_KEY="test",
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME="test_collection",
    )
    monkeypatch.setattr("app.rag.vector_store.get_settings", lambda: test_settings)

    # Import after patching so the instance reads the patched settings.
    from app.rag.vector_store import VectorStore

    return VectorStore()


# Three orthogonal 3-D unit vectors: each is closest to itself.
_VEC_A: List[float] = [1.0, 0.0, 0.0]
_VEC_B: List[float] = [0.0, 1.0, 0.0]
_VEC_C: List[float] = [0.0, 0.0, 1.0]


def test_starts_empty(store) -> None:
    """A new store has zero chunks."""
    assert store.count() == 0


def test_add_and_count(store) -> None:
    """Adding chunks increases the count."""
    chunks = [_make_chunk("h1::p1::c0", "alpha", "h1")]
    store.add_chunks(chunks, [_VEC_A])
    assert store.count() == 1


def test_query_returns_most_similar(store) -> None:
    """A query returns the chunk whose embedding is closest."""
    chunks = [
        _make_chunk("h::p1::c0", "alpha", "h", page=1),
        _make_chunk("h::p2::c0", "beta", "h", page=2),
        _make_chunk("h::p3::c0", "gamma", "h", page=3),
    ]
    store.add_chunks(chunks, [_VEC_A, _VEC_B, _VEC_C])

    results = store.query(_VEC_B, top_k=1)
    assert len(results) == 1
    assert results[0].text == "beta"
    assert results[0].page_number == 2


def test_query_scores_in_unit_range(store) -> None:
    """Similarity scores are clamped to the [0, 1] range."""
    store.add_chunks([_make_chunk("h::p1::c0", "alpha", "h")], [_VEC_A])
    results = store.query(_VEC_A, top_k=1)
    assert 0.0 <= results[0].score <= 1.0
    # An identical vector should score very close to 1.0.
    assert results[0].score > 0.99


def test_query_on_empty_store_returns_empty(store) -> None:
    """Querying an empty store yields no results (and does not raise)."""
    assert store.query(_VEC_A, top_k=3) == []


def test_upsert_is_idempotent(store) -> None:
    """Re-adding the same chunk ids does not create duplicates."""
    chunks = [_make_chunk("h::p1::c0", "alpha", "h")]
    store.add_chunks(chunks, [_VEC_A])
    store.add_chunks(chunks, [_VEC_A])  # same id again
    assert store.count() == 1


def test_mismatched_lengths_raise(store) -> None:
    """A chunk/embedding count mismatch raises VectorStoreError."""
    from app.rag.vector_store import VectorStoreError

    chunks = [_make_chunk("h::p1::c0", "alpha", "h")]
    with pytest.raises(VectorStoreError):
        store.add_chunks(chunks, [_VEC_A, _VEC_B])  # 1 chunk, 2 vectors


def test_document_exists_dedup(store) -> None:
    """document_exists reflects whether a doc hash is present."""
    assert store.document_exists("h1") is False
    store.add_chunks([_make_chunk("h1::p1::c0", "alpha", "h1")], [_VEC_A])
    assert store.document_exists("h1") is True
    assert store.document_exists("other") is False


def test_list_sources_groups_by_document(store) -> None:
    """list_sources returns one entry per unique document hash."""
    store.add_chunks(
        [
            _make_chunk("h1::p1::c0", "a", "h1"),
            _make_chunk("h1::p2::c0", "b", "h1"),
            _make_chunk("h2::p1::c0", "c", "h2"),
        ],
        [_VEC_A, _VEC_B, _VEC_C],
    )
    sources = store.list_sources()
    assert set(sources.keys()) == {"h1", "h2"}


def test_clear_empties_store(store) -> None:
    """clear() removes all stored chunks."""
    store.add_chunks([_make_chunk("h1::p1::c0", "alpha", "h1")], [_VEC_A])
    assert store.count() == 1
    store.clear()
    assert store.count() == 0


def test_all_records_returns_ids_and_metadata(store) -> None:
    """all_records() exposes ids + metadata for diagnostics (read-only)."""
    store.add_chunks(
        [
            _make_chunk("h1::p1::c0", "a", "h1"),
            _make_chunk("h2::p1::c0", "b", "h2"),
        ],
        [_VEC_A, _VEC_B],
    )
    records = store.all_records()
    assert set(records.keys()) == {"ids", "metadatas"}
    assert set(records["ids"]) == {"h1::p1::c0", "h2::p1::c0"}
    assert len(records["metadatas"]) == 2
    assert all("doc_hash" in m for m in records["metadatas"])


def test_all_records_empty_store(store) -> None:
    """all_records() on an empty store returns empty aligned lists."""
    records = store.all_records()
    assert records == {"ids": [], "metadatas": []}
