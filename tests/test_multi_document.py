"""Multi-document ingestion & retrieval tests.

Proves the property that motivated this sprint: when several PDFs are indexed,
*all* of them are stored, none overwrites another, and retrieval can span more
than one document.

These tests are deterministic and fully offline — they use crafted stub
embeddings (no model download, no network), mirroring the style of
``test_vector_store.py``. Real-embedding, semantic end-to-end validation with
ML/OS/DBMS PDFs lives in ``scripts/validate_multidoc.py`` (run manually).
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from app.config.settings import Settings
from app.rag.chunker import Chunk


def _chunk(doc_hash: str, source: str, page: int, index: int, text: str) -> Chunk:
    """Build a Chunk with an id matching the production scheme."""
    return Chunk(
        chunk_id=f"{doc_hash}::p{page}::c{index}",
        text=text,
        source=source,
        page_number=page,
        doc_hash=doc_hash,
        chunk_index=index,
    )


# Three documents, two chunks each. Embeddings are crafted so that each
# document forms its own cluster (intra-doc similar, inter-doc dissimilar).
_DOCS = {
    "ml": ("ML.pdf", [(1, 0, "machine learning supervised"), (1, 1, "neural network training")]),
    "os": ("OS.pdf", [(1, 0, "operating system kernel"), (1, 1, "process scheduling memory")]),
    "db": ("DBMS.pdf", [(1, 0, "relational database sql"), (1, 1, "acid transactions index")]),
}
_EMB: Dict[str, List[float]] = {
    "machine learning supervised": [1.0, 0.0, 0.0],
    "neural network training": [0.98, 0.2, 0.0],
    "operating system kernel": [0.0, 1.0, 0.0],
    "process scheduling memory": [0.0, 0.98, 0.2],
    "relational database sql": [0.0, 0.0, 1.0],
    "acid transactions index": [0.2, 0.0, 0.98],
}


@pytest.fixture
def store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A fresh VectorStore backed by a temporary directory."""
    test_settings = Settings(
        GROQ_API_KEY="test",
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME="test_multidoc",
    )
    monkeypatch.setattr("app.rag.vector_store.get_settings", lambda: test_settings)
    from app.rag.vector_store import VectorStore

    return VectorStore()


def _add_doc(store, key: str) -> List[Chunk]:
    """Add one document's chunks (with their crafted embeddings) to the store."""
    source, rows = _DOCS[key]
    chunks = [_chunk(key, source, page, idx, text) for page, idx, text in rows]
    store.add_chunks(chunks, [_EMB[c.text] for c in chunks])
    return chunks


# ---- Storage: all documents persist -----------------------------------------
def test_all_documents_stored_and_listed(store) -> None:
    """Indexing three PDFs stores all three, not just the first."""
    for key in _DOCS:
        _add_doc(store, key)

    assert store.count() == 6  # 3 docs x 2 chunks
    assert set(store.list_sources().values()) == {"ML.pdf", "OS.pdf", "DBMS.pdf"}

    summary = store.document_chunk_counts()
    assert set(summary.keys()) == {"ml", "os", "db"}
    assert all(entry["chunk_count"] == 2 for entry in summary.values())


def test_incremental_add_never_overwrites_prior_docs(store) -> None:
    """Adding a new PDF preserves every previously indexed PDF."""
    _add_doc(store, "ml")
    assert set(store.list_sources().values()) == {"ML.pdf"}

    _add_doc(store, "os")  # later upload must not drop ML
    assert set(store.list_sources().values()) == {"ML.pdf", "OS.pdf"}

    _add_doc(store, "db")
    assert set(store.list_sources().values()) == {"ML.pdf", "OS.pdf", "DBMS.pdf"}
    assert store.count() == 6


def test_query_candidates_pairs_chunks_with_embeddings(store) -> None:
    """query_candidates returns each chunk alongside its stored vector."""
    for key in _DOCS:
        _add_doc(store, key)

    pairs = store.query_candidates([1.0, 0.0, 0.0], fetch_k=6)
    assert len(pairs) == 6
    for chunk, vector in pairs:
        assert len(vector) == 3  # embeddings round-tripped
        assert chunk.source in {"ML.pdf", "OS.pdf", "DBMS.pdf"}


# ---- Retrieval: MMR spans multiple documents --------------------------------
def test_mmr_selects_across_documents() -> None:
    """_mmr_select diversifies away from a single dominant document."""
    from app.rag.vector_store import RetrievedChunk
    from app.services.retriever import _mmr_select

    def rc(source: str, cid: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=cid, text=cid, source=source, page_number=1,
            chunk_index=0, doc_hash=source, score=0.0,
        )

    query = [1.0, 0.0, 0.0]
    # Three tightly-clustered ML candidates plus one each from OS/DBMS.
    candidates = [
        (rc("ML.pdf", "A1"), [0.97, 0.24, 0.0]),
        (rc("ML.pdf", "A2"), [0.95, 0.31, 0.0]),
        (rc("ML.pdf", "A3"), [0.93, 0.37, 0.0]),
        (rc("OS.pdf", "B1"), [0.0, 1.0, 0.0]),
        (rc("DBMS.pdf", "C1"), [0.0, 0.0, 1.0]),
    ]
    selected = _mmr_select(query, candidates, k=3, lambda_mult=0.5)

    assert selected[0].chunk_id == "A1"  # most relevant first
    # Pure top-3 by relevance would be all ML; MMR must include another doc.
    assert len({c.source for c in selected}) >= 2


def _patch_retriever(monkeypatch, store, settings) -> None:
    """Point the retriever at the temp store, a stub embedder, and settings."""
    class _StubEmbedder:
        def embed_query(self, text: str) -> List[float]:
            return [1.0, 0.0, 0.0]  # query lands in the ML cluster

    monkeypatch.setattr("app.services.retriever.get_settings", lambda: settings)
    monkeypatch.setattr("app.services.retriever.get_embedder", lambda: _StubEmbedder())
    monkeypatch.setattr("app.services.retriever.get_vector_store", lambda: store)


def test_retrieve_with_mmr_spans_documents(store, monkeypatch) -> None:
    """End-to-end retrieve(): MMR on pulls in more than one document."""
    # ML cluster has 3 chunks so a naive top-3 would be ML-only.
    ml_chunks = [
        _chunk("ml", "ML.pdf", 1, 0, "ml a"),
        _chunk("ml", "ML.pdf", 1, 1, "ml b"),
        _chunk("ml", "ML.pdf", 1, 2, "ml c"),
    ]
    store.add_chunks(ml_chunks, [[0.97, 0.24, 0.0], [0.95, 0.31, 0.0], [0.93, 0.37, 0.0]])
    store.add_chunks([_chunk("os", "OS.pdf", 1, 0, "os a")], [[0.0, 1.0, 0.0]])
    store.add_chunks([_chunk("db", "DBMS.pdf", 1, 0, "db a")], [[0.0, 0.0, 1.0]])

    settings = Settings(GROQ_API_KEY="test", USE_MMR=True, TOP_K=3, FETCH_K=10, MMR_LAMBDA=0.5)
    _patch_retriever(monkeypatch, store, settings)

    from app.services.retriever import retrieve

    results = retrieve("alpha question")
    assert len(results) == 3
    assert len({r.source for r in results}) >= 2  # not collapsed onto one doc


def test_retrieve_without_mmr_is_relevance_only(store, monkeypatch) -> None:
    """With MMR disabled, retrieve() returns the pure top-K (single cluster)."""
    ml_chunks = [
        _chunk("ml", "ML.pdf", 1, 0, "ml a"),
        _chunk("ml", "ML.pdf", 1, 1, "ml b"),
        _chunk("ml", "ML.pdf", 1, 2, "ml c"),
    ]
    store.add_chunks(ml_chunks, [[0.97, 0.24, 0.0], [0.95, 0.31, 0.0], [0.93, 0.37, 0.0]])
    store.add_chunks([_chunk("os", "OS.pdf", 1, 0, "os a")], [[0.0, 1.0, 0.0]])
    store.add_chunks([_chunk("db", "DBMS.pdf", 1, 0, "db a")], [[0.0, 0.0, 1.0]])

    settings = Settings(GROQ_API_KEY="test", USE_MMR=False, TOP_K=3, FETCH_K=10, MMR_LAMBDA=0.5)
    _patch_retriever(monkeypatch, store, settings)

    from app.services.retriever import retrieve

    results = retrieve("alpha question")
    assert len(results) == 3
    assert {r.source for r in results} == {"ML.pdf"}  # pure relevance, one doc
