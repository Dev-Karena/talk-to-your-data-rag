"""Sprint 2 reliability tests for the ingestion pipeline.

Covers the ingestion-side failure scenarios from the Phase 1 audit:

    * Scenario 1 — empty PDF upload          -> REJECTED
    * Scenario 2 — corrupted PDF upload      -> FAILED, and NO file left on disk
    * Scenario 3 — unsupported file type     -> REJECTED
    * Scenario 8 — duplicate upload          -> SKIPPED (no re-index)
    * Scenario 9 — large document            -> warning logged, still INDEXED
    + persist-after-parse: a good PDF is persisted; a bad one is not.

The pipeline's collaborators (vector store, embedder, settings) are stubbed so
these run fully offline with an isolated documents directory. PDF parsing uses
the real loader where it matters (corrupted input).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pytest

from app.rag import pipeline
from app.rag.chunker import Chunk
from app.rag.loader import PageDocument
from app.utils.validators import compute_content_hash

# Minimal bytes that pass the magic-byte check (start with the PDF header).
_VALID_HEADER = b"%PDF-1.7\n%stub content for tests\n"


class _FakeStore:
    """In-memory stand-in for the vector store (dedup + add only)."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.add_calls: int = 0

    def document_exists(self, doc_hash: str) -> bool:
        return doc_hash in self.hashes

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        self.add_calls += 1
        if chunks:
            self.hashes.add(chunks[0].doc_hash)


class _FakeEmbedder:
    """Deterministic stub embedder (no model load, no network)."""

    name = "fake"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class _FakeSettings:
    def __init__(self, documents_dir: Path) -> None:
        self.documents_dir = documents_dir


@pytest.fixture
def wired(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Wire the pipeline to isolated, in-memory collaborators.

    Returns the fake store and the isolated documents directory so tests can
    assert on persisted files.
    """
    docs_dir = tmp_path / "documents"
    store = _FakeStore()
    monkeypatch.setattr(pipeline, "get_vector_store", lambda: store)
    monkeypatch.setattr(pipeline, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(pipeline, "get_settings", lambda: _FakeSettings(docs_dir))
    return store, docs_dir


def _list_pdfs(docs_dir: Path) -> list[Path]:
    return list(docs_dir.glob("*")) if docs_dir.exists() else []


# ---- Scenario 1: empty PDF ---------------------------------------------------
def test_empty_pdf_rejected(wired) -> None:
    store, docs_dir = wired
    result = pipeline.ingest_document("empty.pdf", b"")
    assert result.status is pipeline.IngestStatus.REJECTED
    assert "empty" in result.message.lower()
    assert store.add_calls == 0
    assert _list_pdfs(docs_dir) == []  # nothing persisted


# ---- Scenario 3: unsupported file type --------------------------------------
def test_unsupported_type_rejected(wired) -> None:
    store, docs_dir = wired
    # Valid PDF bytes but a non-.pdf name -> rejected on extension.
    result = pipeline.ingest_document("notes.txt", _VALID_HEADER)
    assert result.status is pipeline.IngestStatus.REJECTED
    assert store.add_calls == 0
    assert _list_pdfs(docs_dir) == []


def test_renamed_non_pdf_rejected(wired) -> None:
    store, docs_dir = wired
    # .pdf extension but not actually a PDF -> rejected on magic bytes.
    result = pipeline.ingest_document("fake.pdf", b"just plain text, not a pdf")
    assert result.status is pipeline.IngestStatus.REJECTED
    assert "header" in result.message.lower()
    assert _list_pdfs(docs_dir) == []


# ---- Scenario 2: corrupted PDF (persist-after-parse) ------------------------
def test_corrupted_pdf_fails_and_leaves_no_file(wired) -> None:
    """A file with a PDF header but an unparseable body fails AND is not
    persisted to the documents directory (no orphan to poison re-index)."""
    store, docs_dir = wired
    corrupt = _VALID_HEADER + b"\x00\xff\xfe not a real pdf body \x00\x01"
    result = pipeline.ingest_document("corrupt.pdf", corrupt)

    assert result.status is pipeline.IngestStatus.FAILED
    assert store.add_calls == 0
    # Persist-after-parse guarantee: the bad file is NOT in documents/.
    assert _list_pdfs(docs_dir) == []


# ---- Scenario 8: duplicate upload -------------------------------------------
def test_duplicate_upload_skipped(wired) -> None:
    store, docs_dir = wired
    # Pre-seed the store with this content's hash to simulate "already indexed".
    store.hashes.add(compute_content_hash(_VALID_HEADER))

    result = pipeline.ingest_document("dup.pdf", _VALID_HEADER)
    assert result.status is pipeline.IngestStatus.SKIPPED
    assert store.add_calls == 0  # no re-index work performed


# ---- Happy path + Scenario 9: large-document warning ------------------------
def _stub_loader(monkeypatch: pytest.MonkeyPatch, text: str = "Hello world. " * 50) -> None:
    """Make load_pdf return one page of real text (skip actual PDF parsing)."""
    def _fake_load(file_path, source_name):  # noqa: ANN001
        return [PageDocument(source=source_name, page_number=1, text=text)]
    monkeypatch.setattr(pipeline, "load_pdf", _fake_load)


def test_valid_pdf_indexed_and_persisted(wired, monkeypatch: pytest.MonkeyPatch) -> None:
    store, docs_dir = wired
    _stub_loader(monkeypatch)
    result = pipeline.ingest_document("good.pdf", _VALID_HEADER)

    assert result.status is pipeline.IngestStatus.INDEXED
    assert result.chunk_count > 0
    assert store.add_calls == 1
    # A valid document IS persisted, under a hash-prefixed name.
    persisted = _list_pdfs(docs_dir)
    assert len(persisted) == 1
    assert persisted[0].name.endswith("_good.pdf")


def test_large_document_logs_warning(wired, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """Scenario 9: an unusually large document logs a warning but still indexes."""
    store, docs_dir = wired
    _stub_loader(monkeypatch)
    # Force the warning path without generating a huge PDF.
    monkeypatch.setattr(pipeline, "_LARGE_DOC_CHUNK_WARNING", 0)

    # The app's loggers set propagate=False; enable it so caplog (which listens
    # on the root logger) can capture this module's warning.
    pipeline_logger = logging.getLogger("app.rag.pipeline")
    monkeypatch.setattr(pipeline_logger, "propagate", True)

    with caplog.at_level(logging.WARNING, logger="app.rag.pipeline"):
        result = pipeline.ingest_document("big.pdf", _VALID_HEADER)

    assert result.status is pipeline.IngestStatus.INDEXED
    assert any("large document" in r.getMessage().lower() for r in caplog.records)
