"""Unit tests for app.rag.chunker.

Covers chunk creation, stable chunk-id generation, metadata correctness, and
configurability of size/overlap. Uses small synthetic PageDocuments; no PDF or
external service required.
"""

from __future__ import annotations

from typing import List

import pytest

from app.config.settings import Settings
from app.rag.chunker import Chunk, chunk_pages
from app.rag.loader import PageDocument


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force small, predictable chunk settings for deterministic assertions.

    Patches ``get_settings`` used inside the chunker so tests don't depend on
    the developer's ``.env`` values.
    """
    test_settings = Settings(
        GROQ_API_KEY="test",
        CHUNK_SIZE=50,
        CHUNK_OVERLAP=10,
    )
    monkeypatch.setattr("app.rag.chunker.get_settings", lambda: test_settings)


def _page(text: str, page_number: int = 1, source: str = "doc.pdf") -> PageDocument:
    """Build a synthetic page for testing."""
    return PageDocument(source=source, page_number=page_number, text=text)


def test_empty_pages_produce_no_chunks() -> None:
    """An empty page list yields no chunks."""
    assert chunk_pages([], doc_hash="h") == []


def test_short_text_single_chunk() -> None:
    """Text shorter than the chunk size yields exactly one chunk."""
    chunks = chunk_pages([_page("hello world")], doc_hash="abc")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_long_text_splits_into_multiple_chunks() -> None:
    """Text longer than the chunk size is split into multiple chunks."""
    long_text = " ".join(f"word{i}" for i in range(80))
    chunks = chunk_pages([_page(long_text)], doc_hash="abc")
    assert len(chunks) > 1


def test_chunk_id_format_is_stable() -> None:
    """Chunk ids follow the ``{hash}::p{page}::c{index}`` convention."""
    chunks = chunk_pages([_page("hello world", page_number=3)], doc_hash="deadbeef")
    assert chunks[0].chunk_id == "deadbeef::p3::c0"


def test_chunk_ids_are_unique() -> None:
    """All generated chunk ids within a document are unique."""
    long_text = " ".join(f"token{i}" for i in range(100))
    chunks = chunk_pages([_page(long_text)], doc_hash="abc")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_metadata_carries_source_and_page() -> None:
    """Each chunk preserves its source document and page number."""
    chunks = chunk_pages(
        [_page("some content here", page_number=7, source="report.pdf")],
        doc_hash="abc",
    )
    meta = chunks[0].metadata()
    assert meta["source"] == "report.pdf"
    assert meta["page_number"] == 7
    assert meta["doc_hash"] == "abc"
    assert meta["chunk_id"] == chunks[0].chunk_id


def test_metadata_values_are_primitives() -> None:
    """Chroma requires flat primitive metadata; verify no nested types leak."""
    chunks = chunk_pages([_page("content")], doc_hash="abc")
    for value in chunks[0].metadata().values():
        assert isinstance(value, (str, int, float, bool))


def test_pages_are_chunked_independently() -> None:
    """Chunks never straddle pages; page numbers stay correct per chunk."""
    pages = [
        _page("alpha content one", page_number=1),
        _page("beta content two", page_number=2),
    ]
    chunks = chunk_pages(pages, doc_hash="abc")
    pages_seen = {c.page_number for c in chunks}
    assert pages_seen == {1, 2}


def test_returns_chunk_dataclass_instances() -> None:
    """The chunker returns typed Chunk objects."""
    chunks: List[Chunk] = chunk_pages([_page("hi there")], doc_hash="abc")
    assert all(isinstance(c, Chunk) for c in chunks)
