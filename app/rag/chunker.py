"""Document chunking.

Splits cleaned, per-page text into overlapping chunks suitable for embedding
and retrieval, using LangChain's ``RecursiveCharacterTextSplitter``. Each chunk
carries rich metadata (source, page, position, stable chunk id, document hash)
so that retrieved results can be cited precisely.

Why ``RecursiveCharacterTextSplitter``:
    It tries a prioritized list of separators (paragraph -> line -> sentence ->
    word) and only falls back to a harder split when a piece is still too
    large. This keeps semantically related text together far better than a
    naive fixed-width split.

Recommended values (see README for full rationale):
    * CHUNK_SIZE = 1000 characters — large enough for a coherent idea, small
      enough to keep embeddings focused and retrieval precise.
    * CHUNK_OVERLAP = 150 characters (~15%) — preserves context across chunk
      boundaries so a sentence split in two is not lost.

Usage:
    >>> from app.rag.chunker import chunk_pages
    >>> chunks = chunk_pages(pages, doc_hash="abc123")
    >>> chunks[0].chunk_id
    'abc123::p1::c0'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.rag.loader import PageDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A single embeddable text chunk with citation metadata.

    Attributes:
        chunk_id: Stable, unique identifier of the form
            ``{doc_hash}::p{page}::c{index}``. Used as the ChromaDB record id.
        text: The chunk's text content.
        source: Display name of the source document.
        page_number: 1-based page the chunk was derived from.
        doc_hash: SHA-256 content hash of the source document (dedupe key).
        chunk_index: 0-based index of this chunk within its page.
    """

    chunk_id: str
    text: str
    source: str
    page_number: int
    doc_hash: str
    chunk_index: int

    def metadata(self) -> Dict[str, object]:
        """Return a flat metadata dict for storage in the vector DB.

        ChromaDB metadata values must be primitives (str/int/float/bool), so
        this intentionally returns only flat, JSON-safe fields.
        """
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page_number": self.page_number,
            "doc_hash": self.doc_hash,
            "chunk_index": self.chunk_index,
        }


@dataclass
class _SplitterCache:
    """Holds a lazily-built splitter keyed by its (size, overlap) config."""

    key: tuple[int, int] | None = None
    splitter: RecursiveCharacterTextSplitter | None = field(default=None)


# Module-level cache so we don't rebuild the splitter on every call.
_cache = _SplitterCache()


def _get_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Return a splitter for the given config, rebuilding only when it changes."""
    key = (chunk_size, chunk_overlap)
    if _cache.key != key or _cache.splitter is None:
        _cache.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Prioritized separators: paragraph -> line -> sentence -> word ->
            # character. The splitter only descends when a piece is still too big.
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        _cache.key = key
    return _cache.splitter


def chunk_pages(pages: List[PageDocument], doc_hash: str) -> List[Chunk]:
    """Split a document's pages into metadata-tagged chunks.

    Chunk size and overlap are read from application settings, so the behavior
    is configurable via ``.env``.

    Args:
        pages: Cleaned pages produced by the loader/cleaner stages.
        doc_hash: SHA-256 content hash of the source document; embedded into
            each ``chunk_id`` and stored as metadata for deduplication and
            source tracking.

    Returns:
        A flat list of :class:`Chunk` objects across all pages, in reading
        order.
    """
    settings = get_settings()
    splitter = _get_splitter(settings.chunk_size, settings.chunk_overlap)

    chunks: List[Chunk] = []
    for page in pages:
        page_chunks = splitter.split_text(page.text)
        for index, text in enumerate(page_chunks):
            text = text.strip()
            if not text:
                continue
            chunk_id = f"{doc_hash}::p{page.page_number}::c{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=page.source,
                    page_number=page.page_number,
                    doc_hash=doc_hash,
                    chunk_index=index,
                )
            )

    logger.info(
        "Chunked '%s' into %d chunk(s) (size=%d, overlap=%d).",
        pages[0].source if pages else "<empty>",
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
