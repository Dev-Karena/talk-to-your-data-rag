"""Ingestion pipeline orchestrator.

Ties the RAG ingestion stages together into a single entry point the UI can
call to index a document:

    validate -> hash -> (dedupe check) -> load -> clean -> chunk -> embed -> store

Deduplication: a document's SHA-256 content hash is checked against the vector
store first. If it already exists, indexing is skipped entirely — satisfying
the "avoid re-indexing existing documents" requirement.

Usage:
    >>> from app.rag.pipeline import ingest_document
    >>> result = ingest_document("report.pdf", pdf_bytes)
    >>> result.status
    <IngestStatus.INDEXED: 'indexed'>
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.config.settings import get_settings
from app.rag.chunker import chunk_pages
from app.rag.cleaner import clean_text
from app.rag.embeddings import get_embedder
from app.rag.loader import PDFLoadError, PageDocument, load_pdf
from app.rag.vector_store import get_vector_store
from app.utils.logger import get_logger
from app.utils.validators import compute_content_hash, validate_pdf

logger = get_logger(__name__)

# Soft threshold: above this many chunks for a single document we log a warning
# (very large in-limit PDFs index slowly and use more memory). Not a hard limit —
# size is capped at validation time; this is observability for pathological docs.
_LARGE_DOC_CHUNK_WARNING = 1500


class IngestStatus(str, Enum):
    """Outcome categories for a document ingestion attempt."""

    INDEXED = "indexed"        # Successfully chunked, embedded, and stored.
    SKIPPED = "skipped"        # Already present in the store (deduplicated).
    REJECTED = "rejected"      # Failed validation (bad type/size/empty).
    FAILED = "failed"          # An error occurred during processing.


@dataclass(frozen=True)
class IngestResult:
    """Result of ingesting a single document.

    Attributes:
        source: Document display name.
        status: One of :class:`IngestStatus`.
        doc_hash: Content hash of the document (empty if validation failed).
        chunk_count: Number of chunks stored (0 unless ``INDEXED``).
        message: Human-readable summary suitable for display in the UI.
    """

    source: str
    status: IngestStatus
    doc_hash: str
    chunk_count: int
    message: str


def _persist_to_disk(source_name: str, data: bytes, doc_hash: str) -> Path:
    """Write the uploaded bytes to the documents directory.

    The filename is prefixed with a short hash slice to avoid collisions
    between different files that share a name.

    Args:
        source_name: Original upload filename.
        data: Raw file bytes.
        doc_hash: Content hash (used for a unique on-disk name).

    Returns:
        The path the file was written to.
    """
    settings = get_settings()
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(source_name).name  # strip any path components (traversal guard)
    target = settings.documents_dir / f"{doc_hash[:12]}_{safe_name}"
    target.write_bytes(data)
    return target


def ingest_document(source_name: str, data: bytes) -> IngestResult:
    """Validate, process, and index a single PDF document.

    Args:
        source_name: Original filename of the uploaded document.
        data: Raw bytes of the uploaded file.

    Returns:
        An :class:`IngestResult` describing the outcome. This function does not
        raise for expected failure modes (validation, parse errors); those are
        reported via the result's ``status`` and ``message`` instead.
    """
    # 1. Validate (security: type, size, magic bytes, non-empty).
    validation = validate_pdf(source_name, data)
    if not validation.is_valid:
        return IngestResult(
            source=source_name,
            status=IngestStatus.REJECTED,
            doc_hash="",
            chunk_count=0,
            message=validation.reason,
        )

    # 2. Hash the content (dedupe key + stable chunk ids).
    doc_hash = compute_content_hash(data)
    store = get_vector_store()

    # 3. Deduplicate: skip if this exact content is already indexed.
    try:
        if store.document_exists(doc_hash):
            logger.info("Skipping '%s' — already indexed (hash %s).", source_name, doc_hash[:12])
            return IngestResult(
                source=source_name,
                status=IngestStatus.SKIPPED,
                doc_hash=doc_hash,
                chunk_count=0,
                message=f"'{source_name}' is already indexed — skipped.",
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Dedupe check failed for '%s': %s", source_name, exc)
        return IngestResult(
            source=source_name,
            status=IngestStatus.FAILED,
            doc_hash=doc_hash,
            chunk_count=0,
            message=f"Could not check existing index: {exc}",
        )

    # 4. Parse from a TEMP file first; only persist to documents/ after the PDF
    #    parses and chunks successfully. This keeps corrupt/unusable uploads out
    #    of the documents directory so they can't poison a later re-index.
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(data)

        # Load (from the temp file)
        pages = load_pdf(tmp_path, source_name=source_name)

        # Clean (page by page; drop pages that become empty after cleaning)
        cleaned_pages = [
            PageDocument(source=p.source, page_number=p.page_number, text=cleaned)
            for p in pages
            if (cleaned := clean_text(p.text))
        ]
        if not cleaned_pages:
            raise PDFLoadError("Document contained no usable text after cleaning.")

        # Chunk
        chunks = chunk_pages(cleaned_pages, doc_hash=doc_hash)
        if not chunks:
            raise PDFLoadError("Chunking produced no chunks.")

        # Observability for unusually large in-limit documents (Row 9).
        if len(chunks) > _LARGE_DOC_CHUNK_WARNING:
            logger.warning(
                "Large document '%s': %d chunks across %d page(s) — indexing may "
                "be slow and memory-intensive.",
                source_name, len(chunks), len(cleaned_pages),
            )

        # The document is valid: persist the original bytes for re-indexing.
        _persist_to_disk(source_name, data, doc_hash)

        # Embed
        embedder = get_embedder()
        embeddings = embedder.embed_documents([c.text for c in chunks])

        # Store
        store.add_chunks(chunks, embeddings)

    except PDFLoadError as exc:
        logger.warning("Ingestion of '%s' failed: %s", source_name, exc)
        return IngestResult(
            source=source_name,
            status=IngestStatus.FAILED,
            doc_hash=doc_hash,
            chunk_count=0,
            message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - catch-all so the UI never crashes
        logger.exception("Unexpected error ingesting '%s'.", source_name)
        return IngestResult(
            source=source_name,
            status=IngestStatus.FAILED,
            doc_hash=doc_hash,
            chunk_count=0,
            message=f"Unexpected error: {exc}",
        )
    finally:
        # Always remove the scratch parse file, success or failure.
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Could not remove temp file '%s'.", tmp_path)

    logger.info("Indexed '%s': %d chunk(s).", source_name, len(chunks))
    return IngestResult(
        source=source_name,
        status=IngestStatus.INDEXED,
        doc_hash=doc_hash,
        chunk_count=len(chunks),
        message=f"Indexed '{source_name}' into {len(chunks)} chunk(s).",
    )
