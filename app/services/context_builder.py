"""Context assembly for answer generation.

Transforms retrieved chunks into two things:

    1. A numbered, citation-ready *context block* injected into the LLM prompt.
       Each chunk is labeled ``[Source N]`` with its document, page, and chunk
       id so the model can cite precisely.
    2. A structured list of :class:`SourceCitation` objects the UI renders in
       its "Sources" section.

Keeping these two views in lockstep (same ``N`` ordering) is what lets a
``[Source 2]`` marker in the answer map back to a concrete document/page/chunk.

Usage:
    >>> from app.services.context_builder import build_context
    >>> assembled = build_context(retrieved_chunks)
    >>> print(assembled.context_text)   # goes into the LLM prompt
    >>> assembled.citations[0].source   # 'report.pdf'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.config.settings import get_settings
from app.rag.vector_store import RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SourceCitation:
    """A single citation shown to the user and referenced by the LLM.

    Attributes:
        index: 1-based citation number matching the ``[Source N]`` marker.
        source: Source document display name.
        page_number: 1-based page number.
        chunk_index: 0-based chunk index within the page.
        chunk_id: Stable id of the underlying chunk.
        score: Similarity score in ``[0, 1]``.
        text: The chunk text (for display / "show snippet").
    """

    index: int
    source: str
    page_number: int
    chunk_index: int
    chunk_id: str
    score: float
    text: str

    @property
    def label(self) -> str:
        """Short human-readable label, e.g. ``report.pdf · p.4 · chunk 12``."""
        return f"{self.source} · p.{self.page_number} · chunk {self.chunk_index}"


@dataclass(frozen=True)
class AssembledContext:
    """The product of context assembly.

    Attributes:
        context_text: The numbered context block for the LLM prompt. Empty
            string when there are no chunks.
        citations: Structured citations aligned 1:1 (by ``index``) with the
            ``[Source N]`` markers in ``context_text``.
    """

    context_text: str
    citations: List[SourceCitation]

    @property
    def is_empty(self) -> bool:
        """Whether any context was assembled."""
        return not self.citations


def build_context(chunks: List[RetrievedChunk]) -> AssembledContext:
    """Assemble retrieved chunks into a cited context block and citation list.

    Args:
        chunks: Retrieved chunks, already ordered by descending relevance.

    Returns:
        An :class:`AssembledContext`. When ``chunks`` is empty, both the
        context text and citation list are empty.
    """
    if not chunks:
        logger.info("No chunks to assemble; returning empty context.")
        return AssembledContext(context_text="", citations=[])

    # Optionally group chunks under their source document (best document first),
    # preserving each chunk's relative order within its document. This only
    # reorders/labels the context — it does not change which chunks were
    # retrieved. With the flag off, the original interleaved order is kept.
    grouped = get_settings().group_context_by_document
    if grouped:
        chunks = _group_by_document(chunks)

    blocks: List[str] = []
    citations: List[SourceCitation] = []
    last_source: str | None = None

    for position, chunk in enumerate(chunks, start=1):
        # When grouping, announce each new document once so the model sees a
        # clear per-document boundary (purely a context-formatting aid).
        block = ""
        if grouped and chunk.source != last_source:
            block += f"=== Document: {chunk.source} ===\n"
            last_source = chunk.source

        # The header makes provenance explicit to the model and pins the
        # citation number it should use in the answer.
        header = (
            f"[Source {position}] "
            f"(document: {chunk.source}, page: {chunk.page_number}, "
            f"chunk: {chunk.chunk_index})"
        )
        blocks.append(f"{block}{header}\n{chunk.text}")

        citations.append(
            SourceCitation(
                index=position,
                source=chunk.source,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                text=chunk.text,
            )
        )

    # Blank line between sources keeps them visually distinct in the prompt.
    context_text = "\n\n".join(blocks)
    logger.info("Assembled context from %d source(s).", len(citations))
    return AssembledContext(context_text=context_text, citations=citations)


def _group_by_document(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Group chunks by source document, best document first, stable within a doc.

    Document order is by each document's best (first-seen) chunk, so the most
    relevant document leads. A chunk's position within its document is preserved
    from the input (which is already in relevance order). Pure reordering — no
    chunk is added or dropped.
    """
    order: List[str] = []
    by_source: Dict[str, List[RetrievedChunk]] = {}
    for chunk in chunks:
        if chunk.source not in by_source:
            by_source[chunk.source] = []
            order.append(chunk.source)
        by_source[chunk.source].append(chunk)

    grouped: List[RetrievedChunk] = []
    for source in order:
        grouped.extend(by_source[source])
    return grouped
