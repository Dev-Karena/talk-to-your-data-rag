"""Citation builder for formatting professional document and page range citations."""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Set
from app.rag.vector_store import RetrievedChunk

@dataclass(frozen=True)
class Citation:
    """A clean, grouped citation for a source document.

    Attributes:
        source: Source document name (e.g. 'OS.pdf').
        pages: Unique sorted page numbers referenced in the document.
        page_range: Professional formatted string (e.g. 'p. 4' or 'pp. 86-87').
        chunk_ids: Underlying database chunk IDs grouped in this citation.
        score: Max similarity score among the matching chunks.
    """

    source: str
    pages: List[int]
    page_range: str
    chunk_ids: List[str]
    score: float

    @property
    def label(self) -> str:
        """Formatted display label, e.g. 'OS.pdf (pp. 86-87)'."""
        return f"{self.source} ({self.page_range})"


def extract_pages_from_chunk_id(chunk: RetrievedChunk) -> List[int]:
    """Extract page numbers from a chunk ID, supporting '+' joined merged chunks.

    Falls back to chunk.page_number if ID format is unexpected.
    """
    parts = chunk.chunk_id.split("+")
    pages: Set[int] = set()
    for part in parts:
        match = re.search(r'::p(\d+)::', part)
        if match:
            pages.add(int(match.group(1)))
            
    if not pages:
        pages.add(chunk.page_number)
    return sorted(list(pages))


def format_page_range(pages: List[int]) -> str:
    """Format a list of sorted page numbers into a clean page range string.

    Examples:
        [4] -> "p. 4"
        [86, 87] -> "pp. 86-87"
        [12, 14, 18, 19, 20] -> "pp. 12, 14, 18-20"
    """
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p. {pages[0]}"

    ranges: List[str] = []
    start = pages[0]
    prev = pages[0]

    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = p
            prev = p

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    combined = ", ".join(ranges)
    return f"pp. {combined}"


def build_citations(chunks: List[RetrievedChunk], min_score: float = 0.61) -> List[Citation]:
    """Group retrieved chunks by document name, deduplicate pages, format ranges, preserve order."""
    # Filter out low-similarity chunks to maximize precision on negative/irrelevant results
    filtered_chunks = [c for c in chunks if c.score >= min_score]
    if not filtered_chunks:
        return []

    doc_order: List[str] = []
    doc_groups: dict[str, List[RetrievedChunk]] = {}

    for chunk in filtered_chunks:
        if chunk.source not in doc_groups:
            doc_groups[chunk.source] = []
            doc_order.append(chunk.source)
        doc_groups[chunk.source].append(chunk)

    citations: List[Citation] = []
    for doc in doc_order:
        doc_chunks = doc_groups[doc]
        all_pages: Set[int] = set()
        chunk_ids: List[str] = []
        scores: List[float] = []

        for c in doc_chunks:
            all_pages.update(extract_pages_from_chunk_id(c))
            chunk_ids.append(c.chunk_id)
            scores.append(c.score)

        sorted_pages = sorted(list(all_pages))
        citations.append(
            Citation(
                source=doc,
                pages=sorted_pages,
                page_range=format_page_range(sorted_pages),
                chunk_ids=chunk_ids,
                score=max(scores) if scores else 0.0,
            )
        )
    return citations
