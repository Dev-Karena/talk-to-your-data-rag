"""Reusable Streamlit UI components.

Small, presentation-only render helpers used by the main Streamlit app:
source citation cards, status badges, the indexed-document list, and the
ingestion-result summary. Keeping these here keeps ``streamlit_app.py`` focused
on layout and flow.

These functions render directly into the Streamlit page; they return ``None``.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from app.rag.pipeline import IngestResult, IngestStatus
from app.services.context_builder import SourceCitation

# Emoji badges keep status scannable without custom CSS.
_STATUS_BADGE: Dict[IngestStatus, str] = {
    IngestStatus.INDEXED: "✅",
    IngestStatus.SKIPPED: "⏭️",
    IngestStatus.REJECTED: "🚫",
    IngestStatus.FAILED: "❌",
}


def render_status_badge(label: str, ok: bool) -> None:
    """Render a simple coloured status line.

    Args:
        label: Text to display.
        ok: When ``True`` shows a success style; otherwise a warning style.
    """
    if ok:
        st.success(label, icon="✅")
    else:
        st.warning(label, icon="⚠️")


def render_ingest_results(results: List[IngestResult]) -> None:
    """Render a per-file summary of an indexing run.

    Args:
        results: Ingestion results returned by the pipeline.
    """
    if not results:
        return

    indexed = sum(1 for r in results if r.status is IngestStatus.INDEXED)
    skipped = sum(1 for r in results if r.status is IngestStatus.SKIPPED)
    failed = sum(
        1
        for r in results
        if r.status in (IngestStatus.FAILED, IngestStatus.REJECTED)
    )

    st.caption(
        f"Indexed: {indexed} · Skipped: {skipped} · Failed: {failed}"
    )
    for result in results:
        badge = _STATUS_BADGE.get(result.status, "•")
        st.write(f"{badge} **{result.source}** — {result.message}")


def render_indexed_documents(sources: Dict[str, str], chunk_count: int) -> None:
    """Render the list of documents currently in the vector store.

    Args:
        sources: Mapping of ``doc_hash -> source`` for indexed documents.
        chunk_count: Total number of chunks in the store.
    """
    st.markdown("**📚 Indexed documents**")
    if not sources:
        st.caption("No documents indexed yet. Upload a PDF to get started.")
        return

    st.caption(f"{len(sources)} document(s) · {chunk_count} chunk(s)")
    for name in sorted(sources.values()):
        st.write(f"• {name}")


def render_citations(citations: List[SourceCitation]) -> None:
    """Render the "Sources" section for an answer.

    Each citation is shown as an expandable card with its document, page,
    chunk, similarity score, and the underlying text snippet — fulfilling the
    requirement to show which document and chunk produced the answer.

    Args:
        citations: Citations aligned with the answer's ``[Source N]`` markers.
    """
    if not citations:
        return

    st.markdown("#### 📎 Sources")
    for citation in citations:
        score_pct = f"{citation.score * 100:.0f}%"
        title = f"[Source {citation.index}] {citation.label} — relevance {score_pct}"
        with st.expander(title, expanded=False):
            st.progress(min(max(citation.score, 0.0), 1.0))
            st.markdown(
                f"**Document:** {citation.source}  \n"
                f"**Page:** {citation.page_number}  \n"
                f"**Chunk:** {citation.chunk_index}  \n"
                f"**Chunk ID:** `{citation.chunk_id}`"
            )
            st.markdown("**Snippet:**")
            st.write(citation.text)
