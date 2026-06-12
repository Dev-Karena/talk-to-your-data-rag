"""Streamlit application — "Talk to Your Data".

The presentation layer. Provides:
    * A sidebar to upload and index PDFs, with status indicators.
    * A chat interface with conversation history.
    * A per-answer "Sources" section with document/page/chunk citations.
    * "Clear database" and "Re-index" controls.

It depends only on the service facade (:mod:`app.services.rag_service`) and the
ingestion entry point (:mod:`app.rag.pipeline`) — never on lower-level modules
directly.

Run via the project root entry point:  ``streamlit run main.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from app.config.settings import get_settings
from app.rag.pipeline import IngestResult, ingest_document
from app.rag.vector_store import get_vector_store
from app.services.rag_service import answer_question_stream
from app.ui.components import (
    render_citations,
    render_indexed_documents,
    render_ingest_results,
    render_status_badge,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PAGE_TITLE = "Talk to Your Data"
_PAGE_ICON = "📄"


def _init_page() -> None:
    """Configure the page and ensure required directories exist."""
    st.set_page_config(page_title=_PAGE_TITLE, page_icon=_PAGE_ICON, layout="wide")
    get_settings().ensure_directories()
    if "messages" not in st.session_state:
        # Each message: {"role": str, "content": str, "citations": list}
        st.session_state.messages = []


def _render_environment_status() -> None:
    """Show whether the app is configured to generate answers."""
    settings = get_settings()
    has_key = bool(settings.groq_api_key)
    render_status_badge(
        "Groq API key loaded" if has_key else "GROQ_API_KEY missing — set it in .env",
        ok=has_key,
    )
    st.caption(
        f"LLM: `{settings.llm_model}` · Embeddings: "
        f"`{settings.embedding_backend.value}:{settings.embedding_model}`"
    )


def _index_uploaded_files(uploaded_files: list) -> List[IngestResult]:
    """Validate and ingest each uploaded file, returning per-file results."""
    results: List[IngestResult] = []
    progress = st.progress(0.0, text="Indexing documents...")
    total = len(uploaded_files)
    for position, uploaded in enumerate(uploaded_files, start=1):
        progress.progress(
            position / total, text=f"Indexing '{uploaded.name}' ({position}/{total})"
        )
        data = uploaded.getvalue()
        results.append(ingest_document(uploaded.name, data))
    progress.empty()
    return results


def _reindex_from_disk() -> List[IngestResult]:
    """Clear the store and re-ingest every PDF in the documents directory.

    Returns:
        Per-file ingestion results.
    """
    settings = get_settings()
    store = get_vector_store()
    store.clear()

    results: List[IngestResult] = []
    pdf_paths = sorted(settings.documents_dir.glob("*.pdf"))
    if not pdf_paths:
        return results

    progress = st.progress(0.0, text="Re-indexing documents...")
    total = len(pdf_paths)
    for position, path in enumerate(pdf_paths, start=1):
        progress.progress(
            position / total, text=f"Re-indexing '{path.name}' ({position}/{total})"
        )
        # Strip the hash prefix added at ingest time to recover the display name.
        display_name = _strip_hash_prefix(path.name)
        results.append(ingest_document(display_name, path.read_bytes()))
    progress.empty()
    return results


def _strip_hash_prefix(filename: str) -> str:
    """Recover the original display name from a stored ``{hash}_{name}`` file."""
    parts = filename.split("_", 1)
    return parts[1] if len(parts) == 2 else filename


def _render_sidebar() -> None:
    """Render the upload, status, and database-control sidebar."""
    with st.sidebar:
        st.header("📥 Documents")
        _render_environment_status()
        st.divider()

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to index.",
        )

        if st.button("📌 Index uploaded files", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please choose at least one PDF first.")
            else:
                results = _index_uploaded_files(uploaded_files)
                render_ingest_results(results)

        st.divider()

        # Live store status.
        store = get_vector_store()
        sources = store.list_sources()
        render_indexed_documents(sources, store.count())

        st.divider()

        # Database controls.
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Re-index", use_container_width=True, help="Rebuild the index from stored PDFs."):
                results = _reindex_from_disk()
                if results:
                    render_ingest_results(results)
                else:
                    st.info("No stored PDFs to re-index.")
        with col_b:
            if st.button("🗑️ Clear DB", use_container_width=True, help="Delete all indexed data."):
                get_vector_store().clear()
                st.session_state.messages = []
                st.success("Vector database cleared.")


def _render_history() -> None:
    """Re-render the full conversation history with citations."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations", []))


def _handle_question(question: str) -> None:
    """Process a new user question: stream the answer and show sources."""
    st.session_state.messages.append(
        {"role": "user", "content": question, "citations": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        token_stream, citations = answer_question_stream(question)
        # st.write_stream renders tokens live and returns the full text.
        answer = st.write_stream(token_stream)
        render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


def main() -> None:
    """Application entry point."""
    _init_page()

    st.title(f"{_PAGE_ICON} {_PAGE_TITLE}")
    st.caption("Upload PDFs, then ask questions. Every answer cites its sources.")

    _render_sidebar()
    _render_history()

    question = st.chat_input("Ask a question about your documents...")
    if question:
        _handle_question(question)


if __name__ == "__main__":
    main()
