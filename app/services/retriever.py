"""Retrieval service.

Bridges a natural-language question to the vector store: embeds the query with
the configured embedder and runs a top-K similarity search, returning scored,
metadata-rich chunks ready for context assembly and citation.

This is a thin coordination layer — the embedding and storage details live in
:mod:`app.rag.embeddings` and :mod:`app.rag.vector_store`.

Usage:
    >>> from app.services.retriever import retrieve
    >>> hits = retrieve("What was Q4 revenue?")
    >>> hits[0].source, hits[0].page_number, round(hits[0].score, 2)
    ('report.pdf', 4, 0.83)
"""

from __future__ import annotations

from typing import List, Optional

from app.config.settings import get_settings
from app.rag.embeddings import get_embedder
from app.rag.vector_store import RetrievedChunk, get_vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question.

    Args:
        question: The user's natural-language question.
        top_k: Optional override for the number of chunks to retrieve. Falls
            back to the configured ``TOP_K`` when not provided.

    Returns:
        A list of :class:`RetrievedChunk` ordered by descending similarity
        score. Empty if the question is blank or the store has no documents.
    """
    question = (question or "").strip()
    if not question:
        logger.warning("Empty question passed to retriever.")
        return []

    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k

    # 1. Embed the query (uses the same backend as document embedding so the
    #    vectors live in a comparable space).
    embedder = get_embedder()
    query_embedding = embedder.embed_query(question)

    # 2. Similarity search against the persistent collection.
    store = get_vector_store()
    results = store.query(query_embedding, top_k=k)

    logger.info(
        "Retrieved %d chunk(s) for question (top_k=%d).", len(results), k
    )
    return results
