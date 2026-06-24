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

import math
from typing import List, Optional, Tuple

from app.config.settings import get_settings
from app.rag.embeddings import get_embedder
from app.rag.vector_store import RetrievedChunk, get_vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question.

    When MMR is enabled (the default), a larger candidate pool is fetched and
    re-ranked with Maximal Marginal Relevance so the final ``top_k`` spans
    multiple documents rather than collapsing onto the single most-similar one.
    This is what makes cross-document questions ("compare A and B") pull context
    from more than one source.

    Args:
        question: The user's natural-language question.
        top_k: Optional override for the number of chunks to retrieve. Falls
            back to the configured ``TOP_K`` when not provided.

    Returns:
        A list of :class:`RetrievedChunk` ordered by descending relevance.
        Empty if the question is blank or the store has no documents.
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

    if not settings.use_mmr:
        results = store.query(query_embedding, top_k=k)
        logger.info("Retrieved %d chunk(s) (top_k=%d, mmr=off).", len(results), k)
        return results

    # MMR path: over-fetch, then diversify down to k.
    fetch_k = max(settings.fetch_k, k)
    candidates = store.query_candidates(query_embedding, fetch_k=fetch_k)
    results = _mmr_select(query_embedding, candidates, k, settings.mmr_lambda)
    logger.info(
        "Retrieved %d chunk(s) (top_k=%d, mmr=on, fetched=%d, sources=%d).",
        len(results),
        k,
        len(candidates),
        len({c.source for c in results}),
    )
    return results


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors (0 if either is degenerate)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select(
    query_embedding: List[float],
    candidates: List[Tuple[RetrievedChunk, List[float]]],
    k: int,
    lambda_mult: float,
) -> List[RetrievedChunk]:
    """Greedily pick ``k`` chunks balancing relevance and diversity (MMR).

    At each step the next chunk maximizes
    ``lambda * sim(query, chunk) - (1 - lambda) * max sim(chunk, already_picked)``.
    Because chunks from *different* documents are typically dissimilar to each
    other, the diversity term naturally spreads selections across documents.

    Args:
        query_embedding: The embedded query.
        candidates: ``(chunk, embedding)`` pairs, ordered by descending
            similarity to the query (as returned by ``query_candidates``).
        k: Number of chunks to select.
        lambda_mult: MMR trade-off in ``[0, 1]`` (1 = relevance only).

    Returns:
        Up to ``k`` selected chunks in selection order.
    """
    if not candidates:
        return []
    if k >= len(candidates):
        # Nothing to prune; preserve the relevance ordering.
        return [chunk for chunk, _ in candidates]

    # Precompute query relevance for every candidate.
    query_sim = [_cosine(query_embedding, vec) for _, vec in candidates]

    selected_idx: List[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected_idx) < k:
        best_idx = None
        best_score = -math.inf
        for idx in remaining:
            if not selected_idx:
                mmr = query_sim[idx]
            else:
                max_sim_to_selected = max(
                    _cosine(candidates[idx][1], candidates[s][1]) for s in selected_idx
                )
                mmr = lambda_mult * query_sim[idx] - (1.0 - lambda_mult) * max_sim_to_selected
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected_idx.append(best_idx)  # type: ignore[arg-type]
        remaining.discard(best_idx)

    return [candidates[i][0] for i in selected_idx]
