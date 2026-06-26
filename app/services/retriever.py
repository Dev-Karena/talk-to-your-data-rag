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
from app.rag.query_rewriter import rewrite_query
from app.rag.vector_store import RetrievedChunk, VectorStore, get_vector_store
from app.utils.logger import get_logger
from app.utils.timing import Stopwatch

logger = get_logger(__name__)


def _gather_candidates(
    store: VectorStore, sub_embeddings: List[List[float]], fetch_k: int
) -> List[Tuple[RetrievedChunk, List[float]]]:
    """Fetch and merge candidate pools across one or more sub-query embeddings.

    A single embedding returns the original pool unchanged (baseline). Multiple
    embeddings (from query decomposition) are unioned by ``chunk_id`` keeping the
    best score, then re-sorted — surfacing chunks each sub-query found best.
    """
    if len(sub_embeddings) == 1:
        return store.query_candidates(sub_embeddings[0], fetch_k=fetch_k)

    merged: dict[str, Tuple[RetrievedChunk, List[float]]] = {}
    for vec in sub_embeddings:
        for chunk, cvec in store.query_candidates(vec, fetch_k=fetch_k):
            existing = merged.get(chunk.chunk_id)
            if existing is None or chunk.score > existing[0].score:
                merged[chunk.chunk_id] = (chunk, cvec)
    return sorted(merged.values(), key=lambda cv: cv[0].score, reverse=True)


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

    # Time each stage for observability (Sprint 3). Timing does not alter results.
    sw = Stopwatch()
    embedder = get_embedder()
    store = get_vector_store()

    # Non-MMR path: original single-query top-k (unchanged baseline). Query
    # decomposition needs re-ranking to merge, so it only applies on the MMR path.
    if not settings.use_mmr:
        with sw.stage("embed"):
            query_embedding = embedder.embed_query(question)
        with sw.stage("search"):
            results = store.query(query_embedding, top_k=k)
        sw.log("retrieve(mmr=off)")
        logger.info("Retrieved %d chunk(s) (top_k=%d, mmr=off).", len(results), k)
        return results

    # 1. Query rewriting (Sprint 5). sub_queries[0] is always the original; with
    #    QUERY_REWRITE_MODE=off this is a 1-element list and the path below is
    #    byte-equivalent to the pre-Sprint-5 baseline.
    sub_queries = rewrite_query(question, settings.query_rewrite_mode)

    # 2. Embed each sub-query; the original's embedding drives MMR relevance.
    with sw.stage("embed"):
        sub_embeddings = [embedder.embed_query(sq) for sq in sub_queries]
    query_embedding = sub_embeddings[0]

    # 3. Over-fetch, then gather + merge candidates across sub-queries.
    fetch_k = max(settings.fetch_k, k)
    with sw.stage("search"):
        candidates = _gather_candidates(store, sub_embeddings, fetch_k)

    # 4/5. Diversify with MMR and (optionally) re-rank with a cross-encoder.
    #      When reranking is disabled, MMR selects k directly and this path is
    #      byte-identical to the pre-Sprint-6 baseline.
    if not settings.reranker_enabled:
        with sw.stage("mmr"):
            results = _mmr_select(query_embedding, candidates, k, settings.mmr_lambda)
    else:
        with sw.stage("rerank"):
            results = _rerank_and_select(
                question, query_embedding, candidates, k, settings
            )

    sw.log("retrieve(mmr=on)")
    logger.info(
        "Retrieved %d chunk(s) (top_k=%d, mmr=on, subq=%d, fetched=%d, sources=%d, "
        "rerank=%s/%s).",
        len(results), k, len(sub_queries), len(candidates),
        len({c.source for c in results}), settings.reranker_enabled,
        settings.reranker_strategy if settings.reranker_enabled else "-",
    )
    return results


def _minmax(values: List[float]) -> List[float]:
    """Min-max normalize to ``[0, 1]``; all-equal inputs map to ``1.0``."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _rerank_and_select(
    question: str,
    query_embedding: List[float],
    candidates: List[Tuple[RetrievedChunk, List[float]]],
    k: int,
    settings,
) -> List[RetrievedChunk]:
    """Combine cross-encoder re-ranking with MMR per ``RERANKER_STRATEGY``.

    Strategies (Sprint 6.x):
        * ``post_mmr``      — MMR widens to top_n, cross-encoder reorders, truncate
                              to k. Best ranking; may drop cross-document diversity.
        * ``pre_mmr``       — cross-encoder reorders the pool, keep top_n, then MMR
                              selects k. MMR (last) preserves diversity.
        * ``mmr_relevance`` — cross-encoder scores become the MMR relevance term,
                              so reranker accuracy and MMR diversity combine.

    Fail-open: if the reranker yields no scores, falls back to plain MMR-to-k.
    """
    from app.services.reranker import rerank, rerank_scores

    strategy = settings.reranker_strategy
    top_n = max(settings.reranker_top_n, k)
    lam = settings.mmr_lambda

    if strategy == "post_mmr":
        pool = _mmr_select(query_embedding, candidates, top_n, lam)
        return rerank(question, pool)[:k]

    chunks_only = [c for c, _ in candidates]
    scores = rerank_scores(question, chunks_only)
    if scores is None:  # disabled mid-run or model failure -> plain MMR
        return _mmr_select(query_embedding, candidates, k, lam)

    if strategy == "pre_mmr":
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        pool = [candidates[i] for i in order][:top_n]
        return _mmr_select(query_embedding, pool, k, lam)

    # mmr_relevance: use normalized cross-encoder scores as MMR relevance.
    relevance = _minmax(scores)
    return _mmr_select(query_embedding, candidates, k, lam, relevance=relevance)


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
    relevance: Optional[List[float]] = None,
) -> List[RetrievedChunk]:
    """Greedily pick ``k`` chunks balancing relevance and diversity (MMR).

    At each step the next chunk maximizes
    ``lambda * relevance(chunk) - (1 - lambda) * max sim(chunk, already_picked)``.
    Because chunks from *different* documents are typically dissimilar to each
    other, the diversity term naturally spreads selections across documents.

    Args:
        query_embedding: The embedded query (used for the default relevance term).
        candidates: ``(chunk, embedding)`` pairs, ordered by descending
            similarity to the query (as returned by ``query_candidates``).
        k: Number of chunks to select.
        lambda_mult: MMR trade-off in ``[0, 1]`` (1 = relevance only).
        relevance: Optional per-candidate relevance scores (aligned to
            ``candidates``) to use instead of cosine query similarity — this is how
            the ``mmr_relevance`` strategy injects cross-encoder scores. Defaults
            to cosine similarity between the query and each candidate.

    Returns:
        Up to ``k`` selected chunks in selection order.
    """
    if not candidates:
        return []

    # Relevance term: cross-encoder scores when provided, else cosine to query.
    query_sim = (
        relevance if relevance is not None
        else [_cosine(query_embedding, vec) for _, vec in candidates]
    )

    if k >= len(candidates):
        # Nothing to prune. With the default relevance, preserve the incoming
        # (cosine-sorted) order exactly — this keeps the disabled path byte-
        # identical. With an injected relevance term, order by it.
        if relevance is None:
            return [chunk for chunk, _ in candidates]
        order = sorted(range(len(candidates)), key=lambda i: query_sim[i], reverse=True)
        return [candidates[i][0] for i in order]

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
