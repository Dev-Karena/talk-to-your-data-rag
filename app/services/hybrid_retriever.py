"""Hybrid retrieval service.

Orchestrates the combination of dense vector retrieval (ChromaDB) and sparse
keyword retrieval (rank-bm25) using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Tuple

from app.config.settings import get_settings
from app.rag.bm25_store import get_bm25_store
from app.rag.embeddings import get_embedder
from app.rag.query_rewriter import rewrite_query
from app.rag.vector_store import RetrievedChunk, get_vector_store
from app.utils.logger import get_logger
from app.utils.timing import Stopwatch

logger = get_logger(__name__)


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


def _gather_sub_query_candidates(
    sub_query: str,
    sub_embedding: List[float],
    fetch_k: int,
    bm25_top_k: int,
    rrf_k: int,
) -> List[Tuple[RetrievedChunk, List[float]]]:
    """Retrieve dense and sparse candidates for a sub-query, then fuse via RRF."""
    store = get_vector_store()
    bm25_store = get_bm25_store()

    # 1. Fetch dense candidates
    dense_res = store.query_candidates(sub_embedding, fetch_k=fetch_k)
    dense_rank = {chunk.chunk_id: idx + 1 for idx, (chunk, _) in enumerate(dense_res)}

    # 2. Fetch BM25 candidates
    bm25_res = bm25_store.search(sub_query, top_k=bm25_top_k)
    bm25_rank = {chunk_id: idx + 1 for idx, (chunk_id, _) in enumerate(bm25_res)}

    # 3. Union candidate IDs and identify missing metadata/vectors
    dense_dict = {chunk.chunk_id: (chunk, vec) for chunk, vec in dense_res}
    union_ids = set(dense_rank.keys()) | set(bm25_rank.keys())

    missing_ids = [cid for cid in union_ids if cid not in dense_dict]
    if missing_ids:
        try:
            fetched = store.get_chunks_by_ids(missing_ids)
            for chunk, vec in fetched:
                dense_dict[chunk.chunk_id] = (chunk, vec)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch missing BM25 chunk vectors: %s", exc)

    # 4. Compute RRF score for each chunk in the union
    fused_candidates: List[Tuple[RetrievedChunk, List[float]]] = []
    for cid in union_ids:
        if cid not in dense_dict:
            continue  # Defensive skip if DB failed to resolve missing chunk
        chunk, vec = dense_dict[cid]

        r_dense = dense_rank.get(cid)
        r_bm25 = bm25_rank.get(cid)

        score_dense = 1.0 / (rrf_k + r_dense) if r_dense is not None else 0.0
        score_bm25 = 1.0 / (rrf_k + r_bm25) if r_bm25 is not None else 0.0
        rrf_score = score_dense + score_bm25

        # Create a new RetrievedChunk with RRF score
        fused_chunk = replace(chunk, score=rrf_score)
        fused_candidates.append((fused_chunk, vec))

    # Sort descending by RRF score
    fused_candidates.sort(key=lambda x: x[0].score, reverse=True)
    return fused_candidates


def _merge_candidates(
    sub_query_candidates: List[List[Tuple[RetrievedChunk, List[float]]]]
) -> List[Tuple[RetrievedChunk, List[float]]]:
    """Union candidate lists across sub-queries, keeping the best RRF score."""
    merged: dict[str, Tuple[RetrievedChunk, List[float]]] = {}
    for candidate_list in sub_query_candidates:
        for chunk, cvec in candidate_list:
            existing = merged.get(chunk.chunk_id)
            if existing is None or chunk.score > existing[0].score:
                merged[chunk.chunk_id] = (chunk, cvec)
    return sorted(merged.values(), key=lambda cv: cv[0].score, reverse=True)


def _mmr_select_hybrid(
    candidates: List[Tuple[RetrievedChunk, List[float]]],
    relevance_scores: List[float],
    k: int,
    lambda_mult: float,
) -> List[RetrievedChunk]:
    """Run MMR using precomputed relevance scores and dense vector similarity."""
    if not candidates:
        return []
    if k >= len(candidates):
        # Return sorted by relevance
        sorted_pairs = sorted(
            zip(candidates, relevance_scores), key=lambda x: x[1], reverse=True
        )
        return [pair[0][0] for pair in sorted_pairs][:k]

    selected_idx: List[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected_idx) < k:
        best_idx = None
        best_score = -math.inf
        for idx in remaining:
            if not selected_idx:
                mmr = relevance_scores[idx]
            else:
                max_sim_to_selected = max(
                    _cosine(candidates[idx][1], candidates[s][1]) for s in selected_idx
                )
                mmr = lambda_mult * relevance_scores[idx] - (1.0 - lambda_mult) * max_sim_to_selected
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected_idx.append(best_idx)  # type: ignore[arg-type]
        remaining.discard(best_idx)

    return [candidates[i][0] for i in selected_idx]


def hybrid_retrieve(question: str, top_k: int) -> List[RetrievedChunk]:
    """Retrieve top chunks for a question combining dense and sparse indices."""
    settings = get_settings()
    sw = Stopwatch()

    embedder = get_embedder()

    # 1. Rewrite queries
    sub_queries = rewrite_query(question, settings.query_rewrite_mode)

    # 2. Embed sub-queries
    with sw.stage("embed"):
        sub_embeddings = [embedder.embed_query(sq) for sq in sub_queries]
    original_embedding = sub_embeddings[0]

    # 3. Retrieve candidates for each sub-query
    fetch_k = max(settings.fetch_k, top_k)
    sub_query_candidates: List[List[Tuple[RetrievedChunk, List[float]]]] = []

    with sw.stage("search"):
        for i, sq in enumerate(sub_queries):
            sub_candidates = _gather_sub_query_candidates(
                sub_query=sq,
                sub_embedding=sub_embeddings[i],
                fetch_k=fetch_k,
                bm25_top_k=settings.bm25_top_k,
                rrf_k=settings.rrf_k,
            )
            sub_query_candidates.append(sub_candidates)

    # 4. Merge candidates across sub-queries
    candidates = _merge_candidates(sub_query_candidates)
    # Truncate pool to fetch_k
    candidates = candidates[:fetch_k]

    # 5. Compute relevance scores based on mode
    if settings.hybrid_relevance_mode == "fused":
        # Max-normalize the RRF score in the candidate pool to [0, 1]
        rrf_max = max(c.score for c, _ in candidates) if candidates else 0.0
        relevance_scores = [
            (c.score / rrf_max if rrf_max > 0.0 else 0.0) for c, _ in candidates
        ]
    else:
        # Default: relevance score is cosine similarity to original query embedding
        relevance_scores = [_cosine(original_embedding, vec) for _, vec in candidates]

    # 6. Apply MMR re-ranking
    with sw.stage("mmr"):
        results = _mmr_select_hybrid(
            candidates=candidates,
            relevance_scores=relevance_scores,
            k=top_k,
            lambda_mult=settings.mmr_lambda if settings.use_mmr else 1.0,
        )


    sw.log("retrieve(hybrid=on)")
    logger.info(
        "Hybrid Retrieved %d chunk(s) (top_k=%d, mode=%s, subq=%d, pool=%d, sources=%d).",
        len(results),
        top_k,
        settings.hybrid_relevance_mode,
        len(sub_queries),
        len(candidates),
        len({c.source for c in results}),
    )
    return results
