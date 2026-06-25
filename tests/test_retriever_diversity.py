"""Unit tests for Sprint-5 candidate merging across decomposed sub-queries.

Pure/offline. (Adaptive fetch_k and a per-document MMR cap were prototyped in
Sprint 5 but removed after benchmarking showed no benefit / a precision
regression, so they are no longer tested here.)
"""

from __future__ import annotations

from app.rag.vector_store import RetrievedChunk
from app.services.retriever import _gather_candidates


def _rc(source: str, cid: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=cid, source=source, page_number=1,
        chunk_index=0, doc_hash=source, score=score,
    )


def test_gather_candidates_single_query_is_passthrough() -> None:
    """One sub-query embedding returns the original pool unchanged (baseline)."""
    class _Store:
        def query_candidates(self, vec, fetch_k):
            return [(_rc("A.pdf", "a1", 0.9), [1.0, 0.0])]
    out = _gather_candidates(_Store(), [[1.0, 0.0]], fetch_k=5)
    assert len(out) == 1 and out[0][0].chunk_id == "a1"


def test_gather_candidates_merges_and_dedupes() -> None:
    """Multiple sub-queries union their pools, dedup by chunk_id keeping best score."""
    class _Store:
        def query_candidates(self, vec, fetch_k):
            # First sub-query finds a1+shared; second finds b1+shared(higher).
            if vec == [1.0, 0.0]:
                return [(_rc("A.pdf", "a1", 0.9), [1.0, 0.0]),
                        (_rc("X.pdf", "shared", 0.4), [0.5, 0.5])]
            return [(_rc("B.pdf", "b1", 0.8), [0.0, 1.0]),
                    (_rc("X.pdf", "shared", 0.7), [0.5, 0.5])]
    out = _gather_candidates(_Store(), [[1.0, 0.0], [0.0, 1.0]], fetch_k=5)
    ids = [c.chunk_id for c, _ in out]
    assert set(ids) == {"a1", "b1", "shared"}      # deduped (cross-document union)
    # shared kept with the higher score (0.7) and the pool sorted by score desc.
    shared = next(c for c, _ in out if c.chunk_id == "shared")
    assert shared.score == 0.7
    assert [c.score for c, _ in out] == sorted([c.score for c, _ in out], reverse=True)
