"""Unit tests for Sprint-5 candidate merging across decomposed sub-queries.

Pure/offline. (Adaptive fetch_k and a per-document MMR cap were prototyped in
Sprint 5 but removed after benchmarking showed no benefit / a precision
regression, so they are no longer tested here.)
"""

from __future__ import annotations

from app.rag.vector_store import RetrievedChunk
from app.services.retriever import _gather_candidates, _minmax, _mmr_select


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


# ---- Sprint 6.x: MMR relevance override + normalization -------------------

def test_minmax_normalizes_to_unit_range() -> None:
    assert _minmax([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]


def test_minmax_all_equal_maps_to_one() -> None:
    assert _minmax([2.0, 2.0]) == [1.0, 1.0]


def test_mmr_relevance_override_changes_ranking() -> None:
    """With an injected relevance term, MMR ranks by it, not cosine-to-query.

    Two orthogonal-vector chunks: cosine relevance would favor the one aligned
    with the query, but the override flips the preference.
    """
    q = [1.0, 0.0]
    candidates = [
        (_rc("A.pdf", "a", 0.9), [1.0, 0.0]),   # high cosine to q
        (_rc("B.pdf", "b", 0.1), [0.0, 1.0]),   # low cosine to q
    ]
    # Default (cosine) relevance -> 'a' first.
    assert [c.chunk_id for c in _mmr_select(q, candidates, 1, 0.5)] == ["a"]
    # Override relevance preferring 'b' -> 'b' first.
    out = _mmr_select(q, candidates, 1, 0.5, relevance=[0.0, 1.0])
    assert [c.chunk_id for c in out] == ["b"]


def test_mmr_relevance_override_preserves_diversity() -> None:
    """lambda<1 still spreads picks across documents even with injected relevance."""
    q = [1.0, 0.0]
    candidates = [
        (_rc("A.pdf", "a1", 0.9), [1.0, 0.0]),
        (_rc("A.pdf", "a2", 0.89), [0.99, 0.01]),  # near-duplicate of a1
        (_rc("B.pdf", "b1", 0.5), [0.0, 1.0]),     # different document
    ]
    # All three rank high on the injected relevance, but diversity should still
    # pull the B-document chunk into the top-2 over the a1/a2 near-duplicate.
    out = _mmr_select(q, candidates, 2, 0.5, relevance=[1.0, 0.95, 0.9])
    sources = {c.source for c in out}
    assert sources == {"A.pdf", "B.pdf"}
