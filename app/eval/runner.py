"""Benchmark runner — scores the production retrieve() path (read-only).

For each case it embeds the query, runs the live ``retrieve()`` (the exact path
the app uses — unchanged), and computes ranking metrics at *document* granularity
(keyed on ``doc_hash``) plus optional *chunk* granularity when the case lists
``relevant_chunk_ids``. Per-stage timing is captured with the Sprint-3
``Stopwatch``. Nothing is written to the store.

Aggregation averages each metric across the non-negative cases. Negative cases
(no expected document) are reported separately via their top similarity score,
since the retriever has no abstention threshold (see Sprint-4 audit, Risk 7).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from app.eval import metrics
from app.eval.dataset import Benchmark, Case
from app.services.retriever import retrieve
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CaseResult:
    """Metrics and context for one evaluated case."""

    id: str
    query: str
    type: str
    expected_doc_hashes: List[str]
    retrieved_sources: List[str]
    retrieved_doc_hashes: List[str]
    top_score: float
    matched_rank: Optional[int]            # 1-based rank of first relevant doc (None if missed)
    recall_at_k: float
    precision_at_k: float
    hit_at_1: float
    reciprocal_rank: float
    ndcg_at_k: float
    source_accuracy: float                 # = hit@1 at document granularity
    chunk_recall_at_k: Optional[float]     # only when relevant_chunk_ids given
    elapsed_ms: float


def _score_case(case: Case, k: int) -> CaseResult:
    """Run one case through retrieve() and compute its metrics (read-only)."""
    start = time.perf_counter()
    hits = retrieve(case.query, top_k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    ranked_hashes = [h.doc_hash for h in hits]
    relevant = set(case.expected_doc_hashes)
    top_score = hits[0].score if hits else 0.0

    matched_rank = next(
        (i for i, h in enumerate(ranked_hashes, start=1) if h in relevant), None
    )

    # Optional chunk-level recall (strongest ground truth).
    chunk_recall = None
    if case.relevant_chunk_ids:
        ranked_chunk_ids = [h.chunk_id for h in hits]
        chunk_recall = metrics.recall_at_k(ranked_chunk_ids, set(case.relevant_chunk_ids), k)

    return CaseResult(
        id=case.id,
        query=case.query,
        type=case.type,
        expected_doc_hashes=case.expected_doc_hashes,
        retrieved_sources=[h.source for h in hits],
        retrieved_doc_hashes=ranked_hashes,
        top_score=round(top_score, 4),
        matched_rank=matched_rank,
        recall_at_k=metrics.recall_at_k(ranked_hashes, relevant, k),
        precision_at_k=metrics.precision_at_k(ranked_hashes, relevant, k),
        hit_at_1=metrics.hit_at_1(ranked_hashes, relevant),
        reciprocal_rank=metrics.reciprocal_rank(ranked_hashes, relevant),
        ndcg_at_k=metrics.ndcg_at_k(ranked_hashes, relevant, k),
        source_accuracy=metrics.hit_at_1(ranked_hashes, relevant),
        chunk_recall_at_k=chunk_recall,
        elapsed_ms=round(elapsed_ms, 2),
    )


@dataclass
class BenchmarkReport:
    """Aggregate benchmark result."""

    k: int
    corpus_fingerprint_dataset: str
    corpus_fingerprint_live: str
    corpus_match: bool
    total_cases: int
    scored_cases: int                      # non-negative cases counted in averages
    negative_cases: int
    aggregate: Dict[str, float]
    timing_ms: Dict[str, float]
    cases: List[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_benchmark(
    benchmark: Benchmark,
    live_doc_hashes: List[str],
    k: int,
) -> BenchmarkReport:
    """Run all cases and aggregate. ``live_doc_hashes`` fingerprints the store.

    The runner does not enforce a corpus match — it records both fingerprints so
    a report always states whether it scored the intended corpus.
    """
    from app.eval.dataset import corpus_fingerprint

    results = [_score_case(c, k) for c in benchmark.cases]

    scored = [r for r in results if r.type != "negative"]
    negatives = [r for r in results if r.type == "negative"]

    def _avg(attr: str) -> float:
        vals = [getattr(r, attr) for r in scored]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    chunk_vals = [r.chunk_recall_at_k for r in scored if r.chunk_recall_at_k is not None]
    aggregate = {
        "recall_at_k": _avg("recall_at_k"),
        "precision_at_k": _avg("precision_at_k"),
        "hit_at_1": _avg("hit_at_1"),
        "mrr": _avg("reciprocal_rank"),
        "ndcg_at_k": _avg("ndcg_at_k"),
        "source_accuracy": _avg("source_accuracy"),
        "chunk_recall_at_k": round(sum(chunk_vals) / len(chunk_vals), 4) if chunk_vals else None,
    }

    all_ms = [r.elapsed_ms for r in results] or [0.0]
    timing = {
        "avg_query_ms": round(sum(all_ms) / len(all_ms), 2),
        "min_query_ms": round(min(all_ms), 2),
        "max_query_ms": round(max(all_ms), 2),
    }

    df = benchmark.fingerprint()
    lf = corpus_fingerprint(live_doc_hashes)
    report = BenchmarkReport(
        k=k,
        corpus_fingerprint_dataset=df,
        corpus_fingerprint_live=lf,
        corpus_match=(df == lf),
        total_cases=len(results),
        scored_cases=len(scored),
        negative_cases=len(negatives),
        aggregate=aggregate,
        timing_ms=timing,
        cases=results,
    )
    logger.info(
        "Benchmark complete: %d case(s), recall@%d=%.3f, hit@1=%.3f, corpus_match=%s",
        len(results), k, aggregate["recall_at_k"], aggregate["hit_at_1"], report.corpus_match,
    )
    return report
