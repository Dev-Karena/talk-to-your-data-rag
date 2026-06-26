"""Cross-encoder re-ranking (Sprint 6).

An optional precision stage placed *after* MMR: it re-scores each
``(query, chunk_text)`` pair with a cross-encoder and reorders the candidates by
that score. Unlike the bi-encoder used for retrieval (which embeds query and
chunk independently), a cross-encoder reads both together and is more accurate at
ranking — at the cost of one model forward pass per candidate, so it runs only on
the small MMR-selected pool, never the whole corpus.

Design:
    * **Config-gated** (``RERANKER_ENABLED``, default off). When off this module
      is never imported on the hot path and retrieval is byte-identical.
    * **Device auto-detection** (``RERANKER_DEVICE=auto|cpu|cuda``). ``auto`` uses
      CUDA when available and falls back to CPU — no hard CUDA dependency, so the
      same code runs on CPU-only and GPU machines.
    * **Fail-open.** Any load/inference error logs a warning and returns the input
      order unchanged: a broken reranker must never break retrieval.
    * **Non-destructive.** The cross-encoder score is a different scale from the
      cosine ``score`` on :class:`RetrievedChunk`, so chunks are only *reordered*;
      their ``score`` field is left untouched.

The model is loaded lazily and cached per process (first call pays a cold load).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from app.config.settings import get_settings
from app.rag.vector_store import RetrievedChunk
from app.utils.logger import get_logger
from app.utils.timing import Stopwatch

logger = get_logger(__name__)


def _resolve_device(configured: str) -> str:
    """Resolve ``RERANKER_DEVICE`` to a concrete torch device string.

    ``auto`` -> ``cuda`` when a GPU is available, else ``cpu``. Explicit ``cuda``
    falls back to ``cpu`` with a warning when no GPU is present, so configuration
    never hard-fails on a CPU-only machine.
    """
    configured = (configured or "auto").strip().lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch import/detection failure
        cuda_available = False

    if configured == "cpu":
        return "cpu"
    if configured == "cuda":
        if cuda_available:
            return "cuda"
        logger.warning("RERANKER_DEVICE=cuda but no GPU available; using CPU.")
        return "cpu"
    # auto
    return "cuda" if cuda_available else "cpu"


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str):
    """Load and cache a CrossEncoder. Cached per (model, device) per process."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder '%s' on %s ...", model_name, device)
    return CrossEncoder(model_name, device=device)


def rerank_scores(
    query: str, chunks: List[RetrievedChunk]
) -> Optional[List[float]]:
    """Cross-encoder relevance score for each chunk, aligned to input order.

    Returns ``None`` (fail-open) when reranking is disabled or the model fails to
    load/score — callers fall back to their non-reranked path. The raw scores are
    cross-encoder logits (unbounded, comparable only *within* one call); callers
    that feed them into MMR should normalize first.
    """
    settings = get_settings()
    if not settings.reranker_enabled:
        return None
    if not chunks:
        return None

    device = _resolve_device(settings.reranker_device)
    try:
        model = _load_cross_encoder(settings.reranker_model, device)
        sw = Stopwatch()
        with sw.stage("rerank"):
            pairs = [(query, c.text) for c in chunks]
            scores = [float(s) for s in model.predict(pairs)]
        # DEBUG-only observability: latency, candidate count, model, device.
        logger.debug(
            "Rerank: %d candidate(s), model='%s', device=%s, latency=%.1fms.",
            len(chunks), settings.reranker_model, device, sw.stages["rerank"],
        )
        return scores
    except Exception as exc:  # fail-open: never break retrieval
        logger.warning("Reranking failed (%s); falling back to no rerank.", exc)
        return None


def rerank(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Re-rank ``chunks`` for ``query``, reordered by descending relevance.

    Returns the list unchanged (fail-open) when reranking is disabled, when there
    is nothing to reorder, or when the model fails. Truncation to ``top_k`` is the
    caller's responsibility — this only reorders. Used by the ``post_mmr`` strategy.
    """
    if len(chunks) <= 1:
        return chunks
    scores = rerank_scores(query, chunks)
    if scores is None:
        return chunks
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order]
