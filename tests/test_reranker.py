"""Unit tests for app.services.reranker (Sprint 6 cross-encoder re-ranking).

Fully offline: the CrossEncoder is stubbed so no model is downloaded or loaded.
Covers the disabled passthrough (byte-identical), reordering, fail-open on model
error, the single-candidate short-circuit, and device resolution.
"""

from __future__ import annotations

import pytest

from app.config.settings import get_settings
from app.rag.vector_store import RetrievedChunk
import app.services.reranker as reranker


def _chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="doc.pdf",
        page_number=1,
        chunk_index=0,
        doc_hash="h",
        score=score,
    )


class _FakeModel:
    """Stub CrossEncoder: returns a preset score per chunk text."""

    def __init__(self, scores: dict) -> None:
        self._scores = scores

    def predict(self, pairs):
        return [self._scores[text] for _query, text in pairs]


@pytest.fixture(autouse=True)
def _clear_caches():
    """Isolate settings + model cache between tests."""
    get_settings.cache_clear()
    reranker._load_cross_encoder.cache_clear()
    yield
    get_settings.cache_clear()
    reranker._load_cross_encoder.cache_clear()


def test_disabled_returns_input_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    get_settings.cache_clear()
    chunks = [_chunk("a", "alpha", 0.9), _chunk("b", "beta", 0.8)]
    out = reranker.rerank("q", chunks)
    assert out is chunks  # no copy, no reorder when disabled


def test_reorders_by_cross_encoder_score(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    # Retrieval order is alpha, beta, gamma; cross-encoder prefers gamma > alpha > beta.
    chunks = [
        _chunk("a", "alpha", 0.9),
        _chunk("b", "beta", 0.8),
        _chunk("c", "gamma", 0.7),
    ]
    fake = _FakeModel({"alpha": 0.5, "beta": 0.1, "gamma": 0.9})
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda *a, **k: fake)
    out = reranker.rerank("q", chunks)
    assert [c.chunk_id for c in out] == ["c", "a", "b"]


def test_scores_are_not_mutated(monkeypatch) -> None:
    # Reranking only reorders; the cosine .score field must be preserved.
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    chunks = [_chunk("a", "alpha", 0.9), _chunk("b", "beta", 0.8)]
    fake = _FakeModel({"alpha": 0.1, "beta": 0.2})
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda *a, **k: fake)
    out = reranker.rerank("q", chunks)
    assert [c.chunk_id for c in out] == ["b", "a"]
    assert {c.chunk_id: c.score for c in out} == {"a": 0.9, "b": 0.8}


def test_fail_open_on_model_error(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    chunks = [_chunk("a", "alpha", 0.9), _chunk("b", "beta", 0.8)]

    def _boom(*a, **k):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(reranker, "_load_cross_encoder", _boom)
    out = reranker.rerank("q", chunks)
    assert out is chunks  # original order, no exception


def test_single_candidate_short_circuits(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    chunks = [_chunk("a", "alpha", 0.9)]
    # Would KeyError if the (un-needed) model were called.
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda *a, **k: _FakeModel({}))
    out = reranker.rerank("q", chunks)
    assert out is chunks


def test_resolve_device_cpu_and_auto_without_gpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert reranker._resolve_device("cpu") == "cpu"
    assert reranker._resolve_device("auto") == "cpu"
    # Explicit cuda with no GPU falls back to cpu (no hard failure).
    assert reranker._resolve_device("cuda") == "cpu"


def test_resolve_device_auto_with_gpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert reranker._resolve_device("auto") == "cuda"
    assert reranker._resolve_device("cuda") == "cuda"
    assert reranker._resolve_device("cpu") == "cpu"


def test_invalid_device_rejected(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_DEVICE", "tpu")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


def test_invalid_strategy_rejected(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_STRATEGY", "magic")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


def test_rerank_scores_disabled_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    get_settings.cache_clear()
    chunks = [_chunk("a", "alpha", 0.9)]
    assert reranker.rerank_scores("q", chunks) is None


def test_rerank_scores_aligned_to_input(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    chunks = [_chunk("a", "alpha", 0.9), _chunk("b", "beta", 0.8)]
    fake = _FakeModel({"alpha": 0.3, "beta": 0.7})
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda *a, **k: fake)
    assert reranker.rerank_scores("q", chunks) == [0.3, 0.7]


def test_rerank_scores_fail_open_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(reranker, "_load_cross_encoder", _boom)
    assert reranker.rerank_scores("q", [_chunk("a", "alpha", 0.9),
                                        _chunk("b", "beta", 0.8)]) is None
