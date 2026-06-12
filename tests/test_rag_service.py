"""Unit tests for app.services.rag_service.

Verifies the query-time facade end to end with the retriever and LLM client
stubbed out, so the tests run fully offline (no embeddings, no API calls).
Covers: empty-question handling, the no-context path, citation passthrough,
the happy path, and error handling for both blocking and streaming variants.
"""

from __future__ import annotations

from typing import Iterator, List

import pytest

from app.rag.vector_store import RetrievedChunk
from app.services import rag_service
from app.services.llm_client import LLMError


def _chunk(index: int, score: float) -> RetrievedChunk:
    """Build a retrieved chunk for stubbing the retriever."""
    return RetrievedChunk(
        chunk_id=f"h::p{index}::c0",
        text=f"context text {index}",
        source="doc.pdf",
        page_number=index,
        chunk_index=0,
        doc_hash="h",
        score=score,
    )


class _FakeLLM:
    """Stand-in LLM client with configurable behavior."""

    def __init__(self, answer: str = "stub answer [Source 1]", raise_error: bool = False):
        self._answer = answer
        self._raise = raise_error

    def generate(self, context: str, question: str) -> str:
        if self._raise:
            raise LLMError("simulated failure")
        return self._answer

    def generate_stream(self, context: str, question: str) -> Iterator[str]:
        if self._raise:
            raise LLMError("simulated failure")
        for token in self._answer.split():
            yield token + " "


def _patch(monkeypatch: pytest.MonkeyPatch, chunks: List[RetrievedChunk], llm: _FakeLLM) -> None:
    """Patch the retriever and LLM client used by the service."""
    monkeypatch.setattr(rag_service, "retrieve", lambda q, top_k=None: chunks)
    monkeypatch.setattr(rag_service, "get_llm_client", lambda: llm)


# ---- Blocking variant --------------------------------------------------------
def test_empty_question_prompts_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank question returns a prompt without retrieval or generation."""
    _patch(monkeypatch, [], _FakeLLM())
    result = rag_service.answer_question("   ")
    assert "enter a question" in result.answer.lower()
    assert result.used_context is False


def test_no_context_returns_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """When retrieval finds nothing, the service answers honestly and skips the LLM."""
    _patch(monkeypatch, [], _FakeLLM(answer="should not be used"))
    result = rag_service.answer_question("anything")
    assert "could not find" in result.answer.lower()
    assert result.used_context is False
    assert result.citations == []


def test_happy_path_returns_answer_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """With context, the answer comes from the LLM and citations are populated."""
    chunks = [_chunk(1, 0.9), _chunk(2, 0.8)]
    _patch(monkeypatch, chunks, _FakeLLM(answer="grounded answer [Source 1]"))
    result = rag_service.answer_question("a real question")

    assert result.answer == "grounded answer [Source 1]"
    assert result.used_context is True
    assert result.error is None
    # Citations align 1:1 with retrieved chunks.
    assert len(result.citations) == 2
    assert result.citations[0].index == 1
    assert result.citations[0].source == "doc.pdf"


def test_llm_error_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM failure is surfaced via ``error`` without raising."""
    chunks = [_chunk(1, 0.9)]
    _patch(monkeypatch, chunks, _FakeLLM(raise_error=True))
    result = rag_service.answer_question("a question")

    assert result.error is not None
    assert result.used_context is True
    # Citations are still available even though generation failed.
    assert len(result.citations) == 1


# ---- Streaming variant -------------------------------------------------------
def test_stream_empty_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming a blank question yields a prompt and no citations."""
    _patch(monkeypatch, [], _FakeLLM())
    tokens, citations = rag_service.answer_question_stream("")
    assert "enter a question" in "".join(tokens).lower()
    assert citations == []


def test_stream_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming with no retrieved context yields the 'not found' message."""
    _patch(monkeypatch, [], _FakeLLM())
    tokens, citations = rag_service.answer_question_stream("anything")
    assert "could not find" in "".join(tokens).lower()
    assert citations == []


def test_stream_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming yields the answer tokens and returns citations up front."""
    chunks = [_chunk(1, 0.9)]
    _patch(monkeypatch, chunks, _FakeLLM(answer="streamed grounded answer"))
    tokens, citations = rag_service.answer_question_stream("a question")

    assembled = "".join(tokens).strip()
    assert assembled == "streamed grounded answer"
    assert len(citations) == 1
    assert citations[0].index == 1


def test_stream_llm_error_yields_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A streaming LLM error is surfaced as an inline error token, not a crash."""
    chunks = [_chunk(1, 0.9)]
    _patch(monkeypatch, chunks, _FakeLLM(raise_error=True))
    tokens, citations = rag_service.answer_question_stream("a question")

    text = "".join(tokens)
    assert "error" in text.lower()
    # Citations were resolved before generation, so they are still returned.
    assert len(citations) == 1
