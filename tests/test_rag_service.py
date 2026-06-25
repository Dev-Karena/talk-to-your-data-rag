"""Unit tests for app.services.rag_service.

Verifies the query-time facade end to end with the retriever and LLM client
stubbed out, so the tests run fully offline (no embeddings, no API calls).
Covers: empty-question handling, the no-context path, citation passthrough,
the happy path, and error handling for both blocking and streaming variants.
"""

from __future__ import annotations

from typing import Iterator, List

import pytest

from app.rag.embeddings import EmbeddingError
from app.rag.vector_store import RetrievedChunk, VectorStoreError
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


class _FakeSettings:
    """Minimal settings stand-in exposing only what the service reads."""

    def __init__(self, groq_api_key: str = "test-key") -> None:
        self.groq_api_key = groq_api_key


class _FakeStore:
    """Minimal vector-store stand-in exposing only ``count()``."""

    def __init__(self, count: int = 1) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    chunks: List[RetrievedChunk],
    llm: _FakeLLM,
    *,
    groq_api_key: str = "test-key",
    store_count: int = 1,
    retrieve_error: Exception | None = None,
) -> None:
    """Patch the retriever, LLM client, settings, and store used by the service.

    Keeps tests hermetic: the service now also reads ``get_settings`` (for the
    API key) and ``get_vector_store`` (to distinguish an empty index), so both
    are stubbed here.
    """
    if retrieve_error is not None:
        def _raise(_q: str, top_k: object = None):  # noqa: ANN001
            raise retrieve_error
        monkeypatch.setattr(rag_service, "retrieve", _raise)
    else:
        monkeypatch.setattr(rag_service, "retrieve", lambda q, top_k=None: chunks)
    monkeypatch.setattr(rag_service, "get_llm_client", lambda: llm)
    monkeypatch.setattr(rag_service, "get_settings", lambda: _FakeSettings(groq_api_key))
    monkeypatch.setattr(rag_service, "get_vector_store", lambda: _FakeStore(store_count))


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


# ---- Sprint 2 reliability: scenarios 4, 6, 7 --------------------------------
# (Scenario 5 "empty query" is covered above; scenarios 1/3/8 live in
#  test_validators.py and test_pipeline_reliability.py.)

def test_query_before_indexing_says_no_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4: querying an EMPTY index tells the user to upload first."""
    _patch(monkeypatch, [], _FakeLLM(), store_count=0)
    result = rag_service.answer_question("anything")
    assert "no documents" in result.answer.lower()
    assert result.used_context is False
    assert result.citations == []


def test_no_match_with_documents_says_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4 (counterpart): a populated index that misses says 'not found'."""
    _patch(monkeypatch, [], _FakeLLM(), store_count=5)
    result = rag_service.answer_question("anything")
    assert "could not find" in result.answer.lower()


def test_missing_api_key_returns_specific_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 6: no GROQ_API_KEY -> specific, actionable message; no LLM call."""
    chunks = [_chunk(1, 0.9)]
    _patch(monkeypatch, chunks, _FakeLLM(raise_error=True), groq_api_key="")
    result = rag_service.answer_question("a question")
    assert "groq_api_key" in result.answer.lower()
    assert result.error == "GROQ_API_KEY is not set."


def test_missing_api_key_stream_returns_specific_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 6 (stream): missing key yields the actionable message, not a crash."""
    chunks = [_chunk(1, 0.9)]
    _patch(monkeypatch, chunks, _FakeLLM(raise_error=True), groq_api_key="")
    tokens, citations = rag_service.answer_question_stream("a question")
    assert "groq_api_key" in "".join(tokens).lower()


def test_retrieval_error_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 7: a corrupted store (VectorStoreError) yields a friendly message."""
    _patch(
        monkeypatch, [], _FakeLLM(),
        retrieve_error=VectorStoreError("simulated corrupt store"),
    )
    result = rag_service.answer_question("a question")
    assert result.error is not None
    assert "system error" in result.answer.lower()
    assert result.used_context is False


def test_retrieval_error_stream_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 7 (stream): retrieval failure yields a message instead of a crash."""
    _patch(
        monkeypatch, [], _FakeLLM(),
        retrieve_error=EmbeddingError("simulated embedding backend down"),
    )
    tokens, citations = rag_service.answer_question_stream("a question")
    assert "system error" in "".join(tokens).lower()
    assert citations == []
