"""RAG service facade.

The single public entry point the UI uses to answer questions. It orchestrates
the query-time flow:

    retrieve -> build context -> generate answer (+ carry citations)

By depending only on this facade, the UI is decoupled from retrieval, context
assembly, and the LLM. Both a blocking (`answer_question`) and a streaming
(`answer_question_stream`) variant are provided.

Usage:
    >>> from app.services.rag_service import answer_question
    >>> result = answer_question("What was Q4 revenue?")
    >>> result.answer
    'Revenue grew 18% ... [Source 1].'
    >>> result.citations[0].label
    'report.pdf · p.4 · chunk 12'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from app.services.context_builder import (
    AssembledContext,
    SourceCitation,
    build_context,
)
from app.services.llm_client import LLMError, get_llm_client
from app.services.retriever import retrieve
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Shown when retrieval returns nothing relevant (e.g. empty index).
_NO_CONTEXT_MESSAGE = "I could not find this information in the provided documents."


@dataclass(frozen=True)
class RAGResponse:
    """A complete answer plus the sources that support it.

    Attributes:
        answer: The generated answer text.
        citations: Source citations aligned with the ``[Source N]`` markers in
            ``answer``. Empty when no relevant context was found.
        used_context: Whether any retrieved context backed the answer.
        error: Optional error message if generation failed; ``None`` on success.
    """

    answer: str
    citations: List[SourceCitation] = field(default_factory=list)
    used_context: bool = False
    error: Optional[str] = None


def _retrieve_and_assemble(question: str, top_k: Optional[int]) -> AssembledContext:
    """Run retrieval and assemble the context for a question."""
    chunks = retrieve(question, top_k=top_k)
    return build_context(chunks)


def answer_question(question: str, top_k: Optional[int] = None) -> RAGResponse:
    """Answer a question over the indexed documents (blocking).

    Args:
        question: The user's natural-language question.
        top_k: Optional override for how many chunks to retrieve.

    Returns:
        A :class:`RAGResponse` with the answer and supporting citations. On
        failure, ``error`` is populated and ``answer`` carries a friendly
        message — this function does not raise for generation errors.
    """
    question = (question or "").strip()
    if not question:
        return RAGResponse(answer="Please enter a question.", used_context=False)

    assembled = _retrieve_and_assemble(question, top_k)

    # No relevant context -> answer honestly, skip the LLM call.
    if assembled.is_empty:
        logger.info("No context retrieved; returning 'not found' response.")
        return RAGResponse(
            answer=_NO_CONTEXT_MESSAGE,
            citations=[],
            used_context=False,
        )

    try:
        answer = get_llm_client().generate(assembled.context_text, question)
    except LLMError as exc:
        logger.error("Answer generation failed: %s", exc)
        return RAGResponse(
            answer="Sorry, I couldn't generate an answer due to an error.",
            citations=assembled.citations,
            used_context=True,
            error=str(exc),
        )

    return RAGResponse(
        answer=answer,
        citations=assembled.citations,
        used_context=True,
    )


def answer_question_stream(
    question: str, top_k: Optional[int] = None
) -> tuple[Iterator[str], List[SourceCitation]]:
    """Answer a question with a streamed answer plus its citations.

    Citations are resolved up front (retrieval happens before generation), so
    the UI can render the "Sources" section immediately and stream the answer
    text as it arrives.

    Args:
        question: The user's natural-language question.
        top_k: Optional override for how many chunks to retrieve.

    Returns:
        A tuple of ``(answer_token_iterator, citations)``. When there is no
        relevant context, the iterator yields a single "not found" message and
        ``citations`` is empty.
    """
    question = (question or "").strip()
    if not question:
        return iter(["Please enter a question."]), []

    assembled = _retrieve_and_assemble(question, top_k)

    if assembled.is_empty:
        return iter([_NO_CONTEXT_MESSAGE]), []

    def _token_stream() -> Iterator[str]:
        """Yield answer tokens, converting LLM errors into a visible message."""
        try:
            yield from get_llm_client().generate_stream(
                assembled.context_text, question
            )
        except LLMError as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"\n\n[Error generating answer: {exc}]"

    return _token_stream(), assembled.citations
