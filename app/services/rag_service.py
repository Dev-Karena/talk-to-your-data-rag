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

from app.config.settings import get_settings
from app.rag.embeddings import EmbeddingError
from app.rag.vector_store import VectorStoreError, get_vector_store
from app.services.context_builder import (
    AssembledContext,
    SourceCitation,
    build_context,
)
from app.services.llm_client import LLMError, get_llm_client
from app.services.retriever import retrieve
from app.utils.logger import get_logger
from app.services.answer_generator import AnswerGenerator, AnswerGenerationError

logger = get_logger(__name__)

# Shown when retrieval returns nothing relevant but documents *are* indexed.
_NO_CONTEXT_MESSAGE = "I could not find this information in the provided documents."

# Shown when the store is empty — distinguishes "nothing indexed yet" from
# "indexed, but not found" so the user knows to upload a document first.
_NO_DOCUMENTS_MESSAGE = (
    "No documents have been indexed yet. Upload a PDF in the sidebar to get started."
)

# Shown when retrieval itself fails (corrupted store, embedding backend down).
_RETRIEVAL_ERROR_MESSAGE = (
    "Sorry, I couldn't search your documents due to a system error. "
    "Please try again, and check the application logs if this persists."
)

# Shown when context was found but answer generation is impossible (no API key).
_MISSING_KEY_MESSAGE = (
    "Answer generation is unavailable because GROQ_API_KEY is not set. "
    "Add it to your .env file and restart the app."
)


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
    """Run retrieval and assemble the context for a question.

    Raises:
        VectorStoreError: If the vector store is unreadable.
        EmbeddingError: If the embedding backend cannot embed the query.
    """
    chunks = retrieve(question, top_k=top_k)
    return build_context(chunks)



def _empty_context_message() -> str:
    """Choose the right 'no answer' message for an empty retrieval result.

    Distinguishes an empty index ("nothing uploaded yet") from a populated index
    that simply had no relevant match. A store read failure here is non-fatal —
    we fall back to the generic message rather than surfacing an error.
    """
    try:
        if get_vector_store().count() == 0:
            return _NO_DOCUMENTS_MESSAGE
    except VectorStoreError as exc:  # don't let a status check mask the answer
        logger.warning("Could not check store size for empty-context message: %s", exc)
    return _NO_CONTEXT_MESSAGE


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

    # Generation is impossible without an API key — short-circuit before doing any
    # work, with a specific, actionable message.
    if not get_settings().groq_api_key:
        logger.warning("Question received but GROQ_API_KEY is missing; cannot generate.")
        return RAGResponse(
            answer=_MISSING_KEY_MESSAGE,
            used_context=False,
            error="GROQ_API_KEY is not set.",
        )

    try:
        generator = AnswerGenerator()
        res = generator.generate(question, top_k=top_k)
        
        # If no context was used, answer honestly
        if not res["used_context"]:
            return RAGResponse(
                answer=_empty_context_message(),
                citations=[],
                used_context=False,
            )
            
        return RAGResponse(
            answer=res["answer"],
            citations=res["citations"],
            used_context=res["used_context"]
        )
    except (VectorStoreError, EmbeddingError) as exc:
        logger.error("Retrieval failed: %s", exc)
        return RAGResponse(
            answer=_RETRIEVAL_ERROR_MESSAGE,
            used_context=False,
            error=str(exc),
        )
    except AnswerGenerationError as exc:
        logger.error("Answer generation failed: %s", exc)
        return RAGResponse(
            answer="Sorry, I couldn't generate an answer due to an error.",
            citations=exc.citations,
            used_context=True,
            error=str(exc),
        )
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        return RAGResponse(
            answer="Sorry, I couldn't generate an answer due to an error.",
            citations=[],
            used_context=False,
            error=str(exc),
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

    # Generation is impossible without an API key — short-circuit before retrieval.
    if not get_settings().groq_api_key:
        logger.warning("Question received but GROQ_API_KEY is missing; cannot generate (stream).")
        return iter([_MISSING_KEY_MESSAGE]), []

    try:
        generator = AnswerGenerator()
        stream, citations = generator.generate_stream(question, top_k=top_k)
        if not citations:
            return iter([_empty_context_message()]), []
        return stream, citations
    except (VectorStoreError, EmbeddingError) as exc:
        logger.error("Retrieval failed (stream): %s", exc)
        return iter([_RETRIEVAL_ERROR_MESSAGE]), []
    except Exception as exc:
        logger.error("Streaming generation failed: %s", exc)
        return iter([f"\n\n[Error generating answer: {exc}]"]), []
