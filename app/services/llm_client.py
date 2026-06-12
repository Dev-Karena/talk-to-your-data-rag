"""Groq LLM client wrapper.

Wraps LangChain's ``ChatGroq`` to generate grounded, citation-bearing answers
with a Groq-hosted model (default: Llama 3.3 70B). Responsibilities:

    * Load the API key from configuration (never hardcoded).
    * Enforce a strict, citation-aware system prompt.
    * Offer both a blocking (`generate`) and a streaming (`generate_stream`)
      interface for the UI.
    * Translate SDK errors into a single typed exception.

Usage:
    >>> from app.services.llm_client import get_llm_client
    >>> client = get_llm_client()
    >>> answer = client.generate(context_text, "What was Q4 revenue?")
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# System prompt: pins the model to the supplied context and the [Source N]
# citation convention used by the context builder.
_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant for a document retrieval "
    "system. Answer the user's question using ONLY the information in the "
    "provided context passages. Each passage is labeled with a marker like "
    "[Source 1], [Source 2], and so on.\n\n"
    "Rules:\n"
    "1. Base every claim strictly on the context. Do not use outside "
    "knowledge or make assumptions.\n"
    "2. Cite the supporting passage inline using its marker, e.g. "
    "'Revenue grew 18% [Source 1].' Cite every factual statement.\n"
    "3. If the answer is not contained in the context, reply exactly: "
    "\"I could not find this information in the provided documents.\" Do not "
    "fabricate an answer.\n"
    "4. Be concise and factual. Prefer quoting figures and terms exactly as "
    "they appear in the context."
)

# Message template combining the retrieved context with the user's question.
_USER_TEMPLATE = (
    "Context passages:\n"
    "----------------\n"
    "{context}\n"
    "----------------\n\n"
    "Question: {question}\n\n"
    "Answer the question using only the context above, citing sources with "
    "their [Source N] markers."
)

# Returned when retrieval found nothing — avoids an unnecessary LLM call.
_NO_CONTEXT_ANSWER = "I could not find this information in the provided documents."


class LLMError(Exception):
    """Raised when answer generation fails."""


class LLMClient:
    """Thin wrapper around ``langchain_groq.ChatGroq`` for grounded Q&A."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        self._settings = settings
        # The key is read from validated settings (sourced from the
        # environment) and passed explicitly — never hardcoded.
        self._client = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
        )
        logger.info("LLM client ready (model=%s).", settings.llm_model)

    def _build_messages(self, context: str, question: str) -> List[BaseMessage]:
        """Construct the messages list for a single Q&A turn."""
        user_content = _USER_TEMPLATE.format(context=context, question=question)
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    def generate(self, context: str, question: str) -> str:
        """Generate a complete answer (blocking).

        Args:
            context: The assembled, numbered context block.
            question: The user's question.

        Returns:
            The model's answer text. If ``context`` is empty, returns the
            standard "not found" message without calling the API.

        Raises:
            LLMError: If the API call fails.
        """
        if not context.strip():
            return _NO_CONTEXT_ANSWER

        try:
            response = self._client.invoke(self._build_messages(context, question))
        except Exception as exc:  # normalize SDK/transport errors into LLMError
            logger.error("Groq API error: %s", exc)
            raise LLMError(f"LLM request failed: {exc}") from exc

        answer = (response.content or "").strip()
        logger.info("Generated answer (%d chars).", len(answer))
        return answer

    def generate_stream(self, context: str, question: str) -> Iterator[str]:
        """Generate an answer as a stream of text deltas (for live UI).

        Args:
            context: The assembled, numbered context block.
            question: The user's question.

        Yields:
            Successive text fragments of the answer.

        Raises:
            LLMError: If the API call fails.
        """
        if not context.strip():
            yield _NO_CONTEXT_ANSWER
            return

        try:
            for chunk in self._client.stream(self._build_messages(context, question)):
                text = chunk.content
                if text:
                    yield text
        except Exception as exc:  # normalize SDK/transport errors into LLMError
            logger.error("Groq streaming error: %s", exc)
            raise LLMError(f"LLM streaming request failed: {exc}") from exc


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return a cached, process-wide :class:`LLMClient` instance."""
    return LLMClient(get_settings())
