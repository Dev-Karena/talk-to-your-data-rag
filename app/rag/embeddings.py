"""Embedding model factory.

Provides a single, uniform interface (:class:`BaseEmbedder`) over multiple
embedding backends, and a factory (:func:`get_embedder`) that returns the one
selected in configuration. This isolates the rest of the pipeline from any
specific provider — switching backends is a ``.env`` change, not a code change.

Backends:
    * ``local``  — sentence-transformers, runs fully offline, no API key.
    * ``voyage`` — Voyage AI API (requires ``VOYAGE_API_KEY``).
    * ``openai`` — OpenAI API (requires ``OPENAI_API_KEY``).

Both documents and queries are supported. Some modern models (e.g. BGE) benefit
from an instruction prefix on the *query* side; that asymmetry is handled inside
each embedder.

Usage:
    >>> from app.rag.embeddings import get_embedder
    >>> embedder = get_embedder()
    >>> vectors = embedder.embed_documents(["hello world"])
    >>> query_vec = embedder.embed_query("what is greeted?")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from app.config.settings import EmbeddingBackend, Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Models that retrieve best when the query is prefixed with an instruction.
# (BGE family is trained with this asymmetry.)
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """Raised when an embedding backend fails to initialize or embed."""


class BaseEmbedder(ABC):
    """Abstract embedding interface used throughout the RAG pipeline."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks.

        Args:
            texts: The chunk texts to embed.

        Returns:
            One embedding vector per input text, in the same order.
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query.

        Args:
            text: The query string.

        Returns:
            The query embedding vector.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the active model/backend."""


class LocalEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers (offline, no API key)."""

    def __init__(self, model_name: str) -> None:
        try:
            # Imported lazily so the heavy dependency loads only when used.
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it with "
                "'pip install sentence-transformers'."
            ) from exc

        self._model_name = model_name
        # Detect BGE-style models to apply the query instruction prefix.
        self._is_bge = "bge" in model_name.lower()
        try:
            logger.info("Loading local embedding model '%s'...", model_name)
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - surface load failures uniformly
            raise EmbeddingError(
                f"Failed to load local embedding model '{model_name}': {exc}"
            ) from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        query = f"{_BGE_QUERY_INSTRUCTION}{text}" if self._is_bge else text
        vector = self._model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


class VoyageEmbedder(BaseEmbedder):
    """Embeddings via the Voyage AI API (requires an API key)."""

    def __init__(self, model_name: str, api_key: str) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "voyageai is not installed. Install it with 'pip install voyageai'."
            ) from exc

        self._model_name = model_name
        try:
            self._client = voyageai.Client(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to initialize Voyage client: {exc}") from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._client.embed(
            texts, model=self._model_name, input_type="document"
        )
        return result.embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self._client.embed(
            [text], model=self._model_name, input_type="query"
        )
        return result.embeddings[0]

    @property
    def name(self) -> str:
        return f"voyage:{self._model_name}"


class OpenAIEmbedder(BaseEmbedder):
    """Embeddings via the OpenAI API (requires an API key)."""

    def __init__(self, model_name: str, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "openai is not installed. Install it with 'pip install openai'."
            ) from exc

        self._model_name = model_name
        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to initialize OpenAI client: {exc}") from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            input=texts, model=self._model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            input=[text], model=self._model_name
        )
        return response.data[0].embedding

    @property
    def name(self) -> str:
        return f"openai:{self._model_name}"


def _build_embedder(settings: Settings) -> BaseEmbedder:
    """Instantiate the embedder selected in settings."""
    backend = settings.embedding_backend

    if backend is EmbeddingBackend.LOCAL:
        return LocalEmbedder(settings.embedding_model)

    if backend is EmbeddingBackend.VOYAGE:
        # Settings validation guarantees the key is present, but be explicit.
        if not settings.voyage_api_key:
            raise EmbeddingError("VOYAGE_API_KEY is required for the voyage backend.")
        return VoyageEmbedder(settings.embedding_model, settings.voyage_api_key)

    if backend is EmbeddingBackend.OPENAI:
        if not settings.openai_api_key:
            raise EmbeddingError("OPENAI_API_KEY is required for the openai backend.")
        return OpenAIEmbedder(settings.embedding_model, settings.openai_api_key)

    # Unreachable given the enum, but defensive against future additions.
    raise EmbeddingError(f"Unsupported embedding backend: {backend}")


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """Return a cached embedder for the configured backend.

    The model is built once per process (``lru_cache``), which is important for
    the local backend where loading the model is expensive.

    Returns:
        The active :class:`BaseEmbedder`.

    Raises:
        EmbeddingError: If the backend cannot be initialized.
    """
    settings = get_settings()
    embedder = _build_embedder(settings)
    logger.info("Embedding backend ready: %s", embedder.name)
    return embedder
