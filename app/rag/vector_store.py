"""ChromaDB vector store.

Encapsulates every interaction with ChromaDB: a persistent on-disk client, a
single collection for document chunks, idempotent inserts, similarity search
with scores, deduplication by document hash, and maintenance operations
(clear, list, count).

The store is embedding-agnostic: callers compute embeddings with the
configured embedder (see :mod:`app.rag.embeddings`) and pass vectors in. We set
the collection's distance metric to cosine to match the normalized embeddings
those backends produce.

Usage:
    >>> from app.rag.vector_store import get_vector_store
    >>> store = get_vector_store()
    >>> store.add_chunks(chunks, embeddings)
    >>> results = store.query(query_embedding, top_k=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Set

import chromadb
from chromadb.config import Settings as ChromaClientSettings

from app.config.settings import get_settings
from app.rag.chunker import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a similarity search.

    Attributes:
        chunk_id: Stable id of the matched chunk.
        text: The chunk's text content.
        source: Source document display name.
        page_number: 1-based page number.
        chunk_index: 0-based chunk index within its page.
        doc_hash: Source document content hash.
        score: Similarity score in ``[0, 1]`` (higher = more relevant),
            derived from cosine distance as ``1 - distance``.
    """

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    doc_hash: str
    score: float


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""


class VectorStore:
    """Thin, typed wrapper around a persistent ChromaDB collection."""

    def __init__(self) -> None:
        settings = get_settings()
        self._collection_name = settings.chroma_collection_name
        try:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=ChromaClientSettings(anonymized_telemetry=False),
            )
            # cosine matches our normalized embeddings; metric is fixed at
            # creation time and persisted with the collection.
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001 - surface init failures uniformly
            raise VectorStoreError(
                f"Failed to open vector store at "
                f"'{settings.chroma_persist_dir}': {exc}"
            ) from exc
        logger.info(
            "Vector store ready: collection='%s', count=%d",
            self._collection_name,
            self._collection.count(),
        )

    # ---- Writes --------------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Insert (or update) chunks and their embeddings.

        Uses ``upsert`` keyed on each chunk's stable ``chunk_id``, so
        re-indexing the same document is idempotent (no duplicate records).

        Args:
            chunks: Chunks to store.
            embeddings: One embedding vector per chunk, in the same order.

        Raises:
            VectorStoreError: If the lengths mismatch or the write fails.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}."
            )

        try:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[c.metadata() for c in chunks],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add chunks: {exc}") from exc

        logger.info("Upserted %d chunk(s) into '%s'.", len(chunks), self._collection_name)

    # ---- Reads ---------------------------------------------------------------
    def query(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks to a query embedding.

        Args:
            query_embedding: The embedded query vector.
            top_k: Maximum number of chunks to return.

        Returns:
            Matched chunks ordered by descending similarity score. Empty if the
            collection has no documents.

        Raises:
            VectorStoreError: If the query fails.
        """
        if self._collection.count() == 0:
            logger.warning("Query attempted on an empty vector store.")
            return []

        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        return self._parse_query_result(result)

    @staticmethod
    def _parse_query_result(result: Dict) -> List[RetrievedChunk]:
        """Convert a raw Chroma query result into typed :class:`RetrievedChunk`s."""
        # Chroma returns lists-of-lists (one inner list per query). We send one
        # query, so index 0. Missing keys default to empty for safety.
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: List[RetrievedChunk] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            meta = meta or {}
            # cosine distance in [0, 2]; similarity = 1 - distance in [-1, 1].
            # Clamp to [0, 1] for a clean, user-facing relevance score.
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(meta.get("chunk_id", "")),
                    text=text or "",
                    source=str(meta.get("source", "unknown")),
                    page_number=int(meta.get("page_number", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    doc_hash=str(meta.get("doc_hash", "")),
                    score=score,
                )
            )
        return retrieved

    # ---- Deduplication & introspection --------------------------------------
    def document_exists(self, doc_hash: str) -> bool:
        """Return ``True`` if any chunk with the given document hash is stored.

        Used to skip re-indexing a document that is already in the store.
        """
        try:
            result = self._collection.get(
                where={"doc_hash": doc_hash}, limit=1, include=[]
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Existence check failed: {exc}") from exc
        return bool(result.get("ids"))

    def list_sources(self) -> Dict[str, str]:
        """Return a mapping of ``doc_hash -> source`` for all indexed documents.

        Returns:
            One entry per unique document currently in the store.
        """
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to list sources: {exc}") from exc

        sources: Dict[str, str] = {}
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            doc_hash = str(meta.get("doc_hash", ""))
            if doc_hash and doc_hash not in sources:
                sources[doc_hash] = str(meta.get("source", "unknown"))
        return sources

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()

    # ---- Maintenance ---------------------------------------------------------
    def clear(self) -> None:
        """Delete all documents by dropping and recreating the collection.

        This fully empties the vector store (used by the UI's "Clear database"
        control).
        """
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to clear vector store: {exc}") from exc
        logger.info("Vector store '%s' cleared.", self._collection_name)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return a cached, process-wide :class:`VectorStore` instance."""
    return VectorStore()
