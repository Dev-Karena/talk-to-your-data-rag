"""BM25 sparse index store.

Wraps the rank-bm25 library to build, cache, and search BM25 indexes
generated from the contents of the ChromaDB vector store.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from rank_bm25 import BM25Okapi

from app.rag.vector_store import get_vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", (text or "").lower())


class BM25Store:
    """An in-memory BM25 index that is lazy-loaded and synchronized with ChromaDB."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: List[str] = []
        self._cached_hashes: Set[str] = set()

    def _build_index(self, texts: List[str], chunk_ids: List[str]) -> None:
        """Tokenize documents and build the rank-bm25 index."""
        tokenized_corpus = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_ids = chunk_ids

    def sync(self) -> None:
        """Synchronize the in-memory BM25 index with the on-disk ChromaDB collection.

        Compares cached source document hashes with the store's current hashes
        to avoid redundant rebuilds. Rebuilds from scratch on cache miss/stale.
        """
        store = get_vector_store()
        try:
            sources = store.list_sources()
            current_hashes = set(sources.keys())
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to list sources from vector store for BM25 sync: %s", exc)
            return

        if self._bm25 is not None and current_hashes == self._cached_hashes:
            return  # Cache is valid.

        logger.info("BM25 index cache stale/empty. Rebuilding from vector store...")
        try:
            texts, ids = store.get_all_chunks()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch all chunks from vector store for BM25: %s", exc)
            return

        if not texts:
            logger.warning("No chunks found in vector store. BM25 index cleared.")
            self._bm25 = None
            self._chunk_ids = []
            self._cached_hashes = set()
            return

        self._build_index(texts, ids)
        self._cached_hashes = current_hashes
        logger.info("BM25 index built successfully with %d chunks.", len(ids))

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Score all chunks against the query and return the top_k.

        Args:
            query: The user's search query.
            top_k: Number of results to return.

        Returns:
            A list of (chunk_id, score) pairs ordered by descending BM25 score.
        """
        self.sync()
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Pair chunk_ids with their respective scores
        results = list(zip(self._chunk_ids, scores))
        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Process-wide cached singleton
_bm25_store = BM25Store()


def get_bm25_store() -> BM25Store:
    """Return a cached, process-wide BM25Store instance."""
    return _bm25_store
