"""RAG ingestion package: load, clean, chunk, embed, and persist documents."""

from app.rag.chunker import Chunk, chunk_pages
from app.rag.cleaner import clean_text
from app.rag.embeddings import BaseEmbedder, get_embedder
from app.rag.loader import PageDocument, load_pdf
from app.rag.pipeline import IngestResult, IngestStatus, ingest_document
from app.rag.vector_store import RetrievedChunk, get_vector_store

__all__ = [
    "Chunk",
    "chunk_pages",
    "clean_text",
    "BaseEmbedder",
    "get_embedder",
    "PageDocument",
    "load_pdf",
    "IngestResult",
    "IngestStatus",
    "ingest_document",
    "RetrievedChunk",
    "get_vector_store",
]
