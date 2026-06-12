"""Services package: query-time retrieval, context assembly, and answering."""

from app.services.context_builder import (
    AssembledContext,
    SourceCitation,
    build_context,
)
from app.services.llm_client import LLMClient, get_llm_client
from app.services.rag_service import (
    RAGResponse,
    answer_question,
    answer_question_stream,
)
from app.services.retriever import retrieve

__all__ = [
    "AssembledContext",
    "SourceCitation",
    "build_context",
    "LLMClient",
    "get_llm_client",
    "RAGResponse",
    "answer_question",
    "answer_question_stream",
    "retrieve",
]
