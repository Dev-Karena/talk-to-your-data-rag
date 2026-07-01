"""Service layer coordinating mixed context assembly and answer generation."""

import sys
from typing import Iterator, List, Optional, Tuple
from app.services.context_assembler import ContextAssembler
from app.services.context_builder import SourceCitation
from app.utils.logger import get_logger

logger = get_logger(__name__)

def _get_llm_client():
    rag_service = sys.modules.get("app.services.rag_service")
    if rag_service and hasattr(rag_service, "get_llm_client"):
        return rag_service.get_llm_client()
    from app.services.llm_client import get_llm_client
    return get_llm_client()

class AnswerGenerationError(Exception):
    """Raised when answer generation fails, preserving citations."""
    def __init__(self, message: str, citations: List[SourceCitation]) -> None:
        super().__init__(message)
        self.citations = citations

class AnswerGenerator:
    """Manages context assembly and answer generation via the configured LLM client."""

    def __init__(self, context_assembler: Optional[ContextAssembler] = None) -> None:
        self.assembler = context_assembler or ContextAssembler()

    def generate(self, question: str, top_k: Optional[int] = None) -> dict:
        """Generate a complete response for the given question (blocking).

        Returns a dict:
        {
            "answer": str,
            "citations": List[SourceCitation],
            "used_context": bool
        }
        """
        assembled = self.assembler.assemble(question, top_k=top_k)
        
        # Check if empty context/no tools
        if not assembled["context_text"] and not assembled["citations"]:
            return {
                "answer": "No relevant context or tools were triggered to answer this question.",
                "citations": [],
                "used_context": False
            }

        # Short-circuit pure tool queries (ISSUE 3)
        successful_tools = [r for r in assembled["tool_results"] if r.get("success", False)]
        has_docs = assembled.get("has_document_context", False)
        
        if successful_tools and not has_docs:
            tool_answer = "\n\n".join(t["content"] for t in successful_tools)
            return {
                "answer": tool_answer,
                "citations": assembled["citations"],
                "used_context": True
            }

        llm_client = _get_llm_client()
        try:
            answer = llm_client.generate(assembled["context_text"], question)
            return {
                "answer": answer,
                "citations": assembled["citations"],
                "used_context": True
            }
        except Exception as exc:
            raise AnswerGenerationError(str(exc), assembled["citations"]) from exc

    def generate_stream(self, question: str, top_k: Optional[int] = None) -> Tuple[Iterator[str], List[SourceCitation]]:
        """Generate a streamed response for the given question (streaming).

        Returns a tuple of (token_stream_iterator, citations).
        """
        assembled = self.assembler.assemble(question, top_k=top_k)
        
        if not assembled["context_text"] and not assembled["citations"]:
            empty_msg_iter = iter(["No relevant context or tools were triggered to answer this question."])
            return empty_msg_iter, []

        # Short-circuit pure tool queries (ISSUE 3)
        successful_tools = [r for r in assembled["tool_results"] if r.get("success", False)]
        has_docs = assembled.get("has_document_context", False)
        
        if successful_tools and not has_docs:
            tool_answer = "\n\n".join(t["content"] for t in successful_tools)
            return iter([tool_answer]), assembled["citations"]

        llm_client = _get_llm_client()

        def _safe_stream() -> Iterator[str]:
            try:
                yield from llm_client.generate_stream(assembled["context_text"], question)
            except Exception as exc:
                logger.error("Streaming generation failed: %s", exc)
                yield f"\n\n[Error generating answer: {exc}]"

        return _safe_stream(), assembled["citations"]
