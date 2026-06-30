"""Context assembler for routing and merging mixed tool + RAG queries."""

import sys
from typing import Dict, List, Any, Optional
from app.tools import ToolRouter, Intent
from app.services.tool_executor import ToolExecutor
from app.services.context_builder import build_context, AssembledContext, SourceCitation
from app.rag.query_rewriter import rewrite_query
from app.utils.logger import get_logger

logger = get_logger(__name__)

def _get_retrieve():
    rag_service = sys.modules.get("app.services.rag_service")
    if rag_service and hasattr(rag_service, "retrieve"):
        return rag_service.retrieve
    from app.services.retriever import retrieve
    return retrieve

def _get_settings():
    rag_service = sys.modules.get("app.services.rag_service")
    if rag_service and hasattr(rag_service, "get_settings"):
        return rag_service.get_settings
    from app.config.settings import get_settings
    return get_settings

class ContextAssembler:
    """Decomposes, routes, executes, and assembles multi-intent and mixed contexts."""

    def __init__(self, router: Optional[ToolRouter] = None, executor: Optional[ToolExecutor] = None) -> None:
        self.router = router or ToolRouter()
        self.executor = executor or ToolExecutor()

    def assemble(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Decompose query, route sub-queries, run tools/RAG, deduplicate, and assemble context.

        Guarantees zero duplicate chunks and degrades tool errors gracefully.
        """
        # 1. Decompose query into sub-queries
        sub_queries = rewrite_query(query, mode="heuristic")
        if not sub_queries:
            sub_queries = [query]

        # If decomposed, the first element is the original query, which we can skip
        # because the sub-queries fully cover all intents.
        if len(sub_queries) > 1:
            processing_queries = sub_queries[1:]
        else:
            processing_queries = sub_queries

        tool_results = []
        retrieved_chunks = []

        retrieve_fn = _get_retrieve()

        # 2. Route and process each sub-query
        for sub_q in processing_queries:
            intent = self.router.route(sub_q)
            if intent not in (Intent.RAG, Intent.UNKNOWN):
                # Execute tool
                logger.info("Routing sub-query '%s' to tool intent '%s'", sub_q, intent)
                res = self.executor.execute_intent(intent, sub_q)
                tool_results.append(res)
            else:
                # Retrieve context from vector store
                logger.info("Routing sub-query '%s' to document retrieval (RAG)", sub_q)
                chunks = retrieve_fn(sub_q, top_k=top_k)
                retrieved_chunks.extend(chunks)

        # 3. Deduplicate retrieved chunks by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        # 4. Build retrieved context using existing context builder
        assembled_rag = build_context(unique_chunks)

        # 5. Build tool citations and format tool context text
        tool_citations = []
        tool_context_lines = []

        # Sequential numbering continues from the end of the RAG citations
        curr_cit_index = len(assembled_rag.citations) + 1

        for t_res in tool_results:
            tool_name = t_res.get("tool", "tool")
            
            if not t_res.get("success", False):
                # Graceful degradation on tool failures
                error_msg = t_res.get("error", "unknown error")
                logger.warning("Tool '%s' execution failed: %s", tool_name, error_msg)
                tool_context_lines.append(f"Tool {tool_name} failed: {error_msg}")
                continue

            content = t_res.get("content", "")
            data = t_res.get("data")

            tool_context_lines.append(f"[Tool: {tool_name}]\n{content}")

            # Generate mock SourceCitations for the UI to display tool results
            if tool_name == "web_search" and isinstance(data, list):
                for item in data:
                    href = item.get("href") or "Web Search"
                    title = item.get("title") or "Web Search Result"
                    snippet = item.get("body") or ""
                    tool_citations.append(SourceCitation(
                        index=curr_cit_index,
                        source=href,
                        page_number=1,
                        chunk_index=curr_cit_index,
                        chunk_id=f"web_search::{curr_cit_index}",
                        score=1.0,
                        text=f"{title}\n{snippet}"
                    ))
                    curr_cit_index += 1
            else:
                tool_citations.append(SourceCitation(
                    index=curr_cit_index,
                    source=tool_name.capitalize(),
                    page_number=1,
                    chunk_index=1,
                    chunk_id=f"{tool_name}::{curr_cit_index}",
                    score=1.0,
                    text=content
                ))
                curr_cit_index += 1

        # 6. Combine context texts
        combined_context_parts = []
        if tool_context_lines:
            combined_context_parts.append("TOOL RESULTS:\n" + "\n\n".join(tool_context_lines))
        if assembled_rag.context_text:
            combined_context_parts.append("DOCUMENT CONTEXT:\n" + assembled_rag.context_text)

        combined_context_text = "\n\n".join(combined_context_parts)
        
        # Combine RAG citations and tool citations
        combined_citations = list(assembled_rag.citations) + tool_citations

        return {
            "context_text": combined_context_text,
            "citations": combined_citations,
            "tool_results": tool_results
        }
