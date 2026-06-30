import pytest
from unittest.mock import patch, MagicMock
from app.services.context_assembler import ContextAssembler
from app.rag.vector_store import RetrievedChunk

@pytest.fixture
def assembler():
    return ContextAssembler()

def test_context_assembler_chunk_deduplication(assembler):
    # Setup mock chunks containing duplicates
    chunk1 = RetrievedChunk(
        chunk_id="chunk_1", text="text 1", source="doc.pdf", page_number=1, chunk_index=0, doc_hash="hash1", score=0.9
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk_1", text="text 1", source="doc.pdf", page_number=1, chunk_index=0, doc_hash="hash1", score=0.9
    )
    chunk3 = RetrievedChunk(
        chunk_id="chunk_2", text="text 2", source="doc.pdf", page_number=1, chunk_index=1, doc_hash="hash1", score=0.8
    )

    mock_retrieve = MagicMock(return_value=[chunk1, chunk2, chunk3])
    with patch("app.services.context_assembler._get_retrieve", return_value=mock_retrieve):
        # Mock rewrite_query to return one query that targets retrieval
        with patch("app.services.context_assembler.rewrite_query", return_value=["explain linear regression"]):
            res = assembler.assemble("explain linear regression")
            
            # Verify citations list contains only 2 entries (meaning deduplicated!)
            assert len(res["citations"]) == 2
            assert res["citations"][0].chunk_id == "chunk_1"
            assert res["citations"][1].chunk_id == "chunk_2"

def test_context_assembler_graceful_tool_degradation(assembler):
    # Mock a tool return representing execution failure
    mock_failed_exec = {
        "success": False,
        "tool": "calculator",
        "error": "Division by zero"
    }
    with patch.object(assembler.executor, "execute_intent", return_value=mock_failed_exec):
        with patch("app.services.context_assembler.rewrite_query", return_value=["1/0"]):
            res = assembler.assemble("1/0")
            assert "failed: Division by zero" in res["context_text"]
            assert len(res["citations"]) == 0 # Failed tool does not produce citations
            assert len(res["tool_results"]) == 1
            assert res["tool_results"][0]["success"] is False
