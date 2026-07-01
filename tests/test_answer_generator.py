import pytest
from unittest.mock import patch, MagicMock
from app.services.answer_generator import AnswerGenerator
from app.services.context_builder import SourceCitation

@pytest.fixture
def answer_gen():
    return AnswerGenerator()

def test_generate_answer_empty_context(answer_gen):
    # Setup mock assembler to return empty context
    mock_empty_assemble = {
        "context_text": "",
        "citations": [],
        "tool_results": []
    }
    with patch.object(answer_gen.assembler, "assemble", return_value=mock_empty_assemble):
        res = answer_gen.generate("What is 1+1?")
        assert res["used_context"] is False
        assert "No relevant context" in res["answer"]
        assert len(res["citations"]) == 0

def test_generate_answer_success(answer_gen):
    mock_success_assemble = {
        "context_text": "TOOL RESULTS:\n[Tool: calculator]\n2",
        "citations": [
            SourceCitation(index=1, source="Calculator", page_number=1, chunk_index=1, chunk_id="calculator::1", score=1.0, text="2", is_tool=True)
        ],
        "tool_results": [{"success": True, "tool": "calculator", "content": "2"}],
        "has_document_context": True
    }
    with patch.object(answer_gen.assembler, "assemble", return_value=mock_success_assemble):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The calculation result is 2."
        with patch("app.services.answer_generator._get_llm_client", return_value=mock_llm):
            res = answer_gen.generate("What is 1+1?")
            assert res["used_context"] is True
            assert res["answer"] == "The calculation result is 2."
            assert len(res["citations"]) == 1
            assert res["citations"][0].source == "Calculator"

def test_calculator_tool_direct_answer(answer_gen):
    mock_calc_assemble = {
        "context_text": "### TOOL RESULT ###\nTool: calculator\nAnswer:\n36895",
        "citations": [
            SourceCitation(index=1, source="Calculator", page_number=1, chunk_index=1, chunk_id="calculator::1", score=1.0, text="36895", is_tool=True)
        ],
        "tool_results": [{"success": True, "tool": "calculator", "content": "36895"}],
        "has_document_context": False
    }
    with patch.object(answer_gen.assembler, "assemble", return_value=mock_calc_assemble):
        with patch("app.services.answer_generator._get_llm_client") as mock_client_factory:
            res = answer_gen.generate("Calculate 785*47")
            assert res["answer"] == "36895"
            assert res["used_context"] is True
            assert len(res["citations"]) == 1
            assert res["citations"][0].is_tool is True
            # Verify LLM was NOT invoked
            mock_client_factory.assert_not_called()

def test_datetime_tool_direct_answer(answer_gen):
    mock_time_assemble = {
        "context_text": "### TOOL RESULT ###\nTool: datetime\nAnswer:\n2026-07-01 10:56:00",
        "citations": [
            SourceCitation(index=1, source="Datetime", page_number=1, chunk_index=1, chunk_id="datetime::1", score=1.0, text="2026-07-01 10:56:00", is_tool=True)
        ],
        "tool_results": [{"success": True, "tool": "datetime", "content": "2026-07-01 10:56:00"}],
        "has_document_context": False
    }
    with patch.object(answer_gen.assembler, "assemble", return_value=mock_time_assemble):
        with patch("app.services.answer_generator._get_llm_client") as mock_client_factory:
            res = answer_gen.generate("What is the current time?")
            assert res["answer"] == "2026-07-01 10:56:00"
            assert res["used_context"] is True
            mock_client_factory.assert_not_called()

def test_mixed_query_still_uses_llm(answer_gen):
    mock_mixed_assemble = {
        "context_text": "TOOL RESULTS:\n### TOOL RESULT ###\nTool: calculator\nAnswer:\n2\n\nDOCUMENT CONTEXT:\nSome text",
        "citations": [
            SourceCitation(index=1, source="Calculator", page_number=1, chunk_index=1, chunk_id="calculator::1", score=1.0, text="2", is_tool=True),
            SourceCitation(index=2, source="doc.pdf", page_number=1, chunk_index=0, chunk_id="doc::1", score=0.9, text="Some text", is_tool=False)
        ],
        "tool_results": [{"success": True, "tool": "calculator", "content": "2"}],
        "has_document_context": True
    }
    with patch.object(answer_gen.assembler, "assemble", return_value=mock_mixed_assemble):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "LLM mixed answer"
        with patch("app.services.answer_generator._get_llm_client", return_value=mock_llm):
            res = answer_gen.generate("Calculate 1+1 and show details from doc")
            assert res["answer"] == "LLM mixed answer"
            assert res["used_context"] is True
            assert len(res["citations"]) == 2

def test_tool_citations_rendering():
    from app.ui.components import render_citations
    cit1 = SourceCitation(index=1, source="Calculator", page_number=1, chunk_index=1, chunk_id="calculator::1", score=1.0, text="36895", is_tool=True)
    cit2 = SourceCitation(index=2, source="DBMS.pdf", page_number=2, chunk_index=4, chunk_id="dbms::2", score=0.85, text="A database...", is_tool=False)
    
    with patch("streamlit.expander") as mock_expander, patch("streamlit.markdown") as mock_markdown:
        render_citations([cit1, cit2])
        assert mock_expander.call_count == 2

