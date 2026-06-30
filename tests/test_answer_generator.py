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
            SourceCitation(index=1, source="Calculator", page_number=1, chunk_index=1, chunk_id="calculator::1", score=1.0, text="2")
        ],
        "tool_results": [{"success": True, "tool": "calculator", "content": "2"}]
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
