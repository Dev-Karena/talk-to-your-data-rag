import pytest
from unittest.mock import patch, MagicMock
from app.services.tool_executor import ToolExecutor
from app.tools import Intent

@pytest.fixture
def executor():
    return ToolExecutor()

def test_executor_initialization(executor):
    assert executor.registry is not None
    assert len(executor.registry.list_tools()) == 5

def test_executor_execute_calculator(executor):
    res = executor.execute_intent(Intent.CALCULATOR, "2 + 2")
    assert res["success"] is True
    assert res["data"] == 4

def test_executor_execute_datetime(executor):
    res = executor.execute_intent(Intent.DATETIME, "what time is it?")
    assert res["success"] is True
    assert "local_time" in res["data"]

def test_executor_execute_document_stats(executor):
    res = executor.execute_intent(Intent.DOCUMENT_STATS, "corpus size")
    assert res["success"] is True
    assert "total_chunks" in res["data"]

def test_executor_execute_rag_tool(executor):
    res = executor.execute_intent(Intent.RAG, "some query")
    assert res["success"] is True
    assert "RAG tool execution is processed natively" in res["content"]

def test_executor_execute_unknown_intent(executor):
    res = executor.execute_intent(Intent.UNKNOWN, "what day is today?")
    assert res["success"] is False
    assert "No executable tool found" in res["error"]
