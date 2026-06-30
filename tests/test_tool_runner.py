import pytest
from unittest.mock import patch, MagicMock
from app.eval.tool_runner import ToolRunner

@pytest.fixture
def runner():
    return ToolRunner()

def test_runner_load_cases(runner):
    cases = runner.load_cases()
    assert isinstance(cases, list)
    assert len(cases) > 0
    assert "query" in cases[0]
    assert "expected_intent" in cases[0]

def test_runner_run_eval_structure(runner):
    res = runner.run_eval()
    assert "cases" in res
    assert "total_cases" in res
    assert res["total_cases"] == len(res["cases"])
    
    first_case = res["cases"][0]
    assert "id" in first_case
    assert "query" in first_case
    assert "expected_intent" in first_case
    assert "actual_intent" in first_case
    assert "intent_match" in first_case
    assert "latency_e2e_ms" in first_case
