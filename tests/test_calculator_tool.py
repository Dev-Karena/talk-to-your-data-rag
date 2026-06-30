import pytest
from app.tools.calculator_tool import CalculatorTool

@pytest.fixture
def calculator():
    return CalculatorTool()

def test_calculator_basic_arithmetic(calculator):
    # Test simple addition
    res = calculator.execute("2 + 2")
    assert res["success"] is True
    assert res["data"] == 4

    # Test multiplication
    res = calculator.execute("12 * 17")
    assert res["success"] is True
    assert res["data"] == 204

    # Test division
    res = calculator.execute("100 / 4")
    assert res["success"] is True
    assert res["data"] == 25.0

def test_calculator_sqrt(calculator):
    res = calculator.execute("sqrt(144)")
    assert res["success"] is True
    assert res["data"] == 12.0

def test_calculator_percentage(calculator):
    res = calculator.execute("25% of 400")
    assert res["success"] is True
    assert res["data"] == 100.0

def test_calculator_prefixed_queries(calculator):
    res = calculator.execute("calculate 10 + 20")
    assert res["success"] is True
    assert res["data"] == 30

    res = calculator.execute("what is 5 * 6")
    assert res["success"] is True
    assert res["data"] == 30

def test_calculator_error_handling(calculator):
    # Division by zero
    res = calculator.execute("1 / 0")
    assert res["success"] is False
    assert "ZeroDivisionError" in res["error"] or "division by zero" in res["error"].lower()

    # Malformed expression
    res = calculator.execute("2 + * 3")
    assert res["success"] is False
    assert "Failed to evaluate expression" in res["error"]

    # Unsupported function or call
    res = calculator.execute("eval('import os')")
    assert res["success"] is False
    assert "Unsupported function" in res["error"]
