import pytest
from app.tools import (
    ToolRegistry,
    DuplicateToolError,
    ToolNotFoundError,
    RagTool,
    CalculatorTool,
    DateTimeTool
)

def test_tool_registry_registration_happy_path():
    registry = ToolRegistry()
    rag = RagTool()
    calc = CalculatorTool()

    registry.register(rag)
    registry.register(calc)

    assert registry.get("rag") is rag
    assert registry.get("calculator") is calc

    tools = registry.list_tools()
    assert len(tools) == 2
    assert rag in tools
    assert calc in tools

def test_tool_registry_duplicate_registration():
    registry = ToolRegistry()
    rag1 = RagTool()
    rag2 = RagTool()

    registry.register(rag1)
    with pytest.raises(DuplicateToolError):
        registry.register(rag2)

def test_tool_registry_not_found():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("invalid_tool")

def test_tool_registry_type_safety():
    registry = ToolRegistry()
    with pytest.raises(TypeError):
        registry.register("not a BaseTool instance") # type: ignore
