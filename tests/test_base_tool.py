import pytest
from app.tools.base_tool import BaseTool
from app.tools import (
    RagTool,
    CalculatorTool,
    DateTimeTool,
    WebSearchTool,
    DocumentStatsTool
)

def test_cannot_instantiate_base_tool():
    with pytest.raises(TypeError):
        BaseTool() # Abstract class check

def test_tool_stubs_inherit_base_tool():
    tools = [
        RagTool(),
        CalculatorTool(),
        DateTimeTool(),
        WebSearchTool(),
        DocumentStatsTool()
    ]
    for tool in tools:
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert len(tool.name) > 0
        assert len(tool.description) > 0


