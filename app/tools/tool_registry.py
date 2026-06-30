"""Tool registry for registering and managing tools."""

from typing import Dict, List
from app.tools.base_tool import BaseTool

class ToolRegistryException(Exception):
    """Base exception class for all tool registry operations."""
    pass

class DuplicateToolError(ToolRegistryException):
    """Raised when registering a tool whose name is already registered."""
    pass

class ToolNotFoundError(ToolRegistryException):
    """Raised when looking up a tool name that is not registered."""
    pass

class ToolRegistry:
    """Registry to manage and lookup available tools.

    Supports dependency injection by initializing local registry instances.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry.

        Raises DuplicateToolError if a tool with the same name is already registered.
        Raises TypeError if the tool does not inherit from BaseTool.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError("Only instances inheriting from BaseTool can be registered.")
        
        if tool.name in self._tools:
            raise DuplicateToolError(f"A tool with name '{tool.name}' is already registered.")
        
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool instance by its name.

        Raises ToolNotFoundError if the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool with name '{name}' is not registered.")
        return self._tools[name]

    def list_tools(self) -> List[BaseTool]:
        """List all currently registered tool instances."""
        return list(self._tools.values())

    def get_available_tools(self) -> List[BaseTool]:
        """List all registered tools that are currently available."""
        return [tool for tool in self._tools.values() if tool.available()]

