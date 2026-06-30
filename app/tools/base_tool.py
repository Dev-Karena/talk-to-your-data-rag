"""Abstract base class for all tools in the tool-calling framework."""

from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base class interface for all tools.

    All tools in the framework must inherit from this class and implement
    its abstract properties and methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name identifier of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A description explaining what the tool does and its parameters."""
        pass

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Determine if the tool can handle the given query input."""
        pass

    @abstractmethod
    def execute(self, query: str) -> dict:
        """Execute the tool on the given query.

        Returns a dictionary matching the standard tool response schema:
        {
            "success": bool,
            "tool": str,
            "content": str,
            "sources": List[str],
            "metadata": dict
        }
        """
        pass

    @abstractmethod
    def available(self) -> bool:
        """Determine if the tool is currently available for execution (e.g. dependencies installed)."""
        pass

