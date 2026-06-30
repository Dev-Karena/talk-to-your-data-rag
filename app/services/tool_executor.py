"""Service to manage and execute tools based on routed intents."""

from app.tools import (
    ToolRegistry,
    Intent,
    CalculatorTool,
    DateTimeTool,
    DocumentStatsTool,
    WebSearchTool,
    RagTool
)

class ToolExecutor:
    """Orchestrates tool execution based on the Intent Router classification.

    Decouples UI and presentation layers from registry and tool execution logic.
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        """Initialize the executor, optionally injecting a custom registry."""
        if registry is None:
            registry = ToolRegistry()
            registry.register(CalculatorTool())
            registry.register(DateTimeTool())
            registry.register(DocumentStatsTool())
            registry.register(WebSearchTool())
            registry.register(RagTool())
        self.registry = registry

    def execute_intent(self, intent: Intent, query: str) -> dict:
        """Retrieve and execute the tool corresponding to the given intent.

        Never raises exceptions; returns a standardized error response if any lookup
        or execution failure occurs.
        """
        intent_map = {
            Intent.CALCULATOR: "calculator",
            Intent.DATETIME: "datetime",
            Intent.DOCUMENT_STATS: "document_stats",
            Intent.WEB_SEARCH: "web_search",
            Intent.RAG: "rag"
        }

        tool_name = intent_map.get(intent)
        if not tool_name or intent == Intent.UNKNOWN:
            return {
                "success": False,
                "error": f"No executable tool found for intent '{intent}'",
                "content": "",
                "sources": [],
                "metadata": {}
            }

        try:
            tool = self.registry.get(tool_name)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' retrieval failed: {exc}",
                "content": "",
                "sources": [],
                "metadata": {}
            }

        if not tool.available():
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not currently available.",
                "content": "",
                "sources": [],
                "metadata": {}
            }

        try:
            return tool.execute(query)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' execution crashed: {exc}",
                "content": "",
                "sources": [],
                "metadata": {}
            }
