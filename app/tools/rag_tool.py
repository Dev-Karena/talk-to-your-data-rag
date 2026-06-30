"""Implementation of RagTool placeholder."""

from app.tools.base_tool import BaseTool

class RagTool(BaseTool):
    """Tool stub for document-grounded retrieval and question answering (RAG)."""

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return "Retrieve information and answer questions grounded in indexed PDF documents."

    def can_handle(self, query: str) -> bool:
        return False

    def available(self) -> bool:
        return True

    def execute(self, query: str) -> dict:
        return {
            "success": True,
            "tool": self.name,
            "data": [],
            "content": "RAG tool execution is processed natively by the RAG service.",
            "sources": [],
            "metadata": {}
        }
