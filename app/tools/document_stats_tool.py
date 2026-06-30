"""DocumentStats tool implementation for retrieving database stats."""

from app.tools.base_tool import BaseTool
from app.rag.vector_store import get_vector_store

class DocumentStatsTool(BaseTool):
    """Tool for retrieving collection metrics and database document statistics."""

    @property
    def name(self) -> str:
        return "document_stats"

    @property
    def description(self) -> str:
        return "Retrieve diagnostic and statistical metrics about the indexed PDF document collection."

    def can_handle(self, query: str) -> bool:
        return any(kw in query.lower() for kw in ["stats", "statistics", "collection", "documents"])

    def available(self) -> bool:
        try:
            get_vector_store()
            return True
        except Exception:
            return False

    def execute(self, query: str) -> dict:
        try:
            store = get_vector_store()
            chunk_counts = store.document_chunk_counts()
            
            total_chunks = store._collection.count()
            total_documents = len(chunk_counts)
            collection_name = store._collection_name
            
            per_document_chunks = {}
            for doc_hash, entry in chunk_counts.items():
                source_name = str(entry.get("source", "unknown"))
                per_document_chunks[source_name] = int(entry.get("chunk_count", 0))

            content_lines = [
                f"Collection Name: {collection_name}",
                f"Total Documents: {total_documents}",
                f"Total Chunks: {total_chunks}",
                "Per-Document Chunk Counts:"
            ]
            for source, count in per_document_chunks.items():
                content_lines.append(f" - {source}: {count} chunks")
                
            content = "\n".join(content_lines)

            data = {
                "collection_name": collection_name,
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "per_document_chunk_counts": per_document_chunks
            }

            return {
                "success": True,
                "tool": self.name,
                "data": data,
                "content": content,
                "sources": [],
                "metadata": {}
            }
        except Exception as exc:
            return {
                "success": False,
                "tool": self.name,
                "error": f"Failed to retrieve document statistics: {exc}",
                "content": "",
                "sources": [],
                "metadata": {}
            }
