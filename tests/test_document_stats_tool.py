import pytest
from app.tools.document_stats_tool import DocumentStatsTool

@pytest.fixture
def stats_tool():
    return DocumentStatsTool()

def test_document_stats_execution(stats_tool):
    # Execute stats tool (will read from test database collection)
    res = stats_tool.execute("How many PDFs are indexed?")
    assert res["success"] is True
    assert "collection_name" in res["data"]
    assert "total_documents" in res["data"]
    assert "total_chunks" in res["data"]
    assert "per_document_chunk_counts" in res["data"]
    assert isinstance(res["data"]["total_chunks"], int)
    assert isinstance(res["data"]["total_documents"], int)
