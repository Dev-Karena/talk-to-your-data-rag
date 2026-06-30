import pytest
from unittest.mock import patch, MagicMock

from app.tools.web_search_tool import WebSearchTool
from app.config.settings import get_settings

@pytest.fixture
def search_tool():
    return WebSearchTool()

def test_web_search_disabled(search_tool):
    settings = get_settings()
    with patch.object(settings, "web_search_enabled", False):
        # Even if DDGS is available, it should return False if disabled
        with patch("app.tools.web_search_tool.DDGS_AVAILABLE", True):
            assert search_tool.available() is False
            res = search_tool.execute("latest news about OpenAI")
            assert res["success"] is False
            assert "disabled" in res["error"]

def test_web_search_available_and_execution_mocked(search_tool):
    settings = get_settings()
    with patch.object(settings, "web_search_enabled", True):
        # Setup search text results mock
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = [
            {"title": "OpenAI News", "body": "OpenAI releases new models.", "href": "https://openai.com/news"}
        ]
        
        with patch("app.tools.web_search_tool.DDGS") as mock_ddgs_cls:
            mock_ddgs_cls.return_value = mock_ddgs_instance
            with patch("app.tools.web_search_tool.DDGS_AVAILABLE", True):
                assert search_tool.available() is True
                res = search_tool.execute("latest news about OpenAI")
                assert res["success"] is True
                assert len(res["data"]) == 1
                assert res["data"][0]["title"] == "OpenAI News"
                assert "https://openai.com/news" in res["sources"]
                assert "OpenAI News" in res["content"]
                assert res["metadata"]["search_term"] == "OpenAI"


