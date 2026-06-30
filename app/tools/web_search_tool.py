"""Web search tool implementation using DuckDuckGo search."""

import re
from app.tools.base_tool import BaseTool
from app.config.settings import get_settings

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

class WebSearchTool(BaseTool):
    """Tool for searching the web for real-time external data."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web to retrieve real-time external information and references."

    def can_handle(self, query: str) -> bool:
        return any(kw in query.lower() for kw in ["latest news", "weather", "stock price", "search"])

    def available(self) -> bool:
        settings = get_settings()
        return DDGS_AVAILABLE and getattr(settings, "web_search_enabled", True)

    def execute(self, query: str) -> dict:
        settings = get_settings()
        if not getattr(settings, "web_search_enabled", True):
            return {
                "success": False,
                "tool": self.name,
                "error": "Web search is disabled in configuration settings.",
                "content": "",
                "sources": [],
                "metadata": {}
            }

        if not DDGS_AVAILABLE:
            return {
                "success": False,
                "tool": self.name,
                "error": "duckduckgo-search library is not installed.",
                "content": "",
                "sources": [],
                "metadata": {}
            }

        try:
            # Extract actual query phrase
            search_term = query
            for kw in ["latest news about", "latest news on", "search the web for", "google search for", "news about"]:
                if kw in search_term.lower():
                    idx = search_term.lower().find(kw)
                    search_term = search_term[idx + len(kw):].strip()

            with DDGS() as ddgs:
                results = list(ddgs.text(search_term, max_results=3))

            if not results:
                return {
                    "success": True,
                    "tool": self.name,
                    "data": [],
                    "content": f"No web search results found for query: '{search_term}'",
                    "sources": [],
                    "metadata": {}
                }

            formatted_results = []
            urls = []
            for idx, r in enumerate(results, start=1):
                title = r.get("title", "No Title")
                body = r.get("body", "No Snippet")
                href = r.get("href", "")
                formatted_results.append(f"[{idx}] {title}\nSnippet: {body}\nURL: {href}")
                if href:
                    urls.append(href)

            content = "\n\n".join(formatted_results)
            
            return {
                "success": True,
                "tool": self.name,
                "data": results,
                "content": content,
                "sources": urls,
                "metadata": {"search_term": search_term}
            }
        except Exception as exc:
            return {
                "success": False,
                "tool": self.name,
                "error": f"Web search request failed: {exc}",
                "content": "",
                "sources": [],
                "metadata": {}
            }
