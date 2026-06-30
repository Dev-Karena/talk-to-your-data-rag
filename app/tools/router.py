"""Deterministic intent router for selecting appropriate tools."""

import re
from enum import Enum

class Intent(Enum):
    """Enums representing query execution intents."""
    RAG = "rag"
    CALCULATOR = "calculator"
    DATETIME = "datetime"
    DOCUMENT_STATS = "document_stats"
    WEB_SEARCH = "web_search"
    UNKNOWN = "unknown"


class ToolRouter:
    """Heuristic and regex-based router to classify user intents."""

    def route(self, query: str) -> Intent:
        """Route the query to an Intent based on prioritised heuristics.

        Priority order:
            Calculator -> DateTime -> DocumentStats -> WebSearch -> RAG.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return Intent.UNKNOWN

        query_lower = clean_query.lower()

        # 1. CALCULATOR (Priority 1)
        # Matches pure math expressions or inline expressions like "25*17"
        if re.match(r'^[\d\s+\-*/()^.%]+$', clean_query):
            # Exclude strings that are just single numbers
            if not re.match(r'^\s*\d+\s*$', clean_query):
                return Intent.CALCULATOR
        if re.search(r'\d+\s*[\+\-\*/\^]\s*\d+', clean_query):
            return Intent.CALCULATOR

        math_keywords = {
            "calculate", "add ", "multiply ", "divide ", "subtract ",
            "plus", "minus", "times", "divided by", "sum of", "square root",
            "sqrt", "%", "percent"
        }
        if any(kw in query_lower for kw in math_keywords):
            return Intent.CALCULATOR

        # 2. DATETIME (Priority 2)
        datetime_keywords = {
            "today", "tomorrow", "yesterday", "what time", "what date",
            "current time", "current date", "current clock", "time is it",
            "what is today", "what day is"
        }
        if any(kw in query_lower for kw in datetime_keywords):
            return Intent.DATETIME

        # 3. DOCUMENT_STATS (Priority 3)
        stats_keywords = {
            "document stats", "document statistics", "how many documents",
            "number of documents", "total documents", "collection size",
            "size of my collection", "size of the collection", "size of collection",
            "indexed files", "corpus stats", "stats of the document",
            "how many files", "statistics of my documents",
            "how many pdfs", "number of pdfs", "total pdfs", "pdfs indexed",
            "pdf statistics", "pdfs are indexed"
        }
        if any(kw in query_lower for kw in stats_keywords):
            return Intent.DOCUMENT_STATS

        # 4. WEB_SEARCH (Priority 4)
        web_keywords = {
            "latest news", "weather in", "stock price", "news about",
            "search the web", "google search", "current events",
            "who won the", "real-time info", "live status", "recent news"
        }
        if any(kw in query_lower for kw in web_keywords):
            return Intent.WEB_SEARCH

        # 5. RAG (Default Fallback)
        return Intent.RAG
