import pytest
from app.tools import Intent, ToolRouter

@pytest.fixture
def router():
    return ToolRouter()

def test_router_empty_queries(router):
    assert router.route("") == Intent.UNKNOWN
    assert router.route("    ") == Intent.UNKNOWN
    assert router.route(None) == Intent.UNKNOWN

def test_router_calculator_queries(router):
    # Pure numeric math expressions
    assert router.route("2 + 2") == Intent.CALCULATOR
    assert router.route(" (10 * 5) / 2.5 ") == Intent.CALCULATOR
    assert router.route("100 % 3") == Intent.CALCULATOR
    
    # Exclude single numbers
    assert router.route("42") == Intent.RAG
    
    # Mathematical keywords
    assert router.route("calculate the square root of 256") == Intent.CALCULATOR
    assert router.route("divide 100 by 4") == Intent.CALCULATOR
    assert router.route("sum of 10, 20, and 30") == Intent.CALCULATOR

def test_router_datetime_queries(router):
    assert router.route("what time is it in New York?") == Intent.DATETIME
    assert router.route("tell me today's date") == Intent.DATETIME
    assert router.route("what day is it?") == Intent.DATETIME
    assert router.route("show current time clock info") == Intent.DATETIME

def test_router_document_stats_queries(router):
    assert router.route("how many documents have been uploaded?") == Intent.DOCUMENT_STATS
    assert router.route("give me total documents statistics") == Intent.DOCUMENT_STATS
    assert router.route("show the corpus stats") == Intent.DOCUMENT_STATS
    assert router.route("what is the size of my collection?") == Intent.DOCUMENT_STATS

def test_router_web_search_queries(router):
    assert router.route("what is the weather in Tokyo?") == Intent.WEB_SEARCH
    assert router.route("latest news about local inflation rates") == Intent.WEB_SEARCH
    assert router.route("what is the current stock price of Google?") == Intent.WEB_SEARCH
    assert router.route("search the web for CPU designs") == Intent.WEB_SEARCH

def test_router_rag_fallback_queries(router):
    # Textbook knowledge / General ground truth RAG questions
    assert router.route("What is supervised learning?") == Intent.RAG
    assert router.route("How does a context switch between processes work?") == Intent.RAG
    assert router.route("What are the ACID properties of a transaction?") == Intent.RAG
    assert router.route("Explain virtual memory and database normalization.") == Intent.RAG

def test_router_priority_ordering(router):
    # Calculator vs DateTime (Calculator wins)
    # "calculate today's clock 5 + 5" matches datetime ("today's clock") and calculator ("calculate", "5 + 5")
    assert router.route("calculate today's clock 5 + 5") == Intent.CALCULATOR

    # DateTime vs DocumentStats (DateTime wins)
    # "today's total documents count" matches datetime ("today's") and stats ("total documents")
    assert router.route("today's total documents count") == Intent.DATETIME

    # DocumentStats vs WebSearch (DocumentStats wins)
    # "total documents latest news" matches stats ("total documents") and web ("latest news")
    assert router.route("total documents latest news") == Intent.DOCUMENT_STATS

    # WebSearch vs RAG (WebSearch wins)
    # " ACID properties of transactions latest news" matches RAG and WebSearch ("latest news")
    assert router.route("ACID properties of transactions latest news") == Intent.WEB_SEARCH
