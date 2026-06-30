import pytest
from app.rag.vector_store import RetrievedChunk
from app.services.citation_builder import (
    extract_pages_from_chunk_id,
    format_page_range,
    build_citations,
    Citation
)

def test_extract_pages_single_id():
    chunk = RetrievedChunk(
        chunk_id="defd579::p15::c2",
        text="some text content...",
        source="OS.pdf",
        page_number=1,
        chunk_index=2,
        doc_hash="defd579",
        score=0.9
    )
    assert extract_pages_from_chunk_id(chunk) == [15]

def test_extract_pages_merged_id():
    chunk = RetrievedChunk(
        chunk_id="defd579::p86::c0+defd579::p87::c1+defd579::p89::c0",
        text="some merged text content...",
        source="OS.pdf",
        page_number=86,
        chunk_index=0,
        doc_hash="defd579",
        score=0.9
    )
    assert extract_pages_from_chunk_id(chunk) == [86, 87, 89]

def test_extract_pages_fallback():
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="unexpected id format",
        source="OS.pdf",
        page_number=12,
        chunk_index=0,
        doc_hash="h1",
        score=0.9
    )
    assert extract_pages_from_chunk_id(chunk) == [12]

def test_format_page_range():
    assert format_page_range([4]) == "p. 4"
    assert format_page_range([86, 87]) == "pp. 86-87"
    assert format_page_range([12, 14, 15, 16, 20]) == "pp. 12, 14-16, 20"
    assert format_page_range([]) == ""

def test_build_citations():
    c1 = RetrievedChunk(
        chunk_id="h1::p5::c0", text="text1", source="OS.pdf", page_number=5, chunk_index=0, doc_hash="h1", score=0.9
    )
    c2 = RetrievedChunk(
        chunk_id="h2::p10::c0", text="text2", source="DBMS.pdf", page_number=10, chunk_index=0, doc_hash="h2", score=0.8
    )
    c3 = RetrievedChunk(
        chunk_id="h1::p6::c1", text="text3", source="OS.pdf", page_number=6, chunk_index=1, doc_hash="h1", score=0.85
    )
    c4 = RetrievedChunk(
        chunk_id="h2::p10::c1+h2::p11::c0", text="text4", source="DBMS.pdf", page_number=10, chunk_index=1, doc_hash="h2", score=0.75
    )
    
    citations = build_citations([c1, c2, c3, c4])
    assert len(citations) == 2
    
    # OS.pdf citation (first matched document)
    assert citations[0].source == "OS.pdf"
    assert citations[0].pages == [5, 6]
    assert citations[0].page_range == "pp. 5-6"
    assert citations[0].score == 0.9
    assert citations[0].label == "OS.pdf (pp. 5-6)"
    
    # DBMS.pdf citation
    assert citations[1].source == "DBMS.pdf"
    assert citations[1].pages == [10, 11]
    assert citations[1].page_range == "pp. 10-11"
    assert citations[1].score == 0.8
    assert citations[1].label == "DBMS.pdf (pp. 10-11)"
