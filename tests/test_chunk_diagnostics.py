import pytest
from scripts.chunk_diagnostics import get_overlap, analyze_chunk

def test_get_overlap():
    assert get_overlap("hello world", "world peace") == "world"
    assert get_overlap("abc", "def") == ""
    assert get_overlap("aaaa", "aa") == "aa"
    assert get_overlap("testing", "testing") == "testing"

def test_analyze_chunk_middle():
    chunk_id = "doc1::p1::c1"
    doc_text = "chunk of text. This is the second chunk of text."
    doc_meta = {
        "chunk_id": chunk_id,
        "page_number": 1,
        "chunk_index": 1,
        "source": "book.pdf",
        "doc_hash": "hash123"
    }
    all_doc_chunks = [
        {"id": "doc1::p1::c0", "text": "This is the first chunk of text.", "page_number": 1, "chunk_index": 0},
        {"id": "doc1::p1::c1", "text": "chunk of text. This is the second chunk of text.", "page_number": 1, "chunk_index": 1},
        {"id": "doc1::p1::c2", "text": "second chunk of text. And this is the third chunk.", "page_number": 1, "chunk_index": 2}
    ]
    
    analysis = analyze_chunk(chunk_id, doc_text, doc_meta, all_doc_chunks)
    
    assert analysis["chunk_id"] == chunk_id
    assert analysis["page_number"] == 1
    assert analysis["chunk_index"] == 1
    assert analysis["doc_hash"] == "hash123"
    assert analysis["source"] == "book.pdf"
    
    pred = analysis["neighbors"]["predecessor"]
    assert pred is not None
    assert pred["chunk_id"] == "doc1::p1::c0"
    assert pred["overlap_len"] == len("chunk of text.")
    assert pred["overlap_text"] == "chunk of text."
    
    succ = analysis["neighbors"]["successor"]
    assert succ is not None
    assert succ["chunk_id"] == "doc1::p1::c2"
    assert succ["overlap_len"] == len("second chunk of text.")
    assert succ["overlap_text"] == "second chunk of text."

def test_analyze_chunk_boundaries():
    # First chunk: predecessor is None
    all_chunks = [
        {"id": "c0", "text": "First chunk", "page_number": 1, "chunk_index": 0},
        {"id": "c1", "text": "Second chunk", "page_number": 1, "chunk_index": 1}
    ]
    
    a1 = analyze_chunk("c0", "First chunk", {"page_number": 1, "chunk_index": 0}, all_chunks)
    assert a1["neighbors"]["predecessor"] is None
    assert a1["neighbors"]["successor"]["chunk_id"] == "c1"
    
    # Last chunk: successor is None
    a2 = analyze_chunk("c1", "Second chunk", {"page_number": 1, "chunk_index": 1}, all_chunks)
    assert a2["neighbors"]["predecessor"]["chunk_id"] == "c0"
    assert a2["neighbors"]["successor"] is None
