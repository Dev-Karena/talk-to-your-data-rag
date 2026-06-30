import pytest
from app.rag.vector_store import RetrievedChunk
from app.services.context_compressor import (
    extract_keywords,
    calculate_jaccard,
    get_overlap,
    compress_chunks
)

def test_extract_keywords():
    assert "indexing" in extract_keywords("We study indexing methods.")
    assert "does" not in extract_keywords("What does it do?") # Stopword

def test_calculate_jaccard():
    text1 = "supervised learning uses labeled training datasets"
    text2 = "supervised learning uses labeled training data"
    sim = calculate_jaccard(text1, text2)
    assert sim >= 0.5
    
    sim_diff = calculate_jaccard("CPU scheduling.", "Database locking transactions.")
    assert sim_diff == 0.0

def test_get_overlap():
    assert get_overlap("hello world", "world is nice") == "world"
    assert get_overlap("abcde", "fgh") == ""

def test_compress_chunks_low_info_fallback():
    # If all chunks are low-info, it should fallback to returning the first chunk
    c1 = RetrievedChunk(chunk_id="c1", doc_hash="h1", source="doc1.pdf", page_number=1, chunk_index=0, text="short", score=0.9)
    c2 = RetrievedChunk(chunk_id="c2", doc_hash="h1", source="doc1.pdf", page_number=1, chunk_index=1, text="noise", score=0.8)
    
    compressed = compress_chunks([c1, c2])
    assert len(compressed) == 1
    assert compressed[0].chunk_id == "c1"

def test_compress_chunks_duplicates():
    c1 = RetrievedChunk(
        chunk_id="c1", doc_hash="h1", source="doc1.pdf", page_number=1, chunk_index=0,
        text="Supervised learning is a machine learning paradigm that utilizes labeled datasets to train algorithms. It maps inputs to targets and predicts output labels for unseen test queries.", score=0.9
    )
    c2 = RetrievedChunk(
        chunk_id="c2", doc_hash="h2", source="doc2.pdf", page_number=2, chunk_index=0,
        text="Supervised learning is a machine learning paradigm that utilizes labeled datasets to train algorithms. It maps inputs to targets and predicts output labels for unseen test queries.", score=0.8
    ) # Duplicate of c1
    c3 = RetrievedChunk(
        chunk_id="c3", doc_hash="h3", source="doc3.pdf", page_number=3, chunk_index=0,
        text="Unrelated text discussing CPU schedulers, virtual memory, and process control blocks in operating systems. Schedulers determine which processes are allocated CPU time slices.", score=0.7
    )
    
    compressed = compress_chunks([c1, c2, c3])
    assert len(compressed) == 2
    assert compressed[0].chunk_id == "c1"
    assert compressed[1].chunk_id == "c3"

def test_compress_chunks_merge():
    # Consecutive chunks
    c1 = RetrievedChunk(
        chunk_id="c1", doc_hash="h1", source="doc1.pdf", page_number=5, chunk_index=0,
        text="This is the first segment of text that discusses how concurrency control is managed in databases. We must verify that locking protocols work correctly.", score=0.9
    )
    c2 = RetrievedChunk(
        chunk_id="c2", doc_hash="h1", source="doc1.pdf", page_number=5, chunk_index=1,
        text="is managed in databases. We must verify that locking protocols work correctly. Locking protocols are typically employed to guarantee serializability on database transactions.", score=0.8
    )
    
    compressed = compress_chunks([c1, c2])
    assert len(compressed) == 1
    assert compressed[0].chunk_id == "c1+c2"
    # Merged and then pruned (sentence 3 dropped by 75% limit)
    expected = "This is the first segment of text that discusses how concurrency control is managed in databases. We must verify that locking protocols work correctly."
    assert compressed[0].text == expected


