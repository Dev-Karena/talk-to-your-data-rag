import pytest
from app.rag.vector_store import RetrievedChunk
from app.eval.faithfulness_metrics import (
    calculate_groundedness,
    calculate_hallucination_rate,
    calculate_context_utilization
)

def test_groundedness_perfect():
    c = RetrievedChunk(
        chunk_id="c1", text="Supervised learning uses labeled datasets to train algorithms.",
        source="ML.pdf", page_number=1, chunk_index=0, doc_hash="h1", score=0.9
    )
    # The sentence matches the chunk closely
    answer = "Supervised learning utilizes labeled datasets to train machine learning algorithms."
    assert calculate_groundedness(answer, [c]) == 1.0
    assert calculate_hallucination_rate(answer, [c]) == 0.0

def test_groundedness_hallucinated():
    c = RetrievedChunk(
        chunk_id="c1", text="Supervised learning uses labeled datasets to train algorithms.",
        source="ML.pdf", page_number=1, chunk_index=0, doc_hash="h1", score=0.9
    )
    # Sentence 1 is grounded, sentence 2 is hallucinated
    answer = "Supervised learning uses labeled datasets to train algorithms. France has its capital in Paris."
    assert calculate_groundedness(answer, [c]) == 0.5
    assert calculate_hallucination_rate(answer, [c]) == 0.5

def test_groundedness_abstention():
    # Abstention answers should be 100% grounded
    answer = "I could not find this information in the provided documents."
    assert calculate_groundedness(answer, []) == 1.0
    assert calculate_hallucination_rate(answer, []) == 0.0

def test_context_utilization():
    chunks = [
        RetrievedChunk(chunk_id="c1", text="text1", source="OS.pdf", page_number=1, chunk_index=0, doc_hash="h1", score=0.9),
        RetrievedChunk(chunk_id="c2", text="text2", source="OS.pdf", page_number=2, chunk_index=1, doc_hash="h1", score=0.8),
    ]
    # Cites source 1 but not source 2
    answer = "Process management is detailed in [Source 1]."
    assert calculate_context_utilization(answer, chunks) == 0.5
