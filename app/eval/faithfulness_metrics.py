"""Metrics for evaluating answer faithfulness, hallucination rate, and context utilization."""

import re
from typing import List
from app.rag.vector_store import RetrievedChunk

def calculate_groundedness(answer: str, chunks: List[RetrievedChunk]) -> float:
    """Calculate the groundedness score of the answer.

    The score is the fraction of sentences in the generated answer that are lexically
    supported by the retrieved chunks. Abstention/negative answers are 100% grounded by definition.
    """
    answer_clean = (answer or "").strip()
    if not answer_clean:
        return 1.0

    # Handle standard negative/abstention responses
    abstention_keywords = [
        "could not find this information",
        "no documents have been indexed",
        "please enter a question",
        "answer generation is unavailable"
    ]
    if any(k in answer_clean.lower() for k in abstention_keywords):
        return 1.0

    if not chunks:
        return 0.0

    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 1.0

    grounded_count = 0
    for s in sentences:
        # Strip citation markers like [Source N] for clean overlap check
        s_clean = re.sub(r'\[Source \d+\]', '', s).strip()
        # Find non-stopword alphanumeric terms of length > 2
        words_s = set(w.lower() for w in re.findall(r'\b\w+\b', s_clean) if len(w) > 2)
        if not words_s:
            grounded_count += 1
            continue

        is_grounded = False
        for c in chunks:
            words_c = set(w.lower() for w in re.findall(r'\b\w+\b', c.text) if len(w) > 2)
            if not words_c:
                continue
            # Calculate Jaccard word containment similarity
            intersection = words_s & words_c
            containment = len(intersection) / len(words_s) if len(words_s) > 0 else 0.0
            # If 30% of sentence words or at least 4 words are matched in the chunk, we consider it grounded.
            if containment >= 0.30 or len(intersection) >= 4:
                is_grounded = True
                break
        if is_grounded:
            grounded_count += 1

    return grounded_count / len(sentences)


def calculate_hallucination_rate(answer: str, chunks: List[RetrievedChunk]) -> float:
    """Calculate the hallucination rate (complement of groundedness)."""
    return 1.0 - calculate_groundedness(answer, chunks)


def calculate_context_utilization(answer: str, chunks: List[RetrievedChunk]) -> float:
    """Calculate context utilization based on [Source N] markers in the answer."""
    if not chunks:
        return 0.0

    # Find cited index numbers
    cited_indices = set()
    for m in re.finditer(r'\[Source (\d+)\]', answer):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(chunks):
            cited_indices.add(idx)

    return len(cited_indices) / len(chunks)
