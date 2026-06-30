"""Benchmarking runner for a single configuration in Sprint 9."""

import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup project root import path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.services.retriever import retrieve
from app.services.context_builder import build_context
from app.services.citation_builder import build_citations
from app.eval import metrics
from app.eval.citation_metrics import calculate_citation_metrics
from app.eval.faithfulness_metrics import (
    calculate_groundedness,
    calculate_hallucination_rate,
    calculate_context_utilization
)

# Simulated RAG answer generation
def generate_mock_answer(chunks) -> str:
    import re
    if not chunks:
        return "I could not find this information in the provided documents."
    sentences = []
    for idx, c in enumerate(chunks):
        parts = re.split(r'(?<=[.!?])\s+', c.text)
        first_sent = parts[0].strip() if parts else c.text
        sentences.append(f"{first_sent} [Source {idx+1}].")
    return " ".join(sentences)

def run_benchmark(compression_enabled: bool, citation_filter: bool) -> Dict[str, Any]:
    # Enforce isolated collection
    settings = get_settings()
    settings.chroma_persist_dir = _ROOT / "benchmark_chroma"
    settings.chroma_collection_name = "benchmark_corpus"
    os.environ["CHROMA_PERSIST_DIR"] = str(settings.chroma_persist_dir)
    os.environ["CHROMA_COLLECTION_NAME"] = settings.chroma_collection_name
    
    settings.context_compression_enabled = compression_enabled

    yaml_path = _ROOT / "benchmarks" / "retrieval_cases.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = data.get("cases", [])

    scored_cases = [c for c in cases if c.get("type") != "negative"]
    cross_doc_cases = [c for c in scored_cases if c.get("type") == "cross_document"]
    negative_cases = [c for c in cases if c.get("type") == "negative"]

    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_sa = 0.0
    total_xdoc_recall = 0.0
    
    total_precision = 0.0
    total_recall_cit = 0.0
    total_f1_cit = 0.0
    
    total_groundedness = 0.0
    total_hallucination = 0.0
    total_utilization = 0.0
    
    total_chars = 0.0
    total_latency = 0.0

    for case in cases:
        query = case.get("query")
        expected_sources = case.get("expected_sources", [])
        expected_hashes = case.get("expected_doc_hashes", [])
        relevant = set(expected_hashes)

        # Start timer
        start_time = time.perf_counter()
        
        # 1. Retrieve
        raw_chunks = retrieve(query, top_k=4)
        
        # 2. Context Builder & Compression
        assembled = build_context(raw_chunks)
        
        # 3. Filter citations and build final citations list
        threshold = 0.61 if citation_filter else 0.0
        filtered_cits = [c for c in assembled.citations if c.score >= threshold]
        
        citations = build_citations(assembled.citations, min_score=threshold)
        cited_sources = [cit.source for cit in citations]

        # Stop timer
        latency = time.perf_counter() - start_time
        total_latency += latency

        # Generate simulated answer
        answer = generate_mock_answer(filtered_cits)

        # Calculate faithfulness metrics
        groundedness = calculate_groundedness(answer, filtered_cits)
        hallucination = calculate_hallucination_rate(answer, filtered_cits)
        utilization = calculate_context_utilization(answer, filtered_cits)

        total_groundedness += groundedness
        total_hallucination += hallucination
        total_utilization += utilization
        total_chars += len(assembled.context_text)

        # Calculate citation metrics
        cit_metrics = calculate_citation_metrics(cited_sources, expected_sources)
        total_precision += cit_metrics["precision"]
        total_recall_cit += cit_metrics["recall"]
        total_f1_cit += cit_metrics["f1"]

        # Calculate retrieval metrics (for non-negative cases only)
        if case.get("type") != "negative":
            # Document hashes used for metric evaluation
            retrieved_hashes = [c.chunk_id.split("::")[0] for c in assembled.citations]
            
            r = metrics.recall_at_k(retrieved_hashes, relevant, 4)
            mrr = metrics.reciprocal_rank(retrieved_hashes, relevant)
            ndcg = metrics.ndcg_at_k(retrieved_hashes, relevant, 4)
            sa = metrics.hit_at_1(retrieved_hashes, relevant)

            total_recall += r
            total_mrr += mrr
            total_ndcg += ndcg
            total_sa += sa

            if case.get("type") == "cross_document":
                total_xdoc_recall += r

    num_cases = len(cases)
    num_scored = len(scored_cases)
    num_xdoc = len(cross_doc_cases)

    return {
        "recall": total_recall / num_scored if num_scored > 0 else 0.0,
        "mrr": total_mrr / num_scored if num_scored > 0 else 0.0,
        "ndcg": total_ndcg / num_scored if num_scored > 0 else 0.0,
        "source_accuracy": total_sa / num_scored if num_scored > 0 else 0.0,
        "xdoc_recall": total_xdoc_recall / num_xdoc if num_xdoc > 0 else 0.0,
        "citation_precision": total_precision / num_cases if num_cases > 0 else 0.0,
        "citation_recall": total_recall_cit / num_cases if num_cases > 0 else 0.0,
        "citation_f1": total_f1_cit / num_cases if num_cases > 0 else 0.0,
        "groundedness": total_groundedness / num_cases if num_cases > 0 else 0.0,
        "hallucination": total_hallucination / num_cases if num_cases > 0 else 0.0,
        "utilization": total_utilization / num_cases if num_cases > 0 else 0.0,
        "avg_chars": total_chars / num_cases if num_cases > 0 else 0.0,
        "avg_latency": total_latency / num_cases if num_cases > 0 else 0.0,
    }

if __name__ == "__main__":
    # Test execution
    res = run_benchmark(compression_enabled=True, citation_filter=True)
    print("Test run successfully:", res)
