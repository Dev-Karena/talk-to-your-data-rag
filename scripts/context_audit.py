#!/usr/bin/env python
"""Audits retrieved context quality and citation structures across the benchmark."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup project root import path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.eval.dataset import load_benchmark
from app.rag.vector_store import get_vector_store
from app.services.retriever import retrieve
from app.services.context_builder import build_context
from app.eval.benchmark_validator import STOPWORDS

def tokenize(text: str) -> set:
    """Clean and tokenize text to a set of words."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return {w for w in cleaned.split() if w not in STOPWORDS}

def calculate_jaccard(text1: str, text2: str) -> float:
    """Calculate token Jaccard similarity between two texts."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def main():
    parser = argparse.ArgumentParser(description="Audit quality and redundancy of retrieved contexts.")
    parser.add_argument("--json", action="store_true", help="Print stats as JSON")
    args = parser.parse_args()
    
    # Enforce benchmark database
    settings = get_settings()
    settings.chroma_persist_dir = _ROOT / "benchmark_chroma"
    settings.chroma_collection_name = "benchmark_corpus"
    os.environ["CHROMA_PERSIST_DIR"] = str(settings.chroma_persist_dir)
    os.environ["CHROMA_COLLECTION_NAME"] = settings.chroma_collection_name
    
    dataset_path = _ROOT / "benchmarks" / "retrieval_cases.yaml"
    try:
        benchmark = load_benchmark(dataset_path)
    except Exception as exc:
        print(f"ERROR: Failed to load benchmark: {exc}", file=sys.stderr)
        sys.exit(1)
        
    store = get_vector_store()
    
    # 1. Fetch document-level chunk index maps for consecutiveness checks
    # To avoid repeated Chroma hits, load all chunk metadata for each doc once
    doc_chunk_indices = {}
    try:
        res = store._collection.get(include=["metadatas"])
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") or []
        
        doc_chunks = {}
        for cid, meta in zip(ids, metadatas):
            if not meta:
                continue
            doc_hash = meta.get("doc_hash", "")
            doc_chunks.setdefault(doc_hash, []).append({
                "id": cid,
                "page": int(meta.get("page_number", 0)),
                "index": int(meta.get("chunk_index", 0))
            })
            
        for doc_hash, chunks in doc_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda x: (x["page"], x["index"]))
            doc_chunk_indices[doc_hash] = {c["id"]: idx for idx, c in enumerate(sorted_chunks)}
    except Exception as exc:
        print(f"WARNING: Failed to build global chunk index map: {exc}", file=sys.stderr)
        
    # 2. Iterate and retrieve for each case
    total_cases = len(benchmark.cases)
    scored_cases = [c for c in benchmark.cases if c.type != "negative"]
    
    total_char_len = 0
    total_estimated_tokens = 0
    total_retrieved_chunks = 0
    
    duplicates_count = 0
    near_duplicates_count = 0
    total_pairs_compared = 0
    
    consecutive_pairs_count = 0
    total_adjacent_compared = 0
    
    low_info_chunks_count = 0
    
    redundant_examples = []
    
    for case in scored_cases:
        chunks = retrieve(case.query, top_k=4)
        total_retrieved_chunks += len(chunks)
        
        assembled = build_context(chunks)
        char_len = len(assembled.context_text)
        est_tokens = char_len // 4
        
        total_char_len += char_len
        total_estimated_tokens += est_tokens
        
        # Low information chunks (len < 150 or keywords < 10)
        for chunk in chunks:
            if len(chunk.text) < 150 or len(tokenize(chunk.text)) < 10:
                low_info_chunks_count += 1
                
        # Pairwise duplicate/near-duplicate checks
        n_chunks = len(chunks)
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                c1 = chunks[i]
                c2 = chunks[j]
                sim = calculate_jaccard(c1.text, c2.text)
                total_pairs_compared += 1
                if sim == 1.0:
                    duplicates_count += 1
                elif sim >= 0.8:
                    near_duplicates_count += 1
                    
                # Store redundant example if they overlap significantly
                if sim >= 0.4 and c1.doc_hash == c2.doc_hash and len(redundant_examples) < 5:
                    redundant_examples.append({
                        "case_id": case.id,
                        "query": case.query,
                        "source": c1.source,
                        "similarity": round(sim, 3),
                        "chunk1_id": c1.chunk_id,
                        "chunk1_page": c1.page_number,
                        "chunk2_id": c2.chunk_id,
                        "chunk2_page": c2.page_number,
                        "chunk1_snippet": c1.text[:120].strip().replace("\n", " "),
                        "chunk2_snippet": c2.text[:120].strip().replace("\n", " ")
                    })
                    
        # Consecutiveness check
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                c1 = chunks[i]
                c2 = chunks[j]
                if c1.doc_hash == c2.doc_hash:
                    total_adjacent_compared += 1
                    index_map = doc_chunk_indices.get(c1.doc_hash, {})
                    idx1 = index_map.get(c1.chunk_id)
                    idx2 = index_map.get(c2.chunk_id)
                    if idx1 is not None and idx2 is not None:
                        if abs(idx1 - idx2) == 1:
                            consecutive_pairs_count += 1
                            
    # Averages
    avg_chunks = total_retrieved_chunks / len(scored_cases) if scored_cases else 0.0
    avg_char_len = total_char_len / len(scored_cases) if scored_cases else 0.0
    avg_tokens = total_estimated_tokens / len(scored_cases) if scored_cases else 0.0
    
    dup_percent = (duplicates_count / total_pairs_compared * 100.0) if total_pairs_compared else 0.0
    near_dup_percent = (near_duplicates_count / total_pairs_compared * 100.0) if total_pairs_compared else 0.0
    consec_percent = (consecutive_pairs_count / total_adjacent_compared * 100.0) if total_adjacent_compared else 0.0
    
    report = {
        "dataset_name": benchmark.description,
        "fingerprint": benchmark.fingerprint(),
        "scored_cases_count": len(scored_cases),
        "context_length": {
            "avg_char_len": round(avg_char_len, 2),
            "avg_estimated_tokens": round(avg_tokens, 2),
            "avg_chunks_retrieved": round(avg_chunks, 2)
        },
        "redundancy": {
            "exact_duplicates": duplicates_count,
            "near_duplicates": near_duplicates_count,
            "duplicate_percent": round(dup_percent, 2),
            "near_duplicate_percent": round(near_dup_percent, 2),
            "pairs_compared": total_pairs_compared
        },
        "adjacency": {
            "consecutive_pairs": consecutive_pairs_count,
            "consecutive_percent": round(consec_percent, 2),
            "adjacent_pairs_compared": total_adjacent_compared
        },
        "quality": {
            "low_info_chunks": low_info_chunks_count,
            "low_info_percent": round(low_info_chunks_count / total_retrieved_chunks * 100.0, 2) if total_retrieved_chunks else 0.0
        },
        "redundant_examples": redundant_examples
    }
    
    if args.json:
        print(json.dumps(report, indent=2))
        return
        
    # Print human-readable report
    print("=" * 80)
    print("RAG CONTEXT ASSEMBLY & CITATION AUDIT REPORT")
    print("=" * 80)
    print(f"Benchmark Name   : {report['dataset_name']}")
    print(f"Fingerprint      : {report['fingerprint']}")
    print(f"Scored Cases     : {report['scored_cases_count']}")
    print("-" * 80)
    print("1. PROMPT SIZE & ESTIMATED TOKENS")
    print(f"  * Average Chunks Retrieved : {report['context_length']['avg_chunks_retrieved']} (top_k=4)")
    print(f"  * Average Context Length   : {report['context_length']['avg_char_len']} characters")
    print(f"  * Average Estimated Tokens : {report['context_length']['avg_estimated_tokens']} tokens")
    
    print("\n2. REDUNDANCY & DUPLICATION")
    print(f"  * Exact Duplicate Chunks   : {report['redundancy']['exact_duplicates']} ({report['redundancy']['duplicate_percent']}% of pairs)")
    print(f"  * Near-Duplicate Chunks    : {report['redundancy']['near_duplicates']} ({report['redundancy']['near_duplicate_percent']}% of pairs)")
    
    print("\n3. CHUNK ADJACENCY (CONSECUTIVE SEGMENTS)")
    print(f"  * Consecutive Chunk Pairs  : {report['adjacency']['consecutive_pairs']} ({report['adjacency']['consecutive_percent']}% of same-doc pairs)")
    
    print("\n4. LOW-INFORMATION CHUNKS")
    print(f"  * Low-Information Chunks   : {report['quality']['low_info_chunks']} ({report['quality']['low_info_percent']}% of retrieved chunks)")
    
    print("-" * 80)
    print("5. CONCRETE EXAMPLES OF CONTEXT REDUNDANCY")
    print("-" * 80)
    for idx, ex in enumerate(report["redundant_examples"], start=1):
        print(f"\nExample {idx}: Case {ex['case_id']} - Similarity: {ex['similarity']}")
        print(f"  Query  : '{ex['query']}'")
        print(f"  Chunk 1: Page {ex['chunk1_page']} | Snippet: {ex['chunk1_snippet']}...")
        print(f"  Chunk 2: Page {ex['chunk2_page']} | Snippet: {ex['chunk2_snippet']}...")
    print("=" * 80)

if __name__ == "__main__":
    main()
