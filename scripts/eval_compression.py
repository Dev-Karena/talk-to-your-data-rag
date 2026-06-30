#!/usr/bin/env python
"""Benchmark evaluation script for context compression quality and regression testing."""

import argparse
import os
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
from app.services.retriever import retrieve
from app.services.context_compressor import compress_chunks
from app.services.context_builder import build_context
from app.eval import metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate context compression metrics and check for quality regression.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Keyword match evidence threshold")
    args = parser.parse_args()
    
    # Enforce isolated benchmark collection
    settings = get_settings()
    settings.chroma_persist_dir = _ROOT / "benchmark_chroma"
    settings.chroma_collection_name = "benchmark_corpus"
    os.environ["CHROMA_PERSIST_DIR"] = str(settings.chroma_persist_dir)
    os.environ["CHROMA_COLLECTION_NAME"] = settings.chroma_collection_name
    
    dataset_path = _ROOT / "benchmarks" / "retrieval_cases.yaml"
    try:
        benchmark = load_benchmark(dataset_path)
    except Exception as exc:
        print(f"ERROR: Failed to load benchmark dataset: {exc}", file=sys.stderr)
        sys.exit(1)
        
    scored_cases = [c for c in benchmark.cases if c.type != "negative"]
    cross_doc_cases = [c for c in scored_cases if c.type == "cross_document"]
    
    # Baseline lists
    base_recalls = []
    base_mrrs = []
    base_ndcgs = []
    base_source_accs = []
    base_xdoc_recalls = []
    base_char_lens = []
    
    # Compressed lists
    comp_recalls = []
    comp_mrrs = []
    comp_ndcgs = []
    comp_source_accs = []
    comp_xdoc_recalls = []
    comp_char_lens = []
    
    for case in scored_cases:
        # 1. Retrieve raw chunks
        raw_chunks = retrieve(case.query, top_k=4)
        raw_hashes = [c.doc_hash for c in raw_chunks]
        relevant = set(case.expected_doc_hashes)
        
        # Original context size
        raw_assembled = build_context(raw_chunks)
        base_char_lens.append(len(raw_assembled.context_text))
        
        # Calculate baseline metrics
        r_base = metrics.recall_at_k(raw_hashes, relevant, 4)
        mrr_base = metrics.reciprocal_rank(raw_hashes, relevant)
        ndcg_base = metrics.ndcg_at_k(raw_hashes, relevant, 4)
        sa_base = metrics.hit_at_1(raw_hashes, relevant)
        
        base_recalls.append(r_base)
        base_mrrs.append(mrr_base)
        base_ndcgs.append(ndcg_base)
        base_source_accs.append(sa_base)
        if case.type == "cross_document":
            base_xdoc_recalls.append(r_base)
            
        # 2. Compress chunks
        comp_chunks = compress_chunks(raw_chunks)
        comp_hashes = [c.doc_hash for c in comp_chunks]
        
        # Compressed context size
        # Force settings variable temporarily to test actual build_context formatting
        settings.context_compression_enabled = True
        comp_assembled = build_context(raw_chunks)
        comp_char_lens.append(len(comp_assembled.context_text))
        settings.context_compression_enabled = False
        
        # Calculate compressed metrics
        r_comp = metrics.recall_at_k(comp_hashes, relevant, 4)
        mrr_comp = metrics.reciprocal_rank(comp_hashes, relevant)
        ndcg_comp = metrics.ndcg_at_k(comp_hashes, relevant, 4)
        sa_comp = metrics.hit_at_1(comp_hashes, relevant)
        
        comp_recalls.append(r_comp)
        comp_mrrs.append(mrr_comp)
        comp_ndcgs.append(ndcg_comp)
        comp_source_accs.append(sa_comp)
        if case.type == "cross_document":
            comp_xdoc_recalls.append(r_comp)
            
    # Compute averages
    avg_base_recall = sum(base_recalls) / len(base_recalls) if base_recalls else 0.0
    avg_comp_recall = sum(comp_recalls) / len(comp_recalls) if comp_recalls else 0.0
    
    avg_base_mrr = sum(base_mrrs) / len(base_mrrs) if base_mrrs else 0.0
    avg_comp_mrr = sum(comp_mrrs) / len(comp_mrrs) if comp_mrrs else 0.0
    
    avg_base_ndcg = sum(base_ndcgs) / len(base_ndcgs) if base_ndcgs else 0.0
    avg_comp_ndcg = sum(comp_ndcgs) / len(comp_ndcgs) if comp_ndcgs else 0.0
    
    avg_base_sa = sum(base_source_accs) / len(base_source_accs) if base_source_accs else 0.0
    avg_comp_sa = sum(comp_source_accs) / len(comp_source_accs) if comp_source_accs else 0.0
    
    avg_base_xdoc = sum(base_xdoc_recalls) / len(base_xdoc_recalls) if base_xdoc_recalls else 0.0
    avg_comp_xdoc = sum(comp_xdoc_recalls) / len(comp_xdoc_recalls) if comp_xdoc_recalls else 0.0
    
    avg_base_char = sum(base_char_lens) / len(base_char_lens) if base_char_lens else 0.0
    avg_comp_char = sum(comp_char_lens) / len(comp_char_lens) if comp_char_lens else 0.0
    
    token_reduction = 0.0
    if avg_base_char > 0:
        token_reduction = (1.0 - (avg_comp_char / avg_base_char)) * 100.0
        
    # Check criteria
    reduction_ok = 20.0 <= token_reduction <= 40.0
    no_regression = (
        avg_comp_recall >= avg_base_recall and
        avg_comp_mrr >= avg_base_mrr and
        avg_comp_ndcg >= avg_base_ndcg and
        avg_comp_sa >= avg_base_sa and
        avg_comp_xdoc >= avg_base_xdoc
    )
    
    print("=" * 80)
    print("CONTEXT COMPRESSION BENCHMARK EVALUATION REPORT")
    print("=" * 80)
    print(f"Scored Cases               : {len(scored_cases)}")
    print(f"Cross-Document Cases       : {len(cross_doc_cases)}")
    print("-" * 80)
    print(f"{'Metric':<25}{'Baseline':<15}{'Compressed':<15}{'Delta':<10}")
    print("-" * 80)
    print(f"{'Average Context Chars':<25}{avg_base_char:<15.2f}{avg_comp_char:<15.2f}{avg_comp_char - avg_base_char:<10.2f}")
    print(f"{'Recall@4':<25}{avg_base_recall:<15.4f}{avg_comp_recall:<15.4f}{avg_comp_recall - avg_base_recall:<10.4f}")
    print(f"{'MRR':<25}{avg_base_mrr:<15.4f}{avg_comp_mrr:<15.4f}{avg_comp_mrr - avg_base_mrr:<10.4f}")
    print(f"{'nDCG':<25}{avg_base_ndcg:<15.4f}{avg_comp_ndcg:<15.4f}{avg_comp_ndcg - avg_base_ndcg:<10.4f}")
    print(f"{'Source Accuracy':<25}{avg_base_sa:<15.4f}{avg_comp_sa:<15.4f}{avg_comp_sa - avg_base_sa:<10.4f}")
    print(f"{'Cross-Doc Recall':<25}{avg_base_xdoc:<15.4f}{avg_comp_xdoc:<15.4f}{avg_comp_xdoc - avg_base_xdoc:<10.4f}")
    print("-" * 80)
    print(f"Projected Token Reduction  : {token_reduction:.2f}%")
    print(f"Reduction in bounds (20-40%): {'PASS' if reduction_ok else 'FAIL'}")
    print(f"No metric regressions      : {'PASS' if no_regression else 'FAIL'}")
    print("=" * 80)
    
    if not no_regression:
        print("\nERROR: Context compression caused a regression in retrieval quality!", file=sys.stderr)
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
