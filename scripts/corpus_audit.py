#!/usr/bin/env python
"""Runner script to audit the vector store chunk corpus for structural health."""

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup project root import path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.rag.vector_store import get_vector_store
from app.eval.corpus_validator import audit_corpus

def print_human_report(report: dict, col_name: str):
    """Print a detailed corpus audit report to stdout."""
    print("=" * 80)
    print(f"CORPUS STRUCTURAL AUDIT REPORT (Collection: '{col_name}')")
    print("=" * 80)
    print(f"Total Chunks     : {report['total_chunks']}")
    print(f"Overall Status   : {'HEALTHY' if report['is_healthy'] else 'ANOMALOUS (Anomalies detected)'}")
    print("-" * 80)
    
    print("CHUNK LENGTH STATS:")
    print(f"  Average Length : {report['length_stats']['avg']} characters")
    print(f"  Min Length     : {report['length_stats']['min']} characters")
    print(f"  Max Length     : {report['length_stats']['max']} characters")
    
    print("\nCHUNK OVERLAP STATS:")
    print(f"  Average Overlap: {report['overlap_stats']['avg']} characters")
    print(f"  Min Overlap    : {report['overlap_stats']['min']} characters")
    print(f"  Max Overlap    : {report['overlap_stats']['max']} characters")
    
    print("\nDOCUMENT SUMMARIES:")
    print(f"  {'Document Source':<25}{'Chunks':<10}{'Pages Indexed'}")
    print(f"  {'-'*23:<25}{'-'*6:<10}{'-'*13}")
    for doc_name, summary in report["document_summaries"].items():
        print(f"  {doc_name:<25}{summary['chunk_count']:<10}{summary['pages_indexed']}")
        
    print("-" * 80)
    print("ANOMALIES DETAIL:")
    anom = report["anomalies"]
    print(f"  * Duplicate chunks count      : {anom['duplicates_count']}")
    print(f"  * Extremely short (<50 chars) : {anom['short_chunks_count']}")
    print(f"  * Missing required metadata   : {anom['missing_metadata_count']}")
    
    if anom['duplicates_count'] > 0:
        print("\n  DUPLICATE CHUNKS:")
        for dup in anom['duplicates']:
            print(f"    - '{dup['chunk_id_1']}' ({dup['source_1']}) matches '{dup['chunk_id_2']}' ({dup['source_2']})")
            
    if anom['short_chunks_count'] > 0:
        print("\n  SHORT CHUNKS:")
        for sc in anom['short_chunks']:
            print(f"    - ID: {sc['chunk_id']} | Length: {sc['length']} | Source: {sc['source']} (Page {sc['page']})")
            
    if anom['missing_metadata_count'] > 0:
        print("\n  MISSING METADATA:")
        for mm in anom['missing_metadata']:
            print(f"    - ID: {mm['chunk_id']} | Missing keys: {mm['missing_keys']}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Audit the structure and health of the vector store corpus.")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    parser.add_argument("--benchmark", action="store_true", help="Force audit of the isolated benchmark collection")
    args = parser.parse_args()
    
    settings = get_settings()
    if args.benchmark:
        settings.chroma_persist_dir = _ROOT / "benchmark_chroma"
        settings.chroma_collection_name = "benchmark_corpus"
        os.environ["CHROMA_PERSIST_DIR"] = str(settings.chroma_persist_dir)
        os.environ["CHROMA_COLLECTION_NAME"] = settings.chroma_collection_name
        
    store = get_vector_store()
    report = audit_corpus(store)
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report, settings.chroma_collection_name)
        
    # Exit non-zero on failure (anomalies detected)
    if not report.get("is_healthy", False):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
