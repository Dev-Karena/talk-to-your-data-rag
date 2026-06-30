#!/usr/bin/env python
"""Runner script to validate benchmark case answerability against the corpus."""

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
from app.eval.dataset import load_benchmark
from app.rag.vector_store import get_vector_store
from app.eval.benchmark_validator import validate_benchmark

def print_human_report(report: dict):
    """Print a detailed validation report to stdout."""
    print("=" * 80)
    print("BENCHMARK INTEGRITY VALIDATION REPORT")
    print("=" * 80)
    print(f"Benchmark Name   : {report['description']}")
    print(f"Fingerprint      : {report['fingerprint']}")
    print(f"Total Cases      : {report['total_cases']}")
    print(f"Impossible Cases : {report['impossible_cases_count']}")
    print(f"Overall Status   : {'PASS (All cases valid)' if report['is_valid'] else 'FAIL (Impossible cases detected)'}")
    print("-" * 80)
    
    print(f"{'Case ID':<10}{'Type':<15}{'Score':<8}{'Status':<22}{'Issue'}")
    print("-" * 80)
    for c in report["cases"]:
        issue = ""
        if c["impossible"]:
            issue = c["reason"]
        print(f"{c['case_id']:<10}{c['type']:<15}{c['evidence_score']:<8.2f}{c['status']:<22}{issue[:45]}")
        
    print("-" * 80)
    if not report["is_valid"]:
        print("\nCRITICAL ANOMALIES FOUND:")
        for c in report["cases"]:
            if c["impossible"]:
                print(f"\n  * Case {c['case_id']} ('{c['query']}')")
                print(f"    Status: {c['status']}")
                print(f"    Reason: {c['reason']}")
                if c.get("missing_critical_terms"):
                    print(f"    Missing Critical Terms: {c['missing_critical_terms']}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Validate benchmark cases against the indexed corpus.")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    parser.add_argument("--threshold", type=float, default=0.25, help="Threshold for evidence score (default: 0.25)")
    args = parser.parse_args()
    
    # Force settings to target the isolated benchmark corpus
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
        
    store = get_vector_store()
    report = validate_benchmark(benchmark, store, threshold=args.threshold)
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report)
        
    # Exit non-zero on failure
    if not report["is_valid"]:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
