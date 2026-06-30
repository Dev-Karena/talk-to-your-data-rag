"""Evaluation script for citation precision, recall, and F1-score."""

import os
import sys
import yaml
from typing import List

# Configure environment variables to point to the benchmark collection
os.environ["CHROMA_PERSIST_DIR"] = "benchmark_chroma"
os.environ["CHROMA_COLLECTION_NAME"] = "benchmark_corpus"

from app.config.settings import get_settings
from app.services.retriever import retrieve
from app.services.citation_builder import build_citations
from app.eval.citation_metrics import calculate_citation_metrics

def main() -> None:
    settings = get_settings()
    settings.chroma_persist_dir = "benchmark_chroma"
    settings.chroma_collection_name = "benchmark_corpus"

    yaml_path = os.path.join("benchmarks", "retrieval_cases.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases = data.get("cases", [])
    if not cases:
        print("Error: No benchmark cases found in YAML.")
        sys.exit(1)

    print(f"Evaluating citations on {len(cases)} cases...")
    print("-" * 80)
    print(f"{'Case ID':<10} | {'Expected Sources':<25} | {'Cited Sources':<25} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 80)

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0

    for case in cases:
        case_id = case.get("id")
        query = case.get("query")
        expected_sources = case.get("expected_sources", [])

        # Retrieve chunks using active hybrid retrieval
        chunks = retrieve(query)
        # Build citations
        citations = build_citations(chunks)
        cited_sources = [c.source for c in citations]

        metrics = calculate_citation_metrics(cited_sources, expected_sources)
        
        prec = metrics["precision"]
        rec = metrics["recall"]
        f1 = metrics["f1"]

        total_precision += prec
        total_recall += rec
        total_f1 += f1
        count += 1

        print(f"{case_id:<10} | {str(expected_sources):<25} | {str(cited_sources):<25} | {prec:<6.3f} | {rec:<6.3f} | {f1:<6.3f}")

    avg_precision = total_precision / count if count > 0 else 0.0
    avg_recall = total_recall / count if count > 0 else 0.0
    avg_f1 = total_f1 / count if count > 0 else 0.0

    print("-" * 80)
    print(f"{'AVERAGE':<10} | {'':<25} | {'':<25} | {avg_precision:<6.3f} | {avg_recall:<6.3f} | {avg_f1:<6.3f}")
    print("-" * 80)

    print(f"Citation Precision: {avg_precision:.4f} (Criteria: >= 0.90)")
    print(f"Citation Recall   : {avg_recall:.4f} (Criteria: >= 0.90)")
    print(f"Citation F1-Score : {avg_f1:.4f}")

    # Enforce acceptance criteria
    if avg_precision < 0.90 or avg_recall < 0.90:
        print("FAIL: Citation metrics fell below the 0.90 threshold.")
        sys.exit(1)
    else:
        print("PASS: Citation metrics satisfy acceptance criteria.")
        sys.exit(0)

if __name__ == "__main__":
    main()
