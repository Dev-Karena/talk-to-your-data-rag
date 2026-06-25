"""Build the ISOLATED benchmark corpus from the real textbooks.

Indexes documents/ML.pdf, documents/OS.pdf, documents/DBMS.pdf into a SEPARATE
Chroma store (``benchmark_chroma/``) and a separate documents dir, so production
``chroma_db/`` and ``documents/`` are never touched. This is the only data-
writing step in Sprint 4, and it writes only to the isolated benchmark location.

It prints each document's content hash and chunk count — copy the hashes into
benchmarks/retrieval_cases.yaml as ground truth.

Run:
    python scripts/build_benchmark_corpus.py
    python scripts/build_benchmark_corpus.py --rebuild   # clear & re-index
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Isolate BEFORE importing the app (settings/logger read env once at import).
_ROOT = Path(__file__).resolve().parents[1]
os.environ["CHROMA_PERSIST_DIR"] = str(_ROOT / "benchmark_chroma")
os.environ["CHROMA_COLLECTION_NAME"] = "benchmark_corpus"
os.environ["DOCUMENTS_DIR"] = str(_ROOT / "benchmark_chroma" / "source_docs")
os.environ.setdefault("GROQ_API_KEY", "benchmark")  # generation unused here

sys.path.insert(0, str(_ROOT))

from app.rag.pipeline import IngestStatus, ingest_document   # noqa: E402
from app.rag.vector_store import get_vector_store             # noqa: E402

# The real textbooks (canonical Option-A corpus). Read from the real documents/.
_TEXTBOOKS = ["ML.pdf", "OS.pdf", "DBMS.pdf"]
_SOURCE_DIR = _ROOT / "documents"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the isolated benchmark corpus.")
    parser.add_argument("--rebuild", action="store_true", help="Clear the benchmark store first.")
    args = parser.parse_args()

    store = get_vector_store()
    if args.rebuild:
        store.clear()
        print("Cleared benchmark store.")

    print(f"Indexing real textbooks into isolated store: {os.environ['CHROMA_PERSIST_DIR']}\n")
    for name in _TEXTBOOKS:
        path = _SOURCE_DIR / name
        if not path.is_file():
            print(f"  !! missing {path} — skipping")
            continue
        result = ingest_document(name, path.read_bytes())
        print(f"  {name:<9} status={result.status.value:<8} "
              f"chunks={result.chunk_count:<5} doc_hash={result.doc_hash}")

    print("\nBenchmark corpus summary")
    print("-" * 60)
    summary = store.document_chunk_counts()
    for doc_hash, e in sorted(summary.items(), key=lambda kv: str(kv[1]["source"])):
        print(f"  {str(e['source']):<9} chunks={int(e['chunk_count']):<5} "
              f"pages={len(e['pages']):<4} doc_hash={doc_hash}")
    print(f"\nTotal documents: {len(summary)}   Total chunks: {store.count()}")
    print("\nCopy the doc_hash values above into benchmarks/retrieval_cases.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
