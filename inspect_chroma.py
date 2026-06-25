"""Enhanced ChromaDB inspection utility (read-only).

A read-only diagnostic for the vector store. Run it any time to confirm what is
actually indexed — especially after uploading several PDFs, to verify that
*every* document made it in (not just the first):

    python inspect_chroma.py                 # full summary + integrity one-liner
    python inspect_chroma.py --doc ML.pdf    # per-document chunk detail
    python inspect_chroma.py --json          # machine-readable output

It prints:
    * Collection name, persist dir
    * Total documents (unique source PDFs) and total chunks
    * Per document: name + chunks + pages covered
    * Metadata statistics (chunks-per-document spread)
    * An integrity one-liner (clean vs. N issues) — see metadata_health.py for a
      full scan
    * With --doc: every chunk of one document (page, index, id, text length)

This touches only ChromaDB (no embedding model is loaded), so it runs fast and
works even without the sentence-transformers dependency installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config.settings import get_settings              # noqa: E402
from app.rag.vector_store import get_vector_store          # noqa: E402
from app.utils.metadata_health import EXPECTED_FIELDS, check_records  # noqa: E402


def _build_summary() -> dict:
    """Collect the read-only inspection data into a plain dict."""
    settings = get_settings()
    store = get_vector_store()

    summary = store.document_chunk_counts()
    records = store.all_records()
    health = check_records(records["ids"], records["metadatas"])

    documents = [
        {
            "source": str(e["source"]),
            "doc_hash": doc_hash,
            "chunk_count": int(e["chunk_count"]),
            "pages": list(e["pages"]),
        }
        for doc_hash, e in sorted(summary.items(), key=lambda kv: str(kv[1]["source"]).lower())
    ]

    return {
        "collection": settings.chroma_collection_name,
        "persist_dir": str(settings.chroma_persist_dir),
        "total_documents": len(summary),
        "total_chunks": store.count(),
        "documents": documents,
        "integrity": {
            "ok": health.ok,
            "issue_count": len(health.issues),
            "clean_records": health.clean_records,
        },
    }


def _print_human(data: dict) -> None:
    print("=" * 64)
    print("ChromaDB Inspection (read-only)")
    print("=" * 64)
    print(f"Collection Name : {data['collection']}")
    print(f"Persist Dir     : {data['persist_dir']}")
    print(f"Total Documents : {data['total_documents']}")
    print(f"Total Chunks    : {data['total_chunks']}")

    docs = data["documents"]
    if not docs:
        print("\n(no documents indexed yet - upload PDFs in the app first)")
        return

    print("\nPer Document")
    print("-" * 64)
    print(f"{'Document':<34}{'Chunks':>8}{'Pages':>8}")
    print("-" * 64)
    for d in docs:
        print(f"{d['source']:<34}{d['chunk_count']:>8}{len(d['pages']):>8}")

    chunk_counts = [d["chunk_count"] for d in docs]
    page_counts = [len(d["pages"]) for d in docs]
    print("\nMetadata Statistics")
    print("-" * 64)
    print(f"Documents indexed        : {data['total_documents']}")
    print(f"Chunks total             : {data['total_chunks']}")
    print(f"Chunks per document      : min={min(chunk_counts)}, max={max(chunk_counts)}, "
          f"avg={sum(chunk_counts) / len(chunk_counts):.1f}")
    print(f"Pages per document       : min={min(page_counts)}, max={max(page_counts)}, "
          f"total={sum(page_counts)}")
    print(f"Expected metadata fields : {', '.join(EXPECTED_FIELDS)}")

    integ = data["integrity"]
    print("\nIntegrity")
    print("-" * 64)
    if integ["ok"]:
        print(f"OK - all {data['total_chunks']} chunk(s) have valid metadata.")
    else:
        print(f"!! {integ['issue_count']} issue(s) across "
              f"{data['total_chunks'] - integ['clean_records']} record(s). "
              f"Run: python scripts/metadata_health.py")


def _print_doc_detail(source: str) -> int:
    """Print every chunk of one document by source name. Returns exit code."""
    store = get_vector_store()
    records = store.all_records()
    rows = [
        (rid, meta)
        for rid, meta in zip(records["ids"], records["metadatas"])
        if meta and str(meta.get("source", "")) == source
    ]
    print(f"Document detail: {source}")
    print("-" * 70)
    if not rows:
        print(f"(no chunks found for source '{source}')")
        return 1
    rows.sort(key=lambda r: (int(r[1].get("page_number", 0)), int(r[1].get("chunk_index", 0))))
    print(f"{'page':>5}{'idx':>5}  {'chunk_id':<52}")
    print("-" * 70)
    for rid, meta in rows:
        print(f"{int(meta.get('page_number', 0)):>5}{int(meta.get('chunk_index', 0)):>5}  {rid:<52}")
    print(f"\n{len(rows)} chunk(s) for '{source}'.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only ChromaDB inspector.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--doc", metavar="SOURCE", help="Show per-chunk detail for one document.")
    args = parser.parse_args()

    if args.doc:
        return _print_doc_detail(args.doc)

    data = _build_summary()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_human(data)
    # Non-zero exit if integrity problems exist (handy in CI).
    return 0 if data["integrity"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
