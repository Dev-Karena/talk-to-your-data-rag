"""ChromaDB inspection utility.

A read-only diagnostic for the vector store. Run it any time to confirm what is
actually indexed — especially after uploading several PDFs, to verify that
*every* document made it in (not just the first):

    python inspect_chroma.py

It prints:
    * Collection name
    * Total documents (unique source PDFs)
    * Total chunks
    * Per document: name + number of chunks (+ pages covered)
    * Metadata statistics (schema fields, pages, chunks-per-document spread)

This touches only ChromaDB (no embedding model is loaded), so it runs fast and
works even without the sentence-transformers dependency installed.
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.rag.vector_store import get_vector_store

# Fields every chunk is expected to carry (see app.rag.chunker.Chunk.metadata).
_EXPECTED_METADATA_FIELDS = (
    "chunk_id",
    "source",
    "page_number",
    "doc_hash",
    "chunk_index",
)


def main() -> None:
    """Print a human-readable summary of the current vector store."""
    settings = get_settings()
    store = get_vector_store()

    total_chunks = store.count()
    summary = store.document_chunk_counts()

    print("=" * 60)
    print("ChromaDB Inspection")
    print("=" * 60)
    print(f"Collection Name : {settings.chroma_collection_name}")
    print(f"Persist Dir     : {settings.chroma_persist_dir}")
    print(f"Total Documents : {len(summary)}")
    print(f"Total Chunks    : {total_chunks}")

    if not summary:
        print("\n(no documents indexed yet — upload PDFs in the app first)")
        return

    print("\nPer Document")
    print("-" * 60)
    print(f"{'Document':<34}{'Chunks':>8}{'Pages':>8}")
    print("-" * 60)
    # Sort by source name for stable, scannable output.
    for entry in sorted(summary.values(), key=lambda e: str(e["source"]).lower()):
        pages = entry["pages"]  # sorted list[int]
        print(f"{str(entry['source']):<34}{int(entry['chunk_count']):>8}{len(pages):>8}")

    print("\nMetadata Statistics")
    print("-" * 60)
    chunk_counts = [int(e["chunk_count"]) for e in summary.values()]
    page_counts = [len(e["pages"]) for e in summary.values()]
    print(f"Documents indexed        : {len(summary)}")
    print(f"Chunks total             : {total_chunks}")
    print(f"Chunks per document      : "
          f"min={min(chunk_counts)}, max={max(chunk_counts)}, "
          f"avg={sum(chunk_counts) / len(chunk_counts):.1f}")
    print(f"Pages per document       : "
          f"min={min(page_counts)}, max={max(page_counts)}, "
          f"total={sum(page_counts)}")
    print(f"Expected metadata fields : {', '.join(_EXPECTED_METADATA_FIELDS)}")


if __name__ == "__main__":
    main()
