"""Read-only evidence report for the REAL Chroma collection.

Proves, against your live ``chroma_db/`` (never modified), that:

    1. every uploaded PDF is stored        (documents + chunk counts + metadata)
    2. every uploaded PDF is retrievable    (a content-derived query for each
       document returns that document, ideally as the top hit)

Unlike ``scripts/diagnostic_report.py``, this script is strictly READ-ONLY: it
never calls ``store.clear()`` or ingests anything, so it is safe to run against
the real collection. The per-document retrieval query is derived from each
document's own indexed text, so the proof adapts to whatever is actually stored
— nothing is hard-coded.

It embeds queries with the configured embedder (so retrieval is real), and
writes a Markdown evidence report to ``docs/evidence/`` in addition to printing
to stdout.

Usage:
    python scripts/evidence_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings           # noqa: E402
from app.rag.embeddings import get_embedder            # noqa: E402
from app.rag.vector_store import get_vector_store       # noqa: E402

# How many chunks to pull per retrieval probe.
_TOP_K = 5
# Max characters of a chunk's text used to build the per-document query.
_QUERY_CHARS = 240


def _representative_text(coll, doc_hash: str) -> str:
    """Return text from one chunk of the given document, to use as a query.

    Picks the document's first page/chunk so the query reflects the document's
    own content rather than anything external.
    """
    got = coll.get(where={"doc_hash": doc_hash}, include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    if not docs:
        return ""
    # Prefer the earliest (page_number, chunk_index) chunk for determinism.
    order = sorted(
        range(len(docs)),
        key=lambda i: (
            int((metas[i] or {}).get("page_number", 0)),
            int((metas[i] or {}).get("chunk_index", 0)),
        ),
    )
    return docs[order[0]] or ""


def main() -> int:
    settings = get_settings()
    store = get_vector_store()
    coll = store._collection  # read-only diagnostic access

    summary = store.document_chunk_counts()
    total_chunks = store.count()

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("=" * 70)
    out("CHROMA EVIDENCE REPORT (read-only, real collection)")
    out("=" * 70)
    out(f"Collection      : {settings.chroma_collection_name}")
    out(f"Persist dir     : {settings.chroma_persist_dir}")
    out(f"Total documents : {len(summary)}")
    out(f"Total chunks    : {total_chunks}")

    if not summary:
        out("\n(no documents indexed — nothing to prove)")
        _write_markdown(lines)
        return 1

    # ---- 1. Documents + chunk counts + example metadata --------------------
    out("\n1) DOCUMENTS STORED (chunk counts + example metadata)")
    out("-" * 70)
    out(f"{'document':<14}{'chunks':>8}{'pages':>8}   example chunk_id")
    out("-" * 70)
    # Stable, deterministic ordering by source name.
    ordered = sorted(summary.items(), key=lambda kv: str(kv[1]["source"]).lower())
    example_meta: dict[str, dict] = {}
    for doc_hash, e in ordered:
        got = coll.get(where={"doc_hash": doc_hash}, limit=1, include=["metadatas"])
        meta = (got.get("metadatas") or [{}])[0] or {}
        example_meta[doc_hash] = meta
        out(
            f"{str(e['source']):<14}{int(e['chunk_count']):>8}{len(e['pages']):>8}"
            f"   {meta.get('chunk_id', '<none>')}"
        )

    out("\n   Example metadata (one chunk per document):")
    for doc_hash, e in ordered:
        out(f"     {e['source']}: {example_meta[doc_hash]}")

    # ---- 2. Per-document retrieval proof -----------------------------------
    embedder = get_embedder()
    out("\n2) PER-DOCUMENT RETRIEVAL (query derived from each document's text)")
    out(f"   backend={embedder.name}  top_k={_TOP_K}")
    out("-" * 70)

    all_retrievable = True
    all_top1 = True
    for doc_hash, e in ordered:
        source = str(e["source"])
        seed = _representative_text(coll, doc_hash)[:_QUERY_CHARS].strip()
        qv = embedder.embed_query(seed)
        hits = store.query(qv, top_k=_TOP_K)

        retrieved = any(h.doc_hash == doc_hash for h in hits)
        top1 = bool(hits) and hits[0].doc_hash == doc_hash
        all_retrievable = all_retrievable and retrieved
        all_top1 = all_top1 and top1

        status = "RETRIEVED" if retrieved else "MISSING"
        rank = next((i for i, h in enumerate(hits, 1) if h.doc_hash == doc_hash), None)
        out(
            f"\n   {source}  [{status}"
            f"{f', rank #{rank}' if rank else ''}]"
        )
        out(f"     query: {seed[:90]!r}{'...' if len(seed) > 90 else ''}")
        out(f"     {'rank':<5}{'source_file':<14}{'chunk_id':<30}{'score':>8}")
        out("     " + "-" * 57)
        for i, h in enumerate(hits, 1):
            mark = " <=" if h.doc_hash == doc_hash else ""
            out(f"     {i:<5}{h.source:<14}{h.chunk_id:<30}{h.score:>8.4f}{mark}")

    # ---- 3. Verdict --------------------------------------------------------
    out("\n" + "=" * 70)
    out("VERDICT")
    out("-" * 70)
    out(f"  Documents stored                 : {len(summary)}")
    out(f"  Total chunks                     : {total_chunks}")
    out(f"  Every document retrievable       : {'YES' if all_retrievable else 'NO'}")
    out(f"  Every document is its own top hit: {'YES' if all_top1 else 'NO'}")
    verdict_ok = all_retrievable
    out(f"\n  RESULT: {'PASS — all uploaded PDFs are stored and retrievable.' if verdict_ok else 'FAIL — see MISSING entries above.'}")
    out("=" * 70)

    _write_markdown(lines)
    return 0 if verdict_ok else 1


def _write_markdown(lines: list[str]) -> None:
    """Persist the report as a fenced Markdown file under docs/evidence/."""
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "chroma_evidence_report.md"
    body = "\n".join(lines)
    path.write_text(
        "# Chroma Evidence Report\n\n"
        "Read-only proof that every uploaded PDF is stored and retrievable in the\n"
        "real Chroma collection. Generated by `scripts/evidence_report.py`.\n\n"
        "```\n" + body + "\n```\n",
        encoding="utf-8",
    )
    print(f"\n[written] {path}")


if __name__ == "__main__":
    raise SystemExit(main())
