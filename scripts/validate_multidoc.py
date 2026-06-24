"""End-to-end multi-document validation with REAL embeddings.

Unlike the deterministic unit tests (which use stub vectors), this script runs
the *actual* ingestion + retrieval pipeline — real PDF parsing and real BGE
embeddings — over three documents (ML.pdf, OS.pdf, DBMS.pdf) and checks that:

    ✓ all documents embed and land in Chroma
    ✓ all documents are individually retrievable
    ✓ cross-document retrieval works ("compare ML and DBMS" pulls from both)

It runs against an ISOLATED Chroma directory/collection, so it never touches
your real ``chroma_db/``.

Requirements:
    * sentence-transformers installed (the default LOCAL embedding backend).
    * reportlab installed (only if the sample PDFs need to be generated).
      Alternatively, drop your own ML.pdf / OS.pdf / DBMS.pdf into ``documents/``
      before running and they will be used as-is.

Usage:
    python scripts/validate_multidoc.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

# Isolate the validation run BEFORE importing the app (settings read env once).
_VALIDATION_DIR = Path(tempfile.gettempdir()) / "ttyd_validation_chroma"
os.environ["CHROMA_PERSIST_DIR"] = str(_VALIDATION_DIR)
os.environ["CHROMA_COLLECTION_NAME"] = "validation_multidoc"
os.environ.setdefault("GROQ_API_KEY", "validation")  # retrieval doesn't call the LLM

# Make `import app...` resolve when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil  # noqa: E402

from app.rag.pipeline import IngestStatus, ingest_document  # noqa: E402
from app.services.retriever import retrieve  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

_SAMPLE_TEXT = {
    "ML.pdf": (
        "Machine Learning",
        "Machine learning is a subfield of artificial intelligence that lets "
        "systems learn patterns from data without being explicitly programmed. "
        "Paradigms include supervised, unsupervised, and reinforcement learning. "
        "Supervised learning maps labeled inputs to outputs using algorithms such "
        "as linear regression, decision trees, and neural networks. The goal is "
        "to generalize to unseen data by minimizing a loss function.",
    ),
    "OS.pdf": (
        "Operating Systems",
        "An operating system is system software that manages computer hardware "
        "and software resources and provides common services for programs. The "
        "kernel handles process scheduling, memory management, and device input "
        "and output. Processes and threads are scheduled by the OS scheduler, and "
        "virtual memory gives each process an isolated address space backed by "
        "paging between RAM and disk.",
    ),
    "DBMS.pdf": (
        "Database Management Systems",
        "A database management system (DBMS) is software that stores, retrieves, "
        "and manages data in databases. A relational DBMS organizes data into "
        "tables of rows and columns and uses SQL for queries, while ACID "
        "properties guarantee reliable transactions. Indexes such as B-trees "
        "speed up lookups, normalization reduces redundancy, and locking provides "
        "concurrency control and isolation between transactions.",
    ),
}


def _ensure_sample_pdfs(docs_dir: Path) -> dict[str, bytes]:
    """Return name->bytes for the three sample PDFs, generating any that are missing."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    for name, (title, body) in _SAMPLE_TEXT.items():
        path = docs_dir / name
        if path.is_file():
            out[name] = path.read_bytes()
            continue
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError:
            sys.exit(
                f"Missing '{name}' in {docs_dir} and reportlab is not installed.\n"
                "Either `pip install reportlab` or place ML.pdf/OS.pdf/DBMS.pdf "
                "into the documents/ directory yourself."
            )
        c = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - 72
        c.drawString(72, y, title)
        y -= 28
        for line in textwrap.wrap(body, 90):
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
        c.save()
        out[name] = path.read_bytes()
    return out


def main() -> int:
    print("=" * 64)
    print("Multi-document validation (real embeddings)")
    print("=" * 64)

    # Fresh, isolated store every run.
    shutil.rmtree(_VALIDATION_DIR, ignore_errors=True)
    get_vector_store.cache_clear()  # drop any cached store bound to a stale dir

    docs = _ensure_sample_pdfs(Path("documents"))

    # ---- Ingest ------------------------------------------------------------
    print("\n[1] Ingesting documents")
    for name, data in docs.items():
        result = ingest_document(name, data)
        flag = "OK " if result.status is IngestStatus.INDEXED else "!! "
        print(f"  {flag}{name:<10} status={result.status.value:<8} chunks={result.chunk_count}")

    # ---- Storage check -----------------------------------------------------
    store = get_vector_store()
    summary = store.document_chunk_counts()
    print("\n[2] Stored in Chroma")
    print(f"  documents={len(summary)}  chunks={store.count()}")
    for entry in sorted(summary.values(), key=lambda e: str(e["source"])):
        print(f"    - {entry['source']:<10} chunks={entry['chunk_count']}")

    checks: list[tuple[str, bool]] = []
    checks.append(("all 3 documents embedded & stored", len(summary) == 3))

    # ---- Per-document retrieval -------------------------------------------
    print("\n[3] Per-document retrieval")
    single = {
        "What is Machine Learning?": "ML.pdf",
        "What is an Operating System?": "OS.pdf",
        "What is DBMS?": "DBMS.pdf",
    }
    for question, expected in single.items():
        hits = retrieve(question)
        top = hits[0].source if hits else "<none>"
        ok = top == expected
        checks.append((f"{question!r} -> {expected}", ok))
        print(f"  {'OK ' if ok else '!! '}{question:<32} top={top}")

    # ---- Cross-document retrieval -----------------------------------------
    print("\n[4] Cross-document retrieval")
    q = "Compare Machine Learning and DBMS."
    hits = retrieve(q)
    sources = {h.source for h in hits}
    ok = {"ML.pdf", "DBMS.pdf"}.issubset(sources)
    checks.append((f"{q!r} spans ML.pdf + DBMS.pdf", ok))
    print(f"  {'OK ' if ok else '!! '}{q}")
    print(f"     retrieved sources: {sorted(sources)}")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 64)
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n{passed}/{len(checks)} checks passed")
    print("=" * 64)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
