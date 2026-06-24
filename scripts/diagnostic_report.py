"""Sprint-1 diagnostic: prove multi-document presence + retrievability.

Ingests representative ML/OS/DBMS PDFs through the REAL pipeline (real BGE
embeddings) into the REAL configured Chroma collection, then dumps raw,
unedited evidence:

    1. total documents stored
    2. total chunks stored
    3. per-document chunk counts
    4. example metadata from each document
    5. top-10 RAW retrieval (no MMR) for a question from each PDF, showing
       source_file / chunk_id / similarity score
    + ranking-vs-ingestion evidence: raw top-4 (old behavior) vs MMR top-4

Run:  python scripts/diagnostic_report.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings           # noqa: E402
from app.rag.embeddings import get_embedder            # noqa: E402
from app.rag.pipeline import IngestStatus, ingest_document  # noqa: E402
from app.rag.vector_store import get_vector_store       # noqa: E402
from app.services.retriever import _mmr_select          # noqa: E402

# Multi-paragraph bodies so each PDF yields several chunks (richer evidence).
_DOCS = {
    "ML.pdf": [
        "Machine learning is a subfield of artificial intelligence that enables "
        "computer systems to learn patterns directly from data without being "
        "explicitly programmed for every rule. Instead of hand-coding logic, a "
        "model is trained on examples and improves its performance on a task as it "
        "sees more data. The three classical paradigms are supervised learning, "
        "unsupervised learning, and reinforcement learning, each suited to a "
        "different kind of problem and feedback signal.",
        "In supervised learning the model is given labeled examples, pairs of an "
        "input and the correct output, and it learns a function that maps inputs "
        "to outputs. Common algorithms include linear and logistic regression, "
        "decision trees, support vector machines, and neural networks. Training "
        "proceeds by minimizing a loss function that measures the gap between the "
        "model prediction and the true label, typically using gradient descent.",
        "Unsupervised learning finds structure in unlabeled data, for example by "
        "clustering similar points together or reducing dimensionality. "
        "Reinforcement learning trains an agent to take actions in an environment "
        "to maximize a cumulative reward. Across all paradigms the central "
        "challenge is generalization: performing well on unseen data rather than "
        "memorizing the training set, which is the problem of overfitting.",
    ],
    "OS.pdf": [
        "An operating system is the system software that manages a computer's "
        "hardware and software resources and provides common services for "
        "application programs. It sits between user programs and the bare "
        "hardware, exposing clean abstractions such as files, processes, and "
        "sockets so that applications do not have to manage physical devices "
        "directly. The core of the operating system is called the kernel.",
        "The kernel is responsible for process scheduling, memory management, and "
        "device input and output. A process is a running program with its own "
        "address space, and the scheduler decides which process or thread runs on "
        "the CPU at any moment, switching rapidly between them to give the "
        "illusion of concurrency. Threads share an address space and allow a "
        "single program to run multiple flows of control.",
        "Virtual memory gives each process the illusion of a large, private, "
        "contiguous address space, which the operating system maps onto physical "
        "RAM and backs with paging to disk when memory is scarce. The operating "
        "system also enforces protection and isolation between processes, manages "
        "file systems on storage devices, and mediates access to peripherals "
        "through device drivers.",
    ],
    "DBMS.pdf": [
        "A database management system, or DBMS, is software that lets users "
        "define, store, retrieve, and manage data in an organized way. It "
        "provides a layer between the raw stored bytes and the applications that "
        "use the data, handling concurrency, recovery, and integrity so that many "
        "users can share the same data safely and efficiently.",
        "A relational DBMS organizes data into tables made of rows and columns, "
        "where each table represents an entity and relationships are expressed "
        "through shared key values. Users query and modify the data using SQL, a "
        "declarative language in which you describe the result you want rather "
        "than the procedure to compute it. The query optimizer then chooses an "
        "efficient execution plan.",
        "Transactions group several operations into a single logical unit that is "
        "guaranteed to satisfy the ACID properties: atomicity, consistency, "
        "isolation, and durability. Indexes such as B-trees accelerate lookups by "
        "avoiding full table scans, normalization reduces redundancy by splitting "
        "data into related tables, and locking or multiversion concurrency "
        "control keeps concurrent transactions isolated from one another.",
    ],
}


def _ensure_pdfs(docs_dir: Path) -> dict[str, bytes]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    for name, paras in _DOCS.items():
        path = docs_dir / name
        if not path.is_file():
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(path), pagesize=letter)
            width, height = letter
            for para in paras:  # one paragraph per page -> multi-page docs
                y = height - 72
                for line in textwrap.wrap(para, 88):
                    c.drawString(72, y, line)
                    y -= 15
                c.showPage()
            c.save()
        out[name] = path.read_bytes()
    return out


def main() -> int:
    settings = get_settings()
    store = get_vector_store()

    # Start from a clean, reproducible state in the REAL collection.
    store.clear()

    docs = _ensure_pdfs(Path("documents"))
    print("Ingesting via real pipeline (backend: %s)\n" % get_embedder().name)
    for name, data in docs.items():
        r = ingest_document(name, data)
        print(f"  ingest {name:<9} -> {r.status.value:<8} chunks={r.chunk_count}")

    coll = store._collection  # diagnostic-only direct access

    # ---- 1 & 2. Totals -----------------------------------------------------
    summary = store.document_chunk_counts()
    print("\n" + "=" * 70)
    print("1) TOTAL DOCUMENTS STORED :", len(summary))
    print("2) TOTAL CHUNKS STORED    :", store.count())

    # ---- 3. Per-document chunk counts -------------------------------------
    print("\n3) PER-DOCUMENT CHUNK COUNTS")
    for e in sorted(summary.values(), key=lambda x: str(x["source"])):
        print(f"     {str(e['source']):<10} chunks={e['chunk_count']:<3} pages={e['pages']}")

    # ---- 4. Example metadata from each document ---------------------------
    print("\n4) EXAMPLE METADATA (one chunk per document)")
    for doc_hash, e in summary.items():
        got = coll.get(where={"doc_hash": doc_hash}, limit=1, include=["metadatas"])
        print(f"     {e['source']}:")
        print(f"       {got['metadatas'][0]}")

    # ---- 5. Top-10 RAW retrieval per PDF ----------------------------------
    embedder = get_embedder()
    questions = {
        "A (ML.pdf)":   "What is machine learning?",
        "B (OS.pdf)":   "What is an operating system?",
        "C (DBMS.pdf)": "What is a DBMS?",
    }
    print("\n5) TOP-10 RAW RETRIEVAL (pure cosine similarity, MMR OFF)")
    for label, q in questions.items():
        qv = embedder.embed_query(q)
        hits = store.query(qv, top_k=10)
        print(f"\n   {label}  ::  {q!r}")
        print(f"   {'rank':<5}{'source_file':<12}{'chunk_id':<28}{'score':>8}")
        print("   " + "-" * 53)
        for i, h in enumerate(hits, 1):
            print(f"   {i:<5}{h.source:<12}{h.chunk_id:<28}{h.score:>8.4f}")

    # ---- C evidence: ranking vs ingestion ---------------------------------
    print("\n" + "=" * 70)
    print("RANKING-vs-INGESTION EVIDENCE  (cross-document question)")
    q = "Compare machine learning and databases."
    qv = embedder.embed_query(q)
    raw4 = store.query(qv, top_k=4)  # what the ORIGINAL code returned (no MMR)
    cands = store.query_candidates(qv, fetch_k=settings.fetch_k)
    mmr4 = _mmr_select(qv, cands, k=4, lambda_mult=settings.mmr_lambda)
    print(f"   question: {q!r}")
    print(f"   RAW top-4 (old behavior)  sources: {[h.source for h in raw4]}")
    print(f"   MMR top-4 (new behavior)  sources: {[h.source for h in mmr4]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
