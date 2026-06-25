"""Retrieval Inspector — read-only 'explain this query' tool.

Runs any query against the LIVE vector store and shows exactly what retrieval
does, without mutating anything and without changing retrieval behavior:

    * timing for each stage (embed / search / mmr)
    * the candidate pool (raw cosine similarity, MMR off)
    * the final top-k as the app would return it (MMR on/off per settings)
    * a side-by-side of RAW vs MMR ordering and the documents each surfaces

Use it to debug "why did this answer cite that chunk?" or "is MMR spreading
across documents?".

Run:
    python scripts/retrieval_inspector.py "What is an operating system?"
    python scripts/retrieval_inspector.py "compare ML and databases" --top-k 6
    python scripts/retrieval_inspector.py "..." --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings              # noqa: E402
from app.rag.embeddings import get_embedder               # noqa: E402
from app.rag.vector_store import get_vector_store          # noqa: E402
from app.services.retriever import _mmr_select             # noqa: E402
from app.utils.timing import Stopwatch                     # noqa: E402


def _inspect(query: str, top_k: int) -> dict:
    """Gather read-only retrieval diagnostics for a query."""
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    fetch_k = max(settings.fetch_k, k)

    embedder = get_embedder()
    store = get_vector_store()
    sw = Stopwatch()

    with sw.stage("embed"):
        qv = embedder.embed_query(query)

    # Candidate pool (raw similarity, no MMR) paired with vectors for MMR.
    with sw.stage("search"):
        candidates = store.query_candidates(qv, fetch_k=fetch_k)

    raw_top = [c for c, _ in candidates][:k]

    with sw.stage("mmr"):
        mmr_top = _mmr_select(qv, candidates, k, settings.mmr_lambda)

    final = mmr_top if settings.use_mmr else raw_top

    def _row(chunk) -> dict:
        return {
            "source": chunk.source,
            "page": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "chunk_id": chunk.chunk_id,
            "score": round(chunk.score, 4),
        }

    return {
        "query": query,
        "backend": embedder.name,
        "settings": {
            "use_mmr": settings.use_mmr,
            "top_k": k,
            "fetch_k": fetch_k,
            "mmr_lambda": settings.mmr_lambda,
        },
        "timing_ms": {name: round(ms, 2) for name, ms in sw.items()},
        "candidate_pool": [_row(c) for c, _ in candidates],
        "raw_top_k": [_row(c) for c in raw_top],
        "mmr_top_k": [_row(c) for c in mmr_top],
        "final_top_k": [_row(c) for c in final],
        "final_sources": sorted({c.source for c in final}),
    }


def _print_table(title: str, rows: list) -> None:
    print(f"\n{title}")
    print(f"   {'rank':<5}{'source':<14}{'pg':>3}{'idx':>4}  {'score':>8}  chunk_id")
    print("   " + "-" * 78)
    for i, r in enumerate(rows, 1):
        print(f"   {i:<5}{r['source']:<14}{r['page']:>3}{r['chunk_index']:>4}  "
              f"{r['score']:>8.4f}  {r['chunk_id']}")


def _print_human(data: dict) -> None:
    s = data["settings"]
    print("=" * 84)
    print("Retrieval Inspector (read-only)")
    print("=" * 84)
    print(f"Query    : {data['query']!r}")
    print(f"Backend  : {data['backend']}")
    print(f"Settings : use_mmr={s['use_mmr']}  top_k={s['top_k']}  "
          f"fetch_k={s['fetch_k']}  mmr_lambda={s['mmr_lambda']}")
    timing = data["timing_ms"]
    print(f"Timing   : " + ", ".join(f"{k}={v}ms" for k, v in timing.items())
          + f"  (total={sum(timing.values()):.2f}ms)")

    _print_table(f"Candidate pool (raw cosine, {len(data['candidate_pool'])} fetched):",
                 data["candidate_pool"])
    _print_table("RAW top-k (MMR off - pure relevance):", data["raw_top_k"])
    _print_table("MMR top-k (diversified):", data["mmr_top_k"])

    raw_src = [r["source"] for r in data["raw_top_k"]]
    mmr_src = [r["source"] for r in data["mmr_top_k"]]
    print("\nRanking comparison")
    print("-" * 84)
    print(f"   RAW sources : {raw_src}")
    print(f"   MMR sources : {mmr_src}")
    print(f"   ACTIVE      : {'MMR' if s['use_mmr'] else 'RAW'} "
          f"-> final sources {data['final_sources']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only retrieval inspector.")
    parser.add_argument("query", help="The query to inspect.")
    parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    data = _inspect(args.query, args.top_k)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_human(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
