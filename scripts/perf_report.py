"""Performance Metrics — read-only timing report for retrieval stages.

Measures how long the retrieval path takes against the LIVE store, broken down
by stage (embed / search / mmr), averaged over repeated runs. Read-only: it
issues queries but never writes, clears, or re-indexes.

Run:
    python scripts/perf_report.py                       # default sample queries
    python scripts/perf_report.py --runs 10
    python scripts/perf_report.py -q "what is a kernel?" -q "acid transactions"
    python scripts/perf_report.py --json

Note: the first run includes one-time embedding-model load; the report shows a
warm-up run separately so steady-state timings are not skewed by it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings              # noqa: E402
from app.rag.embeddings import get_embedder               # noqa: E402
from app.rag.vector_store import get_vector_store          # noqa: E402
from app.services.retriever import _mmr_select             # noqa: E402
from app.utils.timing import Stopwatch                     # noqa: E402

# Default probes — one per validation document (ML/OS/DBMS).
_DEFAULT_QUERIES = [
    "What is machine learning?",
    "What is an operating system?",
    "What is a database management system?",
]


def _time_query(query: str, k: int, fetch_k: int, use_mmr: bool, lam: float) -> dict:
    """Run one query and return per-stage timings (ms). Read-only."""
    embedder = get_embedder()
    store = get_vector_store()
    sw = Stopwatch()
    with sw.stage("embed"):
        qv = embedder.embed_query(query)
    with sw.stage("search"):
        candidates = store.query_candidates(qv, fetch_k=fetch_k)
    if use_mmr:
        with sw.stage("mmr"):
            _mmr_select(qv, candidates, k, lam)
    return dict(sw.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only retrieval performance report.")
    parser.add_argument("-q", "--query", action="append", dest="queries",
                        help="A query to time (repeatable). Defaults to ML/OS/DBMS probes.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per query.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    settings = get_settings()
    k = settings.top_k
    fetch_k = max(settings.fetch_k, k)
    queries = args.queries or _DEFAULT_QUERIES

    # Warm up (loads the embedding model once; excluded from measured stats).
    warmup = _time_query(queries[0], k, fetch_k, settings.use_mmr, settings.mmr_lambda)

    per_query = []
    for q in queries:
        runs = [_time_query(q, k, fetch_k, settings.use_mmr, settings.mmr_lambda)
                for _ in range(args.runs)]
        stages = sorted({s for r in runs for s in r})
        avg = {s: mean(r.get(s, 0.0) for r in runs) for s in stages}
        per_query.append({
            "query": q,
            "avg_ms": {s: round(v, 2) for s, v in avg.items()},
            "avg_total_ms": round(sum(avg.values()), 2),
        })

    result = {
        "backend": get_embedder().name,
        "settings": {"use_mmr": settings.use_mmr, "top_k": k, "fetch_k": fetch_k},
        "runs_per_query": args.runs,
        "warmup_ms": {s: round(v, 2) for s, v in warmup.items()},
        "queries": per_query,
        "overall_avg_total_ms": round(mean(p["avg_total_ms"] for p in per_query), 2),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 72)
    print("Retrieval Performance Report (read-only)")
    print("=" * 72)
    print(f"Backend        : {result['backend']}")
    print(f"Settings       : use_mmr={settings.use_mmr}  top_k={k}  fetch_k={fetch_k}")
    print(f"Runs per query : {args.runs}  (plus 1 warm-up, excluded)")
    print(f"Warm-up (cold) : " + ", ".join(f"{s}={v}ms" for s, v in result["warmup_ms"].items()))
    print("\nSteady-state averages")
    print("-" * 72)
    print(f"{'query':<40}{'embed':>8}{'search':>8}{'mmr':>7}{'total':>9}")
    print("-" * 72)
    for p in per_query:
        a = p["avg_ms"]
        print(f"{p['query'][:38]:<40}{a.get('embed', 0):>8.2f}{a.get('search', 0):>8.2f}"
              f"{a.get('mmr', 0):>7.2f}{p['avg_total_ms']:>9.2f}")
    print("-" * 72)
    print(f"Overall avg total per query: {result['overall_avg_total_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
