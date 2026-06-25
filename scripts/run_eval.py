"""Run the retrieval benchmark (read-only) and report metrics.

Scores the production ``retrieve()`` path against benchmarks/retrieval_cases.yaml
and prints a human-readable report (or --json). Targets the ISOLATED benchmark
store by default so it never touches production data; override with env vars
CHROMA_PERSIST_DIR / CHROMA_COLLECTION_NAME to benchmark another corpus.

Read-only: it issues queries only; it never writes, clears, or re-indexes.

Run:
    python scripts/run_eval.py
    python scripts/run_eval.py --top-k 5 --json
    python scripts/run_eval.py --min-recall 0.9     # non-zero exit if below
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Default to the isolated benchmark corpus (overridable via real env vars).
_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CHROMA_PERSIST_DIR", str(_ROOT / "benchmark_chroma"))
os.environ.setdefault("CHROMA_COLLECTION_NAME", "benchmark_corpus")
os.environ.setdefault("GROQ_API_KEY", "benchmark")

sys.path.insert(0, str(_ROOT))

from app.eval.dataset import load_benchmark                # noqa: E402
from app.eval.runner import run_benchmark                  # noqa: E402
from app.rag.vector_store import get_vector_store          # noqa: E402

_DEFAULT_DATASET = _ROOT / "benchmarks" / "retrieval_cases.yaml"


def _print_human(report, dataset_path: Path) -> None:
    agg = report.aggregate
    print("=" * 88)
    print("Retrieval Benchmark Report (read-only)")
    print("=" * 88)
    print(f"Dataset         : {dataset_path}")
    print(f"Cases           : {report.total_cases} "
          f"({report.scored_cases} scored, {report.negative_cases} negative)")
    print(f"top_k           : {report.k}")
    print(f"Corpus match    : {report.corpus_match}  "
          f"(dataset={report.corpus_fingerprint_dataset}, live={report.corpus_fingerprint_live})")
    if not report.corpus_match:
        print("  !! WARNING: live corpus does not match the dataset's expected corpus.")

    print("\nAggregate metrics (averaged over scored cases)")
    print("-" * 88)
    print(f"  Recall@{report.k}        : {agg['recall_at_k']:.3f}")
    print(f"  Precision@{report.k}     : {agg['precision_at_k']:.3f}")
    print(f"  Hit@1            : {agg['hit_at_1']:.3f}")
    print(f"  MRR              : {agg['mrr']:.3f}")
    print(f"  nDCG@{report.k}          : {agg['ndcg_at_k']:.3f}")
    print(f"  Source accuracy  : {agg['source_accuracy']:.3f}")
    if agg.get("chunk_recall_at_k") is not None:
        print(f"  Chunk recall@{report.k}   : {agg['chunk_recall_at_k']:.3f}")

    t = report.timing_ms
    print(f"\nTiming: avg={t['avg_query_ms']}ms  min={t['min_query_ms']}ms  max={t['max_query_ms']}ms")

    print("\nPer-case results")
    print("-" * 88)
    print(f"  {'id':<9}{'type':<15}{'hit@1':>6}{'rank':>6}{'recall':>8}{'ndcg':>7}{'top_score':>11}  query")
    print("  " + "-" * 86)
    for r in report.cases:
        rank = r.matched_rank if r.matched_rank is not None else "-"
        print(f"  {r.id:<9}{r.type:<15}{r.hit_at_1:>6.0f}{str(rank):>6}"
              f"{r.recall_at_k:>8.2f}{r.ndcg_at_k:>7.2f}{r.top_score:>11.4f}  {r.query[:40]}")

    if report.negative_cases:
        print("\nNegative cases (no abstention threshold exists; top_score shown for inspection):")
        for r in report.cases:
            if r.type == "negative":
                print(f"  {r.id:<9} top_source={r.retrieved_sources[0] if r.retrieved_sources else '-':<9} "
                      f"top_score={r.top_score:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark (read-only).")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET), help="Benchmark YAML path.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Fail (exit 2) if aggregate Recall@k is below this.")
    args = parser.parse_args()

    benchmark = load_benchmark(args.dataset)
    store = get_vector_store()
    live_hashes = list(store.list_sources().keys())
    k = args.top_k if args.top_k is not None else None

    # Resolve k: fall back to settings TOP_K via a dummy retrieve only if needed.
    if k is None:
        from app.config.settings import get_settings
        k = get_settings().top_k

    report = run_benchmark(benchmark, live_hashes, k)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report, Path(args.dataset))

    if args.min_recall is not None and report.aggregate["recall_at_k"] < args.min_recall:
        print(f"\nFAIL: Recall@{k} {report.aggregate['recall_at_k']:.3f} < "
              f"min-recall {args.min_recall}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
