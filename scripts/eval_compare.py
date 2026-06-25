"""Before/after retrieval benchmark comparison (read-only).

Runs the benchmark twice against the SAME isolated corpus — once in baseline
mode (all Sprint-5 improvements off) and once in improved mode — and prints the
metric deltas, with a dedicated cross-document breakdown. This is the mandatory
before/after report for Sprint 5.

It switches configuration in-process via environment variables + a settings
cache clear, so both runs hit the identical embedder, store, and dataset.

Run:
    python scripts/eval_compare.py
    python scripts/eval_compare.py --json
    # tune a knob for the 'improved' run:
    python scripts/eval_compare.py --max-per-doc 2 --mmr-lambda 0.3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Default to the isolated benchmark corpus (never production).
_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CHROMA_PERSIST_DIR", str(_ROOT / "benchmark_chroma"))
os.environ.setdefault("CHROMA_COLLECTION_NAME", "benchmark_corpus")
os.environ.setdefault("GROQ_API_KEY", "benchmark")

import sys  # noqa: E402

sys.path.insert(0, str(_ROOT))

from app.eval.dataset import load_benchmark                # noqa: E402
from app.eval.runner import run_benchmark                  # noqa: E402

_DATASET = _ROOT / "benchmarks" / "retrieval_cases.yaml"

# Baseline = exact pre-Sprint-5 behavior.
_BASELINE = {
    "QUERY_REWRITE_MODE": "off",
    "GROUP_CONTEXT_BY_DOCUMENT": "false",
    "MMR_LAMBDA": "0.5",
    "FETCH_K": "20",
}


def _apply(cfg: dict) -> None:
    """Set env knobs and clear the settings cache so the next call re-reads them."""
    for key, val in cfg.items():
        os.environ[key] = str(val)
    from app.config.settings import get_settings
    get_settings.cache_clear()


def _run(cfg: dict):
    _apply(cfg)
    from app.config.settings import get_settings
    from app.rag.vector_store import get_vector_store
    settings = get_settings()
    benchmark = load_benchmark(_DATASET)
    live = list(get_vector_store().list_sources().keys())
    return run_benchmark(benchmark, live, settings.top_k)


def _subset(report, ctype: str) -> dict:
    """Average key metrics over cases of one type (e.g. cross_document)."""
    rows = [c for c in report.cases if c.type == ctype]
    if not rows:
        return {}
    n = len(rows)
    return {
        "n": n,
        "recall_at_k": round(sum(c.recall_at_k for c in rows) / n, 4),
        "hit_at_1": round(sum(c.hit_at_1 for c in rows) / n, 4),
        "mrr": round(sum(c.reciprocal_rank for c in rows) / n, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Before/after benchmark comparison.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--query-rewrite", default="heuristic", help="improved QUERY_REWRITE_MODE")
    parser.add_argument("--mmr-lambda", default="0.5")
    parser.add_argument("--group-context", default="true")
    args = parser.parse_args()

    improved = {
        "QUERY_REWRITE_MODE": args.query_rewrite,
        "GROUP_CONTEXT_BY_DOCUMENT": args.group_context,
        "MMR_LAMBDA": args.mmr_lambda,
        "FETCH_K": "20",
    }

    base = _run(_BASELINE)
    impr = _run(improved)

    metrics = ["recall_at_k", "precision_at_k", "hit_at_1", "mrr", "ndcg_at_k", "source_accuracy"]
    comparison = {
        m: {"baseline": base.aggregate[m], "improved": impr.aggregate[m],
            "delta": round(impr.aggregate[m] - base.aggregate[m], 4)}
        for m in metrics
    }
    xdoc = {"baseline": _subset(base, "cross_document"), "improved": _subset(impr, "cross_document")}

    if args.json:
        print(json.dumps({
            "improved_config": improved,
            "overall": comparison,
            "cross_document": xdoc,
            "timing_ms": {"baseline": base.timing_ms, "improved": impr.timing_ms},
        }, indent=2))
        return 0

    print("=" * 78)
    print("Sprint 5 — Retrieval Before/After Comparison (read-only)")
    print("=" * 78)
    print(f"Corpus match    : {impr.corpus_match}  | cases: {impr.total_cases} "
          f"({impr.scored_cases} scored, {impr.negative_cases} negative)  top_k={impr.k}")
    print(f"Improved config : {improved}")

    print("\nOverall metrics")
    print("-" * 78)
    print(f"  {'metric':<18}{'baseline':>12}{'improved':>12}{'delta':>10}")
    print("  " + "-" * 50)
    for m in metrics:
        c = comparison[m]
        flag = "" if c["delta"] == 0 else ("  up" if c["delta"] > 0 else "  DOWN")
        print(f"  {m:<18}{c['baseline']:>12.4f}{c['improved']:>12.4f}{c['delta']:>+10.4f}{flag}")

    print("\nCross-document cases only")
    print("-" * 78)
    if xdoc["baseline"]:
        b, i = xdoc["baseline"], xdoc["improved"]
        print(f"  (n={b['n']})        {'baseline':>12}{'improved':>12}{'delta':>10}")
        for m in ("recall_at_k", "hit_at_1", "mrr"):
            print(f"  {m:<18}{b[m]:>12.4f}{i[m]:>12.4f}{i[m] - b[m]:>+10.4f}")

    print("\nPer-case changes (cases whose matched rank or hit@1 changed)")
    print("-" * 78)
    base_by_id = {c.id: c for c in base.cases}
    any_change = False
    for ic in impr.cases:
        bc = base_by_id.get(ic.id)
        if bc and (bc.matched_rank != ic.matched_rank or bc.hit_at_1 != ic.hit_at_1
                   or round(bc.recall_at_k, 3) != round(ic.recall_at_k, 3)):
            any_change = True
            print(f"  {ic.id:<9} {ic.type:<15} rank {bc.matched_rank}->{ic.matched_rank}  "
                  f"hit@1 {bc.hit_at_1:.0f}->{ic.hit_at_1:.0f}  "
                  f"recall {bc.recall_at_k:.2f}->{ic.recall_at_k:.2f}  | {ic.query[:34]}")
    if not any_change:
        print("  (no per-case ranking changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
