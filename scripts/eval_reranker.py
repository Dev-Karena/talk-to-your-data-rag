"""Sprint 6 / 6.x retrieval benchmark: Baseline / MMR / reranker strategies.

Read-only. Runs the benchmark against the SAME isolated corpus under several
configurations and prints metric deltas, with the cross-document Recall@4
guardrail required by Sprint 6 (must stay exactly 1.000).

Configs:
    Baseline             — pure dense top-k (USE_MMR off, no rewrite, no reranker).
    MMR                  — current `main` (MMR + heuristic rewrite, no reranker).
    RR:post_mmr          — Strategy A: MMR top-N, then cross-encoder, truncate.
    RR:pre_mmr           — Strategy B1: rerank pool, keep top-N, then MMR.
    RR:mmr_relevance     — Strategy B2: cross-encoder scores as MMR relevance.

Acceptance (Sprint 6.x): keep a reranker strategy only if it beats MMR on Hit@1
or MRR AND holds cross-document Recall@4 == 1.000.

It switches configuration in-process via environment variables + a settings
cache clear, so all runs hit the identical embedder, store, and dataset. The
reranker downloads its model from HuggingFace on first use (needs internet once).

Run:
    python scripts/eval_reranker.py
    python scripts/eval_reranker.py --json
    python scripts/eval_reranker.py --reranker-model BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CHROMA_PERSIST_DIR", str(_ROOT / "benchmark_chroma"))
os.environ.setdefault("CHROMA_COLLECTION_NAME", "benchmark_corpus")
os.environ.setdefault("GROQ_API_KEY", "benchmark")

import sys  # noqa: E402

sys.path.insert(0, str(_ROOT))

from app.eval.dataset import load_benchmark    # noqa: E402
from app.eval.runner import run_benchmark      # noqa: E402

_DATASET = _ROOT / "benchmarks" / "retrieval_cases.yaml"

_METRICS = [
    "recall_at_k", "precision_at_k", "hit_at_1", "mrr", "ndcg_at_k",
    "source_accuracy",
]


def _apply(cfg: dict) -> None:
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


def _xdoc_recall(report) -> float:
    rows = [c for c in report.cases if c.type == "cross_document"]
    if not rows:
        return float("nan")
    return round(sum(c.recall_at_k for c in rows) / len(rows), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 6 three-way benchmark.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--reranker-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-top-n", default="10")
    args = parser.parse_args()

    _RR = {
        "USE_MMR": "true", "QUERY_REWRITE_MODE": "heuristic",
        "GROUP_CONTEXT_BY_DOCUMENT": "true", "RERANKER_ENABLED": "true",
        "RERANKER_MODEL": args.reranker_model,
        "RERANKER_TOP_N": args.reranker_top_n, "RERANKER_DEVICE": "auto",
    }
    strategies = ["post_mmr", "pre_mmr", "mmr_relevance"]
    configs = {
        "Baseline": {
            "USE_MMR": "false", "QUERY_REWRITE_MODE": "off",
            "GROUP_CONTEXT_BY_DOCUMENT": "false", "RERANKER_ENABLED": "false",
        },
        "MMR": {
            "USE_MMR": "true", "QUERY_REWRITE_MODE": "heuristic",
            "GROUP_CONTEXT_BY_DOCUMENT": "true", "RERANKER_ENABLED": "false",
        },
    }
    for s in strategies:
        configs[f"RR:{s}"] = {**_RR, "RERANKER_STRATEGY": s}

    reports = {name: _run(cfg) for name, cfg in configs.items()}
    rows = {
        name: {m: r.aggregate[m] for m in _METRICS}
        for name, r in reports.items()
    }
    for name, r in reports.items():
        rows[name]["xdoc_recall_at_k"] = _xdoc_recall(r)
        t = r.timing_ms
        rows[name]["timing_ms"] = round(
            sum(t.values()) if isinstance(t, dict) else t, 2
        )

    # Acceptance (Sprint 6.x): a strategy is acceptable iff it beats MMR on Hit@1
    # or MRR AND holds cross-doc Recall@4 == 1.000. The winner additionally must
    # outperform the *baseline* on Hit@1 or MRR.
    mmr, base = rows["MMR"], rows["Baseline"]
    verdicts = {}
    for s in strategies:
        rr = rows[f"RR:{s}"]
        gain_vs_mmr = (rr["hit_at_1"] > mmr["hit_at_1"]) or (rr["mrr"] > mmr["mrr"])
        gain_vs_base = (rr["hit_at_1"] > base["hit_at_1"]) or (rr["mrr"] > base["mrr"])
        xdoc_ok = rr["xdoc_recall_at_k"] == 1.0
        verdicts[s] = {
            "acceptable": bool(gain_vs_mmr and gain_vs_base and xdoc_ok),
            "gain_vs_mmr": gain_vs_mmr, "gain_vs_base": gain_vs_base,
            "xdoc_ok": xdoc_ok,
        }
    acceptable = [s for s in strategies if verdicts[s]["acceptable"]]
    # Winner = acceptable strategy with the best (Hit@1, MRR, nDCG).
    winner = max(
        acceptable,
        key=lambda s: (rows[f"RR:{s}"]["hit_at_1"], rows[f"RR:{s}"]["mrr"],
                       rows[f"RR:{s}"]["ndcg_at_k"]),
        default=None,
    )

    if args.json:
        print(json.dumps({"rows": rows, "verdicts": verdicts,
                          "winner": winner}, indent=2))
        return 0

    names = list(rows.keys())
    print("=" * 104)
    print("Sprint 6.x — Reranker Strategy Benchmark (read-only)")
    print("=" * 104)
    print(f"Corpus match: {reports['MMR'].corpus_match} | cases: "
          f"{reports['MMR'].total_cases} ({reports['MMR'].scored_cases} scored, "
          f"{reports['MMR'].negative_cases} negative) | top_k={reports['MMR'].k}")
    print(f"Reranker model: {args.reranker_model} | top_n={args.reranker_top_n}\n")

    metric_labels = _METRICS + ["xdoc_recall_at_k", "timing_ms"]
    print(f"  {'metric':<20}" + "".join(f"{n:>16}" for n in names))
    print("  " + "-" * (20 + 16 * len(names)))
    for m in metric_labels:
        line = f"  {m:<20}"
        for n in names:
            val = rows[n][m]
            line += f"{val:>16.4f}" if m != "timing_ms" else f"{val:>16.2f}"
        print(line)

    print("\nAcceptance per strategy (vs MMR; cross-doc must stay 1.000)")
    print("  " + "-" * 72)
    for s in strategies:
        v = verdicts[s]
        rr = rows[f"RR:{s}"]
        print(f"  RR:{s:<14} acceptable={str(v['acceptable']):<5}  "
              f"Hit@1={rr['hit_at_1']:.4f}  MRR={rr['mrr']:.4f}  "
              f"xdoc={rr['xdoc_recall_at_k']:.4f}  "
              f"(gain_vs_mmr={v['gain_vs_mmr']}, xdoc_ok={v['xdoc_ok']})")

    print("\n" + "=" * 104)
    if winner:
        print(f"WINNER: RR:{winner}  -> KEEP (set RERANKER_ENABLED=true "
              f"RERANKER_STRATEGY={winner})")
    else:
        print("WINNER: none acceptable -> REMOVE the reranker entirely.")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
