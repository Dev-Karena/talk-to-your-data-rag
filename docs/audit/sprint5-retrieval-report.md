# Sprint 5 — Retrieval Improvements Report

> **Superseded in part by Sprint 5.x** — see
> `docs/audit/sprint5x-conjunctive-multipart-report.md`. Sprint 5 shipped
> **comparative-only** decomposition; Sprint 5.x extends it to a 4-class
> classifier (adds conjunctive + multi-part) and grows the benchmark to 31 cases.
> The numbers below are the **25-case Sprint 5** results and remain the record for
> that milestone.

**Goal:** Improve the benchmark numbers, especially cross-document retrieval.
**Method:** every lever was implemented behind config, measured with the Sprint-4
benchmark (before/after on the isolated real-textbook corpus), and **kept only if
it improved metrics without regressing others** — anything that lowered metrics
was *removed*, not flagged.
**Out of scope (Sprint 6/7):** BM25, hybrid search, cross-encoder reranking.
**Tests:** `120 passed`.

---

## Headline result (final config)

| Metric | Baseline | Improved | Δ |
|---|---|---|---|
| Recall@4 | 0.978 | **1.000** | **+0.022** |
| Cross-document Recall@4 | 0.833 | **1.000** | **+0.167** |
| Precision@4 | 0.913 | 0.913 | 0.000 |
| Hit@1 | 0.957 | 0.957 | 0.000 |
| MRR | 0.971 | 0.971 | 0.000 |
| nDCG@4 | 0.976 | 0.976 | 0.000 |
| Source accuracy | 0.957 | 0.957 | 0.000 |

A **pure improvement**: cross-document recall went from 0.833 → 1.000 (case
`xdoc-02` "Compare neural-network training … with query optimization in
databases" went 0.50 → 1.00) with **zero regression** on any other metric.

## What was kept

1. **Heuristic query rewriting** (`QUERY_REWRITE_MODE=off|heuristic|llm`, default
   `heuristic`). Comparative questions ("compare X with Y", "X versus Y") are
   decomposed into sub-queries; their candidate pools are unioned (dedup by
   `chunk_id`, best score) before MMR. Single-intent questions are untouched.
   *Ablation: +0.022 overall recall, +0.167 cross-doc recall, 0 regression.*
   *(Sprint 5.x later adds conjunctive + multi-part classes — see its report.)*
2. **Document-grouped context assembly** (`GROUP_CONTEXT_BY_DOCUMENT`, default
   on). Retrieved chunks are grouped under their source document (best document
   first) with a per-document header. This changes only the LLM context layout,
   not retrieval, so it does not move the retrieval benchmark (it targets answer
   readability/grounding; an answer-quality eval is Sprint 6/7 territory).

## What was implemented, measured, and REMOVED (per the "no flagged regressions" rule)

| Lever | Measured effect | Decision |
|---|---|---|
| **Adaptive fetch_k** (scale pool with top_k × docs) | No recall benefit; precision −0.033, nDCG −0.009 (partly ANN noise). | **Removed** — fixed `FETCH_K` restored. No upside on this corpus; revisit only with a much larger corpus. |
| **Per-document MMR cap** (`max_per_document`) | Recall unchanged vs decomposition; precision **−0.087**. | **Removed** — forces off-topic chunks into single-document answers. |
| **Lower MMR λ (0.3)** | Precision −0.011, nDCG −0.006, no recall gain. | **Rejected** — λ left at 0.5. |

Because decomposition already lifts cross-document recall to 1.000, none of the
diversity levers added value — they only traded precision away. They are gone
from the code, not hidden behind a default-off flag.

## LLM query rewriting (configured, NOT executed)

`QUERY_REWRITE_MODE=llm` is wired through settings and `query_rewriter.py` but
**does not call any LLM** — it logs a warning and falls back to the heuristic
path. This provides the architecture/config seam for a future sprint with no
production use now, and keeps the offline benchmark meaningful.

## Baseline reproducibility

The exact pre-Sprint-5 behavior is recoverable at any time:
```
QUERY_REWRITE_MODE=off  GROUP_CONTEXT_BY_DOCUMENT=false
```
`scripts/eval_compare.py` uses this to produce the before/after numbers above.

## Files

**Modified:** `app/config/settings.py` (2 new knobs + validator),
`app/services/retriever.py` (rewrite + merged candidates; adaptive/cap reverted),
`app/services/context_builder.py` (document grouping).
**New:** `app/rag/query_rewriter.py`, `scripts/eval_compare.py`,
`tests/test_query_rewriter.py`, `tests/test_retriever_diversity.py`.

## How to reproduce

```bash
python scripts/build_benchmark_corpus.py --rebuild   # isolated corpus (1131 chunks)
python scripts/eval_compare.py                       # before/after table
python -m pytest tests/ -q                           # 120 passed
```

## Remaining gaps / Sprint 6–7 candidates

1. **BM25 / hybrid search** — lexical recall for exact terms/identifiers the
   dense embedder misses (explicitly deferred).
2. **Cross-encoder reranking** — precision lift on the final top-k (deferred).
3. **LLM query rewriting** — enable the configured path for vague/paraphrased
   queries the heuristic can't split.
4. **Answer-quality eval** — the context-grouping improvement isn't captured by
   retrieval metrics; a faithfulness/answer-relevance eval would measure it.
5. **ANN determinism** — precision wobbled ±0.01–0.03 across runs (HNSW
   approximation); pin search params or report confidence intervals for tighter
   before/after deltas on larger corpora.
