# Sprint 5.x — Conjunctive & Multi-part Query Decomposition

> Extends the Sprint 5 heuristic query rewriter (commit `b7abec5`) from a
> two-class (comparative vs. single) splitter to a four-class classifier.
> Design-then-implement; benchmarked before/after; backward-compatible.

## Context / starting point

Sprint 5 already shipped heuristic decomposition, but it only recognized
**comparative** questions (a `compare|versus|difference|…` cue plus a connector).
Plain conjunctions ("virtual memory **and** database normalization") and
multi-question prompts ("What is X? Also, how does Y work?") fell through to the
`single` path and were never decomposed.

## What changed

### `app/rag/query_rewriter.py`
- Added a public `classify(question)` → `single | comparative | conjunctive |
  multi_part` with precedence **multi_part > comparative > conjunctive > single**.
- **Conjunctive**: split on coordinators (`and`, `as well as`, `along with`,
  `plus`) **only when both halves are topic noun phrases**. A `_is_topic` guard
  rejects halves that begin with an interrogative/auxiliary/pronoun word, so
  clausal "and" stays whole. This is what protects existing single-doc cases:
  - `ml-03` "What is overfitting **and how is it** prevented?" → not split
  - `db-04` "What is a SQL join **and what types** exist?" → not split
  - `os-03` "What is a deadlock **and how can it** be avoided?" → not split
- **Multi-part**: split on sentence boundaries (terminal punctuation + capital,
  so `vs.`/`e.g.` don't trigger) and on discourse markers (`also`,
  `additionally`, `separately`, `furthermore`, `moreover`).
- Comparative path unchanged. `off`/`llm` modes and the "first element is always
  the original question" contract are preserved exactly.

### `benchmarks/retrieval_cases.yaml` (25 → 31 cases)
Added 6 cases so the new classes are benchmarkable:
- `conj-01`, `conj-02` — conjunctive, cross-document (coverage)
- `conj-03` — conjunctive, single-doc (over-split regression guard)
- `mp-01`, `mp-02` — multi-part, cross-document (coverage)
- `mp-03` — multi-part, single-doc (over-split regression guard)

### `tests/test_query_rewriter.py`
9 → 14 tests. Added coverage for all four classes, the topic guard (the three
`*-03`-style clausal cases), and multi-part splitting. The old
`test_single_intent_query_not_decomposed` (which asserted conjunctive queries are
*not* split) was replaced — that behavior is the feature being added.

### Explicitly NOT changed
- **Adaptive fetch_k** stays removed (rejected in Sprint 5 for lowering
  precision/nDCG with no recall benefit). `fetch_k = max(FETCH_K, k)`.
- `app/services/retriever.py` — **zero changes**. `_gather_candidates` already
  unions/dedups across an arbitrary number of sub-queries; the new classes just
  produce more of them. Low blast radius.

## Benchmark (before/after, isolated `benchmark_chroma`, 31 cases, top_k=4)

Baseline = `QUERY_REWRITE_MODE=off`; improved = `heuristic`.

| Metric | Baseline | Improved | Δ |
|---|---|---|---|
| Recall@4 | 0.9828 | **1.0000** | +0.0172 |
| Precision@4 | 0.9224 | 0.9138 | **−0.0086** |
| Hit@1 | 0.9655 | 0.9655 | 0 |
| MRR | 0.9770 | 0.9770 | 0 |
| nDCG@4 | 0.9802 | 0.9813 | +0.0011 |
| Source accuracy | 0.9655 | 0.9655 | 0 |
| **Cross-doc Recall@4** (n=7) | 0.9286 | **1.0000** | +0.0714 |

## Honest interpretation

- **The new cross-document conjunctive/multi-part cases (`conj-01/02`,
  `mp-01/02`) already reach Recall@4 = 1.0 in the baseline** — MMR over the single
  query embedding retrieves both documents without decomposition. The only case
  that genuinely needed splitting remains the pre-existing comparative `xdoc-02`
  (recall 0.50 → 1.00). So on *this* corpus the decomposition's measurable benefit
  is still concentrated on comparative questions; the new classes are
  **correctness insurance** for harder corpora, not a present-day metric win.
- **Precision −0.0086** is the documented cost of decomposition: extra sub-queries
  occasionally pull one off-document chunk into a single-doc case's top-4. It is
  within the noted HNSW run-to-run noise band (±0.01–0.03) and Hit@1/MRR/source
  accuracy are unchanged. The single-doc guards (`conj-03`, `mp-03`) showed **no
  rank/hit/recall regression**.

## Backward compatibility

`QUERY_REWRITE_MODE=off` reproduces the exact pre-Sprint-5 baseline. Single-intent
questions (including clausal "and" questions) are returned unchanged, so simple-
query retrieval is byte-identical. 125/125 tests pass.
