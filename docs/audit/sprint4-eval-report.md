# Sprint 4 — Phase 2–4: Retrieval Evaluation Framework Report

**Goal:** Build a retrieval evaluation & benchmarking framework.
**Constraints honored:** evaluation is **read-only**; no retrieval, chunking,
embedding, or production behavior changed. The benchmark runs against an
**isolated** Chroma store (`benchmark_chroma/`), never `chroma_db/`.
**Test suite:** `109 passed` (was 93; +16 eval tests).

---

## 1. Files modified / created

**New — evaluation framework (`app/eval/`):**
- `metrics.py` — pure ranking metrics: recall@k, precision@k, hit@1, MRR, nDCG@k.
- `dataset.py` — YAML loader + validation + corpus fingerprint (keyed on `doc_hash`).
- `runner.py` — scores the production `retrieve()` path; aggregates + timing.

**New — terminal tools (`scripts/`):**
- `collision_audit.py` — read-only source-name collision audit (no deletion).
- `build_benchmark_corpus.py` — indexes the real textbooks into the **isolated** store.
- `run_eval.py` — runs the benchmark; human + `--json`; `--min-recall` exit gate.

**New — data & tests:**
- `benchmarks/retrieval_cases.yaml` — 25 labeled cases.
- `tests/test_eval_metrics.py` — 16 metric/dataset unit tests.

**Modified:** `.gitignore` (ignore generated `benchmark_chroma/`). **No production
application code was modified** — the framework only *reads* via `retrieve()`.

## 2. Collision audit results

| Scope | Result |
|---|---|
| Live collection | Clean — 3 sources, each → a single `doc_hash`. |
| `documents/` on disk | **3 collisions** — `ML.pdf`, `OS.pdf`, `DBMS.pdf` each map to a synthetic 3-page file **and** a real textbook (different content hashes). |

**Recommendation (no data changed):** keep the textbooks, remove the 3 synthetic
hash-prefixed duplicates, rebuild a clean index — in an isolated store, which is
exactly what `build_benchmark_corpus.py` does. Deletion is left to the operator.

## 3. Benchmark dataset

- **Corpus (Option A):** real textbooks, isolated store — 3 docs, **1131 chunks**
  (ML 379 / OS 289 / DBMS 463), 485 pages total.
- **Cases:** 25 — 20 single-source, 3 cross-document, 2 negative.
- **Ground truth:** document-level, keyed on `doc_hash` (collision-proof). No
  page/chunk labels (intentional — document/source accuracy is the reliable signal).
- **Corpus fingerprint:** `5f0b5b73613c832e`; report confirms `corpus_match: True`.

## 4. Example report output

```
Retrieval Benchmark Report (read-only)
Dataset      : benchmarks/retrieval_cases.yaml
Cases        : 25 (23 scored, 2 negative)   top_k: 4
Corpus match : True  (dataset=5f0b5b73613c832e, live=5f0b5b73613c832e)

Aggregate metrics (averaged over scored cases)
  Recall@4        : 0.978
  Precision@4     : 0.913
  Hit@1           : 0.957
  MRR             : 0.971
  nDCG@4          : 0.976
  Source accuracy : 0.957
Timing: avg=600ms  min=59ms  max=13234ms   (max = one-time cold model load)

Negative cases (no abstention threshold; top_score for inspection):
  neg-01  capital of France        top_score=0.4470
  neg-02  cookie recipe            top_score=0.5297
```
JSON output (`--json`) carries the same data plus 16 per-case fields
(`recall_at_k`, `hit_at_1`, `reciprocal_rank`, `ndcg_at_k`, `top_score`,
`matched_rank`, `elapsed_ms`, …) for aggregation/CI.

## 5. Validation results

- **Tests:** `109 passed` (16 new eval tests: metrics on crafted rankings +
  dataset validation incl. malformed/duplicate/missing inputs).
- **Bug found & fixed in validation:** nDCG initially reported **2.207** (>1) —
  IDCG was derived from relevant-*document* count while the ranked list is
  *chunks* (many per document). Fixed to textbook/sklearn semantics
  (IDCG = actual gains sorted descending); now **0.976**, bounded in [0, 1].
- **Real findings the framework surfaced** (not labeling errors):
  - `db-03` ("B-tree index") — Hit@1 **miss**, relevant doc at **rank 3**
    (recall still 1.0). A concrete ranking weakness to investigate.
  - `xdoc-02` — cross-document **recall 0.50** (retrieved only ML, missed DBMS).
  - **Negatives separate cleanly** — out-of-corpus top scores 0.45–0.53 vs
    0.68–0.90 in-corpus, suggesting a score threshold (~0.6) *could* power
    abstention later (would be a retrieval-behavior change — out of scope here).

## 6. Remaining gaps / recommendations (Sprint 5 candidates)

1. **No abstention at retrieval layer** (audit Risk 7 stands): `retrieve()` always
   returns top-k. The negative-case scores show a threshold is feasible; adding
   one is a deliberate retrieval-behavior change for a future sprint.
2. **MMR-on only:** the report scores the production path (MMR on). A RAW-vs-MMR
   A/B column would isolate embedder vs. diversifier quality.
3. **Document-level only:** page/chunk-level ground truth (deliberately omitted)
   would enable finer precision analysis if ever needed.
4. **Answer-quality (RAGAS-style) eval** — faithfulness/answer-relevance — is out
   of scope here (this sprint is retrieval-only) and a natural next layer.

> All Sprint 4 changes are additive and read-only. Retrieval, chunking, and
> embeddings are untouched; the isolated benchmark store leaves production data
> intact.
