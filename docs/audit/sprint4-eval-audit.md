# Sprint 4 — Phase 1: Retrieval Evaluation Framework Audit

**Goal:** Build a retrieval evaluation & benchmarking framework.
**Status:** Audit only — **no code modified.** Awaiting approval for Phase 2.
**Constraint carried from prior sprints:** evaluation must be read-only and must
not change retrieval behavior — it measures `retrieve()`, it does not alter it.

---

## What exists today (review)

### 1. Metadata schema (`app/rag/chunker.py` → `Chunk.metadata()`)
Every stored chunk carries: `chunk_id` (`{doc_hash}::p{page}::c{index}`),
`source` (display filename), `page_number` (1-based), `doc_hash` (SHA-256),
`chunk_index` (0-based). This is **sufficient ground-truth granularity** for
evaluation at three levels: document (`source`/`doc_hash`), page (`page_number`),
and exact chunk (`chunk_id`).

### 2. Retrieval pipeline (`app/services/retriever.py`)
`retrieve(question, top_k)` → embed query → `query_candidates(fetch_k)` → MMR
re-rank to `top_k` (MMR on by default; raw top-k when off). Returns
`RetrievedChunk` with `source`, `page_number`, `chunk_index`, `chunk_id`,
`doc_hash`, `score`. **Deterministic** for the local BGE backend at this scale,
so runs are repeatable. Important: MMR is a re-ranker, so "retrieval quality"
must be measured **both** MMR-on and MMR-off to separate embedder quality from
the diversifier.

### 3. Retrieval inspector (`scripts/retrieval_inspector.py`, Sprint 3)
Already produces, read-only, exactly the per-query signal an evaluator needs:
candidate pool, RAW vs MMR top-k, per-chunk `source`/score, and stage timings.
The eval runner can **reuse this pattern** (and the `Stopwatch`) but should call
the production `retrieve()` to score real behavior.

### 4. Existing test documents — **⚠ critical finding**
`documents/` contains **two distinct corpora**, and only one is indexed:

| File | Pages | Indexed? | doc_hash |
|---|---|---|---|
| `0fec69462619_ML.pdf` (synthetic) | 3 | ✅ yes (3 chunks) | `0fec6946…` |
| `887eed34d7de_OS.pdf` (synthetic) | 3 | ✅ yes (3 chunks) | `887eed34…` |
| `75d1fbdd4db7_DBMS.pdf` (synthetic) | 3 | ✅ yes (3 chunks) | `75d1fbdd…` |
| `ML.pdf` (real textbook) | 120 | ❌ **not indexed** | — |
| `OS.pdf` (real textbook) | 133 | ❌ **not indexed** | — |
| `DBMS.pdf` (real textbook) | 232 | ❌ **not indexed** | — |

The live store holds **9 chunks of the small synthetic PDFs**, not the
textbooks. Worse, both sets resolve to the same display `source` ("ML.pdf",
etc.) — so a "Re-index from disk" would index **both** under the same `source`
name with different `doc_hash`es, creating source-name collisions. **Benchmark
ground truth must be pinned to one known, indexed corpus** (see Risks).

### 5. Source attribution logic (`app/services/context_builder.py`)
`SourceCitation` is built 1:1 from each retrieved chunk: `index` (the
`[Source N]` marker), `source`, `page_number`, `chunk_index`, `chunk_id`,
`score`. The LLM is pinned to cite `[Source N]`. So **source accuracy = whether
the retrieved chunk's `source`/`doc_hash` matches the expected document** — no
extra plumbing needed; attribution is a direct function of retrieval.

---

## How accuracy can be measured

- **Retrieval accuracy** — given a query with known relevant target(s), score
  the ranked `retrieve()` output: Recall@k (target in top-k?), Precision@k, MRR
  (rank of first hit), Hit@1, nDCG@k if graded. Targets can be at document, page,
  or chunk granularity (the schema supports all three).
- **Source accuracy** — top-1 source-correct rate, and source-precision@k
  (fraction of top-k chunks whose `source`/`doc_hash` ∈ expected). Because
  citations derive directly from retrieval, this equals citation correctness.
- **Automating expected-source benchmarks — yes.** With a fixed indexed corpus
  and deterministic embeddings, a labeled `{query → expected_sources}` dataset
  can be run through `retrieve()` and scored automatically, with a pass/fail
  threshold and non-zero exit (CI-friendly), exactly like `metadata_health.py`.

---

## Deliverable 1 — Proposed evaluation architecture

Read-only, additive; mirrors the Sprint-3 "pure logic + thin CLI + tests" shape:

```
benchmarks/
  retrieval_cases.yaml      # ground-truth dataset (human-authored, version-controlled)
app/eval/                   # pure, unit-testable logic (no I/O)
  dataset.py                # load + validate cases; pin to a corpus fingerprint
  metrics.py                # recall@k, precision@k, MRR, hit@1, nDCG, source-acc
  runner.py                 # for each case: call retrieve() (raw + mmr), score
scripts/run_eval.py         # CLI: human table + --json, thresholds, exit code
tests/test_eval_metrics.py  # metric correctness on synthetic rankings
docs/eval/                  # generated benchmark reports
```

Flow: `run_eval.py` → load cases → for each, `retrieve()` (and a raw variant for
comparison) → `metrics.py` aggregates → report + exit code. The runner records
a **corpus fingerprint** (sorted `doc_hash`es + chunk count) so a report always
states which corpus it scored — directly addressing the corpus-mismatch risk.

## Deliverable 2 — Metrics to collect

| Group | Metric | Meaning |
|---|---|---|
| Retrieval | Recall@k | expected target appears in top-k |
| Retrieval | Precision@k | fraction of top-k that are relevant |
| Retrieval | MRR | 1/rank of first relevant hit |
| Retrieval | Hit@1 | top result is relevant |
| Retrieval | nDCG@k | rank-weighted (only if graded relevance is labeled) |
| Source | Top-1 source accuracy | top chunk's `source` ∈ expected |
| Source | Source precision@k | fraction of top-k from expected source(s) |
| Source | Page accuracy | top chunk's `page_number` ∈ expected (optional) |
| Coverage | Multi-source coverage | for cross-doc cases, all expected sources present |
| Diversity | Distinct sources @k | MMR effect (compare on vs off) |
| Negative | False-positive rate | for "no answer" queries (see Risk 7) |
| Perf | embed/search/mmr ms | reuse `Stopwatch`; report alongside accuracy |

Report at three levels: per-case, per-document, aggregate (with MMR-on vs
MMR-off columns).

## Deliverable 3 — Required files (to be created in Phase 2)

1. `benchmarks/retrieval_cases.yaml` — the labeled dataset.
2. `app/eval/{__init__,dataset,metrics,runner}.py` — pure evaluation logic.
3. `scripts/run_eval.py` — terminal entry point (human + `--json` + thresholds).
4. `tests/test_eval_metrics.py` — metric unit tests on crafted rankings.
5. `docs/eval/` — generated reports (and a short README on authoring cases).

No production files are modified; the framework only *reads* via `retrieve()`
and the store.

## Deliverable 4 — Risks and limitations

1. **Corpus mismatch (highest priority).** Indexed = small synthetic 3-page PDFs
   (9 chunks); the real textbooks on disk are un-indexed. A benchmark is only
   meaningful against a known corpus. **Decision required before Phase 2:** which
   corpus is canonical? Re-indexing the textbooks first is recommended for
   realistic difficulty.
2. **Source-name collisions.** Synthetic and textbook files share display
   `source` names. Ground truth should key on **`doc_hash`** (or unique
   filenames), not the ambiguous `source` string.
3. **Trivial difficulty on the synthetic corpus.** ML/OS/DBMS topics are fully
   disjoint, so source accuracy ≈ 100% trivially and won't discriminate. Real
   textbooks (and within-topic queries) are needed for a discriminating
   benchmark.
4. **Small N.** 3 docs / 9 chunks → metric denominators are tiny and not
   statistically meaningful; needs more queries and a richer corpus.
5. **MMR coupling.** `retrieve()` applies MMR by default; measure MMR-off and
   MMR-on separately or you score the diversifier, not the embedder.
6. **Labeling cost/subjectivity.** Document-level labels are cheap (topic-
   disjoint); page/chunk-level relevance needs manual judgment and is the main
   authoring cost.
7. **No abstention at the retrieval layer.** `retrieve()` always returns top-k
   regardless of score (no similarity threshold); "correctly return nothing" for
   out-of-corpus queries **cannot be measured at retrieval today** without adding
   a score cutoff — which would be a *retrieval-behavior change* (out of scope).
   Negative cases can only be evaluated at the answer layer, or by reporting the
   top score for inspection. Flag, don't fix, this sprint.
8. **Approximate ANN at scale.** Chroma HNSW is exact at 9 chunks but approximate
   on large corpora; recall may vary slightly run-to-run once textbooks are
   indexed — report the corpus size with results.

## Deliverable 5 — Example benchmark format

Primary (human-friendly YAML), keyed on `doc_hash` to avoid name collisions
(Risk 2), with a fingerprint header pinning the corpus (Risk 1):

```yaml
# benchmarks/retrieval_cases.yaml
corpus:
  description: "Synthetic ML/OS/DBMS 3-page primers"
  expected_doc_hashes:        # corpus this dataset is valid against
    - "0fec69462619543278e99149616f30e25348c4a071938630306ac30632d089ae"  # ML.pdf
    - "887eed34d7deb3fe79450ba4f5b143a2d819c72e211b7798dec0b60ef39fd4a6"  # OS.pdf
    - "75d1fbdd4db7382e90bbd142c544529b8795f630a8fa5fe2160b81d096dfa985"  # DBMS.pdf

cases:
  - id: ml-001
    query: "What is supervised learning?"
    type: single                 # single | cross_document | negative
    expected_sources: ["ML.pdf"] # human-readable
    expected_doc_hashes: ["0fec6946...089ae"]
    expected_pages: [2]          # optional, page-level grading
    relevant_chunk_ids:          # optional, strongest ground truth
      - "0fec6946...089ae::p2::c0"

  - id: xdoc-001
    query: "Compare machine learning and databases."
    type: cross_document
    expected_sources: ["ML.pdf", "DBMS.pdf"]   # coverage: both must appear in top-k

  - id: neg-001
    query: "What is the capital of France?"
    type: negative               # not in corpus; evaluated for top-score / abstention
    expected_sources: []
```

Equivalent JSONL (one case per line) is also viable for programmatic generation:

```json
{"id":"ml-001","query":"What is supervised learning?","type":"single","expected_doc_hashes":["0fec6946...089ae"],"expected_pages":[2]}
```

A run produces, per case: matched rank, Recall@k/Hit@1/MRR, source-correct,
top score, and timing — then aggregates with MMR-on vs MMR-off columns and a
corpus fingerprint, exiting non-zero if aggregate Recall@k (or any threshold)
falls below a configured bar.

---

## Recommended decision before Phase 2

**Pick the canonical benchmark corpus** (Risk 1):
- **Option A (recommended):** re-index the real textbooks (ML/OS/DBMS), remove
  the synthetic duplicates, and author ~15–30 labeled cases against them →
  realistic, discriminating benchmark.
- **Option B:** keep the synthetic 3-page corpus → trivial but fast/deterministic
  smoke-test; fine as a CI gate, weak as a quality benchmark.

> No code will be changed until this audit is approved and the corpus decision is
> made.
