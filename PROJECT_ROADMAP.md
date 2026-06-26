# Project Roadmap — Talk To Your Data (RAG)

> Living roadmap. Tracks what shipped, what's in flight, and what's next.
> Companion to `SESSION_SUMMARY.md` (detailed handoff state). Last updated for
> **Sprint 6** (branch `sprint6-reranker`; Sprint 5.x merged at `d2beb03`).

A local, citation-grounded RAG app over user PDFs: Streamlit → `rag_service` →
retriever + context builder + Groq generation; ingestion pipeline → Chroma. Fully
offline except answer generation. Embeddings: local BGE. Vector DB: ChromaDB.

---

## Status at a glance

| Sprint | Theme | Status | Commit / branch |
|---|---|---|---|
| 1 | Multi-document correctness | ✅ Done | `0b58c19` |
| 2 | Reliability & error handling | ✅ Done | `3fbc275` |
| 3 | Observability & diagnostics | ✅ Done | `343a269` |
| 4 | Retrieval evaluation framework | ✅ Done | `7881641` |
| 5 | Cross-document retrieval (decomposition) | ✅ Done | `b7abec5` |
| 5.x | Conjunctive & multi-part decomposition | ✅ Done | `d2beb03` (merged) |
| 6 | Cross-encoder reranking | 🔶 Implemented, **default OFF** (failed keep-rule) | `sprint6-reranker` (unmerged) |
| 6.x | Reranking Strategy B (rerank → MMR) | ⬜ Proposed | — |
| 7 | Lexical recall (BM25) + hybrid search | ⬜ Planned | — |
| 8 | Answer-quality evaluation | ⬜ Planned | — |

Legend: ✅ merged to `main` · 🔶 in flight / unmerged · ⬜ not started.

---

## Completed

### Sprint 1 — Multi-document correctness
Per-document `doc_hash`, stable chunk ids, `upsert` idempotency, MMR re-ranking
for cross-document spread, first diagnostics. *Lesson: naive top-k collapses onto
one document; MMR is required.*

### Sprint 2 — Reliability & error handling
Service-layer error boundary, UI store-guard, persist-after-parse ingestion,
specific error messages, aligned upload limits. *No new features — robustness only.*

### Sprint 3 — Observability & diagnostics
Opt-in JSON logging + correlation ids, `Stopwatch` stage timing, metadata-health
checker, enhanced Chroma inspector. *Default output unchanged.*

### Sprint 4 — Retrieval evaluation framework
`app/eval/` (metrics/dataset/runner), 25-case labeled dataset, `doc_hash` ground
truth, isolated benchmark corpus, before/after harness. *Lesson: nDCG IDCG must be
gains sorted desc; ground truth must key on `doc_hash`, not display name.*

### Sprint 5 — Cross-document retrieval
Heuristic **comparative** query decomposition + document-grouped context.
Cross-doc Recall@4 **0.833 → 1.000**, overall Recall@4 **→ 1.000**, zero
regression. Adaptive fetch_k and per-doc MMR cap were tried and **removed** (hurt
precision). LLM rewrite path wired but disabled.

### Sprint 5.x — Conjunctive & multi-part decomposition (merged `d2beb03`)
Rewriter extended from 2 to 4 classes (`single | comparative | conjunctive |
multi_part`) via `classify()`, with an `_is_topic` guard so clausal "and" doesn't
over-split. Benchmark grew 25 → 31 cases; Recall@4 → 1.000, cross-doc → 1.000,
Precision@4 −0.0086 (within HNSW noise). *Lesson: on this corpus MMR already
covers the new cross-doc cases — the new classes are correctness insurance for
harder corpora.*

---

## In flight

### Sprint 6 — Cross-encoder reranking 🔶
**Branch `sprint6-reranker` — implemented, tested (133 passing), default OFF, not
merged.**

Optional cross-encoder stage after MMR: MMR widens to `RERANKER_TOP_N`, the
cross-encoder re-scores, then truncate to `top_k`. New `app/services/reranker.py`
(lazy/cached `CrossEncoder`, device auto-detect `auto|cpu|cuda`, fail-open,
reorder-only). Knobs `RERANKER_ENABLED|MODEL|DEVICE|TOP_N`. New
`scripts/eval_reranker.py` (3-way Baseline/MMR/MMR+Reranker + verdict).
`RERANKER_ENABLED=false` → retrieval byte-identical.

**Benchmark (31 cases, MMR vs MMR+Reranker, ms-marco-MiniLM-L-6-v2):** Hit@1
0.9655→**1.000**, MRR 0.9770→**1.000**, Source Acc 0.9655→**1.000**, Precision@4
+0.069 — **but cross-doc Recall@4 1.000→0.9286**. Warm CPU latency median 36.4 ms
(<150 ms target met).

**Verdict: REMOVE/DISABLE.** The Sprint-6 hard rule requires cross-doc Recall@4 ==
1.000 exactly; reranking the diverse top-N to top-k by pure relevance collapses
one cross-doc case. Deterministic, not HNSW noise.

**Decision pending:** (1) merge default-OFF as opt-in, (2) remove entirely
("no dead config"), or (3) **Sprint 6.x Strategy B** (recommended). See
`docs/audit/sprint6-reranker-report.md`.

---

## Planned

### Sprint 6.x — Reranking Strategy B (rerank → MMR) ⬜
**Why:** Strategy A (rerank after MMR) wins on Hit@1/MRR but loses cross-doc
recall. **Plan:** rerank the candidate *pool* first, then let MMR do the final
diversification to `top_k` — should keep the ranking gain while preserving
cross-doc Recall@4 = 1.000. Config-gated + benchmarked; keep only if it satisfies
the Sprint-6 hard rule.

### Sprint 7 — Lexical recall (BM25) + hybrid search ⬜
**Why:** dense retrieval saturates recall but misses exact terms/identifiers
(`db-03` "B-tree index" lands at rank 2–3). **Plan:** add a BM25 lexical scorer,
fuse with dense via reciprocal-rank fusion, config-gated + benchmarked exactly
like Sprint 5 (remove if it doesn't beat current numbers). **Prereq:** corpus
hygiene — delete the 3 synthetic hash-prefixed PDFs and index the real textbooks
into production `chroma_db/` so production matches the benchmark corpus.

### Sprint 8 — Answer-quality evaluation ⬜
Retrieval metrics don't capture the document-grouped-context improvement. Add a
faithfulness / answer-relevance eval so generation/context changes are measurable.

---

## Cross-cutting backlog

- **No abstention threshold** — `retrieve()` always returns top-k; out-of-corpus
  queries still return chunks. A score threshold is a deliberate behavior change
  (deferred).
- **HNSW nondeterminism** — precision/nDCG wobble ±0.01–0.03 across runs; pin ANN
  params or report confidence intervals for tighter before/after deltas.
- **Tech debt** — pin `PyYAML` in `requirements.txt`; add new knobs to
  `.env.example` (`LOG_FORMAT`, `QUERY_REWRITE_MODE`, `GROUP_CONTEXT_BY_DOCUMENT`);
  automated Streamlit test for the UI store-guard.
- **Larger/harder benchmark** — 31 cases / 3 topic-disjoint docs saturate
  document-level metrics; a bigger corpus keeps them discriminating.
- **Semantic deduplication** — see `docs/backlog/semantic-deduplication.md`.
- **LLM query rewriting** — `QUERY_REWRITE_MODE=llm` is wired but disabled; enable
  for vague/paraphrased queries the heuristic can't split.

---

## Operating principles (carried across sprints)

1. **Metric-driven keep/remove** — any lever that lowers metrics is deleted, not
   hidden behind a default-off flag.
2. **Config-gated + benchmarked** — every retrieval change has a knob and a
   before/after run; `QUERY_REWRITE_MODE=off` reproduces the pre-S5 baseline.
3. **Backward-compatible by default** — simple-query retrieval stays byte-identical;
   observability is opt-in.
4. **`doc_hash` is ground truth** — display names collide; never match on `source`.
5. **Read-only evaluation** — benchmarks run in the isolated `benchmark_chroma/`,
   never production `chroma_db/`.
