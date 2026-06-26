# Project Roadmap — Talk To Your Data (RAG)

> Living roadmap. Tracks what shipped, what's in flight, and what's next.
> Companion to `SESSION_SUMMARY.md` (detailed handoff state). Last updated for
> **Sprint 5.x** (branch `sprint5x-conjunctive-multipart`, `97d355f`).

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
| 5.x | Conjunctive & multi-part decomposition | 🔶 Done, **unmerged** | `sprint5x-conjunctive-multipart` `97d355f` |
| 6 | Lexical recall (BM25) + hybrid search | ⬜ Planned | — |
| 7 | Cross-encoder reranking | ⬜ Planned | — |
| 8 | Answer-quality evaluation | ⬜ Planned | — |

Legend: ✅ merged to `main` · 🔶 in flight · ⬜ not started.

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

---

## In flight

### Sprint 5.x — Conjunctive & multi-part decomposition 🔶
**Branch `sprint5x-conjunctive-multipart` (`97d355f`) — committed, not merged.**

Extends the rewriter from 2 classes to 4: `single | comparative | conjunctive |
multi_part`, via a public `classify()` (precedence `multi_part > comparative >
conjunctive > single`). Conjunctive splitting is gated by an `_is_topic` guard so
clausal "and" doesn't over-split single-doc questions. `retriever.py` unchanged;
adaptive fetch_k stays removed. Benchmark grew 25 → 31 cases.

**Benchmark (31 cases, off vs heuristic):** Recall@4 0.983 → **1.000**, cross-doc
Recall@4 (n=7) 0.929 → **1.000**, Precision@4 **−0.0086** (within HNSW noise),
Hit@1/MRR/source-accuracy unchanged. 125/125 tests pass.

**Honest caveat:** on this corpus MMR already retrieves both docs for the new
cross-doc cases — decomposition's measurable benefit is still concentrated on the
one comparative case. The new classes are **correctness insurance for harder
corpora**, not a present-day metric win.

**Decision pending:** merge to `main` (accept the −0.0086 precision trade) or
revert the added cases. See `docs/audit/sprint5x-conjunctive-multipart-report.md`.

---

## Planned

### Sprint 6 — Lexical recall (BM25) + hybrid search ⬜
**Why:** dense retrieval saturates recall but misses exact terms/identifiers
(`db-03` "B-tree index" lands at rank 2–3). **Plan:** add a BM25 lexical scorer,
fuse with dense via reciprocal-rank fusion, config-gated + benchmarked exactly
like Sprint 5 (remove if it doesn't beat current numbers). **Prereq:** corpus
hygiene — delete the 3 synthetic hash-prefixed PDFs and index the real textbooks
into production `chroma_db/` so production matches the benchmark corpus.

### Sprint 7 — Cross-encoder reranking ⬜
Precision lift on the final top-k by re-scoring candidates with a cross-encoder.
Targets the remaining precision/Hit@1 misses on hard single-doc queries.

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
