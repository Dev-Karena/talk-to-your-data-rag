# SESSION SUMMARY — Talk To Your Data (RAG)

> **Handoff document.** Written for a developer (including future-me) with **no
> memory of prior work**. Everything needed to continue without re-reading old
> chats is here. Sprint 5.x is now **merged** to `main` (`d2beb03`). Last updated
> on branch `sprint6-reranker` (Sprint 6 — cross-encoder re-ranking, implemented
> default-OFF, **not merged**; did not meet keep-criteria — see Sprint 6 below).

---

# Project Overview

* **Project name:** Talk To Your Data — a local, citation-grounded RAG app that
  answers questions over user-uploaded PDFs.
* **Current architecture:** Layered Python app. UI (Streamlit) → service facade
  (`rag_service`) → retrieval (`retriever`) + context assembly (`context_builder`)
  + generation (`llm_client`); ingestion pipeline (`pipeline`) → loader → cleaner
  → chunker → embeddings → vector store. Configuration is centralized and typed
  (`app/config/settings.py`, pydantic-settings). Everything is dependency-light
  and runs **fully offline except answer generation** (which calls Groq).
* **LLM provider:** **Groq** via `langchain-groq` (`ChatGroq`). Default model
  `llama-3.3-70b-versatile`. **Only used for answer generation** — retrieval and
  the whole eval pipeline run without it.
* **Embedding model:** **Local** `BAAI/bge-small-en-v1.5` via
  `sentence-transformers` (offline, no API key). BGE query-instruction prefix is
  applied on the query side. Optional API backends (Voyage, OpenAI) are wired but
  off by default.
* **Vector database:** **ChromaDB** (local, persistent, on-disk). Cosine distance
  (`hnsw:space=cosine`). One collection of chunk records.
* **UI framework:** **Streamlit** (`streamlit run main.py`).
* **Current retrieval pipeline (post-Sprint-5.x):**
  `query → classify (single|comparative|conjunctive|multi_part) → heuristic
  rewrite (decompose into sub-queries) → embed each sub-query → per-sub-query
  candidate search (Chroma) → merge/dedup candidate pool → MMR re-rank to top_k →
  group context by document → cite → Groq answer`.

---

# Repository Structure

| Path | Responsibility |
|---|---|
| `main.py` | Entry point; bootstraps Streamlit (`streamlit run main.py` or `python main.py`). |
| `app/config/settings.py` | **Single source of truth** for all settings (typed, validated, `.env`-bound, `lru_cache`d). |
| `app/rag/` | Ingestion + storage. `loader.py` (PDF→pages), `cleaner.py` (text cleanup), `chunker.py` (chunks + metadata), `embeddings.py` (pluggable backends), `vector_store.py` (Chroma wrapper), `pipeline.py` (orchestrates ingest), `query_rewriter.py` (**Sprint 5/5.x** heuristic 4-class classify + decomposition). |
| `app/services/` | Query-time. `retriever.py` (embed + search + MMR + rewrite), `context_builder.py` (numbered cited context), `llm_client.py` (Groq wrapper), `rag_service.py` (facade the UI calls). |
| `app/eval/` | **Sprint 4** evaluation framework. `metrics.py` (pure ranking metrics), `dataset.py` (YAML loader + corpus fingerprint), `runner.py` (scores `retrieve()`). |
| `app/ui/` | `streamlit_app.py` (page, sidebar, chat), `components.py` (render helpers). |
| `app/utils/` | `logger.py` (text/JSON logging + correlation id), `timing.py` (Stopwatch), `metadata_health.py` (integrity checker), `validators.py` (upload validation + content hash). |
| `scripts/` | Terminal diagnostics & eval (see **Files Added**). |
| `tests/` | Pytest suite (**133 passing**). Offline; stubs embeddings/LLM. |
| `benchmarks/retrieval_cases.yaml` | Labeled benchmark dataset (25 cases). |
| `docs/audit/` | Per-sprint audit + report docs (the written trail of decisions). |
| `docs/backlog/`, `docs/evidence/` | Backlog items; generated evidence reports. |
| `inspect_chroma.py` | Root-level enhanced Chroma inspector. |
| `chroma_db/` | **Production** vector store (git-ignored contents). |
| `benchmark_chroma/` | **Isolated** benchmark store (git-ignored; built on demand). |
| `documents/` | Uploaded/source PDFs (git-ignored contents). |

---

# Current Environment

* **Python version:** 3.10.9 (tested 3.10–3.11; Windows).
* **Main dependencies** (`requirements.txt`, pinned):
  `streamlit==1.40.2`, `langchain-groq==0.2.1`, `chromadb==0.5.23`,
  `langchain==0.3.13` + `langchain-community==0.3.13` +
  `langchain-text-splitters==0.3.4`, `pypdf==5.1.0`,
  `sentence-transformers==3.3.1`, `pydantic==2.10.4` +
  `pydantic-settings==2.7.0`, `python-dotenv==1.0.1`, `pytest==8.3.4`.
  `PyYAML` (6.0.3) is present (used by the eval dataset loader) — **not yet pinned
  in requirements.txt** (see Known Issues).
* **Active `.env` settings** (defaults; **no secrets printed**):
  * `GROQ_API_KEY` — **required for answer generation only** (value not shown).
  * `LLM_MODEL=llama-3.3-70b-versatile`, `LLM_MAX_TOKENS=2048`
  * `EMBEDDING_BACKEND=local`, `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`
  * `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`
  * `CHROMA_PERSIST_DIR=./chroma_db`, `CHROMA_COLLECTION_NAME=talk_to_your_data`
  * `DOCUMENTS_DIR=./documents`, `MAX_FILE_SIZE_MB=25`
  * `LOG_LEVEL=INFO`, `LOG_FILE=./app.log`
* **Retrieval-related configuration:**
  * `TOP_K=4`, `USE_MMR=true`, `FETCH_K=20`, `MMR_LAMBDA=0.5`
  * **Sprint 5 knobs (in `settings.py`, may be absent from `.env.example`):**
    `QUERY_REWRITE_MODE=heuristic` (`off|heuristic|llm`),
    `GROUP_CONTEXT_BY_DOCUMENT=true`.
  * **Sprint 3 knob:** `LOG_FORMAT=text` (`text|json`).
  * To reproduce the exact **pre-Sprint-5 baseline**:
    `QUERY_REWRITE_MODE=off GROUP_CONTEXT_BY_DOCUMENT=false`.

---

# Completed Work

> Sprints 2–5 were all completed and merged to `main` in the current session.
> Sprint 1 predates it (commit `0b58c19`). All five are DONE.

## Sprint 1 — Multi-document correctness (pre-session, commit `0b58c19`)
* **Goals:** Index multiple PDFs without one overwriting another; cross-document
  retrieval; metadata validation; Chroma inspection.
* **Implemented:** Per-document `doc_hash`; stable chunk ids `{doc_hash}::p{page}::c{index}`;
  `upsert` idempotency; MMR re-ranking for cross-document spread; first diagnostics
  (`inspect_chroma.py`, `scripts/diagnostic_report.py`, `scripts/validate_multidoc.py`).
* **Important files:** `vector_store.py`, `chunker.py`, `retriever.py`.
* **Lessons:** A naive top-k collapses onto one document; MMR is needed for
  cross-document questions. Per-document metadata is the backbone of everything later.

## Sprint 2 — Reliability & error handling (commit `3fbc275`)
* **Goals:** Robustness only — no new features, no architecture change.
* **Implemented (9 audited failure points):** Service-layer error boundary
  (`rag_service` now catches `VectorStoreError`/`EmbeddingError`, not just
  `LLMError`); UI guard `_get_store_or_error()` so a corrupted Chroma DB shows a
  message instead of crashing; **persist-after-parse** (uploads are parsed from a
  temp file and only written to `documents/` on success — corrupt files leave no
  residue); specific early missing-API-key message; "nothing indexed" vs "not
  found" distinction; large-document warning; aligned Streamlit upload limit
  (`.streamlit/config.toml`).
* **Important files:** `rag_service.py`, `ui/streamlit_app.py`, `pipeline.py`,
  `validators.py`, `tests/test_pipeline_reliability.py`.
* **Lessons:** The only **High**-severity bug was a corrupted-store traceback
  crashing the page; root cause was the UI calling cached singletons unguarded.
  Catching too narrowly (only `LLMError`) hid retrieval-time failures.

## Sprint 3 — Observability & diagnostics (commit `343a269`)
* **Goals:** Visibility only; no behavior change (default logs unchanged).
* **Implemented:** Opt-in **JSON structured logging** + per-operation
  **correlation id** (`LOG_FORMAT=json`); **timing** helper (`app/utils/timing.py`
  `Stopwatch`) with DEBUG stage timings in retriever/pipeline; **metadata health**
  checker (`app/utils/metadata_health.py` + `scripts/metadata_health.py`,
  non-zero exit on problems); **enhanced Chroma inspector** (`--doc`, `--json`,
  integrity line); **retrieval inspector** and **perf report** scripts;
  read-only `VectorStore.all_records()`.
* **Important files:** `logger.py`, `timing.py`, `metadata_health.py`,
  `retriever.py`/`pipeline.py` (timing), `inspect_chroma.py`.
* **Lessons:** App loggers set `propagate=False`, so `caplog`/root handlers don't
  see them (tests must enable propagation). Retrieval is **embed-dominated**
  (~23 ms of ~39 ms steady state; first query pays ~270 ms cold model load).

## Sprint 4 — Retrieval evaluation framework (commit `7881641`)
* **Goals:** Measure retrieval quality; read-only; no production changes.
* **Implemented:** `app/eval/` (metrics, dataset, runner); `benchmarks/retrieval_cases.yaml`
  (25 cases); `scripts/run_eval.py` (human + `--json` + `--min-recall` gate);
  `scripts/build_benchmark_corpus.py` (indexes the **real textbooks** into an
  **isolated** store); `scripts/collision_audit.py`. Ground truth keyed on
  `doc_hash`; **corpus fingerprint** pins dataset↔corpus.
* **Important files:** `app/eval/*`, `scripts/run_eval.py`,
  `scripts/build_benchmark_corpus.py`, `tests/test_eval_metrics.py`.
* **Lessons:** **nDCG bug** — IDCG must be the actual gains sorted descending, not
  relevant-document count (a chunk-level ranking has many chunks per relevant
  doc; the wrong IDCG produced nDCG > 1). **Collision finding:** the small
  synthetic PDFs and the real textbooks share display names (`ML.pdf`, …) but
  differ by content — ground truth must key on `doc_hash`, never `source`.

## Sprint 5 — Retrieval improvements (commit `b7abec5`)
* **Goals:** Improve benchmark numbers, **especially cross-document recall**.
* **Implemented & KEPT:** Heuristic **query decomposition** (split comparative/
  conjunctive questions, union their candidate pools before MMR) — default
  `QUERY_REWRITE_MODE=heuristic`; **document-grouped context assembly**
  (`GROUP_CONTEXT_BY_DOCUMENT=true`). LLM rewrite path is **configured but
  disabled** (warns + falls back to heuristic; never calls an LLM).
* **Implemented, benchmarked, then REMOVED** (lowered metrics, no benefit — per
  the "no flagged regressions" rule): **adaptive fetch_k** and a **per-document
  MMR cap (diversity tuning)**; lower MMR λ=0.3 also rejected.
* **Important files:** `query_rewriter.py`, `retriever.py` (rewrite + merged
  candidates), `context_builder.py` (grouping), `scripts/eval_compare.py`.
* **Lessons:** Decomposition alone closed the cross-document gap (recall 0.833→1.0)
  with **zero regression**; diversity levers only traded precision away once
  decomposition existed. HNSW is **approximate** — precision wobbles ±0.01–0.03
  across process runs, which masquerades as small regressions.

## Sprint 5.x — Conjunctive & multi-part decomposition (branch `sprint5x-conjunctive-multipart`, commit `97d355f`, NOT yet merged)
* **Goal:** Extend Sprint 5's rewriter from 2 classes (comparative/single) to the
  **4-class** taxonomy: `single | comparative | conjunctive | multi_part`.
* **Implemented & KEPT:** public `classify()` (precedence
  `multi_part > comparative > conjunctive > single`); **conjunctive** splitting on
  coordinators (`and`, `as well as`, …) gated by an `_is_topic` guard so clausal
  "and" stays whole; **multi-part** splitting on sentence boundaries + discourse
  markers (`also`, `additionally`, …). 6 new benchmark cases (25→**31**). `off`
  reproduces the pre-Sprint-5 baseline exactly.
* **Deliberately NOT changed:** `retriever.py` (candidate merge/dedup already
  handles N sub-queries — **zero changes**); **adaptive fetch_k stays removed**.
* **Important files:** `query_rewriter.py` (rewritten), `tests/test_query_rewriter.py`
  (9→14 tests), `benchmarks/retrieval_cases.yaml` (+6 cases),
  `docs/audit/sprint5x-conjunctive-multipart-report.md`.
* **Lessons:** On this corpus **MMR alone already retrieves both documents** for
  the new conjunctive/multi-part cross-doc cases — decomposition's measurable
  benefit is still concentrated on the one comparative case (`xdoc-02`). The new
  classes are **correctness insurance for harder corpora**, not a present-day
  metric win. The guard is what prevents single-doc "...and how is it prevented?"
  questions from over-splitting.

## Sprint 6 — Cross-encoder re-ranking (branch `sprint6-reranker`, NOT merged, default OFF)
* **Goal:** lift Hit@1 / MRR / Source Accuracy with an optional cross-encoder
  stage **after** MMR, preserving Recall@4 and cross-document retrieval.
* **Implemented (config-gated, default OFF):** `app/services/reranker.py`
  (lazy/cached `CrossEncoder`, device auto-detect `auto|cpu|cuda`, fail-open,
  reorder-only). `retriever.py` MMR path widens to `RERANKER_TOP_N` then reranks
  → truncates to `top_k`. New knobs `RERANKER_ENABLED|MODEL|DEVICE|TOP_N`. New
  `scripts/eval_reranker.py` (3-way Baseline/MMR/MMR+Reranker + verdict).
* **Benchmark verdict: REMOVE/DISABLE.** Reranker (ms-marco-MiniLM-L-6-v2)
  improved Hit@1 0.9655→**1.000**, MRR 0.9770→**1.000**, Source Acc
  0.9655→**1.000**, Precision@4 +0.069 — **but regressed cross-doc Recall@4
  1.000→0.9286** (one cross-doc case lost its 2nd document). The Sprint-6 hard
  rule requires cross-doc Recall@4 == 1.000 exactly, so it stays **default OFF**.
* **Latency:** warm CPU rerank of 10 candidates **median 36.4 ms** (<150 ms
  target met). No usable GPU here (torch is `+cpu` build; `auto`→cpu).
* **Lessons:** Reranking the MMR top-N to top-k by *pure relevance* discards MMR's
  diversity — exactly the predicted cross-doc risk. Deterministic (not HNSW
  noise). **Mitigation = Strategy B** (rerank pool *then* MMR last) — Phase 3,
  needs approval. See `docs/audit/sprint6-reranker-report.md`.

---

# Current Benchmark Results

Measured by `scripts/run_eval.py` / `scripts/eval_compare.py` against the
**isolated real-textbook benchmark corpus** (`benchmark_chroma/`: ML 379 + OS 289
+ DBMS 463 = **1131 chunks**, `top_k=4`). **NOT** the production `chroma_db`
(which currently holds only the small synthetic corpus). Sprint 5 measured on
**25 cases**; Sprint 5.x added 6 cases → **31 cases** (so its baseline/improved
numbers differ slightly from the 25-case set — they are not directly comparable
to the S5 column).

**Sprint 5 (25 cases, `main` @ `b7abec5`):**

| Metric | Baseline (pre-S5) | `main` (S5) | Meaning |
|---|---|---|---|
| **Recall@4** | 0.978 | **1.000** | Fraction of expected documents found in the top-4. |
| **Precision@4** | 0.913 | 0.913 | Fraction of top-4 chunks belonging to an expected document. |
| **Hit@1** | 0.957 | 0.957 | Top-1 chunk is from an expected document. |
| **MRR** | 0.971 | 0.971 | Mean reciprocal rank of the first relevant chunk. |
| **nDCG@4** | 0.976 | 0.976 | Rank-quality of relevant chunks within the top-4. |
| **Source Accuracy** | 0.957 | 0.957 | Top-1 `doc_hash` matches expected (citation correctness). |
| **Cross-doc Recall@4** | 0.833 | **1.000** | Headline S5 win — comparative questions retrieve **all** expected docs. |

**Sprint 5.x (31 cases, branch `97d355f`; baseline = rewrite off, improved = heuristic):**

| Metric | Baseline | Improved (S5.x) | Δ |
|---|---|---|---|
| **Recall@4** | 0.9828 | **1.0000** | +0.0172 |
| **Precision@4** | 0.9224 | 0.9138 | −0.0086 (within HNSW noise) |
| **Hit@1** | 0.9655 | 0.9655 | 0 |
| **MRR** | 0.9770 | 0.9770 | 0 |
| **nDCG@4** | 0.9802 | 0.9813 | +0.0011 |
| **Source Accuracy** | 0.9655 | 0.9655 | 0 |
| **Cross-doc Recall@4** (n=7) | 0.9286 | **1.0000** | +0.0714 |

**Interpretation:** Retrieval is strong and now **complete on recall** for this
corpus. Remaining headroom is in **precision/Hit@1 on hard single-doc queries**
(e.g. `db-03` "B-tree index" lands its relevant doc at rank 2–3, not 1). Negative
(out-of-corpus) queries score 0.45–0.53 vs 0.68–0.90 in-corpus — separable, but
the retriever has **no abstention threshold** (always returns top-k).

---

# Current Known Issues

1. **Production vs benchmark corpus mismatch.** `chroma_db/` (production) holds the
   **small synthetic** 3-page PDFs (9 chunks). The **real textbooks** live only in
   the isolated `benchmark_chroma/`. Benchmarks do **not** reflect production
   content until the textbooks are indexed into production.
2. **Source-name collisions on disk.** `documents/` contains both synthetic
   (`0fec69462619_ML.pdf`) and real (`ML.pdf`) files mapping to the same display
   name with different `doc_hash`. A "Re-index from disk" would index **both**
   under `ML.pdf`. Cleanup recommended (delete synthetic duplicates) — `collision_audit.py`
   reports this and deletes nothing.
3. **No abstention at the retrieval layer.** `retrieve()` always returns top-k
   regardless of score; out-of-corpus questions still return chunks. A score
   threshold would fix it but is a deliberate retrieval-behavior change (deferred).
4. **Hard single-document ranking misses.** `db-03` ("B-tree index") is Hit@1=0
   (relevant doc at rank 2–3). Lexical/exact-term recall (BM25) or a reranker would
   likely fix it (Sprint 6/7).
5. **HNSW approximate-search nondeterminism.** Precision/nDCG fluctuate ±0.01–0.03
   across process runs; tighter before/after deltas need pinned ANN params or CIs.
6. **Context-grouping benefit is unmeasured.** `GROUP_CONTEXT_BY_DOCUMENT` improves
   answer context but is invisible to the retrieval benchmark — needs an
   answer-quality (faithfulness/relevance) eval.
7. **Technical debt:**
   * `PyYAML` used by `app/eval/dataset.py` but **not pinned** in `requirements.txt`.
   * `.env.example` is **missing the new knobs** (`LOG_FORMAT`,
     `QUERY_REWRITE_MODE`, `GROUP_CONTEXT_BY_DOCUMENT`).
   * The UI store-guard (`_get_store_or_error`) is verified by inspection, not an
     automated Streamlit test.
8. **Future risks:** Benchmark is small (25 cases, 3 docs) and topic-disjoint, so
   document-level metrics are easy to saturate; needs a larger/harder corpus to
   stay discriminating. Embedding model load is a ~270 ms cold-start tax per process.

---

# Current Retrieval Architecture

* **Chunking** (`app/rag/chunker.py`): LangChain `RecursiveCharacterTextSplitter`,
  separators paragraph→line→sentence→word→char. `CHUNK_SIZE=1000`,
  `CHUNK_OVERLAP=150`. Each chunk gets metadata at write time (single writer:
  `Chunk.metadata()`).
* **Embeddings** (`app/rag/embeddings.py`): pluggable `BaseEmbedder`; default
  `LocalEmbedder` (sentence-transformers, BGE, normalized vectors, query-instruction
  prefix). Voyage/OpenAI backends available behind config. Cached per process.
* **Chroma** (`app/rag/vector_store.py`): `PersistentClient`; one collection;
  cosine metric; `upsert` keyed on `chunk_id` (idempotent re-index);
  `query`/`query_candidates` (the latter returns vectors for MMR);
  `document_exists` (dedup), `list_sources`, `document_chunk_counts`,
  `all_records` (read-only diagnostics), `clear`.
* **Retrieval** (`app/services/retriever.py`): `retrieve(question, top_k)` →
  rewrite → embed each sub-query → `fetch_k=max(FETCH_K, top_k)` candidates per
  sub-query → **merge/dedup by chunk_id keeping best score** → MMR. Non-MMR path
  is the original single-query top-k.
* **MMR** (`_mmr_select`): greedy `λ·sim(query,chunk) − (1−λ)·max sim(chunk,picked)`;
  λ=0.5. Diversity term spreads picks across documents. (Per-document **cap was
  removed** in Sprint 5 — hurt precision.)
* **Multi-PDF support:** every chunk carries `source` + `doc_hash`; `upsert` keeps
  documents independent; MMR + query decomposition produce cross-document results.
* **Metadata schema** (every stored chunk): `chunk_id` (`{doc_hash}::p{page}::c{index}`),
  `source` (display filename), `page_number` (1-based int), `doc_hash` (64-hex
  SHA-256), `chunk_index` (0-based int). Validated by `metadata_health`.

---

# Sprint 5 Planning (approved direction — now IMPLEMENTED)

> Recorded as approved, then reconciled with the actual outcome.

* **Query Rewriting Strategy:**
  * **Option 3 selected** (heuristic default + LLM optional).
  * **Heuristic decomposition implemented first** — DONE, kept (default).
  * **LLM rewriting optional and disabled** — DONE (`QUERY_REWRITE_MODE=llm` is
    wired but warns + falls back; never calls an LLM).
* **Retrieval Rollout Strategy:**
  * **Config-gated** — DONE (every lever is a setting).
  * **Default ON after validation** — DONE for the winner (decomposition,
    grouping); **losers removed entirely**, not flagged.
  * **Benchmark comparison required** — DONE (`scripts/eval_compare.py`,
    before/after report in `docs/audit/sprint5-retrieval-report.md`).
* **Planned vs delivered:**
  1. **Query decomposition** → delivered & kept (cross-doc recall 0.833→1.0).
  2. **Adaptive fetch_k** → delivered, benchmarked, **removed** (no benefit;
     precision/nDCG down).
  3. **Diversity tuning** (per-doc cap + λ) → delivered, benchmarked, **removed**
     (precision −0.087 for the cap; λ=0.3 no gain).
  4. **Benchmark comparison** → delivered (`eval_compare.py`).

---

# Important Decisions Made

* **Real textbook corpus chosen over synthetic** for benchmarking (Option A) —
  the synthetic 3-page PDFs are topic-disjoint and trivial, so they can't
  discriminate retrieval quality. Real ML/OS/DBMS textbooks (485 pages, 1131
  chunks) give realistic difficulty.
* **`doc_hash` used as evaluation ground truth** — content hashes are stable and
  collision-proof; display `source` names collide between synthetic and real files.
* **Source names used only for human-readable reporting**, never for matching.
* **Benchmark runs in an isolated store** (`benchmark_chroma/`) so evaluation
  never mutates production `chroma_db/` (read-only/no-production-change rule).
* **Multi-document correctness prioritized before advanced RAG** — MMR +
  decomposition first; BM25/hybrid/cross-encoder explicitly deferred to Sprint 6/7.
* **Metric-driven keep/remove** — any lever that lowered metrics was deleted, not
  hidden behind a default-off flag (avoids silent regressions and dead config).
* **Observability is opt-in & backward-compatible** — JSON logging and timing are
  off/at-DEBUG by default; production text output is byte-identical.
* **Persist-after-parse** ingestion — never leave unusable files in `documents/`.

---

# Files Added During Development

**Diagnostics / inspection (read-only):**
* `inspect_chroma.py` — enhanced Chroma inspector: doc/chunk counts, per-doc
  detail (`--doc`), integrity line, `--json`.
* `scripts/metadata_health.py` — scans stored chunks for schema/format problems;
  non-zero exit on failure (CI gate). Logic in `app/utils/metadata_health.py`.
* `scripts/retrieval_inspector.py` — "explain this query": per-chunk scores, RAW
  vs MMR ordering, stage timing.
* `scripts/perf_report.py` — per-stage retrieval timing (embed/search/mmr) averaged.
* `scripts/collision_audit.py` — detects display-name→multiple-`doc_hash` collisions
  (live collection + disk); recommends cleanup, **deletes nothing**.
* `scripts/evidence_report.py` — proof that every indexed PDF is stored & retrievable.

**Evaluation:**
* `app/eval/{metrics,dataset,runner}.py` — pure metrics, YAML loader + corpus
  fingerprint, read-only runner over `retrieve()`.
* `scripts/run_eval.py` — run the benchmark (human/`--json`, `--min-recall` gate).
* `scripts/build_benchmark_corpus.py` — index the real textbooks into the isolated store.
* `scripts/eval_compare.py` — before/after baseline-vs-improved comparison.
* `benchmarks/retrieval_cases.yaml` — **31-case** labeled dataset (Sprint 5.x
  added 3 conjunctive + 3 multi-part cases to the original 25).

**Retrieval / infra:**
* `app/rag/query_rewriter.py` — heuristic 4-class `classify()` + decomposition
  (+ disabled LLM stub).
* `app/utils/timing.py` — `Stopwatch` timing helper.
* `.streamlit/config.toml` — upload-size alignment.

**Docs:** `docs/audit/sprint{2,3,4,5}-*.md`,
`docs/audit/sprint5x-conjunctive-multipart-report.md`, `PROJECT_ROADMAP.md`,
`docs/backlog/semantic-deduplication.md`, `docs/evidence/chroma_evidence_report.md`.
(Pre-existing from Sprint 1: `scripts/diagnostic_report.py`,
`scripts/validate_multidoc.py`.)

---

# Recommended Next Action

**First: decide on Sprint 6 (reranker) at the merge gate.** Branch
`sprint6-reranker` is implemented, tested (133 passing), default OFF. The
benchmark verdict is **REMOVE/DISABLE** (cross-doc Recall@4 1.000→0.9286 fails the
hard rule, despite Hit@1/MRR/Source-Accuracy gains). Three options:
  1. **Merge default-OFF** as opt-in (keeps the gains available; documented caveat).
  2. **Remove entirely** (repo's "no dead config" convention).
  3. **Sprint 6.x — Strategy B** (rerank pool *then* MMR last) to recover cross-doc
     recall while keeping the ranking gain — the recommended path; needs approval.

**Then, the longer-standing next step — Sprint 7: lexical recall (BM25) + hybrid
search** — but first do the **5-minute corpus hygiene** that unblocks everything:

1. Run `python scripts/collision_audit.py`, then **delete the 3 synthetic
   hash-prefixed PDFs** from `documents/` and re-index the **real textbooks** into
   **production** `chroma_db/` (so production matches the benchmark corpus).
2. Re-run `python scripts/run_eval.py` to confirm the production numbers.
3. Then implement **BM25 hybrid retrieval** targeting the known weakness
   (`db-03` exact-term miss): add a lexical scorer, fuse with dense scores
   (e.g. reciprocal-rank fusion), keep it **config-gated + benchmarked** exactly
   like Sprint 5 (remove if it doesn't beat the current numbers).

This is the highest-value next step because dense retrieval already saturates
recall; the remaining misses are **exact-term/precision** problems BM25 addresses.

---

# Git Status Recommendation

* **Working tree:** Sprint 5.x is **merged & pushed** to `main` (`d2beb03`).
  Sprint 6 is on branch `sprint6-reranker` — committed work pending, **not merged**
  (awaiting the keep/remove/Strategy-B decision above). Per instruction, Sprint 6
  was **not auto-committed or merged**.
* **To land Sprint 6 default-OFF (if chosen):**
  ```bash
  git checkout main && git merge --no-ff sprint6-reranker
  git push origin main
  ```
* **Branches:** `sprint3-4-observability-eval` and `sprint5-retrieval-improvements`
  are fully merged into `main` and can be deleted:
  ```bash
  git branch -d sprint3-4-observability-eval sprint5-retrieval-improvements
  git push origin --delete sprint3-4-observability-eval sprint5-retrieval-improvements
  ```
* **Recommended tag** (mark the end-of-Sprint-5 milestone):
  ```bash
  git tag -a v0.5.0 -m "Sprints 1-5: reliability, observability, eval, cross-doc retrieval"
  git push origin v0.5.0
  ```
* **Next commit message (for the corpus-hygiene step):**
  `Re-index production corpus to real textbooks; remove synthetic duplicates`
* **Co-author trailer** used in this repo:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

# Executive Summary

**Where it stands.** Talk To Your Data is a working, local, citation-grounded RAG
app (Streamlit + Groq + local BGE embeddings + ChromaDB). Over Sprints 1–5 it grew
from a basic multi-PDF RAG into a **reliable, observable, and benchmarked** system.
Sprint 2 hardened every failure path (no more crashes on corrupt input/stores);
Sprint 3 added structured logging, timing, and read-only diagnostics; Sprint 4 built
a real retrieval-evaluation harness with `doc_hash` ground truth and an isolated
benchmark corpus; Sprint 5 closed the cross-document gap via heuristic query
decomposition (cross-doc Recall@4 **0.833 → 1.000**, overall Recall@4 **→ 1.000**,
no regressions). Sprint 5.x extended the rewriter to a 4-class classifier
(conjunctive + multi-part) and is **merged** to `main` (`d2beb03`). Sprint 6 added
an optional cross-encoder reranker (branch `sprint6-reranker`, default OFF) that
lifted Hit@1/MRR/Source-Accuracy to 1.000 but **regressed cross-doc Recall@4** —
so it fails the keep-rule and stays disabled pending Strategy B. **133 tests pass.**

**Maturity level.** **Mid/solid.** Production-grade error handling, diagnostics,
and an automated benchmark put it well past prototype. It is **not yet** advanced
RAG: retrieval is pure dense + MMR + heuristic rewrite; there is no lexical/hybrid
retrieval, no reranker, and no answer-quality evaluation.

**Biggest remaining weaknesses.** (1) **Production corpus is stale** — the real
textbooks are only in the isolated benchmark store, and synthetic duplicates cause
source-name collisions on disk. (2) **Precision/exact-term misses** on hard
single-doc queries (e.g. B-tree index) that dense retrieval ranks at #2–3. (3) **No
abstention** for out-of-corpus questions. (4) **No answer-quality eval**, so
context/generation improvements are unmeasured.

**Highest-value next improvement.** After the quick corpus-hygiene fix, add
**BM25 + hybrid (dense+lexical) retrieval** — it directly targets the exact-term
precision misses that dense retrieval cannot fix, and the Sprint-4 benchmark +
Sprint-5 before/after harness are already in place to validate it config-gated and
remove it if it doesn't beat the current numbers. Cross-encoder reranking is the
natural Sprint 7 follow-up.
