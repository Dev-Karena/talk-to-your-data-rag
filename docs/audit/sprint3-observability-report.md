# Sprint 3 — Phase 4: Observability & Diagnostics Report

**Goal:** Improve observability, diagnostics, and debugging visibility.
**Constraints honored:** no retrieval-quality change, no new AI features, no
chunking/embedding changes, no architecture redesign. All additions are
read-only or behavior-preserving (timing/logging are observation, not control).
**Test suite:** `93 passed` (was 68; +25 new). All five tools validated against
the real `talk_to_your_data` collection (ML.pdf / OS.pdf / DBMS.pdf).

---

## 1. Files modified

| File | Change | Behavior impact |
|---|---|---|
| `app/config/settings.py` | Added `log_format` (`text`\|`json`, default `text`) + validator. | None (new opt-in setting). |
| `app/utils/logger.py` | Added `_JsonFormatter`, correlation-id contextvar + `_CorrelationIdFilter`, `set/get/new_correlation_id`; `_build_formatter` honors `LOG_FORMAT`. | None by default — text output unchanged; JSON is opt-in. |
| `app/rag/vector_store.py` | Added read-only `all_records()` (ids + metadata). | None (additive read method). |
| `app/services/retriever.py` | Wrapped embed/search/mmr in a `Stopwatch`; logs timing at **DEBUG**. | **None** — same calls, order, and results; only DEBUG logs added. |
| `app/rag/pipeline.py` | Wrapped load/clean/chunk/embed/store in a `Stopwatch`; logs ingest timing at DEBUG. | None — same pipeline, timing only. |

## 2. New utilities created

| File | Deliverable | What it does |
|---|---|---|
| `app/utils/timing.py` | Performance Metrics (core) | `Stopwatch` (named stage timings, total, safe-on-exception) + `timed()` context manager. Reusable, dependency-free. |
| `app/utils/metadata_health.py` | Metadata integrity (core) | Pure `check_records()` → `HealthReport`; detects missing/empty fields, malformed `chunk_id`, id↔`doc_hash` mismatch, bad page/index, duplicate ids. Unit-tested. |
| `inspect_chroma.py` (enhanced) | **Enhanced Chroma Inspector** | Counts + per-doc chunk/page stats + integrity one-liner; `--doc SOURCE` per-chunk detail; `--json`. Read-only. |
| `scripts/retrieval_inspector.py` | **Retrieval Inspector** | Any query → embed/search/mmr timing, candidate pool, RAW vs MMR top-k, per-chunk scores, final sources. Read-only, never mutates the store. |
| `scripts/metadata_health.py` | **Metadata Health Inspector** | Thin CLI over `check_records()`; human or `--json`; **non-zero exit** on problems (CI-friendly). |
| `scripts/perf_report.py` | **Performance Metrics** | Per-stage retrieval timing averaged over N runs, with a separate cold warm-up; default probes one per ML/OS/DBMS. Read-only. |
| Tests | — | `test_timing.py`, `test_logger.py`, `test_metadata_health.py`, plus `all_records` cases in `test_vector_store.py`. |

## 3. Example outputs (real ML/OS/DBMS collection)

**Document & chunk counts + integrity** (`python inspect_chroma.py`):
```
Total Documents : 3
Total Chunks    : 9
DBMS.pdf  3 chunks / 3 pages   ML.pdf  3 chunks / 3 pages   OS.pdf  3 chunks / 3 pages
Chunks per document: min=3, max=3, avg=3.0
Integrity: OK - all 9 chunk(s) have valid metadata.        (exit 0)
```

**Metadata integrity** (`python scripts/metadata_health.py`):
```
Records scanned : 9   Distinct sources: 3   Distinct hashes : 3
Clean records   : 9/9
RESULT: PASS - all metadata is valid and consistent.        (exit 0)
```

**Retrieval visibility + timing** (`retrieval_inspector.py "What is an operating system?"`):
```
Timing : embed=270.14ms, search=41.49ms, mmr=7.3ms  (total=318.93ms)   # cold (1st query)
RAW top-k sources : ['OS.pdf', 'OS.pdf', 'OS.pdf', 'DBMS.pdf']
MMR top-k sources : ['OS.pdf', 'OS.pdf', 'ML.pdf', 'DBMS.pdf']         # diversified
top hit OS.pdf p1 score=0.8948
```

**Timing metrics, steady state** (`scripts/perf_report.py --runs 5`):
```
query                                   embed  search   mmr   total
What is machine learning?               23.18    5.85  9.44   38.48
What is an operating system?            23.20    6.13  9.84   39.17
What is a database management system?   22.77    6.48  9.27   38.52
Overall avg total per query: 38.72 ms   (cold warm-up shown separately)
```

**Structured logging** (`LOG_FORMAT=json`, one correlation id per operation):
```json
{"time":"2026-06-25 16:43:28","level":"INFO","logger":"app.demo","message":"Retrieved 4 chunk(s) (top_k=4, mmr=on, fetched=9, sources=3).","correlation_id":"query-0952707b"}
{"time":"2026-06-25 16:43:28","level":"INFO","logger":"app.demo","message":"Generated answer (812 chars).","correlation_id":"query-0952707b"}
```

## 4. Validation summary (Phase 3)

| Demonstrated | Tool | Result |
|---|---|---|
| Document counts | `inspect_chroma.py` | 3 documents ✔ |
| Chunk counts | `inspect_chroma.py` (+ `--doc`) | 9 chunks (3×3), per-doc detail ✔ |
| Retrieval visibility | `retrieval_inspector.py` | scores + RAW vs MMR + timing ✔ |
| Metadata integrity | `metadata_health.py` | 9/9 clean, exit 0 ✔ |
| Timing metrics | `perf_report.py` | ~38.7ms/query steady state ✔ |

Observed: retrieval is **embed-dominated** (~23ms of ~39ms steady state); the
first query pays a one-time model-load (~270ms embed) that warms to ~23ms — a
useful, previously invisible fact.

## 5. Remaining observability gaps

1. **No end-to-end query trace through generation.** Timing/correlation cover
   retrieval; the LLM `generate` call is not yet timed or correlation-wrapped in
   the live service path (the UI does not call `new_correlation_id()` per turn).
2. **Metrics are point-in-time, not persisted.** No counters/histograms over
   time (e.g. p50/p95 latency, query volume); each run is a snapshot.
3. **JSON logging is opt-in and not wired in the UI** — no log shipping or
   rotation policy for structured logs; correlation id must be set by callers.
4. **No ingestion-side perf CLI.** `perf_report.py` covers retrieval; ingest
   timing is logged at DEBUG but has no dedicated report.
5. **Inspectors require the embedding model** for retrieval/perf tools (slow cold
   start); only the Chroma/metadata inspectors are model-free.

## 6. Recommendations for Sprint 4

1. **Correlation id per UI turn + time `generate`:** call `new_correlation_id()`
   in `rag_service`/UI and add a `Stopwatch` stage around generation, so one
   query's retrieve→assemble→generate is traceable end-to-end in JSON logs.
2. **Latency aggregation:** persist per-stage timings (append-only JSONL) and add
   a small `perf_summary` that reports p50/p95 — moves from snapshots to trends.
3. **Wire structured logging in the app:** default the UI to JSON in
   production, document a log-shipping target, and set the correlation id at the
   request boundary.
4. **Ingestion perf report** mirroring `perf_report.py` (load/clean/chunk/embed/
   store per document).
5. **`make diagnostics` / unified entrypoint** that runs inspect + metadata-health
   + a sample retrieval as one read-only health check (CI gate).
6. **Surface a read-only "diagnostics" panel in the UI** (counts + integrity) so
   non-terminal users get the same visibility.

> All Sprint 3 changes are additive and behavior-preserving. Retrieval quality,
> chunking, and embeddings are untouched; the 25 new tests plus the existing 68
> all pass.
