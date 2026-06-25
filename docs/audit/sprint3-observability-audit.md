# Sprint 3 — Phase 1: Observability & Diagnostics Audit

**Goal:** Improve observability, diagnostics, and debugging visibility.
**Constraints:** No change to retrieval quality, no new AI features, no chunking
or embedding-model changes, no architecture redesign. **Visibility only.**
**Status:** Audit only — **no code modified.** Awaiting approval for Phase 2.

Maturity scale: **None** · **Partial** (exists but limited/unsafe/ad-hoc) ·
**Good** (fit for purpose).

---

## 1. Existing logging capabilities

**What exists** (`app/utils/logger.py`):
- Central `get_logger(name)` factory; console (stderr) + rotating file handler
  (`app.log`, 5 MB × 3 backups). Format: `ts | LEVEL | name | message`.
- Level from `LOG_LEVEL` setting; `propagate=False`; handlers attached once.
- Modules log meaningful events: ingestion outcomes, upsert counts, retrieval
  counts (`retriever.py` logs `top_k`, `mmr`, `fetched`, `sources`), LLM
  errors, dedupe skips.

**Gaps:**
- **Plain text only — not structured.** No JSON/key-value option, so logs can't
  be filtered/aggregated by field (doc_hash, source, latency) by a log tool.
- **No timing/latency anywhere.** Confirmed: no `time`/`perf_counter`/`elapsed`
  in `app/`. Ingestion and retrieval emit counts but never durations.
- **No correlation id** to tie a query's retrieve → assemble → generate steps
  together across log lines.
- Maturity: **Partial.**

## 2. Existing Chroma inspection tools

**What exists:**
- `inspect_chroma.py` (read-only): collection name, persist dir, total docs,
  total chunks, per-document chunk/page counts, and basic metadata stats
  (min/max/avg chunks per doc). Loads no embedding model — fast.
- `VectorStore.document_chunk_counts()` / `list_sources()` provide the
  underlying aggregation.

**Gaps:**
- Lists the **expected** metadata fields but does **not verify** them — a chunk
  missing `doc_hash` or with a malformed `chunk_id` would not be flagged
  (see §4).
- No per-document detail view (e.g. inspect one source's chunks), no orphan/empty
  detection, no JSON output for tooling.
- Maturity: **Partial → Good** for high-level counts; **None** for integrity.

## 3. Existing retrieval debugging capabilities

**What exists:**
- `scripts/evidence_report.py` (read-only, added Sprint-1 follow-up): per-document
  retrieval probe with scores; good for proving recall.
- `scripts/diagnostic_report.py`: shows top-10 raw retrieval and raw-vs-MMR
  source spread — **but it is destructive** (`store.clear()` + re-ingests
  synthetic PDFs), so it cannot be run against a live store for debugging.
- `retriever.py` logs counts (not scores) per query.

**Gaps:**
- **No ad-hoc, read-only "explain this query" tool**: type any query, see the
  embedded-query path, candidate pool, per-chunk scores, MMR vs raw ordering,
  and which documents won — without mutating the store.
- No timing (embed time vs. search time vs. MMR time).
- Maturity: **Partial** (proof scripts exist; interactive debugging does not).

## 4. Existing metadata validation

**What exists:**
- `inspect_chroma._EXPECTED_METADATA_FIELDS` names the schema, and
  `tests/test_multi_document.py` + `scripts/validate_multidoc.py` assert document
  presence and cross-document retrieval with crafted/real embeddings.
- `Chunk.metadata()` is the single writer of metadata, so the schema is
  consistent by construction at write time.

**Gaps:**
- **No runtime integrity check** over what is *actually stored*: nothing scans
  the live collection for missing/empty fields, malformed `chunk_id`
  (`{hash}::p{n}::c{n}`), `doc_hash` mismatches between id and metadata, bad page
  numbers, or duplicate ids.
- Tests validate the *write path* on fixtures, not the *persisted state* of the
  real DB.
- Maturity: **None** for stored-state validation.

---

## Gap analysis summary

| Area | Today | Maturity | Key missing capability |
|---|---|---|---|
| Logging | Plain text, counts, file+console | Partial | Structured (JSON) option; latency; correlation id |
| Chroma inspection | Read-only counts script | Partial | Per-doc detail; integrity flags; JSON |
| Retrieval debugging | Proof scripts (one destructive) | Partial | Ad-hoc read-only "explain query" with scores + timing |
| Metadata validation | Write-path tests only | None | Runtime scan of stored chunks for schema/format errors |
| Performance metrics | — | **None** | Any timing for ingest/embed/search/generate |

**Cross-cutting:** the project has good *counts* and *proof* tooling but no
*timing*, no *integrity verification of stored state*, no *interactive query
debugging*, and no *machine-readable logs*. None of the gaps require touching
retrieval/chunking/embedding behavior — they are pure read-only/observability
additions.

---

## Proposed Phase 2 scope (for approval — not yet implemented)

All additive, read-only, terminal-runnable; **zero** change to retrieval,
chunking, or embedding behavior:

1. **Enhanced Chroma Inspector** — extend inspection with per-document detail,
   empty/orphan detection, and a `--json` mode. (Builds on `inspect_chroma.py`.)
2. **Retrieval Inspector** — new read-only CLI: run any query against the live
   store, print embedded-query timing, candidate pool, per-chunk scores, and
   raw-vs-MMR ordering. Never mutates the store.
3. **Metadata Health Inspector** — new read-only CLI that scans every stored
   chunk and reports missing/empty fields, malformed `chunk_id`, id↔metadata
   `doc_hash` mismatches, bad page numbers, and duplicate ids; non-zero exit on
   problems (CI-friendly).
4. **Performance Metrics** — a small, reusable timing helper plus opt-in `DEBUG`
   timing logs at the existing seams (embed / search / MMR / generate / ingest
   stages). No behavior change; timings are logged, not acted on.
5. **Structured Logging** — opt-in JSON log format (env-toggled, default stays
   human-readable) and an optional per-query correlation id, so the existing log
   lines become machine-parseable without changing what is logged.

**Phase 3** will validate with ML/OS/DBMS PDFs (counts, retrieval visibility,
metadata integrity, timings). **Phase 4** delivers the final report.

> No code will be changed until this audit is approved.
