# Sprint 2 — Phase 1: Failure-Point Audit

**Goal:** Improve error handling and reliability of the existing RAG app.
**Constraints:** No new features. No architecture redesign. No retrieval-logic
changes unless required for reliability. Robustness only.
**Status:** Audit only — **no code modified.** Awaiting approval for Phase 2.

Risk scale: **Low** (cosmetic / already mostly handled) · **Medium** (degraded
UX or unclear error, but contained) · **High** (unhandled crash / data-integrity
or security concern).

---

## Audit results

### 1. Empty PDF uploads
- **Scenario:** User uploads a 0-byte file or an empty `.pdf`.
- **Current behavior:** `validate_pdf` rejects on `if not data` →
  `IngestStatus.REJECTED`, message `"'x.pdf' is empty."` Shown in the UI results.
  (`app/utils/validators.py:77`)
- **Expected behavior:** Reject with a clear message; no crash. ✔ Already met.
- **Risk:** **Low** — handled correctly today.

### 2. Corrupted PDF uploads
- **Scenario:** File begins with `%PDF-` but the body is truncated/garbled.
- **Current behavior:** Passes validation, gets **persisted to `documents/`**,
  then `load_pdf` → `PyPDFLoader.load()` raises → caught as `PDFLoadError` →
  `IngestStatus.FAILED` with the parser message. No crash.
  (`app/rag/loader.py:71`, `app/rag/pipeline.py:164`)
- **Gap:** The corrupt file is written to disk **before** parsing and is **left
  there** on failure. A later "Re-index from disk" will re-attempt and re-fail on
  it every time (orphaned bad file). Parser error text is leaked verbatim to the
  UI.
- **Expected behavior:** Fail gracefully with a friendly message **and** do not
  leave an unusable file that poisons future re-indexing.
- **Risk:** **Medium** — handled at the boundary, but leaves persistent residue.

### 3. Unsupported file types
- **Scenario:** `.txt`/`.docx`, or a non-PDF renamed to `.pdf`.
- **Current behavior:** Streamlit `file_uploader(type=["pdf"])` blocks selection
  in the UI; `validate_pdf` independently checks extension **and** magic bytes, so
  a renamed file is rejected (`"does not have a valid PDF header."`).
  (`app/ui/streamlit_app.py:120`, `app/utils/validators.py:83,100`)
- **Expected behavior:** Reject non-PDFs with a clear message. ✔ Already met
  (defense in depth).
- **Risk:** **Low** — handled correctly today.

### 4. Query before indexing
- **Scenario:** User asks a question with an empty vector store.
- **Current behavior:** `store.query` / `query_candidates` short-circuit on
  `count() == 0` → `[]` → `assembled.is_empty` → returns the standard
  "I could not find this information…" message. No LLM call, no crash.
  (`app/rag/vector_store.py:140,204`, `app/services/rag_service.py:84`)
- **Gap (minor):** The query embedding model still loads/embeds before the empty
  result is known — wasted work, not a failure. Also the user isn't told *why*
  (nothing indexed) vs. *not found in docs* — same message for both.
- **Expected behavior:** Friendly "no documents indexed yet" guidance; no crash.
- **Risk:** **Low** — no crash; message could be more specific.

### 5. Empty queries
- **Scenario:** User submits blank or whitespace-only input.
- **Current behavior:** `answer_question` / `_stream` strip and return
  `"Please enter a question."`; `retrieve` also guards blank input → `[]`.
  (`app/services/rag_service.py:77,128`, `app/services/retriever.py:48`)
- **Expected behavior:** Prompt the user to enter a question; no crash. ✔ Met.
- **Risk:** **Low** — handled correctly today.

### 6. Missing API key
- **Scenario:** `GROQ_API_KEY` unset; user asks a question that *does* retrieve
  context.
- **Current behavior:** `LLMClient.__init__` raises `LLMError`; in
  `answer_question` the `get_llm_client().generate(...)` call is inside
  `try/except LLMError`, so it returns a generic *"Sorry, I couldn't generate an
  answer due to an error."* with the raw error in `error`. The sidebar shows a
  red "GROQ_API_KEY missing" badge up front.
  (`app/services/llm_client.py:73`, `app/services/rag_service.py:92`,
  `app/ui/streamlit_app.py:53`)
- **Gap:** The failure only surfaces **after** retrieval and **only** when context
  is found; the message is generic and doesn't say "set your API key." No
  pre-flight block on submitting questions when the key is absent.
- **Expected behavior:** Clear, specific "API key missing — add GROQ_API_KEY to
  .env" message at question time (not a generic error). No crash. ✔ no crash today.
- **Risk:** **Medium** — handled but the guidance is unclear/late.

### 7. Missing / corrupted Chroma database
- **Scenario A (missing dir):** `chroma_db/` absent.
- **Scenario B (corrupted store):** dir exists but the SQLite/index is unreadable.
- **Current behavior:**
  - A: `ensure_directories` + `get_or_create_collection` recreate it as empty —
    works fine.
  - B: `VectorStore.__init__` raises `VectorStoreError`. **`get_vector_store()` is
    called directly in `_render_sidebar` and `_handle_question` with no
    try/except**, so a corrupted store throws an **unhandled exception that
    crashes the Streamlit page** with a raw traceback.
    (`app/rag/vector_store.py:81`, `app/ui/streamlit_app.py:137,177`)
- **Related gap:** `rag_service` only catches `LLMError`. Retrieval-time
  `VectorStoreError`/`EmbeddingError` (from `_retrieve_and_assemble`, which is
  **outside** the `try`) propagate uncaught to the UI.
  (`app/services/rag_service.py:81,132`)
- **Expected behavior:** Missing → silently initialize empty (already true).
  Corrupted/unreadable → show a friendly error and remain usable; never dump a
  traceback.
- **Risk:** **High** — corrupted store causes an unhandled UI crash.

### 8. Duplicate uploads
- **Scenario:** Same PDF (same bytes, any filename) uploaded again.
- **Current behavior:** SHA-256 content hash → `store.document_exists` →
  `IngestStatus.SKIPPED`; deterministic chunk ids + `upsert` make re-indexing
  idempotent even if the check is bypassed.
  (`app/rag/pipeline.py:114`, `app/utils/validators.py:109`,
  `app/rag/vector_store.py:114,228`)
- **Expected behavior:** Detect and skip; no duplicate records. ✔ Met. (Semantic
  near-duplicates are intentionally **out of scope** — see backlog.)
- **Risk:** **Low** — handled correctly today.

### 9. Large PDF handling
- **Scenario:** File over the size limit, or a very large in-limit file.
- **Current behavior:** `validate_pdf` rejects `len(data) > max_file_size_bytes`
  (default 25 MB) → `REJECTED` with a size message. In-limit-but-large files
  index successfully (per-file progress bar shown).
  (`app/utils/validators.py:89`)
- **Gap:** The whole file is read into memory (`uploaded.getvalue()`) **before**
  the size check, so the cap doesn't bound peak memory; Streamlit's own
  `server.maxUploadSize` (default 200 MB) is a separate, larger limit not aligned
  with ours. No upper bound on page/chunk count → a pathological in-limit PDF can
  embed very slowly with no feedback beyond the file-level bar.
- **Expected behavior:** Reject oversize with a clear message (✔), and degrade
  predictably (no OOM / indefinite hang) on large in-limit files.
- **Risk:** **Medium** — limit enforced, but memory/throughput edges are unguarded.

---

## Cross-cutting findings (root causes behind several rows above)

- **C1 — Service layer catches too narrowly.** `rag_service` wraps only the LLM
  call in `try/except LLMError`; retrieval/embedding/vector-store errors thrown by
  `_retrieve_and_assemble` are uncaught. → drives rows **4, 7**. *(Reliability fix;
  does not alter retrieval *logic*, only its error handling.)*
- **C2 — UI calls cached singletons unguarded.** `get_vector_store()`,
  (and transitively `get_embedder()` / `get_llm_client()`) are invoked in the UI
  with no error boundary, so any init failure becomes a raw traceback. → rows
  **6, 7**.
- **C3 — Persist-before-validate-parse.** Files are written to `documents/` before
  successful parsing, leaving orphaned bad files. → row **2**.

---

## Proposed Phase 2 scope (for approval — not yet implemented)

All items are robustness-only and stay within the stated constraints:

1. **Row 7 / C1 / C2:** Add a narrow error boundary in `rag_service` (catch
   `VectorStoreError`/`EmbeddingError` alongside `LLMError`, return a friendly
   `RAGResponse.error`); wrap `get_vector_store()` use in the UI so a corrupted
   store shows a message instead of crashing.
2. **Row 2 / C3:** Only persist the uploaded file **after** it parses (or clean up
   on failure) so failed/corrupt uploads don't poison `documents/` and re-index.
3. **Row 6:** Make the missing-API-key path return a specific, actionable message
   at question time (and short-circuit before retrieval when generation is known
   to be impossible).
4. **Row 9:** Enforce the size cap defensively and align/socument the Streamlit
   upload limit; log a warning for unusually large page/chunk counts.
5. **Row 4 (minor):** Distinguish "nothing indexed yet" from "not found in
   documents" in the user-facing message.

**Phase 3** will add automated tests for all nine scenarios; **Phase 4** will
produce a validation report proving each is handled.

> No code will be changed until this audit is approved.
