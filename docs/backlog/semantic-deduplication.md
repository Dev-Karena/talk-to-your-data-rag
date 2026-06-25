# Backlog: Semantic Duplicate Detection

**Status:** Proposed (not scheduled)
**Created:** 2026-06-25
**Type:** Future enhancement
**Explicitly out of scope for:** Sprint 1

---

## Sprint 1 scope (for context — this item is NOT part of it)

Sprint 1 is deliberately limited to the following. This backlog item must **not**
be pulled into Sprint 1, to avoid scope creep.

1. Multi-document ingestion correctness.
2. Multi-document retrieval correctness.
3. Metadata validation.
4. Chroma inspection.
5. Retrieval validation.

---

## Summary

The system already performs **byte-exact** duplicate detection: an uploaded PDF is
hashed with SHA-256 over its raw bytes (`compute_content_hash`,
`app/utils/validators.py`), checked against the vector store before any work
(`VectorStore.document_exists`, `app/rag/vector_store.py`), and skipped if already
present. Chunk IDs are deterministic (`{doc_hash}::p{page}::c{index}`) and writes
use `upsert`, so re-indexing is idempotent.

**Gap:** dedup is byte-exact only. A document that is re-saved, re-exported,
re-OCR'd, watermarked, or changed by a single byte produces a different hash and is
re-indexed as a brand-new document — even though its *content* is identical or
near-identical. **Semantic deduplication** would detect duplicates by extracted/
cleaned text content (or embeddings) rather than raw bytes.

## Proposed approaches (for future design — not decided here)

1. **Text-content hash (low complexity).** Hash the cleaned, normalized extracted
   text instead of (or in addition to) the raw bytes. Catches re-saves and
   metadata-only changes where the extracted text is identical. Does **not** catch
   minor wording differences.
2. **Near-duplicate via embeddings (higher complexity).** Compare a
   document-level embedding (or chunk-overlap signature, e.g. MinHash/SimHash) and
   flag documents above a similarity threshold as probable duplicates. Catches
   paraphrased / lightly-edited versions but requires a threshold policy and a
   human-resolvable "is this really a duplicate?" decision.

## Estimates

### Complexity

| Approach | Complexity | Notes |
|---|---|---|
| Text-content hash | **Low** (~1–2 days) | Reuses existing hashing + `document_exists` patterns; the extracted text already flows through the pipeline. Mainly: hash cleaned text, store as `content_hash` metadata, add a second existence check. |
| Embedding / near-dup | **Medium–High** (~1–2 weeks) | New similarity-threshold logic, a tuning/evaluation step, decision policy for borderline matches, and UI to surface "possible duplicate" rather than silently skipping. Risk of false positives makes silent skipping unsafe. |

### Benefits

- Avoids redundant storage and embedding cost for documents that are
  semantically identical but not byte-identical (common: "Save As" from a viewer,
  re-exported PDFs, OCR re-runs, downloaded-again copies).
- Cleaner retrieval results — fewer near-identical chunks competing in the top-k,
  which improves answer quality and reduces repetition.
- Better corpus hygiene / accurate document counts in Chroma inspection.

### Risks

- **False positives.** Near-duplicate detection can wrongly merge two genuinely
  different documents (e.g. v1 vs v2 of a contract). Silent skipping would then
  *lose data*. Mitigation: surface as a warning + require confirmation rather than
  auto-skip.
- **Threshold tuning.** Embedding-based similarity needs a defensible threshold;
  too tight misses dupes, too loose drops real documents. Requires an eval set.
- **Performance.** Naive all-pairs comparison scales O(n²); needs an index/LSH
  approach as the corpus grows.
- **Scope creep into retrieval logic.** Document-level embeddings overlap with
  existing chunk-embedding code — risk of entangling dedup with retrieval if not
  cleanly separated.

## Recommended sprint placement

- **Sprint 2 — text-content hash variant only.** Low complexity, high signal,
  builds directly on Sprint 1's metadata-validation and Chroma-inspection work
  (it adds one more metadata field and one more existence check). Low risk because
  an exact text-hash match is a safe auto-skip.
- **Sprint 3+ (or later) — embedding/near-duplicate variant.** Defer until there
  is a measured need and an evaluation set. Its false-positive risk and threshold
  tuning warrant dedicated design and a human-in-the-loop UX, so it should not
  ride along with the cheaper text-hash work.

**Do not implement now.** This document records the enhancement only.
