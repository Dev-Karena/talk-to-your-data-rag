# Sprint 6 / 6.x — Cross-Encoder Re-ranking Report (REMOVED)

**Goal:** improve ranking quality (Hit@1, MRR, Source Accuracy) with an optional
cross-encoder re-ranking stage, while preserving Recall@4 and cross-document
retrieval (cross-doc Recall@4 must stay exactly **1.000**).
**Outcome:** investigated three integration strategies. **None preserved
cross-document Recall@4 = 1.000.** Per the Sprint-6 hard rule and the Sprint-6.x
acceptance criteria, the reranker was **removed entirely** (not merged, not
shipped default-off).
**Tests after removal:** `125 passed` (the pre-reranker baseline).

---

## What was tried

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores
`(query, chunk_text)` pairs. Three ways to combine it with MMR were implemented
behind `RERANKER_STRATEGY` and benchmarked:

- **`post_mmr` (Strategy A):** MMR widens to `top_n`, cross-encoder reorders,
  truncate to `top_k`. Maximizes ranking; reranks by pure relevance last.
- **`pre_mmr` (Strategy B1):** cross-encoder reorders the candidate pool, keep
  `top_n`, then MMR selects `top_k` (MMR last → meant to preserve diversity).
- **`mmr_relevance` (Strategy B2):** cross-encoder scores (min-max normalized)
  replace cosine as the MMR **relevance** term, so reranker accuracy and MMR
  diversity combine in one objective.

## Benchmark (isolated `benchmark_chroma`, 31 cases, top_k=4, MiniLM-L-6-v2)

`python scripts/eval_reranker.py`

| Metric | Baseline | MMR (`main`) | RR:post_mmr | RR:pre_mmr | RR:mmr_relevance |
|---|---|---|---|---|---|
| Recall@4 | 0.9310 | **1.0000** | 0.9828 | 0.9828 | 0.9655 |
| Precision@4 | 0.9569 | 0.9138 | 0.9828 | 0.9655 | 0.9397 |
| Hit@1 | 0.9655 | 0.9655 | **1.0000** | 0.9655 | 0.9310 |
| MRR | 0.9741 | 0.9770 | **1.0000** | 0.9828 | 0.9655 |
| nDCG@4 | 0.9761 | 0.9813 | **0.9935** | 0.9853 | 0.9742 |
| Source Accuracy | 0.9655 | 0.9655 | **1.0000** | 0.9655 | 0.9310 |
| **Cross-doc Recall@4** | 0.7143 | **1.0000** | 0.9286 | 0.9286 | **0.8571** |

**Latency** (warm, CPU, no GPU in this env — torch is `+cpu`): cross-encoder
re-score of ~10 candidates ≈ **36 ms median** (< 150 ms target). Not the
limiting factor — correctness was.

## Verdict — REMOVE (all strategies fail acceptance)

| Strategy | Beats MMR on Hit@1/MRR? | Cross-doc Recall@4 = 1.000? | Acceptable |
|---|---|---|---|
| post_mmr | yes | **no** (0.9286) | ❌ |
| pre_mmr | yes (MRR) | **no** (0.9286) | ❌ |
| mmr_relevance | no | **no** (0.8571) | ❌ |

**Root cause.** A cross-encoder scores each chunk's relevance to the *full*
question independently. For a comparative/cross-document question, the phrasing
usually leans toward one document, so that document's chunks dominate the
cross-encoder scores. Re-ranking by those scores — whether last (`post_mmr`),
as a pre-filter (`pre_mmr`), or fused into MMR's relevance term
(`mmr_relevance`) — out-ranks the *second* document's single relevant chunk and
drops it from the top-4. `mmr_relevance` is worst because the cross-encoder's
peaked scores overpower MMR's diversity penalty more than cosine does. The effect
is deterministic (cross-encoder is deterministic), so it is a real property of the
method on this corpus, not HNSW noise.

**Conclusion.** On a corpus where cross-document recall is a first-class
requirement, a relevance-only cross-encoder re-ranker is the wrong tool: it trades
the Sprint-5 cross-document win for single-document ranking gains. Removed
entirely per the "no dead config / metric-driven keep-remove" rule.

## What this leaves / next ideas (not implemented)

- The remaining single-document ranking miss (`db-03` B-tree, Hit@1=0 under MMR)
  is better addressed by **lexical/exact-term retrieval (BM25 + hybrid fusion)** —
  the next planned sprint — which improves ranking *without* a global relevance
  re-sort that collapses diversity.
- A reranker could be revisited only if made **diversity-aware** (e.g. rerank
  *within* each document, or only among same-document ties), but that is
  speculative and out of scope here.

## Audit trail

The full implementation and three-strategy benchmark are preserved in the
`sprint6-reranker` branch history (commit prior to removal) for reproducibility.
The feature code (`app/services/reranker.py`, the `retriever.py` integration, the
`RERANKER_*` settings, `tests/test_reranker.py`, and `scripts/eval_reranker.py`)
was then removed; this report is the permanent record of the result.
