# Sprint 7 — Hybrid Retrieval Implementation Report

**Goal:** Improve retrieval precision (particularly on exact-term/identifier queries like `db-03` "B-tree index") by combining dense vector retrieval with sparse keyword retrieval (BM25) using Reciprocal Rank Fusion (RRF), without regressing cross-document Recall@4.

**Outcome:** Implemented hybrid retrieval using the `rank-bm25` library. Evaluated both Mode A (`fused` relevance) and Mode B (`cosine` relevance). 

- **Mode A (`fused` relevance)** successfully improves MRR ($0.9770 \rightarrow 0.9830$), nDCG@4 ($0.9813 \rightarrow 0.9850$), and Precision@4 ($0.9138 \rightarrow 0.9400$), with **zero regressions** on Recall@4 or cross-document Recall@4 ($1.0000$).
- **Mode B (`cosine` relevance)** improves MRR ($0.9770 \rightarrow 0.9830$) but regresses nDCG@4 ($0.9813 \rightarrow 0.9770$) and Precision@4 ($0.9138 \rightarrow 0.8970$).
- **Recommendation**: Merge Mode A (`HYBRID_RELEVANCE_MODE=fused` with max-normalization) as it satisfies the acceptance criteria of improving MRR/Hit@1 with no cross-document recall regressions.

---

## Benchmark Results (31 cases, top_k=4)

Evaluated against the isolated `benchmark_corpus` (1,131 chunks).

| Metric | Baseline (Dense Only) | Mode A (`fused` relevance) | Mode B (`cosine` relevance) | Fused Δ | Cosine Δ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall@4** | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| **Precision@4** | 0.9138 | **0.9400** | 0.8970 | **+0.0262** | -0.0168 |
| **Hit@1** | 0.9655 | 0.9655 | 0.9655 | 0.0000 | 0.0000 |
| **MRR** | 0.9770 | **0.9830** | **0.9830** | **+0.0060** | **+0.0060** |
| **nDCG@4** | 0.9813 | **0.9850** | 0.9770 | **+0.0037** | -0.0043 |
| **Source Accuracy** | 0.9655 | 0.9655 | 0.9655 | 0.0000 | 0.0000 |
| **Cross-doc Recall@4** | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Detailed Findings by Case
- **`db-03` ("How does a B-tree index speed up queries?")**:
  - **Baseline**: Retrieved the correct document at **Rank 3** (Hit@1 = 0).
  - **Mode A (`fused`)**: Elevated the correct document to **Rank 2** (Hit@1 = 0, but MRR for this query improved from $0.33 \rightarrow 0.50$, lifting the aggregate MRR to $0.9830$ and nDCG to $0.65$).
  - **Mode B (`cosine`)**: Elevated the correct document to **Rank 2** (Hit@1 = 0, MRR improved to $0.50$, query nDCG improved to $0.73$).
- **Aggregate Impact**:
  - In **Mode A**, exact keyword matches from BM25 are max-normalized and fused into the relevance score. This increases the ranking rank-positions of highly-relevant chunks, lifting overall precision by $+0.026$ and nDCG by $+0.004$.
  - In **Mode B**, BM25 acts only as a candidate pool booster. Once candidates are gathered, they are scored purely on dense cosine similarity. This causes some highly relevant keyword chunks to get pushed down by dense retrieval, resulting in a regression in nDCG and Precision compared to the baseline.

---

## Verdict & Recommendation

- **Verdict**: **Accept Mode A (`fused` mode)**. It is config-gated and defaults to OFF (`HYBRID_ENABLED=false` in `.env`), meeting all criteria.
- **Why it passed**: Mode A meets the acceptance criteria of improving MRR ($0.9770 \rightarrow 0.9830$) without any regressions in cross-document Recall@4 ($1.0000$).
- **Why Mode B is rejected**: Mode B regresses on both precision and nDCG compared to the dense-only baseline.

---

## Rollback & Configuration

The feature is fully config-gated:
```env
HYBRID_ENABLED=false
BM25_TOP_K=20
RRF_K=60
HYBRID_RELEVANCE_MODE=fused
```
To disable, set `HYBRID_ENABLED=false`. This completely gates the retrieval pipeline back to dense-only.
