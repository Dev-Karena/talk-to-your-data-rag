"""Retrieval evaluation framework (read-only).

A small, self-contained framework for benchmarking retrieval quality against a
labeled dataset of queries with known relevant documents. It measures the
production ``retrieve()`` path without modifying it — no retrieval, chunking, or
embedding behavior is changed.

Modules:
    metrics  -- pure ranking metrics (recall@k, precision@k, hit@1, MRR, nDCG).
    dataset  -- load/validate the benchmark YAML (cases + corpus fingerprint).
    runner   -- run cases through retrieve() and aggregate metrics + timing.
"""
