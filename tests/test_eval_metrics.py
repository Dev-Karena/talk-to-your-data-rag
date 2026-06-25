"""Unit tests for app.eval.metrics and dataset loading.

Pure and offline: metrics are checked on crafted rankings with known answers,
and the dataset loader is checked on valid and malformed inputs.
"""

from __future__ import annotations

import math

import pytest

from app.eval import metrics
from app.eval.dataset import DatasetError, corpus_fingerprint, load_benchmark


# ---- Ranking metrics ---------------------------------------------------------
def test_recall_single_relevant_found() -> None:
    assert metrics.recall_at_k(["a", "b", "c"], {"b"}, 3) == 1.0


def test_recall_single_relevant_missed_outside_k() -> None:
    assert metrics.recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_recall_multi_relevant_partial_coverage() -> None:
    # 1 of 2 relevant present in top-3 -> 0.5
    assert metrics.recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5


def test_precision_at_k_counts_relevant_over_k() -> None:
    # 2 relevant in top-4 -> 0.5
    assert metrics.precision_at_k(["a", "b", "x", "y"], {"a", "b"}, 4) == 0.5


def test_hit_at_1_true_and_false() -> None:
    assert metrics.hit_at_1(["a", "b"], {"a"}) == 1.0
    assert metrics.hit_at_1(["a", "b"], {"b"}) == 0.0
    assert metrics.hit_at_1([], {"a"}) == 0.0


def test_reciprocal_rank_positions() -> None:
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_and_zero() -> None:
    # Relevant item at rank 1 -> ideal -> 1.0
    assert metrics.ndcg_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1.0)
    # No relevant retrieved -> 0.0
    assert metrics.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_ndcg_discounts_lower_ranks() -> None:
    # One relevant at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2)=1 -> ndcg<1
    val = metrics.ndcg_at_k(["x", "a", "y"], {"a"}, 3)
    assert val == pytest.approx(1.0 / math.log2(3))
    assert 0.0 < val < 1.0


def test_ndcg_with_graded_relevance() -> None:
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    # Perfect order a,b,c should yield ndcg 1.0
    assert metrics.ndcg_at_k(["a", "b", "c"], set(grades), 3, grades=grades) == pytest.approx(1.0)
    # Reversed order should be < 1.0
    assert metrics.ndcg_at_k(["c", "b", "a"], set(grades), 3, grades=grades) < 1.0


def test_empty_relevant_set_is_zero() -> None:
    assert metrics.recall_at_k(["a"], set(), 1) == 0.0
    assert metrics.ndcg_at_k(["a"], set(), 1) == 0.0


# ---- Dataset loading ---------------------------------------------------------
_VALID_YAML = """
corpus:
  description: "test"
  expected_doc_hashes: ["h1", "h2"]
cases:
  - id: c1
    query: "what is x?"
    type: single
    expected_sources: ["A.pdf"]
    expected_doc_hashes: ["h1"]
  - id: c2
    query: "out of scope?"
    type: negative
    expected_doc_hashes: []
"""


def test_load_valid_benchmark(tmp_path) -> None:
    path = tmp_path / "b.yaml"
    path.write_text(_VALID_YAML, encoding="utf-8")
    bench = load_benchmark(path)
    assert len(bench.cases) == 2
    assert bench.cases[0].id == "c1"
    assert bench.cases[1].is_negative
    assert bench.expected_doc_hashes == ["h1", "h2"]


def test_fingerprint_is_order_independent(tmp_path) -> None:
    # Same hashes in any order produce the same fingerprint.
    assert corpus_fingerprint(["h1", "h2"]) == corpus_fingerprint(["h2", "h1"])


def test_benchmark_fingerprint_matches_helper(tmp_path) -> None:
    path = tmp_path / "b.yaml"
    path.write_text(_VALID_YAML, encoding="utf-8")
    bench = load_benchmark(path)
    assert bench.fingerprint() == corpus_fingerprint(["h2", "h1"])


def test_non_negative_case_without_hashes_rejected(tmp_path) -> None:
    bad = """
corpus: {description: "t", expected_doc_hashes: ["h1"]}
cases:
  - id: c1
    query: "q"
    type: single
    expected_doc_hashes: []
"""
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_benchmark(path)


def test_duplicate_case_ids_rejected(tmp_path) -> None:
    dup = """
corpus: {description: "t", expected_doc_hashes: ["h1"]}
cases:
  - {id: c1, query: "q1", type: single, expected_doc_hashes: ["h1"]}
  - {id: c1, query: "q2", type: single, expected_doc_hashes: ["h1"]}
"""
    path = tmp_path / "dup.yaml"
    path.write_text(dup, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_benchmark(path)


def test_missing_file_rejected() -> None:
    with pytest.raises(DatasetError):
        load_benchmark("does/not/exist.yaml")
