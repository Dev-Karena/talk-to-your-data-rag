"""Unit tests for app.rag.query_rewriter (heuristic decomposition).

Pure and offline — no model, no network. Verifies the four query classes
(single / comparative / conjunctive / multi_part) decompose correctly, that
single-intent questions are left untouched, and that the (not-yet-enabled) llm
mode falls back to heuristic without calling an LLM.
"""

from __future__ import annotations

from app.rag.query_rewriter import (
    COMPARATIVE,
    CONJUNCTIVE,
    MULTI_PART,
    SINGLE,
    classify,
    rewrite_query,
)


# --- mode handling ---------------------------------------------------------

def test_off_mode_returns_query_unchanged() -> None:
    assert rewrite_query("What is a kernel?", "off") == ["What is a kernel?"]


def test_first_element_is_always_original() -> None:
    q = "Compare X versus Y"
    assert rewrite_query(q, "heuristic")[0] == q


def test_empty_query() -> None:
    assert rewrite_query("   ", "heuristic") == [""]


def test_llm_mode_falls_back_to_heuristic_without_llm() -> None:
    # llm is configured-only: it must NOT raise and must behave like heuristic
    # (decompose), proving no LLM call is required.
    out = rewrite_query("Compare apples with oranges", "llm")
    assert out[0] == "Compare apples with oranges"
    assert len(out) == 3


# --- single (untouched) ----------------------------------------------------

def test_plain_single_query_not_decomposed() -> None:
    assert classify("What is a kernel?") == SINGLE
    assert rewrite_query("What is a kernel?", "heuristic") == ["What is a kernel?"]


def test_clausal_and_not_split_as_conjunctive() -> None:
    # "X and how is it prevented?" / "X and what types exist?" are single intent:
    # the second half is a clause, not a topic, so the guard keeps them whole.
    # These mirror real benchmark cases (ml-03, db-04, os-03) and must not regress.
    for q in (
        "What is overfitting and how is it prevented?",
        "What is a SQL join and what types exist?",
        "What is a deadlock and how can it be avoided?",
    ):
        assert classify(q) == SINGLE, q
        assert rewrite_query(q, "heuristic") == [q], q


def test_cue_without_usable_split_stays_single() -> None:
    out = rewrite_query("What is the difference?", "heuristic")
    assert out == ["What is the difference?"]


# --- comparative -----------------------------------------------------------

def test_compare_with_decomposes() -> None:
    q = "Compare neural network training with query optimization in databases"
    assert classify(q) == COMPARATIVE
    out = rewrite_query(q, "heuristic")
    assert len(out) == 3
    assert "neural network training" in out[1].lower()
    assert "query optimization in databases" in out[2].lower()


def test_versus_decomposes() -> None:
    out = rewrite_query("Concurrency in operating systems versus databases", "heuristic")
    assert len(out) == 3
    assert "operating systems" in out[1].lower()
    assert "databases" in out[2].lower()


def test_difference_between_and_decomposes() -> None:
    # Comparative cue ("difference") wins over conjunctive even though " and "
    # is present.
    q = "What is the difference between a process and a thread?"
    assert classify(q) == COMPARATIVE
    out = rewrite_query(q, "heuristic")
    assert len(out) == 3
    assert "process" in out[1].lower()
    assert "thread" in out[2].lower()


# --- conjunctive (new) -----------------------------------------------------

def test_conjunctive_two_topics_decomposes() -> None:
    q = "Explain virtual memory and database normalization"
    assert classify(q) == CONJUNCTIVE
    out = rewrite_query(q, "heuristic")
    assert out == [q, "virtual memory", "database normalization"]


def test_conjunctive_as_well_as() -> None:
    q = "Describe paging as well as deadlocks"
    assert classify(q) == CONJUNCTIVE
    out = rewrite_query(q, "heuristic")
    assert len(out) == 3
    assert "paging" in out[1].lower()
    assert "deadlocks" in out[2].lower()


# --- multi-part (new) ------------------------------------------------------

def test_multi_part_two_sentences() -> None:
    q = "What is overfitting? How does gradient descent work?"
    assert classify(q) == MULTI_PART
    out = rewrite_query(q, "heuristic")
    assert out[0] == q
    assert "overfitting" in out[1].lower()
    assert "gradient descent" in out[2].lower()


def test_multi_part_discourse_marker() -> None:
    q = "Explain ACID properties. Also, how do B-trees speed up queries?"
    assert classify(q) == MULTI_PART
    out = rewrite_query(q, "heuristic")
    assert len(out) >= 3
    assert any("acid" in s.lower() for s in out[1:])
    assert any("b-tree" in s.lower() for s in out[1:])
