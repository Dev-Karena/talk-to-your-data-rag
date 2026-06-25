"""Unit tests for app.rag.query_rewriter (heuristic decomposition).

Pure and offline — no model, no network. Verifies that comparative questions
decompose into sub-queries while single-intent questions are left untouched, and
that the (not-yet-enabled) llm mode falls back to heuristic without calling an LLM.
"""

from __future__ import annotations

from app.rag.query_rewriter import rewrite_query


def test_off_mode_returns_query_unchanged() -> None:
    assert rewrite_query("What is a kernel?", "off") == ["What is a kernel?"]


def test_single_intent_query_not_decomposed() -> None:
    # No comparative cue -> unchanged, even though it contains 'and'.
    out = rewrite_query("What is supervised and reinforcement learning?", "heuristic")
    assert out == ["What is supervised and reinforcement learning?"]


def test_compare_with_decomposes() -> None:
    out = rewrite_query(
        "Compare neural network training with query optimization in databases",
        "heuristic",
    )
    assert out[0].startswith("Compare neural network training")
    assert len(out) == 3
    assert "neural network training" in out[1].lower()
    assert "query optimization in databases" in out[2].lower()


def test_versus_decomposes() -> None:
    out = rewrite_query("Concurrency in operating systems versus databases", "heuristic")
    assert len(out) == 3
    assert "operating systems" in out[1].lower()
    assert "databases" in out[2].lower()


def test_difference_between_and_decomposes() -> None:
    out = rewrite_query("What is the difference between a process and a thread?", "heuristic")
    assert len(out) == 3
    # Leading 'what is the difference between' is stripped from the first half.
    assert out[1].lower().startswith("a process") or "process" in out[1].lower()
    assert "thread" in out[2].lower()


def test_first_element_is_always_original() -> None:
    q = "Compare X versus Y"
    assert rewrite_query(q, "heuristic")[0] == q


def test_empty_query() -> None:
    assert rewrite_query("   ", "heuristic") == [""]


def test_llm_mode_falls_back_to_heuristic_without_llm() -> None:
    # llm is configured-only in Sprint 5: it must NOT raise and must behave like
    # heuristic (decompose), proving no LLM call is required.
    out = rewrite_query("Compare apples with oranges", "llm")
    assert out[0] == "Compare apples with oranges"
    assert len(out) == 3


def test_cue_without_usable_split_stays_single() -> None:
    # Has a cue word but no connector to split on -> unchanged.
    out = rewrite_query("What is the difference?", "heuristic")
    assert out == ["What is the difference?"]
