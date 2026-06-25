"""Query rewriting for retrieval.

Turns a single user question into one or more *sub-queries* that are retrieved
independently and merged. The goal is cross-document recall: a comparative
question like "compare X in databases with Y in operating systems" embeds toward
one topic and can miss the other; decomposing it into per-topic sub-queries and
unioning their candidates surfaces both documents.

Modes (see ``QUERY_REWRITE_MODE``):
    * ``off``        — return the query unchanged (baseline).
    * ``heuristic``  — deterministic decomposition of comparative/conjunctive
                       questions. No model, no network, fully offline.
    * ``llm``        — reserved for an LLM-based rewriter. NOT yet enabled; it is
                       wired for configuration only and currently falls back to
                       the heuristic path with a warning (never calls an LLM).

The returned list always starts with the original question, so callers that take
``[0]`` still get the user's exact query (and an ``off`` result is a 1-element
list identical to the input).

Usage:
    >>> rewrite_query("Compare ML training with database indexing", "heuristic")
    ['Compare ML training with database indexing',
     'ML training', 'database indexing']
"""

from __future__ import annotations

import re
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)

# A question is only decomposed when it shows a comparative/contrastive cue —
# this keeps ordinary single-intent questions untouched.
_CUE = re.compile(
    r"\b(compare|comparison|compared|versus|vs\.?|difference|differences|"
    r"contrast|both)\b",
    re.IGNORECASE,
)

# Connectors to split on, longest/most-specific first so " vs. " wins over " vs ".
_CONNECTORS = (
    " versus ",
    " vs. ",
    " vs ",
    " compared with ",
    " compared to ",
    " with ",
    " and ",
)

# Leading boilerplate stripped from a split half to make a cleaner sub-query.
_LEAD = re.compile(
    r"^(compare|comparison of|contrast|what is the difference between|"
    r"how does|how do|how is|how are|what is|what are|describe|explain)\s+",
    re.IGNORECASE,
)


def _strip_lead(text: str) -> str:
    """Remove a leading question/boilerplate phrase from a split half."""
    return _LEAD.sub("", text).strip(" ?.")


def _heuristic_decompose(question: str) -> List[str]:
    """Decompose a comparative question into sub-queries (original + halves).

    Returns ``[question]`` unchanged when there is no comparative cue or no
    usable split — so single-intent questions are never altered.
    """
    if not _CUE.search(question):
        return [question]

    lowered = question.lower()
    for connector in _CONNECTORS:
        idx = lowered.find(connector)
        if idx == -1:
            continue
        left = _strip_lead(question[:idx])
        right = _strip_lead(question[idx + len(connector):])
        # Require both halves to carry real content; otherwise don't split.
        if len(left) >= 3 and len(right) >= 3:
            logger.debug("Decomposed query into: %r | %r", left, right)
            return [question, left, right]
    return [question]


def rewrite_query(question: str, mode: str) -> List[str]:
    """Return sub-queries for a question according to ``mode``.

    Args:
        question: The user's question.
        mode: ``off`` | ``heuristic`` | ``llm``.

    Returns:
        A list whose first element is always the original question. ``off``
        yields a 1-element list; ``heuristic`` may add decomposed halves; ``llm``
        is not yet enabled and falls back to heuristic.
    """
    question = (question or "").strip()
    if not question:
        return [question]

    if mode == "off":
        return [question]
    if mode == "heuristic":
        return _heuristic_decompose(question)
    if mode == "llm":
        # Configured but intentionally not executed in Sprint 5. Fall back to the
        # heuristic path so retrieval keeps working if someone sets this early.
        logger.warning(
            "QUERY_REWRITE_MODE='llm' is configured but not enabled; "
            "falling back to heuristic decomposition."
        )
        return _heuristic_decompose(question)

    # Unknown mode (settings validation should prevent this) — be safe.
    logger.warning("Unknown query_rewrite_mode '%s'; using query as-is.", mode)
    return [question]
