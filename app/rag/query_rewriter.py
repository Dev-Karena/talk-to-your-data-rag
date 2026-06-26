"""Query rewriting for retrieval.

Turns a single user question into one or more *sub-queries* that are retrieved
independently and merged. The goal is cross-document recall: a question that
embeds toward one topic can miss a second topic that lives in another document;
decomposing it into per-topic sub-queries and unioning their candidates surfaces
all the relevant documents.

Query classification (heuristic, deterministic — no model, no network):
    * ``single``       — one intent; retrieved as-is (baseline behavior).
    * ``comparative``  — a comparative/contrastive cue ("compare", "versus",
                         "difference between A and B") joining two topics.
    * ``conjunctive``  — two distinct *topics* coordinated by "and"/"as well as"
                         WITHOUT a comparative cue ("virtual memory and database
                         normalization").
    * ``multi_part``   — several questions in one ("What is X? Also, how does Y
                         work?") split across sentences or discourse markers.

Modes (see ``QUERY_REWRITE_MODE``):
    * ``off``        — return the query unchanged (baseline).
    * ``heuristic``  — classify and decompose as above. Fully offline.
    * ``llm``        — reserved for an LLM-based rewriter. NOT yet enabled; wired
                       for configuration only and currently falls back to the
                       heuristic path with a warning (never calls an LLM).

The returned list always starts with the original question, so callers that take
``[0]`` still get the user's exact query (and an ``off`` result, or any
``single`` query, is a 1-element list identical to the input).

Usage:
    >>> rewrite_query("Compare ML training with database indexing", "heuristic")
    ['Compare ML training with database indexing',
     'ML training', 'database indexing']
    >>> rewrite_query("Explain virtual memory and database normalization",
    ...               "heuristic")
    ['Explain virtual memory and database normalization',
     'virtual memory', 'database normalization']
"""

from __future__ import annotations

import re
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Query types (also used as log/diagnostic labels).
SINGLE = "single"
COMPARATIVE = "comparative"
CONJUNCTIVE = "conjunctive"
MULTI_PART = "multi_part"

# --- Comparative detection (Sprint 5, unchanged) ---------------------------
# A comparative cue marks an explicit compare/contrast intent. Required for the
# comparative class so ordinary single-intent questions stay untouched.
_COMPARATIVE_CUE = re.compile(
    r"\b(compare|comparison|compared|versus|vs\.?|difference|differences|"
    r"contrast|both)\b",
    re.IGNORECASE,
)

# Connectors a comparative question is split on, longest/most-specific first so
# " vs. " wins over " vs ".
_COMPARATIVE_CONNECTORS = (
    " versus ",
    " vs. ",
    " vs ",
    " compared with ",
    " compared to ",
    " with ",
    " and ",
)

# --- Conjunctive detection (Sprint 5.x) ------------------------------------
# Coordinators that may join two independent topics. No comparative cue is
# required; instead each half must look like a *topic* (see ``_is_topic``), which
# is what prevents clausal "and" ("X and how is it prevented?") from splitting.
_CONJUNCTIVE_CONNECTORS = (
    " as well as ",
    " along with ",
    " together with ",
    " and also ",
    " and ",
    " plus ",
)

# --- Multi-part detection (Sprint 5.x) -------------------------------------
# Sentence boundary: terminal punctuation followed by whitespace and a capital
# letter. Requiring a capital avoids splitting abbreviations like "vs." or
# "e.g." whose following token is lowercase.
_SENTENCE_SPLIT = re.compile(r"(?<=[?.!;])\s+(?=[A-Z])")

# Discourse markers that introduce an additional question within one sentence.
_DISCOURSE_SPLIT = re.compile(
    r"[,;]?\s+(?:also|additionally|separately|furthermore|moreover)\b[,:]?\s+",
    re.IGNORECASE,
)

# Leading boilerplate stripped from a split half to make a cleaner sub-query.
_LEAD = re.compile(
    r"^(compare|comparison of|contrast|what is the difference between|"
    r"how does|how do|how is|how are|what is|what are|describe|explain|"
    r"tell me about|list)\s+",
    re.IGNORECASE,
)

# A half that begins with one of these is a clause/fragment, not a topic noun
# phrase — so it must NOT be treated as a conjunctive sub-topic. This guards
# single-intent questions like "What is a SQL join and what types exist?" and
# "What is overfitting and how is it prevented?".
_NON_TOPIC_LEAD = re.compile(
    r"^(how|what|why|when|where|which|who|whom|whose|is|are|was|were|do|does|"
    r"did|can|could|should|would|will|has|have|had|"
    r"it|they|them|this|that|these|those|he|she|its|their)\b",
    re.IGNORECASE,
)


def _strip_lead(text: str) -> str:
    """Remove a leading question/boilerplate phrase from a split half."""
    return _LEAD.sub("", text).strip(" ?.!,;")


def _is_topic(text: str) -> bool:
    """True if ``text`` reads like a standalone topic noun phrase.

    Used to gate conjunctive splitting: both halves must be topics, otherwise the
    "and" is joining clauses (not topics) and the question stays whole.
    """
    text = text.strip()
    if len(text) < 3:
        return False
    if _NON_TOPIC_LEAD.match(text):
        return False
    return True


def _split_on_first(text: str, connectors) -> tuple[str, str] | None:
    """Split ``text`` on the first matching connector; return stripped halves."""
    lowered = text.lower()
    for connector in connectors:
        idx = lowered.find(connector)
        if idx == -1:
            continue
        left = _strip_lead(text[:idx])
        right = _strip_lead(text[idx + len(connector):])
        if len(left) >= 3 and len(right) >= 3:
            return left, right
    return None


def _comparative_decompose(question: str) -> List[str] | None:
    """Decompose an explicit comparative question into ``[original, A, B]``."""
    halves = _split_on_first(question, _COMPARATIVE_CONNECTORS)
    if halves is None:
        return None
    left, right = halves
    logger.debug("Comparative split: %r | %r", left, right)
    return [question, left, right]


def _conjunctive_decompose(question: str) -> List[str] | None:
    """Decompose ``A and B`` into ``[original, A, B]`` when both are topics."""
    halves = _split_on_first(question, _CONJUNCTIVE_CONNECTORS)
    if halves is None:
        return None
    left, right = halves
    if not (_is_topic(left) and _is_topic(right)):
        return None
    logger.debug("Conjunctive split: %r | %r", left, right)
    return [question, left, right]


def _multi_part_decompose(question: str) -> List[str] | None:
    """Split a multi-question prompt into ``[original, part1, part2, ...]``.

    Splits on sentence boundaries and on discourse markers ("also",
    "additionally", ...). Returns ``None`` unless at least two substantive parts
    result, so ordinary single-sentence questions are never touched.
    """
    parts = _SENTENCE_SPLIT.split(question)
    expanded: List[str] = []
    for part in parts:
        expanded.extend(_DISCOURSE_SPLIT.split(part))

    subs = [_strip_lead(p) for p in expanded]
    subs = [s for s in subs if len(s) >= 3]
    if len(subs) < 2:
        return None
    logger.debug("Multi-part split into %d parts: %r", len(subs), subs)
    return [question, *subs]


def classify(question: str) -> str:
    """Classify a question as single/comparative/conjunctive/multi_part.

    Precedence is ``multi_part > comparative > conjunctive > single`` so that an
    explicit comparison is never mistaken for a plain conjunction, and a
    genuinely multi-question prompt is split before either.
    """
    if _multi_part_decompose(question) is not None:
        return MULTI_PART
    if _COMPARATIVE_CUE.search(question) and _comparative_decompose(question):
        return COMPARATIVE
    if _conjunctive_decompose(question) is not None:
        return CONJUNCTIVE
    return SINGLE


def _heuristic_decompose(question: str) -> List[str]:
    """Decompose a question into sub-queries according to its class.

    Returns ``[question]`` unchanged for single-intent questions, so existing
    retrieval behavior for simple queries is preserved exactly.
    """
    qtype = classify(question)
    if qtype == MULTI_PART:
        result = _multi_part_decompose(question)
    elif qtype == COMPARATIVE:
        result = _comparative_decompose(question)
    elif qtype == CONJUNCTIVE:
        result = _conjunctive_decompose(question)
    else:
        result = None

    if result is None:
        return [question]
    logger.debug("Decomposed (%s) into %d sub-quer(ies).", qtype, len(result))
    return result


def rewrite_query(question: str, mode: str) -> List[str]:
    """Return sub-queries for a question according to ``mode``.

    Args:
        question: The user's question.
        mode: ``off`` | ``heuristic`` | ``llm``.

    Returns:
        A list whose first element is always the original question. ``off``
        yields a 1-element list; ``heuristic`` may add decomposed sub-queries;
        ``llm`` is not yet enabled and falls back to heuristic.
    """
    question = (question or "").strip()
    if not question:
        return [question]

    if mode == "off":
        return [question]
    if mode == "heuristic":
        return _heuristic_decompose(question)
    if mode == "llm":
        # Configured but intentionally not executed. Fall back to the heuristic
        # path so retrieval keeps working if someone sets this early.
        logger.warning(
            "QUERY_REWRITE_MODE='llm' is configured but not enabled; "
            "falling back to heuristic decomposition."
        )
        return _heuristic_decompose(question)

    # Unknown mode (settings validation should prevent this) — be safe.
    logger.warning("Unknown query_rewrite_mode '%s'; using query as-is.", mode)
    return [question]
