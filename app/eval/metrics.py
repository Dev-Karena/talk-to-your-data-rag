"""Pure ranking metrics for retrieval evaluation.

Each function takes a *ranked* list of retrieved item ids (most-relevant first)
and a *set* of relevant ids, and returns a score. Ids can be at any granularity
the caller chooses — document hash, source name, page, or chunk id — so the same
metrics serve both "retrieval accuracy" and "source accuracy".

All functions are pure and dependency-free for easy unit testing.

Conventions:
    * ``k`` truncates the ranked list to its first ``k`` entries.
    * An empty relevant set yields 0.0 (undefined relevance scores as 0).
    * Ranks are 1-based in MRR.
"""

from __future__ import annotations

import math
from typing import Hashable, List, Optional, Sequence, Set


def _topk(ranked: Sequence[Hashable], k: int) -> List[Hashable]:
    return list(ranked[: max(0, k)])


def recall_at_k(ranked: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    """Fraction of relevant items that appear in the top-k.

    For a single relevant id this is 1.0 if it is retrieved within k, else 0.0.
    For multiple (cross-document) it is the coverage of expected items.
    """
    if not relevant:
        return 0.0
    found = set(_topk(ranked, k)) & relevant
    return len(found) / len(relevant)


def precision_at_k(ranked: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Uses ``k`` as the denominator (standard precision@k), so retrieving fewer
    than k items is penalized proportionally.
    """
    if k <= 0:
        return 0.0
    top = _topk(ranked, k)
    hits = sum(1 for item in top if item in relevant)
    return hits / k


def hit_at_1(ranked: Sequence[Hashable], relevant: Set[Hashable]) -> float:
    """1.0 if the top-ranked item is relevant, else 0.0."""
    if not ranked or not relevant:
        return 0.0
    return 1.0 if ranked[0] in relevant else 0.0


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Set[Hashable]) -> float:
    """Reciprocal of the 1-based rank of the first relevant item (0.0 if none).

    Averaged across queries by the caller, this is MRR.
    """
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of a gain sequence, truncated to k.

    DCG = sum_i gain_i / log2(i + 1), with i 1-based.
    """
    total = 0.0
    for index, gain in enumerate(_topk(gains, k), start=1):
        total += gain / math.log2(index + 1)
    return total


def ndcg_at_k(
    ranked: Sequence[Hashable],
    relevant: Set[Hashable],
    k: int,
    grades: Optional[dict] = None,
) -> float:
    """Normalized DCG at k.

    Binary relevance by default (gain 1 if id in ``relevant``); if ``grades`` is
    given (id -> gain) those graded gains are used instead. Normalized by the
    ideal DCG so the result is in [0, 1]; 0.0 when there is no attainable gain.
    """
    if grades:
        actual_gains = [float(grades.get(item, 0.0)) for item in ranked]
        ideal_gains = sorted((float(g) for g in grades.values()), reverse=True)
    else:
        if not relevant:
            return 0.0
        actual_gains = [1.0 if item in relevant else 0.0 for item in ranked]
        # Ideal ranking = the same gains sorted best-first. This keeps nDCG in
        # [0, 1] even when several ranked items map to one relevant document
        # (e.g. multiple retrieved chunks from the same source).
        ideal_gains = sorted(actual_gains, reverse=True)

    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(actual_gains, k) / idcg
