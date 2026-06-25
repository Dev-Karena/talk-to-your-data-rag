"""Lightweight timing utilities for performance observability.

A tiny, dependency-free helper for measuring how long named stages of an
operation take, and logging the breakdown. This is *observability only* — timing
a block does not change its behavior or results.

Usage:
    >>> from app.utils.timing import Stopwatch
    >>> sw = Stopwatch()
    >>> with sw.stage("embed"):
    ...     vector = embed(query)
    >>> with sw.stage("search"):
    ...     hits = store.query(vector)
    >>> sw.format()
    'embed=12.4ms, search=3.1ms'
    >>> sw.total_ms
    15.5

The :class:`Stopwatch` is intended to be created per logical operation (one
query, one ingest) so its stage timings describe a single unit of work.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Stopwatch:
    """Accumulates the wall-clock duration of named stages, in order.

    Attributes:
        stages: Ordered mapping of stage name -> elapsed milliseconds. Insertion
            order reflects the order stages completed.
    """

    stages: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the wrapped block and record its duration under ``name``.

        The duration is recorded even if the block raises, so partial timings
        are still available for debugging a failure.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - start) * 1000.0

    @property
    def total_ms(self) -> float:
        """Total recorded time across all stages, in milliseconds."""
        return sum(self.stages.values())

    def items(self) -> List[Tuple[str, float]]:
        """Return ``(stage, ms)`` pairs in completion order."""
        return list(self.stages.items())

    def format(self) -> str:
        """Return a compact 'stage=12.3ms, ...' summary string."""
        return ", ".join(f"{name}={ms:.1f}ms" for name, ms in self.stages.items())

    def log(self, label: str, level: int = logging.DEBUG) -> None:
        """Log the stage breakdown and total at the given level (DEBUG default).

        Logging at DEBUG keeps default (INFO) output unchanged while making
        timings available when DEBUG is enabled or when callers raise the level.
        """
        logger.log(
            level, "%s timing: %s (total=%.1fms)", label, self.format(), self.total_ms
        )


@contextmanager
def timed(label: str, level: int = logging.DEBUG) -> Iterator[Stopwatch]:
    """Time a whole operation, logging its total on exit.

    Yields a :class:`Stopwatch` so the caller may also record sub-stages.

    Usage:
        >>> with timed("retrieve") as sw:
        ...     with sw.stage("embed"):
        ...         ...
    """
    sw = Stopwatch()
    start = time.perf_counter()
    try:
        yield sw
    finally:
        total = (time.perf_counter() - start) * 1000.0
        if sw.stages:
            logger.log(level, "%s timing: %s (total=%.1fms)", label, sw.format(), total)
        else:
            logger.log(level, "%s timing: total=%.1fms", label, total)
