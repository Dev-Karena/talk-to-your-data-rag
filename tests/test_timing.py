"""Unit tests for app.utils.timing.

Verifies the Stopwatch records named stages, accumulates a total, records timing
even when a stage raises, and formats a readable summary. Timing values are
asserted only for ordering/presence (not absolute durations) to stay robust.
"""

from __future__ import annotations

import logging

import pytest

from app.utils.timing import Stopwatch, timed


def test_stage_records_named_durations() -> None:
    sw = Stopwatch()
    with sw.stage("a"):
        pass
    with sw.stage("b"):
        pass
    assert set(sw.stages) == {"a", "b"}
    assert all(v >= 0.0 for v in sw.stages.values())


def test_total_is_sum_of_stages() -> None:
    sw = Stopwatch()
    with sw.stage("a"):
        pass
    with sw.stage("b"):
        pass
    assert sw.total_ms == pytest.approx(sum(sw.stages.values()))


def test_items_preserve_completion_order() -> None:
    sw = Stopwatch()
    with sw.stage("first"):
        pass
    with sw.stage("second"):
        pass
    assert [name for name, _ in sw.items()] == ["first", "second"]


def test_duration_recorded_even_on_exception() -> None:
    sw = Stopwatch()
    with pytest.raises(ValueError):
        with sw.stage("boom"):
            raise ValueError("x")
    # The stage time is still captured despite the error.
    assert "boom" in sw.stages


def test_format_is_readable() -> None:
    sw = Stopwatch()
    with sw.stage("embed"):
        pass
    text = sw.format()
    assert "embed=" in text and "ms" in text


def test_timed_logs_total(caplog) -> None:
    # app loggers don't propagate to root; enable so caplog can capture.
    tlogger = logging.getLogger("app.utils.timing")
    tlogger.propagate = True
    with caplog.at_level(logging.DEBUG, logger="app.utils.timing"):
        with timed("op") as sw:
            with sw.stage("step"):
                pass
    assert any("op timing" in r.getMessage() for r in caplog.records)
