"""Unit tests for structured logging in app.utils.logger.

Covers the JSON formatter, the correlation-id context helpers, and the filter
that injects the id onto records. Does not touch settings/handlers, so it is
fully offline and side-effect free.
"""

from __future__ import annotations

import json
import logging

from app.utils import logger as log_mod
from app.utils.logger import (
    _CorrelationIdFilter,
    _JsonFormatter,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)


def _record(msg: str = "hello", **extra) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


def test_correlation_id_set_and_get() -> None:
    set_correlation_id("abc")
    assert get_correlation_id() == "abc"


def test_new_correlation_id_has_prefix_and_sets_context() -> None:
    cid = new_correlation_id("query")
    assert cid.startswith("query-")
    assert get_correlation_id() == cid


def test_filter_injects_correlation_id() -> None:
    set_correlation_id("cid-123")
    rec = _record()
    assert _CorrelationIdFilter().filter(rec) is True
    assert rec.correlation_id == "cid-123"


def test_json_formatter_emits_expected_keys() -> None:
    rec = _record("indexed 3 chunks", correlation_id="cid-9")
    out = _JsonFormatter().format(rec)
    payload = json.loads(out)  # must be valid JSON
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "indexed 3 chunks"
    assert payload["correlation_id"] == "cid-9"
    assert "time" in payload


def test_json_formatter_defaults_correlation_id() -> None:
    rec = _record("no cid set")  # no correlation_id attribute
    payload = json.loads(_JsonFormatter().format(rec))
    assert payload["correlation_id"] == "-"


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = logging.LogRecord(
            name="app.test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(_JsonFormatter().format(rec))
    assert "exception" in payload and "ValueError" in payload["exception"]


def test_build_formatter_switches_on_setting(monkeypatch) -> None:
    """_build_formatter returns JSON vs text per the log_format setting."""
    class _S:
        log_format = "json"
    monkeypatch.setattr(log_mod, "get_settings", lambda: _S())
    assert isinstance(log_mod._build_formatter(), _JsonFormatter)

    class _T:
        log_format = "text"
    monkeypatch.setattr(log_mod, "get_settings", lambda: _T())
    assert not isinstance(log_mod._build_formatter(), _JsonFormatter)
