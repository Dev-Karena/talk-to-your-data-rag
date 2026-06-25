"""Centralized logging configuration.

Provides a single ``get_logger`` factory so every module emits logs in a
consistent format to both the console and a rotating log file. Configuration
(level, file path) is sourced from the application settings.

Usage:
    >>> from app.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Indexing started")
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.config.settings import get_settings

# Shared format across all handlers: timestamp | level | logger name | message.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotating file handler limits: 5 MB per file, keep 3 backups.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Track configured logger names so handlers are attached only once per process.
_configured_loggers: set[str] = set()

# Per-operation correlation id, set via set_correlation_id(); threads a single id
# across the log lines of one logical operation (e.g. a query: embed -> search ->
# generate). Default "-" means "not in a correlated operation".
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(value: str) -> None:
    """Set the correlation id for the current context (and its log lines)."""
    _correlation_id.set(value)


def get_correlation_id() -> str:
    """Return the current correlation id ('-' if none is set)."""
    return _correlation_id.get()


def new_correlation_id(prefix: str = "op") -> str:
    """Generate, set, and return a short correlation id for a new operation."""
    cid = f"{prefix}-{uuid.uuid4().hex[:8]}"
    set_correlation_id(cid)
    return cid


class _CorrelationIdFilter(logging.Filter):
    """Inject the current correlation id onto every record as ``correlation_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class _JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object (structured logging).

    Emits stable top-level keys (time, level, logger, message, correlation_id)
    plus exception text when present. This changes only the *encoding* of the
    same events — not what gets logged.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_formatter() -> logging.Formatter:
    """Create the shared log formatter based on the configured LOG_FORMAT.

    Defaults to the human-readable text format; JSON is opt-in and leaves the
    default output unchanged.
    """
    if get_settings().log_format == "json":
        return _JsonFormatter()
    return logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _build_console_handler(level: int, formatter: logging.Formatter) -> logging.Handler:
    """Create a stream handler that writes to stderr/console."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(
    log_file: Path, level: int, formatter: logging.Formatter
) -> Optional[logging.Handler]:
    """Create a rotating file handler.

    Returns ``None`` if the file handler cannot be created (e.g. the path is
    not writable); logging then falls back to console only rather than crashing
    the application.
    """
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler
    except OSError:
        # Don't let a logging failure take down the app; console still works.
        return None


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    On first use for a given name, attaches a console handler and a rotating
    file handler using the level and file path from application settings.
    Subsequent calls return the already-configured logger without adding
    duplicate handlers.

    Args:
        name: Logger name, typically the calling module's ``__name__``.

    Returns:
        A ``logging.Logger`` ready for use.
    """
    logger = logging.getLogger(name)

    # Avoid attaching handlers more than once (e.g. on Streamlit reruns).
    if name in _configured_loggers:
        return logger

    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)
    formatter = _build_formatter()

    logger.setLevel(level)
    # Don't propagate to the root logger; we manage our own handlers.
    logger.propagate = False

    # Inject the correlation id onto every record (surfaced in JSON output).
    logger.addFilter(_CorrelationIdFilter())

    logger.addHandler(_build_console_handler(level, formatter))

    file_handler = _build_file_handler(settings.log_file, level, formatter)
    if file_handler is not None:
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger
