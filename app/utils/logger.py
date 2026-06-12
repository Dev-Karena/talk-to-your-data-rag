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

import logging
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


def _build_formatter() -> logging.Formatter:
    """Create the shared log formatter."""
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

    logger.addHandler(_build_console_handler(level, formatter))

    file_handler = _build_file_handler(settings.log_file, level, formatter)
    if file_handler is not None:
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger
