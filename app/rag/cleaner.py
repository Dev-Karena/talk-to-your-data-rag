"""Text cleaning and normalization.

Deterministic, side-effect-free functions that normalize raw text extracted
from PDFs before it is chunked and embedded. Cleaning improves embedding
quality and retrieval relevance by removing extraction artifacts (broken
hyphenation, stray control characters, irregular whitespace).

Every function is pure (same input → same output), which keeps the module
trivial to unit test.

Usage:
    >>> from app.rag.cleaner import clean_text
    >>> clean_text("The quick brown\\nfox jumps  over   the lazy dog.")
    'The quick brown fox jumps over the lazy dog.'
"""

from __future__ import annotations

import re
import unicodedata

# Hyphenation at a line break: "inter-\nnational" -> "international".
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

# A single newline between two non-empty lines (a soft wrap) -> space.
_SOFT_WRAP = re.compile(r"(?<=\S)\n(?=\S)")

# Two or more blank lines collapse to a single paragraph break.
_MULTI_BLANKLINE = re.compile(r"\n\s*\n\s*(?:\n\s*)+")

# Runs of spaces/tabs collapse to one space.
_MULTI_SPACE = re.compile(r"[ \t]+")

# Control characters except tab (\t), newline (\n), carriage return (\r).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _normalize_unicode(text: str) -> str:
    """Apply NFKC normalization to unify visually-equivalent characters.

    Converts things like full-width characters and ligatures to canonical
    forms so the same content embeds consistently.
    """
    return unicodedata.normalize("NFKC", text)


def _repair_hyphenation(text: str) -> str:
    """Re-join words split by a hyphen at a line break."""
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def _strip_control_chars(text: str) -> str:
    """Remove non-printable control characters that survive extraction."""
    return _CONTROL_CHARS.sub("", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse irregular whitespace while preserving paragraph breaks.

    Order matters:
        1. Soft single newlines (line wraps) become spaces.
        2. Repeated blank lines collapse to one paragraph break.
        3. Repeated spaces/tabs collapse to one space.
    """
    text = _SOFT_WRAP.sub(" ", text)
    text = _MULTI_BLANKLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text


def clean_text(text: str) -> str:
    """Normalize raw extracted text into clean, embedding-ready text.

    The pipeline is:
        unicode normalize -> repair hyphenation -> strip control chars ->
        normalize whitespace -> trim.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text. Returns an empty string if the input is empty or becomes
        empty after cleaning.
    """
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = _repair_hyphenation(text)
    text = _strip_control_chars(text)
    text = _normalize_whitespace(text)
    return text.strip()
