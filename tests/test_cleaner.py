"""Unit tests for app.rag.cleaner.

Covers the deterministic text-normalization pipeline: unicode normalization,
hyphenation repair, control-character stripping, and whitespace collapsing.
Pure functions, no external services.
"""

from __future__ import annotations

from app.rag.cleaner import clean_text


def test_empty_input_returns_empty() -> None:
    """Empty input yields an empty string."""
    assert clean_text("") == ""


def test_whitespace_only_returns_empty() -> None:
    """Input that is only whitespace collapses to an empty string."""
    assert clean_text("   \n\n  \t ") == ""


def test_collapses_multiple_spaces() -> None:
    """Runs of spaces and tabs collapse to a single space."""
    assert clean_text("hello     world\t\tagain") == "hello world again"


def test_soft_linebreak_becomes_space() -> None:
    """A single newline between words (a soft wrap) becomes a space."""
    assert clean_text("The quick brown\nfox") == "The quick brown fox"


def test_repairs_hyphenation_across_linebreak() -> None:
    """A word split by a hyphen at a line break is rejoined."""
    assert clean_text("inter-\nnational") == "international"


def test_repairs_hyphenation_with_surrounding_space() -> None:
    """Hyphenation repair tolerates spaces around the line break."""
    assert clean_text("co- \n operation") == "cooperation"


def test_preserves_paragraph_breaks() -> None:
    """Blank lines separating paragraphs are preserved as a single break."""
    cleaned = clean_text("First paragraph.\n\n\n\nSecond paragraph.")
    assert cleaned == "First paragraph.\n\nSecond paragraph."


def test_strips_control_characters() -> None:
    """Non-printable control characters are removed."""
    assert clean_text("a\x00b\x07c") == "abc"


def test_keeps_normal_punctuation() -> None:
    """Ordinary punctuation and casing are left intact."""
    text = "Revenue grew 18% to $2.4B (up YoY)."
    assert clean_text(text) == text


def test_unicode_normalization_ligature() -> None:
    """NFKC normalization expands ligatures to their canonical characters."""
    # U+FB01 LATIN SMALL LIGATURE FI -> "fi"
    assert clean_text("ﬁle") == "file"


def test_trims_leading_and_trailing_whitespace() -> None:
    """Leading and trailing whitespace is stripped from the result."""
    assert clean_text("   padded text   ") == "padded text"


def test_idempotent_on_clean_text() -> None:
    """Cleaning already-clean text returns it unchanged (idempotency)."""
    once = clean_text("The quick brown\nfox jumps over the lazy dog.")
    twice = clean_text(once)
    assert once == twice
