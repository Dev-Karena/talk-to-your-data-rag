"""Unit tests for app.utils.validators.

Covers the upload security policy (extension, size, magic bytes, empty) and the
content-hash dedupe key. No external services are required.
"""

from __future__ import annotations

from app.utils.validators import (
    ValidationResult,
    compute_content_hash,
    validate_pdf,
)

# A minimal byte stream that begins with the PDF magic header.
_VALID_PDF = b"%PDF-1.7\n%minimal test content\n"


def test_valid_pdf_passes() -> None:
    """A well-formed PDF under the size limit is accepted."""
    result = validate_pdf("report.pdf", _VALID_PDF)
    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
    assert result.reason == ""


def test_empty_file_rejected() -> None:
    """A zero-byte upload is rejected."""
    result = validate_pdf("empty.pdf", b"")
    assert result.is_valid is False
    assert "empty" in result.reason.lower()


def test_non_pdf_extension_rejected() -> None:
    """A non-.pdf extension is rejected even if the bytes look like a PDF."""
    result = validate_pdf("notes.txt", _VALID_PDF)
    assert result.is_valid is False
    assert "pdf" in result.reason.lower()


def test_extension_check_is_case_insensitive() -> None:
    """Uppercase .PDF extensions are accepted."""
    result = validate_pdf("REPORT.PDF", _VALID_PDF)
    assert result.is_valid is True


def test_bad_magic_bytes_rejected() -> None:
    """A .pdf file whose content is not a real PDF is rejected."""
    result = validate_pdf("fake.pdf", b"this is not a pdf at all")
    assert result.is_valid is False
    assert "header" in result.reason.lower()


def test_oversized_file_rejected() -> None:
    """A file larger than the configured limit is rejected."""
    from app.config.settings import get_settings

    limit = get_settings().max_file_size_bytes
    # Valid header + padding that pushes the size just over the limit.
    oversized = _VALID_PDF + b"0" * (limit + 1)
    result = validate_pdf("big.pdf", oversized)
    assert result.is_valid is False
    assert "limit" in result.reason.lower()


def test_content_hash_is_stable() -> None:
    """The same bytes always hash to the same value."""
    assert compute_content_hash(_VALID_PDF) == compute_content_hash(_VALID_PDF)


def test_content_hash_ignores_filename() -> None:
    """Hashing depends only on content, not on the upload filename."""
    # Two different "uploads" with identical bytes must share a hash.
    hash_a = compute_content_hash(_VALID_PDF)
    hash_b = compute_content_hash(bytes(_VALID_PDF))
    assert hash_a == hash_b


def test_different_content_differs() -> None:
    """Different content produces different hashes."""
    assert compute_content_hash(b"%PDF-1.7 A") != compute_content_hash(b"%PDF-1.7 B")


def test_content_hash_is_sha256_hex() -> None:
    """The hash is a 64-character hex SHA-256 digest."""
    digest = compute_content_hash(_VALID_PDF)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
