"""Upload validation and content hashing.

Enforces the application's file-upload security policy and provides the
content hash used to deduplicate documents (so the same PDF is never indexed
twice).

Security checks performed:
    * Extension allowlist (``.pdf`` only).
    * Maximum file size (from settings).
    * Magic-byte / MIME sniff (file must actually start with a PDF header).
    * Empty-file rejection.

Usage:
    >>> from app.utils.validators import validate_pdf, compute_content_hash
    >>> result = validate_pdf("report.pdf", data)
    >>> if result.is_valid:
    ...     doc_hash = compute_content_hash(data)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Only PDFs are accepted.
_ALLOWED_EXTENSIONS = {".pdf"}

# A valid PDF stream begins with the magic bytes "%PDF-".
_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating an uploaded file.

    Attributes:
        is_valid: Whether the file passed every check.
        reason: Human-readable explanation when ``is_valid`` is ``False``;
            empty string when valid.
    """

    is_valid: bool
    reason: str = ""


def _has_allowed_extension(filename: str) -> bool:
    """Return ``True`` if the filename's extension is in the allowlist."""
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


def _looks_like_pdf(data: bytes) -> bool:
    """Return ``True`` if the byte stream starts with the PDF magic header."""
    return data[: len(_PDF_MAGIC)] == _PDF_MAGIC


def validate_pdf(filename: str, data: bytes) -> ValidationResult:
    """Validate an uploaded PDF against the security policy.

    Args:
        filename: Original name of the uploaded file (used for the extension
            check).
        data: Raw bytes of the uploaded file.

    Returns:
        A :class:`ValidationResult`. When invalid, ``reason`` describes the
        first failed check.
    """
    settings = get_settings()

    # 1. Reject empty files.
    if not data:
        reason = f"'{filename}' is empty."
        logger.warning("Rejected upload: %s", reason)
        return ValidationResult(is_valid=False, reason=reason)

    # 2. Extension allowlist.
    if not _has_allowed_extension(filename):
        reason = f"'{filename}' is not a PDF (only .pdf files are allowed)."
        logger.warning("Rejected upload: %s", reason)
        return ValidationResult(is_valid=False, reason=reason)

    # 3. Size cap.
    if len(data) > settings.max_file_size_bytes:
        size_mb = len(data) / (1024 * 1024)
        reason = (
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{settings.max_file_size_mb} MB limit."
        )
        logger.warning("Rejected upload: %s", reason)
        return ValidationResult(is_valid=False, reason=reason)

    # 4. Magic-byte sniff: confirm the content really is a PDF, not just a
    #    renamed file. Guards against a malicious file with a .pdf extension.
    if not _looks_like_pdf(data):
        reason = f"'{filename}' does not have a valid PDF header."
        logger.warning("Rejected upload: %s", reason)
        return ValidationResult(is_valid=False, reason=reason)

    logger.info("Validated upload: '%s' (%d bytes)", filename, len(data))
    return ValidationResult(is_valid=True)


def compute_content_hash(data: bytes) -> str:
    """Compute a stable SHA-256 hash of file content.

    The hash is derived from the file's bytes only (not its name), so the same
    document uploaded under different filenames produces the same hash. This is
    used as the deduplication key to avoid re-indexing documents that are
    already in the vector store.

    Args:
        data: Raw bytes of the file.

    Returns:
        The hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()
