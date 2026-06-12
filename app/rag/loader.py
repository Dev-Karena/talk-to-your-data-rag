"""PDF loading and text extraction.

Turns a PDF file on disk into a list of :class:`PageDocument` objects — one per
page — carrying the extracted text plus page-level metadata (source filename
and 1-based page number). This is pure extraction: no cleaning or chunking
happens here.

Backed by LangChain's ``PyPDFLoader`` (which uses ``pypdf`` under the hood).

Usage:
    >>> from app.rag.loader import load_pdf
    >>> pages = load_pdf("documents/report.pdf", source_name="report.pdf")
    >>> pages[0].page_number, pages[0].text[:50]
    (1, 'Annual Report 2025 ...')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PageDocument:
    """A single extracted PDF page.

    Attributes:
        source: Display name of the source document (e.g. ``report.pdf``).
        page_number: 1-based page index within the document.
        text: Raw text extracted from the page.
    """

    source: str
    page_number: int
    text: str


class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


def load_pdf(file_path: str | Path, source_name: str) -> List[PageDocument]:
    """Extract text from a PDF, one :class:`PageDocument` per page.

    Args:
        file_path: Path to the PDF file on disk.
        source_name: Human-friendly document name to record in metadata
            (typically the original upload filename).

    Returns:
        A list of :class:`PageDocument` objects in page order. Pages whose
        extracted text is empty (e.g. scanned image-only pages) are skipped.

    Raises:
        PDFLoadError: If the file does not exist or cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise PDFLoadError(f"PDF not found: {path}")

    try:
        loader = PyPDFLoader(str(path))
        raw_pages = loader.load()
    except Exception as exc:  # noqa: BLE001 - surface any parser failure uniformly
        logger.error("Failed to parse PDF '%s': %s", source_name, exc)
        raise PDFLoadError(f"Could not parse '{source_name}': {exc}") from exc

    pages: List[PageDocument] = []
    for index, doc in enumerate(raw_pages):
        text = (doc.page_content or "").strip()
        if not text:
            # Skip blank / image-only pages — they add no retrievable content.
            logger.debug(
                "Skipping empty page %d of '%s'", index + 1, source_name
            )
            continue

        # PyPDFLoader exposes a 0-based "page" in metadata; prefer it when
        # present, otherwise fall back to enumeration. Store as 1-based.
        raw_page = doc.metadata.get("page", index)
        page_number = int(raw_page) + 1

        pages.append(
            PageDocument(
                source=source_name,
                page_number=page_number,
                text=text,
            )
        )

    if not pages:
        # A PDF with zero extractable text is almost certainly scanned images.
        logger.warning(
            "No extractable text found in '%s' (scanned/image-only PDF?).",
            source_name,
        )
        raise PDFLoadError(
            f"No extractable text found in '{source_name}'. "
            "It may be a scanned/image-only PDF requiring OCR."
        )

    logger.info(
        "Loaded '%s': %d page(s) with text.", source_name, len(pages)
    )
    return pages
