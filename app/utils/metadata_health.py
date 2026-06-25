"""Metadata integrity checks for stored chunks.

Pure, dependency-light validation of the metadata that is *actually persisted*
in the vector store, separated from any CLI so it can be unit-tested. The CLI
wrapper lives in ``scripts/metadata_health.py``.

The expected per-chunk schema is produced by :meth:`app.rag.chunker.Chunk.metadata`:

    chunk_id    -- "{doc_hash}::p{page}::c{index}"
    source      -- display filename
    page_number -- 1-based int
    doc_hash    -- 64-char hex SHA-256
    chunk_index -- 0-based int

Usage:
    >>> from app.utils.metadata_health import check_records
    >>> report = check_records(ids, metadatas)
    >>> report.ok, len(report.issues)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Fields every stored chunk must carry (and which must be non-empty).
EXPECTED_FIELDS = ("chunk_id", "source", "page_number", "doc_hash", "chunk_index")

# Stable id shape: "{64-hex doc_hash}::p{page}::c{index}".
_CHUNK_ID_RE = re.compile(r"^(?P<hash>[0-9a-f]{64})::p(?P<page>\d+)::c(?P<index>\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class MetadataIssue:
    """A single integrity problem found on one stored record.

    Attributes:
        record_id: The Chroma record id the problem was found on.
        field: The metadata field involved ('-' for record-level problems).
        problem: Human-readable description of what is wrong.
    """

    record_id: str
    field: str
    problem: str


@dataclass
class HealthReport:
    """Aggregate result of a metadata health scan.

    Attributes:
        total_records: Number of records scanned.
        issues: Every problem found (empty when healthy).
        sources: Count of distinct ``source`` values seen.
        doc_hashes: Count of distinct ``doc_hash`` values seen.
    """

    total_records: int = 0
    issues: List[MetadataIssue] = field(default_factory=list)
    sources: int = 0
    doc_hashes: int = 0

    @property
    def ok(self) -> bool:
        """True when no issues were found."""
        return not self.issues

    @property
    def clean_records(self) -> int:
        """Number of records with no issue against them."""
        bad = {issue.record_id for issue in self.issues}
        return self.total_records - len(bad)


def _check_one(record_id: str, meta: Optional[dict]) -> List[MetadataIssue]:
    """Validate a single record's metadata, returning any issues found."""
    issues: List[MetadataIssue] = []

    if not meta:
        return [MetadataIssue(record_id, "-", "metadata is missing or empty")]

    # 1. Required fields present and non-empty.
    for fld in EXPECTED_FIELDS:
        if fld not in meta or meta[fld] in (None, ""):
            issues.append(MetadataIssue(record_id, fld, "missing or empty"))

    # 2. chunk_id should equal the record id (ids are derived from chunk_id).
    chunk_id = str(meta.get("chunk_id", ""))
    if chunk_id and chunk_id != record_id:
        issues.append(
            MetadataIssue(record_id, "chunk_id", f"does not match record id ('{chunk_id}')")
        )

    # 3. chunk_id format + internal consistency with doc_hash/page/index.
    match = _CHUNK_ID_RE.match(chunk_id)
    if chunk_id and not match:
        issues.append(
            MetadataIssue(record_id, "chunk_id", f"malformed (expected hash::pN::cN): '{chunk_id}'")
        )

    doc_hash = str(meta.get("doc_hash", ""))
    if doc_hash and not _SHA256_RE.match(doc_hash):
        issues.append(MetadataIssue(record_id, "doc_hash", "not a 64-char hex SHA-256"))

    if match and doc_hash and match.group("hash") != doc_hash:
        issues.append(
            MetadataIssue(record_id, "doc_hash", "does not match the hash in chunk_id")
        )

    # 4. Numeric sanity for page_number (>=1) and chunk_index (>=0).
    page = meta.get("page_number")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        if "page_number" not in (i.field for i in issues):  # don't double-report empties
            issues.append(MetadataIssue(record_id, "page_number", f"invalid (expected int >= 1): {page!r}"))

    index = meta.get("chunk_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        if "chunk_index" not in (i.field for i in issues):
            issues.append(MetadataIssue(record_id, "chunk_index", f"invalid (expected int >= 0): {index!r}"))

    return issues


def check_records(ids: List[str], metadatas: List[Optional[dict]]) -> HealthReport:
    """Scan all stored records and return a :class:`HealthReport`.

    Args:
        ids: Record ids, aligned by index with ``metadatas``.
        metadatas: Per-record metadata dicts (may contain ``None``).

    Returns:
        A populated :class:`HealthReport`. Detects per-record schema/format
        problems plus collection-level duplicate ids.
    """
    report = HealthReport(total_records=len(ids))

    seen_ids: Dict[str, int] = {}
    sources: set = set()
    hashes: set = set()

    for record_id, meta in zip(ids, metadatas):
        # Collection-level: duplicate record ids.
        seen_ids[record_id] = seen_ids.get(record_id, 0) + 1
        if seen_ids[record_id] == 2:  # report each duplicate id once
            report.issues.append(MetadataIssue(record_id, "-", "duplicate record id"))

        if meta:
            if meta.get("source"):
                sources.add(str(meta["source"]))
            if meta.get("doc_hash"):
                hashes.add(str(meta["doc_hash"]))

        report.issues.extend(_check_one(record_id, meta))

    report.sources = len(sources)
    report.doc_hashes = len(hashes)
    return report
