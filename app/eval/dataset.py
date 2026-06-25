"""Benchmark dataset loading, validation, and corpus fingerprinting.

A benchmark file is YAML with two sections:

    corpus:                      # which indexed corpus this dataset is valid for
      description: "..."
      expected_doc_hashes: [ ... ]
    cases:                       # the labeled queries
      - id: ...
        query: "..."
        type: single | cross_document | negative
        expected_sources: [ ... ]
        expected_doc_hashes: [ ... ]
        expected_pages: [ ... ]        # optional
        relevant_chunk_ids: [ ... ]    # optional

Ground truth is keyed on ``doc_hash`` (stable, collision-proof) — display
``source`` names are kept only for human-readable reports.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


class DatasetError(Exception):
    """Raised when a benchmark dataset is malformed."""


@dataclass(frozen=True)
class Case:
    """A single labeled benchmark query."""

    id: str
    query: str
    type: str  # "single" | "cross_document" | "negative"
    expected_sources: List[str] = field(default_factory=list)
    expected_doc_hashes: List[str] = field(default_factory=list)
    expected_pages: List[int] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)

    @property
    def is_negative(self) -> bool:
        return self.type == "negative"


@dataclass(frozen=True)
class Benchmark:
    """A loaded benchmark: corpus expectations plus the cases."""

    description: str
    expected_doc_hashes: List[str]
    cases: List[Case]

    def fingerprint(self) -> str:
        """Stable hash of the corpus this dataset targets (sorted doc hashes)."""
        joined = "|".join(sorted(self.expected_doc_hashes))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


_VALID_TYPES = {"single", "cross_document", "negative"}


def load_benchmark(path: str | Path) -> Benchmark:
    """Load and validate a benchmark YAML file.

    Raises:
        DatasetError: If the file is missing required structure or a case is
            internally inconsistent (e.g. a non-negative case with no expected
            documents).
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Benchmark file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # malformed YAML
        raise DatasetError(f"Could not parse YAML '{path}': {exc}") from exc

    corpus = raw.get("corpus") or {}
    cases_raw = raw.get("cases") or []
    if not cases_raw:
        raise DatasetError("Benchmark has no 'cases'.")

    cases: List[Case] = []
    seen_ids: set = set()
    for i, c in enumerate(cases_raw):
        cid = str(c.get("id") or f"case-{i}")
        if cid in seen_ids:
            raise DatasetError(f"Duplicate case id: '{cid}'.")
        seen_ids.add(cid)

        if not c.get("query"):
            raise DatasetError(f"Case '{cid}' has no query.")

        ctype = str(c.get("type", "single"))
        if ctype not in _VALID_TYPES:
            raise DatasetError(f"Case '{cid}' has invalid type '{ctype}'.")

        expected_hashes = [str(h) for h in (c.get("expected_doc_hashes") or [])]
        if ctype != "negative" and not expected_hashes:
            raise DatasetError(
                f"Case '{cid}' is '{ctype}' but lists no expected_doc_hashes."
            )

        cases.append(Case(
            id=cid,
            query=str(c["query"]),
            type=ctype,
            expected_sources=[str(s) for s in (c.get("expected_sources") or [])],
            expected_doc_hashes=expected_hashes,
            expected_pages=[int(p) for p in (c.get("expected_pages") or [])],
            relevant_chunk_ids=[str(x) for x in (c.get("relevant_chunk_ids") or [])],
        ))

    return Benchmark(
        description=str(corpus.get("description", "")),
        expected_doc_hashes=[str(h) for h in (corpus.get("expected_doc_hashes") or [])],
        cases=cases,
    )


def corpus_fingerprint(doc_hashes: List[str]) -> str:
    """Fingerprint a live corpus's set of doc hashes (matches Benchmark.fingerprint)."""
    joined = "|".join(sorted(set(doc_hashes)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
