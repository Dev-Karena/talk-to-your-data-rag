"""Unit tests for app.utils.metadata_health.

Validates the stored-state integrity checker against healthy and deliberately
broken metadata. Pure and offline — no Chroma, no embeddings.
"""

from __future__ import annotations

from app.utils.metadata_health import EXPECTED_FIELDS, check_records

# A valid 64-char hex SHA-256-shaped hash.
_HASH = "a" * 64


def _good(page: int = 1, index: int = 0):
    """Return a (record_id, metadata) pair that should pass every check."""
    cid = f"{_HASH}::p{page}::c{index}"
    return cid, {
        "chunk_id": cid,
        "source": "ML.pdf",
        "page_number": page,
        "doc_hash": _HASH,
        "chunk_index": index,
    }


def test_healthy_records_pass() -> None:
    ids, metas = zip(_good(1, 0), _good(1, 1), _good(2, 0))
    report = check_records(list(ids), list(metas))
    assert report.ok
    assert report.total_records == 3
    assert report.clean_records == 3
    assert report.sources == 1
    assert report.doc_hashes == 1


def test_missing_field_flagged() -> None:
    cid, meta = _good()
    del meta["doc_hash"]
    report = check_records([cid], [meta])
    assert not report.ok
    assert any(i.field == "doc_hash" and "missing" in i.problem for i in report.issues)


def test_empty_field_flagged() -> None:
    cid, meta = _good()
    meta["source"] = ""
    report = check_records([cid], [meta])
    assert any(i.field == "source" for i in report.issues)


def test_none_metadata_flagged() -> None:
    report = check_records(["someid"], [None])
    assert not report.ok
    assert report.issues[0].problem.startswith("metadata is missing")


def test_chunk_id_mismatch_with_record_id() -> None:
    cid, meta = _good()
    report = check_records(["different-record-id"], [meta])
    assert any(i.field == "chunk_id" and "record id" in i.problem for i in report.issues)


def test_malformed_chunk_id() -> None:
    bad = "not-a-valid-chunk-id"
    meta = {
        "chunk_id": bad, "source": "ML.pdf", "page_number": 1,
        "doc_hash": _HASH, "chunk_index": 0,
    }
    report = check_records([bad], [meta])
    assert any(i.field == "chunk_id" and "malformed" in i.problem for i in report.issues)


def test_doc_hash_not_matching_chunk_id() -> None:
    other = "b" * 64
    cid = f"{_HASH}::p1::c0"
    meta = {
        "chunk_id": cid, "source": "ML.pdf", "page_number": 1,
        "doc_hash": other, "chunk_index": 0,
    }
    report = check_records([cid], [meta])
    assert any(i.field == "doc_hash" and "chunk_id" in i.problem for i in report.issues)


def test_bad_page_and_index_numbers() -> None:
    cid = f"{_HASH}::p1::c0"
    meta = {
        "chunk_id": cid, "source": "ML.pdf", "page_number": 0,
        "doc_hash": _HASH, "chunk_index": -1,
    }
    report = check_records([cid], [meta])
    fields = {i.field for i in report.issues}
    assert "page_number" in fields
    assert "chunk_index" in fields


def test_duplicate_record_ids_flagged() -> None:
    (cid, meta) = _good()
    report = check_records([cid, cid], [meta, dict(meta)])
    assert any(i.problem == "duplicate record id" for i in report.issues)


def test_expected_fields_constant_is_complete() -> None:
    # Guards against the schema and checker drifting apart.
    assert set(EXPECTED_FIELDS) == {
        "chunk_id", "source", "page_number", "doc_hash", "chunk_index"
    }
