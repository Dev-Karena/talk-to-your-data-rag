"""Metadata Health Inspector — read-only integrity scan of the live store.

Scans every chunk actually persisted in ChromaDB and reports metadata problems:
missing/empty required fields, malformed ``chunk_id``, ``doc_hash`` that does not
match the id, bad page/index numbers, and duplicate record ids.

Exits non-zero when problems are found, so it can gate CI or a pre-deploy check.

Run:
    python scripts/metadata_health.py            # human-readable report
    python scripts/metadata_health.py --json     # machine-readable
    python scripts/metadata_health.py --limit 50 # cap issues printed

The integrity logic lives in app/utils/metadata_health.py (unit-tested); this is
a thin CLI over it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings              # noqa: E402
from app.rag.vector_store import get_vector_store          # noqa: E402
from app.utils.metadata_health import check_records        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only metadata health scan.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--limit", type=int, default=40, help="Max issues to print (human mode).")
    args = parser.parse_args()

    settings = get_settings()
    store = get_vector_store()
    records = store.all_records()
    report = check_records(records["ids"], records["metadatas"])

    if args.json:
        print(json.dumps({
            "collection": settings.chroma_collection_name,
            "total_records": report.total_records,
            "clean_records": report.clean_records,
            "sources": report.sources,
            "doc_hashes": report.doc_hashes,
            "ok": report.ok,
            "issue_count": len(report.issues),
            "issues": [
                {"record_id": i.record_id, "field": i.field, "problem": i.problem}
                for i in report.issues
            ],
        }, indent=2))
        return 0 if report.ok else 2

    print("=" * 70)
    print("Metadata Health Inspector (read-only)")
    print("=" * 70)
    print(f"Collection      : {settings.chroma_collection_name}")
    print(f"Records scanned : {report.total_records}")
    print(f"Distinct sources: {report.sources}")
    print(f"Distinct hashes : {report.doc_hashes}")
    print(f"Clean records   : {report.clean_records}/{report.total_records}")

    if report.ok:
        print("\nRESULT: PASS - all metadata is valid and consistent.")
        return 0

    print(f"\nRESULT: FAIL - {len(report.issues)} issue(s) found:")
    print("-" * 70)
    for issue in report.issues[: args.limit]:
        print(f"  [{issue.field}] {issue.record_id}\n      -> {issue.problem}")
    if len(report.issues) > args.limit:
        print(f"  ... and {len(report.issues) - args.limit} more (use --limit or --json).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
