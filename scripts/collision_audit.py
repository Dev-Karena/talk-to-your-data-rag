"""Source-name collision audit (read-only).

Detects when one display ``source`` name maps to more than one ``doc_hash`` —
i.e. two physically different documents that would be cited under the same name,
making source attribution ambiguous. Checks both:

    1. the LIVE Chroma collection (what is indexed), and
    2. the documents/ directory on disk (what a re-index would ingest).

This NEVER deletes or modifies anything — it only reports and recommends.

Run:
    python scripts/collision_audit.py
    python scripts/collision_audit.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings              # noqa: E402
from app.rag.vector_store import get_vector_store          # noqa: E402


def _strip_hash_prefix(filename: str) -> str:
    """Recover the display name from a stored ``{hash}_{name}`` file."""
    parts = filename.split("_", 1)
    # Only treat as a hash prefix if it looks like one (hex, 8+ chars).
    if len(parts) == 2 and len(parts[0]) >= 8 and all(c in "0123456789abcdef" for c in parts[0]):
        return parts[1]
    return filename


def _audit_collection() -> dict:
    """Map source -> set(doc_hash) for the live collection."""
    store = get_vector_store()
    records = store.all_records()
    by_source: dict[str, set] = defaultdict(set)
    for meta in records["metadatas"]:
        if meta and meta.get("source"):
            by_source[str(meta["source"])].add(str(meta.get("doc_hash", "")))
    collisions = {s: sorted(h) for s, h in by_source.items() if len(h) > 1}
    return {
        "sources": {s: sorted(h) for s, h in by_source.items()},
        "collisions": collisions,
    }


def _audit_disk(documents_dir: Path) -> dict:
    """Map display-name -> {actual file -> content sha256} for documents/."""
    by_display: dict[str, dict] = defaultdict(dict)
    for path in sorted(documents_dir.glob("*.pdf")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        display = _strip_hash_prefix(path.name)
        by_display[display][path.name] = digest
    collisions = {
        display: files
        for display, files in by_display.items()
        if len({d for d in files.values()}) > 1
    }
    return {"by_display": by_display, "collisions": collisions}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only source-name collision audit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    settings = get_settings()
    collection = _audit_collection()
    disk = _audit_disk(settings.documents_dir)

    has_collisions = bool(collection["collisions"]) or bool(disk["collisions"])

    if args.json:
        print(json.dumps({
            "collection": collection,
            "disk": disk,
            "has_collisions": has_collisions,
        }, indent=2))
        return 2 if has_collisions else 0

    print("=" * 72)
    print("Source-Name Collision Audit (read-only)")
    print("=" * 72)

    # --- Live collection ---
    print("\n[1] Live collection")
    print("-" * 72)
    if collection["collisions"]:
        print("!! Collisions found (one source -> multiple doc_hash):")
        for src, hashes in collection["collisions"].items():
            print(f"   {src}: {[h[:12] for h in hashes]}")
    else:
        print(f"OK - {len(collection['sources'])} source(s), each maps to a single doc_hash.")
        for src, hashes in sorted(collection["sources"].items()):
            print(f"   {src:<14} -> {hashes[0][:12]}")

    # --- Disk (what a re-index would ingest) ---
    print("\n[2] documents/ on disk (what 'Re-index from disk' would ingest)")
    print("-" * 72)
    if disk["collisions"]:
        print("!! Collisions found (one display name -> different file contents):")
        for display, files in disk["collisions"].items():
            print(f"   {display}:")
            for fname, digest in files.items():
                print(f"       {fname:<28} sha256={digest[:12]}")
    else:
        print("OK - no display-name collisions on disk.")

    # --- Recommendation ---
    print("\n[3] Recommendation (no data was changed)")
    print("-" * 72)
    if not has_collisions:
        print("No action needed.")
        return 0
    print(textwrap_recommendation())
    return 2


def textwrap_recommendation() -> str:
    return (
        "Display-name collisions make source attribution ambiguous: two different\n"
        "documents would be cited under the same name. For the Sprint-4 benchmark\n"
        "(Option A = real textbooks), make the textbooks the single canonical copy:\n"
        "\n"
        "  1. Keep the real textbooks:   ML.pdf, OS.pdf, DBMS.pdf\n"
        "  2. Remove the synthetic 3-page duplicates (the hash-prefixed files):\n"
        "         documents/0fec69462619_ML.pdf\n"
        "         documents/887eed34d7de_OS.pdf\n"
        "         documents/75d1fbdd4db7_DBMS.pdf\n"
        "  3. Rebuild a clean index from the textbooks only.\n"
        "\n"
        "IMPORTANT: do this in an ISOLATED benchmark store, not the production\n"
        "chroma_db/. The provided scripts/build_benchmark_corpus.py does exactly\n"
        "that and never touches production data. Only delete the synthetic files\n"
        "yourself after confirming - this tool will not delete anything."
    )


if __name__ == "__main__":
    raise SystemExit(main())
