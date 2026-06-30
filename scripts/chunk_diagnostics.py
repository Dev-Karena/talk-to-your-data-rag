#!/usr/bin/env python
"""Diagnostics script to inspect individual chunks and their neighborhoods.

Supports retrieval by chunk ID, source/page, or search query.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr to prevent cp1252 encoding crashes on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup project root import path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.services.retriever import retrieve
from app.rag.vector_store import get_vector_store

def get_overlap(text1: str, text2: str) -> str:
    """Find the longest common suffix of text1 that is a prefix of text2."""
    max_len = min(len(text1), len(text2))
    for i in range(max_len, 0, -1):
        if text1.endswith(text2[:i]):
            return text2[:i]
    return ""

def analyze_chunk(chunk_id: str, doc_text: str, doc_meta: dict, all_doc_chunks: list) -> dict:
    """Extract metadata, overlaps, and neighbors for a given chunk."""
    doc_hash = doc_meta.get("doc_hash", "")
    page_number = int(doc_meta.get("page_number", 0))
    chunk_index = int(doc_meta.get("chunk_index", 0))
    source = doc_meta.get("source", "")
    
    # Sort all document chunks to find predecessor and successor
    sorted_chunks = sorted(all_doc_chunks, key=lambda x: (x["page_number"], x["chunk_index"]))
    
    pred = None
    succ = None
    target_idx = -1
    for idx, c in enumerate(sorted_chunks):
        if c["id"] == chunk_id:
            target_idx = idx
            break
            
    if target_idx != -1:
        if target_idx > 0:
            pred = sorted_chunks[target_idx - 1]
        if target_idx < len(sorted_chunks) - 1:
            succ = sorted_chunks[target_idx + 1]
            
    result = {
        "chunk_id": chunk_id,
        "text": doc_text,
        "length": len(doc_text),
        "page_number": page_number,
        "chunk_index": chunk_index,
        "doc_hash": doc_hash,
        "source": source,
        "metadata": doc_meta,
        "neighbors": {
            "predecessor": None,
            "successor": None
        }
    }
    
    if pred:
        overlap_txt = get_overlap(pred["text"], doc_text)
        result["neighbors"]["predecessor"] = {
            "chunk_id": pred["id"],
            "page_number": pred["page_number"],
            "chunk_index": pred["chunk_index"],
            "text": pred["text"],
            "overlap_len": len(overlap_txt),
            "overlap_text": overlap_txt
        }
        
    if succ:
        overlap_txt = get_overlap(doc_text, succ["text"])
        result["neighbors"]["successor"] = {
            "chunk_id": succ["id"],
            "page_number": succ["page_number"],
            "chunk_index": succ["chunk_index"],
            "text": succ["text"],
            "overlap_len": len(overlap_txt),
            "overlap_text": overlap_txt
        }
        
    return result

def print_human_report(analysis: dict):
    """Print human-readable report."""
    print("=" * 80)
    print("CHUNK DIAGNOSTICS REPORT")
    print("=" * 80)
    print(f"Chunk ID      : {analysis['chunk_id']}")
    print(f"Source Document: {analysis['source']}")
    print(f"Document Hash : {analysis['doc_hash']}")
    print(f"Page Number   : {analysis['page_number']}")
    print(f"Chunk Index   : {analysis['chunk_index']}")
    print(f"Text Length   : {analysis['length']} characters")
    print("-" * 80)
    print("CHUNK TEXT:")
    print("-" * 80)
    print(analysis["text"])
    print("-" * 80)
    
    pred = analysis["neighbors"]["predecessor"]
    if pred:
        print("\nPREDECESSOR NEIGHBOR:")
        print(f"  ID         : {pred['chunk_id']}")
        print(f"  Page/Index : Page {pred['page_number']}, Index {pred['chunk_index']}")
        print(f"  Overlap    : {pred['overlap_len']} characters")
        if pred['overlap_len'] > 0:
            print(f"  Overlap Text: {repr(pred['overlap_text'])}")
        print("  Snippet    :")
        print("    " + pred["text"][-200:].replace("\n", "\n    "))
    else:
        print("\nPREDECESSOR NEIGHBOR: None (Start of document)")
        
    succ = analysis["neighbors"]["successor"]
    if succ:
        print("\nSUCCESSOR NEIGHBOR:")
        print(f"  ID         : {succ['chunk_id']}")
        print(f"  Page/Index : Page {succ['page_number']}, Index {succ['chunk_index']}")
        print(f"  Overlap    : {succ['overlap_len']} characters")
        if succ['overlap_len'] > 0:
            print(f"  Overlap Text: {repr(succ['overlap_text'])}")
        print("  Snippet    :")
        print("    " + succ["text"][:200].replace("\n", "\n    "))
    else:
        print("\nSUCCESSOR NEIGHBOR: None (End of document)")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Inspect chunks and context in ChromaDB.")
    parser.add_argument("--query", type=str, help="Find chunk using a similarity query")
    parser.add_argument("--source", type=str, help="Source document name (must be exact)")
    parser.add_argument("--page", type=int, help="1-based page number")
    parser.add_argument("--chunk-id", type=str, help="Explicit chunk ID to look up")
    parser.add_argument("--json", action="store_true", help="Output cleanly as JSON")
    
    args = parser.parse_args()
    
    if not (args.query or args.chunk_id or (args.source and args.page)):
        parser.error("Must specify at least one target: --query, --chunk-id, or (--source AND --page).")
        
    store = get_vector_store()
    
    target_id = None
    target_text = None
    target_meta = None
    
    # 1. Resolve target chunk
    if args.chunk_id:
        result = store._collection.get(ids=[args.chunk_id], include=["documents", "metadatas"])
        if result and result.get("ids"):
            target_id = result["ids"][0]
            target_text = result["documents"][0]
            target_meta = result["metadatas"][0]
    elif args.source and args.page:
        result = store._collection.get(
            where={"$and": [{"source": args.source}, {"page_number": args.page}]},
            include=["documents", "metadatas"]
        )
        if result and result.get("ids"):
            target_id = result["ids"][0]
            target_text = result["documents"][0]
            target_meta = result["metadatas"][0]
    elif args.query:
        hits = retrieve(args.query, top_k=1)
        if hits:
            hit = hits[0]
            target_id = hit.chunk_id
            result = store._collection.get(ids=[target_id], include=["documents", "metadatas"])
            if result and result.get("ids"):
                target_id = result["ids"][0]
                target_text = result["documents"][0]
                target_meta = result["metadatas"][0]
                
    if not target_id or not target_meta:
        if args.json:
            print(json.dumps({"error": "Chunk not found"}, indent=2))
        else:
            print("ERROR: No matching chunk found in the vector store.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Get all document chunks to resolve neighborhood context
    doc_hash = target_meta.get("doc_hash", "")
    all_chunks_res = store._collection.get(
        where={"doc_hash": doc_hash},
        include=["documents", "metadatas"]
    )
    
    all_doc_chunks = []
    if all_chunks_res and all_chunks_res.get("ids"):
        for cid, text, meta in zip(all_chunks_res["ids"], all_chunks_res["documents"], all_chunks_res["metadatas"]):
            all_doc_chunks.append({
                "id": cid,
                "text": text,
                "page_number": int(meta.get("page_number", 0)),
                "chunk_index": int(meta.get("chunk_index", 0))
            })
            
    analysis = analyze_chunk(target_id, target_text, target_meta, all_doc_chunks)
    
    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_human_report(analysis)

if __name__ == "__main__":
    main()
