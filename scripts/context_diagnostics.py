#!/usr/bin/env python
"""Context diagnostics tool to audit context assembly quality and project compression savings."""

import argparse
import os
import re
import sys
from pathlib import Path

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup project root import path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config.settings import get_settings
from app.rag.vector_store import get_vector_store
from app.services.retriever import retrieve
from app.services.context_builder import build_context
from app.eval.benchmark_validator import STOPWORDS
from scripts.chunk_diagnostics import get_overlap

def tokenize(text: str) -> set:
    """Clean and tokenize text to a set of words."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return {w for w in cleaned.split() if w not in STOPWORDS}

def calculate_jaccard(text1: str, text2: str) -> float:
    """Calculate token Jaccard similarity between two texts."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def run_diagnostics(query: str) -> dict:
    """Run search retrieval and perform a simulated context compression audit."""
    store = get_vector_store()
    
    # 1. Fetch chunks for the query
    chunks = retrieve(query, top_k=4)
    if not chunks:
        return {}
        
    # 2. Build original context
    original_assembled = build_context(chunks)
    original_text = original_assembled.context_text
    original_tokens = len(original_text) // 4
    
    # 3. Retrieve global chunk index mapping for consecutiveness
    doc_chunk_indices = {}
    try:
        res = store._collection.get(include=["metadatas"])
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") or []
        
        doc_chunks = {}
        for cid, meta in zip(ids, metadatas):
            if not meta:
                continue
            doc_hash = meta.get("doc_hash", "")
            doc_chunks.setdefault(doc_hash, []).append({
                "id": cid,
                "page": int(meta.get("page_number", 0)),
                "index": int(meta.get("chunk_index", 0))
            })
            
        for doc_hash, clist in doc_chunks.items():
            sorted_chunks = sorted(clist, key=lambda x: (x["page"], x["index"]))
            doc_chunk_indices[doc_hash] = {c["id"]: idx for idx, c in enumerate(sorted_chunks)}
    except Exception:
        pass
        
    def are_consecutive(c1, c2) -> bool:
        if c1.doc_hash != c2.doc_hash:
            return False
        # Try global index map first
        idx_map = doc_chunk_indices.get(c1.doc_hash, {})
        idx1 = idx_map.get(c1.chunk_id)
        idx2 = idx_map.get(c2.chunk_id)
        if idx1 is not None and idx2 is not None:
            return abs(idx1 - idx2) == 1
        # Fallback to page/index check
        return c1.page_number == c2.page_number and abs(c1.chunk_index - c2.chunk_index) == 1

    # 4. Analyze statistics
    duplicate_chunks_count = 0
    low_info_chunks_count = 0
    consecutive_pairs_count = 0
    
    # Check duplicates and low information
    for i, c in enumerate(chunks):
        # Low info
        if len(c.text) < 150 or len(tokenize(c.text)) < 10:
            low_info_chunks_count += 1
        # Duplicate/Near-duplicate of any higher-ranked chunk
        for prev in chunks[:i]:
            if calculate_jaccard(c.text, prev.text) >= 0.8:
                duplicate_chunks_count += 1
                break
                
    # Check consecutiveness
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if are_consecutive(chunks[i], chunks[j]):
                consecutive_pairs_count += 1
            
    # 5. Project compression opportunities
    # Filter low-info and duplicates
    kept_chunks = []
    opportunities = []
    
    for c in chunks:
        is_low_info = len(c.text) < 150 or len(tokenize(c.text)) < 10
        if is_low_info:
            opportunities.append(f"- Remove low-information chunk: {c.source} p{c.page_number}")
            continue
            
        is_dup = False
        for prev in kept_chunks:
            if calculate_jaccard(c.text, prev.text) >= 0.8:
                is_dup = True
                break
        if is_dup:
            opportunities.append(f"- Remove duplicate chunk: {c.source} p{c.page_number}")
            continue
            
        kept_chunks.append(c)
        
    # Group by document to merge consecutive chunks
    by_doc = {}
    for c in kept_chunks:
        by_doc.setdefault(c.doc_hash, []).append(c)
        
    merged_blocks = []
    for doc_hash, doc_list in by_doc.items():
        # Sort chunks from same doc in reading sequence
        sorted_doc_list = sorted(doc_list, key=lambda x: (x.page_number, x.chunk_index))
        i = 0
        while i < len(sorted_doc_list):
            c1 = sorted_doc_list[i]
            merged_text = c1.text
            pages = [c1.page_number]
            
            j = i + 1
            while j < len(sorted_doc_list):
                c2 = sorted_doc_list[j]
                if are_consecutive(sorted_doc_list[j - 1], c2):
                    overlap = get_overlap(merged_text, c2.text)
                    merged_text += c2.text[len(overlap):]
                    pages.append(c2.page_number)
                    j += 1
                else:
                    break
                    
            if len(pages) > 1:
                doc_display = c1.source.replace(".pdf", "")
                pages_str = "+p".join(str(p) for p in sorted(list(set(pages))))
                opportunities.append(f"- Merge {doc_display} p{pages_str}")
                
            merged_blocks.append({
                "source": c1.source,
                "page_number": c1.page_number,
                "chunk_index": c1.chunk_index,
                "text": merged_text
            })
            i = j
        
    # Build simulated compressed context text
    comp_blocks = []
    for pos, mb in enumerate(merged_blocks, start=1):
        header = f"[Source {pos}] (document: {mb['source']}, page: {mb['page_number']}, chunk: {mb['chunk_index']})"
        comp_blocks.append(f"{header}\n{mb['text']}")
        
    compressed_text = "\n\n".join(comp_blocks)
    compressed_tokens = len(compressed_text) // 4 if compressed_text else 0
    
    reduction_pct = 0
    if original_tokens > 0:
        reduction_pct = int(round((1.0 - (compressed_tokens / original_tokens)) * 100.0))
        if reduction_pct < 0:
            reduction_pct = 0
            
    # Extract unique documents
    unique_docs = len({c.source for c in chunks})
    
    return {
        "query": query,
        "chunks": chunks,
        "original_tokens": original_tokens,
        "unique_docs": unique_docs,
        "duplicates": duplicate_chunks_count,
        "low_info": low_info_chunks_count,
        "consecutive": consecutive_pairs_count,
        "opportunities": opportunities,
        "reduction_pct": reduction_pct,
        "original_text": original_text
    }

def print_human_report(diag: dict):
    """Print the formatted console output matching the requirement."""
    print("=================================================")
    print("QUERY")
    print("=================================================")
    print(diag["query"])
    print()
    print("=================================================")
    print("RETRIEVED CHUNKS")
    print("=================================================")
    for idx, c in enumerate(diag["chunks"], start=1):
        print(f"{idx}. {c.source} p{c.page_number} score={c.score:.2f}")
    print()
    print("=================================================")
    print("STATISTICS")
    print("=================================================")
    print(f"Retrieved chunks: {len(diag['chunks'])}")
    print(f"Documents involved: {diag['unique_docs']}")
    print(f"Estimated tokens: {diag['original_tokens']}")
    print(f"Duplicate chunks: {diag['duplicates']}")
    print(f"Low-information chunks: {diag['low_info']}")
    print(f"Consecutive chunks: {diag['consecutive']}")
    print()
    print("Compression opportunities:")
    if diag["opportunities"]:
        for opp in diag["opportunities"]:
            print(opp)
    else:
        print("None identified")
    print(f"Estimated token reduction: {diag['reduction_pct']}%")
    print()
    print("=================================================")
    print("CONTEXT PREVIEW")
    print("=================================================")
    preview = diag["original_text"][:400]
    print(f"{preview}...")

def main():
    parser = argparse.ArgumentParser(description="Run RAG context diagnostics.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--benchmark", action="store_true", help="Force benchmark collection")
    args = parser.parse_args()
    
    settings = get_settings()
    if args.benchmark:
        settings.chroma_persist_dir = _ROOT / "benchmark_chroma"
        settings.chroma_collection_name = "benchmark_corpus"
        os.environ["CHROMA_PERSIST_DIR"] = str(settings.chroma_persist_dir)
        os.environ["CHROMA_COLLECTION_NAME"] = settings.chroma_collection_name
        
    diag = run_diagnostics(args.query)
    if not diag:
        print("No chunks retrieved.")
        return
        
    print_human_report(diag)

if __name__ == "__main__":
    main()
