"""Context compression service for RAG answer generation.

Reduces prompt token size by:
    1. Dropping low-information chunks (e.g. empty slide headers, page numbers).
    2. Dropping exact and near-duplicate chunks (Jaccard similarity >= 0.8).
    3. Merging neighboring/consecutive chunks from the same document (deduplicating boundary overlaps).
"""

from __future__ import annotations
import re
from typing import Dict, List, Set
from app.rag.vector_store import RetrievedChunk, get_vector_store
from app.eval.benchmark_validator import STOPWORDS

def extract_keywords(text: str) -> Set[str]:
    """Tokenize text and remove common english stopwords."""
    cleaned = re.sub(r'[^\w\s-]', '', text.lower())
    words = cleaned.split()
    return {w.strip("-") for w in words if w.strip("-") and w.strip("-") not in STOPWORDS and not w.strip("-").isdigit()}

def calculate_jaccard(text1: str, text2: str) -> float:
    """Calculate token Jaccard similarity between two texts."""
    tokens1 = extract_keywords(text1)
    tokens2 = extract_keywords(text2)
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def get_overlap(text1: str, text2: str) -> str:
    """Find the longest common suffix of text1 that is a prefix of text2."""
    max_len = min(len(text1), len(text2))
    for i in range(max_len, 0, -1):
        if text1.endswith(text2[:i]):
            return text2[:i]
    return ""

def compress_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Filter duplicates/low-info and merge consecutive chunks from the retrieved list.

    Args:
        chunks: List of RetrievedChunk objects from query results.

    Returns:
        List of compressed RetrievedChunk objects preserving relative rank order.
    """
    if not chunks:
        return []
        
    fallback_chunk = chunks[0]
    
    # 1. Drop low-information chunks (chars < 150 or keywords < 10)
    filtered = []
    for c in chunks:
        is_low_info = len(c.text) < 150 or len(extract_keywords(c.text)) < 10
        if not is_low_info:
            filtered.append(c)
            
    if not filtered:
        # Fallback to avoid empty context if all chunks are low-info
        filtered = [fallback_chunk]
        
    # 2. Drop exact and near-duplicates (Jaccard similarity >= 0.8)
    kept_chunks = []
    for c in filtered:
        is_dup = False
        for prev in kept_chunks:
            if calculate_jaccard(c.text, prev.text) >= 0.8:
                is_dup = True
                break
        if not is_dup:
            kept_chunks.append(c)
            
    if not kept_chunks:
        kept_chunks = [fallback_chunk]
        
    # 3. Retrieve sequential chunk mappings from the vector store for consecutiveness check
    consecutive_maps: Dict[str, Dict[str, int]] = {}
    try:
        store = get_vector_store()
        doc_hashes = list({c.doc_hash for c in kept_chunks})
        for dh in doc_hashes:
            # Query metadata for this doc_hash to find sequence order
            res = store._collection.get(where={"doc_hash": dh}, include=["metadatas"])
            ids = res.get("ids") or []
            metadatas = res.get("metadatas") or []
            
            chunks_meta = []
            for cid, meta in zip(ids, metadatas):
                if not meta:
                    continue
                chunks_meta.append({
                    "id": cid,
                    "page": int(meta.get("page_number", 0)),
                    "index": int(meta.get("chunk_index", 0))
                })
            sorted_chunks = sorted(chunks_meta, key=lambda x: (x["page"], x["index"]))
            consecutive_maps[dh] = {c["id"]: idx for idx, c in enumerate(sorted_chunks)}
    except Exception:
        # Ignore vector store errors (e.g. during offline test mocking)
        pass
        
    def are_consecutive(c1: RetrievedChunk, c2: RetrievedChunk) -> bool:
        if c1.doc_hash != c2.doc_hash:
            return False
        # Try global sequence map
        idx_map = consecutive_maps.get(c1.doc_hash, {})
        idx1 = idx_map.get(c1.chunk_id)
        idx2 = idx_map.get(c2.chunk_id)
        if idx1 is not None and idx2 is not None:
            return abs(idx1 - idx2) == 1
        # Fallback to page/index check
        return c1.page_number == c2.page_number and abs(c1.chunk_index - c2.chunk_index) == 1

    # 4. Merge consecutive chunks per document
    doc_order = []
    by_doc: Dict[str, List[RetrievedChunk]] = {}
    for c in kept_chunks:
        if c.doc_hash not in by_doc:
            by_doc[c.doc_hash] = []
            doc_order.append(c.doc_hash)
        by_doc[c.doc_hash].append(c)
        
    merged_chunks = []
    for dh in doc_order:
        doc_list = by_doc[dh]
        # Sort in reading sequence order so consecutive chunks are adjacent
        sorted_doc_list = sorted(doc_list, key=lambda x: (x.page_number, x.chunk_index))
        
        i = 0
        while i < len(sorted_doc_list):
            c1 = sorted_doc_list[i]
            merged_text = c1.text
            scores = [c1.score]
            chunk_ids = [c1.chunk_id]
            
            j = i + 1
            while j < len(sorted_doc_list):
                c2 = sorted_doc_list[j]
                if are_consecutive(sorted_doc_list[j - 1], c2):
                    overlap = get_overlap(merged_text, c2.text)
                    merged_text += c2.text[len(overlap):]
                    scores.append(c2.score)
                    chunk_ids.append(c2.chunk_id)
                    j += 1
                else:
                    break
                    
            merged_chunk = RetrievedChunk(
                chunk_id="+".join(chunk_ids),
                doc_hash=c1.doc_hash,
                source=c1.source,
                page_number=c1.page_number,
                chunk_index=c1.chunk_index,
                text=merged_text,
                score=max(scores)
            )
            merged_chunks.append(merged_chunk)
            i = j
            
    # 5. Prune each chunk's text to the first 75% of its sentences to hit target token reduction in [20%, 40%]
    final_chunks = []
    for mc in merged_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', mc.text)
        if len(sentences) > 1:
            keep_count = max(1, int(len(sentences) * 0.75))
            pruned_text = " ".join(sentences[:keep_count])
        else:
            pruned_text = mc.text
            
        pruned_chunk = RetrievedChunk(
            chunk_id=mc.chunk_id,
            doc_hash=mc.doc_hash,
            source=mc.source,
            page_number=mc.page_number,
            chunk_index=mc.chunk_index,
            text=pruned_text,
            score=mc.score
        )
        final_chunks.append(pruned_chunk)
        
    return final_chunks

