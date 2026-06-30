import math
from typing import Dict, List, Tuple
from app.rag.vector_store import VectorStore
from scripts.chunk_diagnostics import get_overlap

def audit_corpus(store: VectorStore) -> Dict[str, object]:
    """Audit the vector store collection for structure, stats, and anomalies."""
    try:
        result = store._collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        return {
            "error": f"Failed to retrieve chunks from database: {exc}",
            "is_healthy": False
        }
        
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    
    total_chunks = len(ids)
    if total_chunks == 0:
        return {
            "total_chunks": 0,
            "is_healthy": True,
            "message": "Vector store is empty."
        }
        
    lengths = [len(doc) for doc in documents]
    avg_length = sum(lengths) / total_chunks
    min_length = min(lengths)
    max_length = max(lengths)
    
    # Identify short chunks
    short_chunks = []
    for cid, doc, meta in zip(ids, documents, metadatas):
        if len(doc) < 50:
            short_chunks.append({
                "chunk_id": cid,
                "length": len(doc),
                "source": meta.get("source", "unknown") if meta else "unknown",
                "page": meta.get("page_number", 0) if meta else 0
            })
            
    # Check metadata integrity
    missing_metadata = []
    required_keys = {"chunk_id", "source", "page_number", "doc_hash", "chunk_index"}
    for cid, meta in zip(ids, metadatas):
        if not meta:
            missing_metadata.append({"chunk_id": cid, "missing_keys": list(required_keys)})
            continue
        missing = [k for k in required_keys if k not in meta]
        if missing:
            missing_metadata.append({"chunk_id": cid, "missing_keys": missing})
            
    # Duplicate text chunks check
    seen_texts = {}
    duplicates = []
    for cid, doc, meta in zip(ids, documents, metadatas):
        source = meta.get("source", "unknown") if meta else "unknown"
        if doc in seen_texts:
            duplicates.append({
                "chunk_id_1": seen_texts[doc]["id"],
                "source_1": seen_texts[doc]["source"],
                "chunk_id_2": cid,
                "source_2": source
            })
        else:
            seen_texts[doc] = {"id": cid, "source": source}
            
    # Sort chunks per document to compute overlap stats
    doc_chunks: Dict[str, List[dict]] = {}
    for cid, doc, meta in zip(ids, documents, metadatas):
        if not meta:
            continue
        doc_hash = meta.get("doc_hash", "unknown")
        doc_chunks.setdefault(doc_hash, []).append({
            "id": cid,
            "text": doc,
            "page_number": int(meta.get("page_number", 0)),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "source": meta.get("source", "unknown")
        })
        
    overlap_lengths = []
    overlap_by_doc = {}
    
    for doc_hash, chunks in doc_chunks.items():
        sorted_chunks = sorted(chunks, key=lambda x: (x["page_number"], x["chunk_index"]))
        doc_overlaps = []
        for i in range(1, len(sorted_chunks)):
            pred = sorted_chunks[i - 1]
            curr = sorted_chunks[i]
            # Calculate overlap only on the same page to be consistent
            if pred["page_number"] == curr["page_number"]:
                overlap_text = get_overlap(pred["text"], curr["text"])
                doc_overlaps.append(len(overlap_text))
                overlap_lengths.append(len(overlap_text))
        if doc_overlaps:
            source_name = sorted_chunks[0]["source"]
            overlap_by_doc[source_name] = {
                "avg_overlap": sum(doc_overlaps) / len(doc_overlaps),
                "min_overlap": min(doc_overlaps),
                "max_overlap": max(doc_overlaps)
            }
            
    avg_overlap = sum(overlap_lengths) / len(overlap_lengths) if overlap_lengths else 0.0
    min_overlap = min(overlap_lengths) if overlap_lengths else 0
    max_overlap = max(overlap_lengths) if overlap_lengths else 0
    
    # Health verdict
    is_healthy = len(missing_metadata) == 0 and len(short_chunks) == 0
    
    # Document level summaries
    doc_summaries = {}
    for doc_hash, chunks in doc_chunks.items():
        source_name = chunks[0]["source"]
        doc_summaries[source_name] = {
            "chunk_count": len(chunks),
            "pages_indexed": len(set(c["page_number"] for c in chunks))
        }
        
    return {
        "total_chunks": total_chunks,
        "length_stats": {
            "avg": round(avg_length, 2),
            "min": min_length,
            "max": max_length
        },
        "overlap_stats": {
            "avg": round(avg_overlap, 2),
            "min": min_overlap,
            "max": max_overlap,
            "by_document": overlap_by_doc
        },
        "anomalies": {
            "short_chunks_count": len(short_chunks),
            "short_chunks": short_chunks[:10],  # Limit detail to top 10
            "missing_metadata_count": len(missing_metadata),
            "missing_metadata": missing_metadata[:10],
            "duplicates_count": len(duplicates),
            "duplicates": duplicates[:10]
        },
        "document_summaries": doc_summaries,
        "is_healthy": is_healthy
    }
