import re
from typing import Dict, List, Set, Tuple
from app.rag.vector_store import VectorStore
from app.eval.dataset import Case, Benchmark

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could",
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here",
    "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in",
    "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that",
    "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd",
    "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasnt", "we", "wed", "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres",
    "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd",
    "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}

def extract_keywords(query: str) -> Set[str]:
    """Tokenize the query and remove common english stopwords. Preserves hyphens."""
    # Remove standard punctuation but keep alphanumeric and hyphens
    cleaned = re.sub(r'[^\w\s-]', '', query.lower())
    words = cleaned.split()
    
    keywords = set()
    for w in words:
        # Strip leading/trailing hyphens
        w_strip = w.strip("-")
        if w_strip and w_strip not in STOPWORDS and not w_strip.isdigit():
            keywords.add(w_strip)
    return keywords

def get_word_roots(word: str) -> List[str]:
    """Return potential word roots by stripping common English inflections."""
    roots = [word]
    if word.endswith("s"):
        if word.endswith("es"):
            roots.append(word[:-2])
        roots.append(word[:-1])
    if word.endswith("ed"):
        roots.append(word[:-2])
        if word.endswith("ied"):
            roots.append(word[:-3] + "y")
        roots.append(word[:-1])
    if word.endswith("ing"):
        roots.append(word[:-3])
        roots.append(word[:-3] + "e")
    if word.endswith("ly"):
        roots.append(word[:-2])
    if word.endswith("tion"):
        roots.append(word[:-4])
        roots.append(word[:-4] + "te")
    return list(set(roots))

def is_word_in_text(word: str, text: str) -> bool:
    """Return True if any root version of the word appears as a substring in text."""
    roots = get_word_roots(word)
    return any(root in text for root in roots)

def verify_case_answerability(
    case: Case, store: VectorStore, threshold: float = 0.25
) -> Dict[str, object]:
    """Verify that a single case's expectations can be matched in the vector store."""
    if case.is_negative:
        return {
            "case_id": case.id,
            "query": case.query,
            "type": case.type,
            "impossible": False,
            "evidence_score": 1.0,
            "status": "PASS",
            "reason": "Negative case (out-of-corpus validation skipped)"
        }
        
    missing_docs = []
    missing_pages = []
    missing_critical_terms = []
    all_texts = []
    
    q_words = extract_keywords(case.query)
    
    for doc_hash in case.expected_doc_hashes:
        # 1. Fetch chunks for document
        res = store._collection.get(
            where={"doc_hash": doc_hash},
            include=["documents", "metadatas"]
        )
        
        docs = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        
        if not docs:
            missing_docs.append(doc_hash)
            continue
            
        # 2. Filter by expected pages if specified
        if case.expected_pages:
            target_chunks = [
                text for text, meta in zip(docs, metadatas)
                if int(meta.get("page_number", 0)) in case.expected_pages
            ]
            if not target_chunks:
                missing_pages.extend(case.expected_pages)
                continue
            all_texts.extend(target_chunks)
        else:
            all_texts.extend(docs)
            
    # 3. Compute keyword evidence score over the combined text pool
    if missing_docs or missing_pages:
        avg_evidence = 0.0
    else:
        text_pool = " ".join(all_texts).lower()
        if not q_words:
            avg_evidence = 1.0
        else:
            matched_words = {w for w in q_words if is_word_in_text(w, text_pool)}
            avg_evidence = len(matched_words) / len(q_words)
            
            # 4. Critical term presence check
            # Flag words longer than 4 chars or containing a hyphen that are completely missing
            for w in q_words:
                if (len(w) >= 4 or "-" in w) and not is_word_in_text(w, text_pool):
                    missing_critical_terms.append(w)

    
    impossible = False
    status = "PASS"
    reason = "Case is answerable based on corpus contents."
    
    if missing_docs:
        impossible = True
        status = "FAIL_MISSING_DOC"
        reason = f"Missing expected documents in corpus: {missing_docs}"
    elif missing_pages:
        impossible = True
        status = "FAIL_MISSING_PAGES"
        reason = f"Expected pages {missing_pages} are not indexed."
    elif missing_critical_terms:
        impossible = True
        status = "FAIL_MISSING_KEYWORDS"
        reason = f"Critical query keywords {missing_critical_terms} are completely missing from the target document text."
    elif avg_evidence < threshold:
        impossible = True
        status = "FAIL_LOW_EVIDENCE"
        reason = f"Average evidence score ({avg_evidence:.2f}) is below threshold ({threshold:.2f})."
        
    return {
        "case_id": case.id,
        "query": case.query,
        "type": case.type,
        "impossible": impossible,
        "evidence_score": avg_evidence,
        "status": status,
        "reason": reason,
        "missing_docs": missing_docs,
        "missing_pages": missing_pages,
        "missing_critical_terms": missing_critical_terms,
        "keywords_extracted": list(q_words)
    }

def validate_benchmark(
    benchmark: Benchmark, store: VectorStore, threshold: float = 0.25
) -> Dict[str, object]:
    """Run validation across all cases in the benchmark."""
    cases_verdicts = []
    impossible_count = 0
    
    for case in benchmark.cases:
        verdict = verify_case_answerability(case, store, threshold)
        cases_verdicts.append(verdict)
        if verdict["impossible"]:
            impossible_count += 1
            
    return {
        "description": benchmark.description,
        "fingerprint": benchmark.fingerprint(),
        "total_cases": len(benchmark.cases),
        "impossible_cases_count": impossible_count,
        "is_valid": impossible_count == 0,
        "cases": cases_verdicts
    }
