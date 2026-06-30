# Project Status — Document-Grounded RAG Assistant

This document summarizes the current capabilities, limitations, known issues, roadmap, and production readiness checklist of the **Talk to Your Data** application as of `v1.0.0`.

---

## 1. Current Capabilities

- **Multi-PDF Ingestion Pipeline**:
  - Secure uploads with magic-byte check, format allowlist, and file-size constraints.
  - Ingestion steps: Text extraction (PyPDF) -> cleaning -> recursive splitting -> chunk deduplication (SHA-256 hash) -> index storage.
- **Hybrid Retrieval Engine**:
  - Semantic vector search (ChromaDB with cosine similarity and local `BAAI/bge-small-en-v1.5` embeddings).
  - Lexical keyword search (BM25 store).
  - Maximize Marginal Relevance (MMR) query diversification.
- **Query Decomposition & Intent Routing**:
  - Heuristically breaks down multi-part, comparative, or conjunctive questions into sub-queries.
  - Intent router categorizes queries into Calculator, DateTime, Document Stats, Web Search, or RAG intents using high-performance regex rules.
- **Tool-Augmented Context Assembly**:
  - Dynamically runs tools for mathematical computations, datetime checks, file metrics, or DuckDuckGo searches.
  - Combines multiple tool results and RAG document chunks, removing all duplicate passages before feed-in.
- **Context Compressor**:
  - Gated under `CONTEXT_COMPRESSION_ENABLED` configurations.
  - Achieves **20–40% token reduction** by removing duplicates/near-duplicates and conservatively merging adjacent document chunks.
- **Grounded Answer Generation & Citations**:
  - Answers are strictly grounded in context, outputting honest "not found" messages when no matching context is retrieved.
  - Returns sequentially-numbered Source Citations linking back to the document source, page, chunk index, similarity score, and snippet.
- **Offline Evaluation Suite**:
  - Verification benchmarks scoring intent classification accuracy, tool execution success, citation precision/recall, and hallucination rates.

---

## 2. Limitations

- **Stateless/Single-Turn Only**:
  - The chat interface operates as a stateless single-turn search. It does not carry forward conversational context or support multi-turn dialogues (this is scheduled for future agentic sprints).
- **Heuristic Query Decomposition**:
  - Intent parsing relies on regex rules and keywords. While highly efficient (sub-millisecond latencies) and fully offline-capable, it is less flexible than model-based semantic planners.
- **Sequential Executions**:
  - When query decomposition splits a question into multiple sub-queries, execution occurs sequentially. This scales the cumulative execution time relative to the number of sub-queries.

---

## 3. Known Issues & Warnings

- **ChromaDB Telemetry Warnings**:
  - During startup, Chroma DB internals may log telemetry capture warnings:
    `Failed to send telemetry event: capture() takes 1 positional argument but 3 were given`
    *Status*: These warnings are harmless, do not affect application functionality, and can be safely ignored.
- **DuckDuckGo Package Warnings**:
  - When importing the DuckDuckGo search library, a package rename warning is displayed:
    `RuntimeWarning: This package (duckduckgo_search) has been renamed to ddgs! Use pip install ddgs instead.`
    *Status*: The import succeeds and search functions work correctly. This is non-fatal.

---

## 4. Future Work Roadmap

- **Sprint 11**: Agentic RAG Foundation (Multi-agent autonomous execution loops).
- **Sprint 12**: LLM Planner (Reasoning model planning for sub-task routing).
- **Sprint 13**: Self-Evaluation (Pre-release automated validation checks on LLM answers).
- **Sprint 14**: Reflection (Critique-and-refine generation steps to reduce hallucinations).
- **Sprint 15**: Knowledge Graph RAG (Graph DB mapping for complex relational logical searches).
- **Sprint 16**: Multimodal RAG (Image, chart, and schema parsing inside documents).

---

## 5. Production Readiness Checklist

- [x] **Unit Testing**: 204 unit tests fully operational and passing (100% green).
- [x] **Dependency Isolation**: All used libraries declared in `requirements.txt` (including `duckduckgo-search`).
- [x] **Gitignore Setup**: Vector database folders (`chroma_db/`, `benchmark_chroma/`) and document storage directories are correctly ignored.
- [x] **Secrets Scan**: Verified that no Groq API keys, passwords, absolute system paths, or tokens are checked into version control.
- [ ] **Configure Environment Settings**: Paste your active production `GROQ_API_KEY` into the deployment `.env` file before booting the production container.
