# 📄 Talk to Your Data — Document-Grounded RAG Assistant

> Upload your PDFs and ask questions in plain English. Every answer is **grounded in your documents** and **cites its exact sources** — document, page, and chunk — or calls real-time utility tools.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/Orchestration-LangChain-1C3C3C">
  <img alt="ChromaDB" src="https://img.shields.io/badge/VectorDB-ChromaDB-FCA121">
  <img alt="Groq" src="https://img.shields.io/badge/LLM-Llama%203.3%2070B%20on%20Groq-F55036">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-204%20passing-success">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

---

## 📘 Overview

**Talk to Your Data** is a production-quality, fully local **Retrieval-Augmented Generation (RAG)** application. It turns a pile of PDFs into a conversational knowledge base: you upload documents, the system indexes them into a persistent vector database, and you ask natural-language questions. The LLM answers **only** from the retrieved content or calls built-in deterministic tools (Calculator, DateTime, Document Stats, Web Search) to augment its answers.

It is built on clean software architecture principles, featuring pluggable embedding backends, hybrid search, context compression, citation generation, and a robust offline-friendly evaluation framework.

---

## ✨ Features

- 📥 **Multi-PDF Ingestion** — Upload and index multiple files simultaneously.
- 🔍 **Hybrid Retrieval** — Combines dense semantic vector search (ChromaDB) with lexical keyword matching (BM25).
- ⚖️ **MMR Diversification** — Maximal Marginal Relevance (MMR) ensures retrieval results are non-redundant and cover diverse sections.
- 🗜️ **Context Compressor** — Removes exact/near duplicates, merges adjacent chunks, and trims low-information text to reduce prompt size by 20–40% without losing critical facts.
- 🔀 **Query Decomposition** — Splits complex conjunctive, comparative, or multi-part questions into sub-queries.
- 🛠️ **Tool Calling** — Supports deterministic heuristics for Calculator, DateTime, Document Stats, and Web Search (DuckDuckGo).
- 🤝 **Tool + RAG Integration** — Seamlessly combines tool executions and document chunks into a single, unified generation pipeline.
- 📎 **Citation Generation** — Professional source citations showing exact pages and matching text snippets.
- 🧪 **Evaluation Framework** — Metrics for Intent Router accuracy, tool execution success rate, citation precision/recall, and hallucination rate.
- 📊 **Benchmarking** — Automated scripts to test pipeline changes against ground truth case files.
- 🩺 **Diagnostics** — Run-time visualization of duplicates, token usage, and consecutive chunks.
- 👁️ **Faithfulness Evaluation** — Automated checks to measure groundedness, context utilization, and compression ratio.

---

## 🏛️ Architecture Diagram

```
       User Question
            │
            ▼
        [Planner]
            │
            ▼
     [Tool Router] ──────(Deterministic Heuristics)
            │
      ┌─────┴──────────────┐
      ▼                    ▼
  [Retriever]       [Utility Tools] (Calculator, DateTime, Web Search, etc.)
      │                    │
      ▼                    │
[Hybrid Retrieval]         │
 (Dense + BM25)            │
      │                    │
      ▼                    ▼
   [Context Assembler / Compressor]
            │
            ▼
          [LLM] (Llama 3.3 70B via Groq)
            │
            ▼
     Answer + Citations
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 / 3.11 |
| **UI** | Streamlit (chat interface, upload controls, citation panels) |
| **LLM** | Llama 3.3 70B on **Groq** via `langchain-groq` (`ChatGroq`) |
| **Embeddings** | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) — pluggable to Voyage AI / OpenAI |
| **Vector DB** | **ChromaDB** (local, persistent, cosine similarity) |
| **Lexical Search** | `rank-bm25` (BM25 keyword search) |
| **Orchestration** | LangChain — `RecursiveCharacterTextSplitter`, `PyPDFLoader` |
| **PDF parsing** | `pypdf` |
| **Config & validation** | `pydantic` + `pydantic-settings` (typed settings loaded from `.env`) |
| **Testing** | `pytest` (204 tests, fully offline) |

---

## ⚙️ Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/talk-to-your-data.git
cd talk-to-your-data
```

### 2. Create and activate an environment
```powershell
conda create -n talk-to-your-data python=3.11 -y
conda activate talk-to-your-data
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Copy the template to create a `.env` file:
```powershell
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
```

Paste your Groq API key inside `.env`:
```ini
GROQ_API_KEY=gsk_your-real-key-here
```

---

## ▶️ Running the Application

Start the web application interface:
```powershell
streamlit run main.py
```
Or use the launcher:
```powershell
python main.py
```
The app will open automatically at <http://localhost:8501>.

---

## 📊 Running Benchmarks

Evaluate context compression and retrieval metrics:
```powershell
$env:PYTHONPATH="."
python scripts/benchmark_sprint9.py
```

Evaluate tool-calling accuracy, router routing tables, and latencies:
```powershell
python scripts/run_tool_eval.py
```

Evaluate mixed tool + RAG queries:
```powershell
python scripts/run_mixed_eval.py
```

---

## 🩺 Running Diagnostics

Run diagnostics on retrieve quality and duplicate percentages:
```powershell
python scripts/context_diagnostics.py "How does virtual memory work?"
```

---

## 📂 Folder Structure

```
talk-to-your-data/
│
├── app/
│   ├── config/
│   │   └── settings.py          # Configuration loading & validation
│   ├── utils/
│   │   ├── logger.py            # Central logging config
│   │   └── validators.py        # Upload size & magic-byte checkers
│   ├── rag/
│   │   ├── cleaner.py           # Text cleaning rules
│   │   ├── chunker.py           # Text splitter wrapper
│   │   ├── embeddings.py        # Embedding model factory
│   │   ├── bm25_store.py        # Lexical keyword storage
│   │   ├── vector_store.py      # ChromaDB storage layer
│   │   └── pipeline.py          # Ingestion pipelines
│   ├── tools/
│   │   ├── base_tool.py         # Abstract base tool
│   │   ├── router.py            # Heuristic intent router
│   │   ├── calculator_tool.py   # Inline math calculator
│   │   ├── datetime_tool.py     # System time provider
│   │   ├── document_stats_tool.py # File counts provider
│   │   └── web_search_tool.py   # DuckDuckGo search integration
│   ├── services/
│   │   ├── retriever.py         # Dense/BM25 retrieval engine
│   │   ├── context_compressor.py # Prompt optimizer
│   │   ├── context_assembler.py # Sub-query aggregator
│   │   ├── answer_generator.py  # Prompt generation controller
│   │   └── rag_service.py       # Facade interface
│   └── eval/
│       ├── metrics.py           # Groundedness/Precision/Recall formulas
│       ├── tool_runner.py       # Tool evaluation execution loops
│       └── faithfulness_metrics.py # Hallucination scoring checks
│
├── benchmarks/                  # Ground truth dataset directories
├── docs/                        # Specifications and audit/evaluation reports
├── tests/                       # Unit tests (pytest)
├── requirements.txt             # Python dependencies
├── main.py                      # Streamlit entry point
├── LICENSE
└── README.md
```

---

## 🧪 Evaluation Metrics

Our framework measures:
1. **Recall@4** (Semantic Search recall)
2. **Context Precision** (Relevance of retrieved chunks)
3. **Citation Precision & Recall** (Accurate citation matching)
4. **Hallucination Rate** (Percentage of answers not grounded in retrieved contexts)
5. **Groundedness Score** (Answer alignment with documents)
6. **Compression Effectiveness** (Token reduction percentage)
7. **Intent Router Accuracy** (Intent routing classification success)

---

## 🚀 Future Roadmap

- **Sprint 11**: Agentic RAG — multi-agent autonomous execution loops.
- **Sprint 12**: LLM Planner — reasoning models for sub-task routing.
- **Sprint 13**: Self-Evaluation — real-time verification of answers before returning.
- **Sprint 14**: Reflection — critique-and-refine generation steps.
- **Sprint 15**: Knowledge Graph RAG — graph DB lookups for complex relational logic.
- **Sprint 16**: Multimodal RAG — image/chart parsing in documents.

---

## 🖼️ Screenshots Placeholder

*Screenshots demonstrating upload, chat logs, source citation expansion cards, and sidebar system configurations will be placed here prior to release.*

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for detail.
