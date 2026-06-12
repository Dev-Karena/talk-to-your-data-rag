# 📄 Talk to Your Data — Document-Grounded RAG Assistant

> Upload your PDFs and ask questions in plain English. Every answer is **grounded in your documents** and **cites its exact sources** — document, page, and chunk — so you can trust where it came from.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/Orchestration-LangChain-1C3C3C">
  <img alt="ChromaDB" src="https://img.shields.io/badge/VectorDB-ChromaDB-FCA121">
  <img alt="Groq" src="https://img.shields.io/badge/LLM-Llama%203.3%2070B%20on%20Groq-F55036">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-49%20passing-success">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

---

## 📘 Overview

**Talk to Your Data** is a production-quality, fully local **Retrieval-Augmented Generation (RAG)** application. It turns a pile of PDFs into a conversational knowledge base: you upload documents, the system indexes them into a persistent vector database, and you ask natural-language questions. The LLM answers **only** from the retrieved content and attaches a precise citation to every claim — or honestly says *"I couldn't find this in the provided documents"* when the answer isn't there.

It was built to demonstrate **clean software architecture applied to an AI/ML system**: strict layering, typed and validated configuration, centralized logging, comprehensive error handling, a pluggable embedding backend, and a unit-tested RAG engine that runs **fully offline** except for the answer-generation call.

**Why it stands out**
- 🎯 **Grounded, not hallucinated** — answers are constrained to retrieved context, with an explicit "not found" path.
- 📎 **Verifiable** — every answer maps back to `document · page · chunk · similarity %`.
- 🔌 **Pluggable** — swap the embedding provider (local / Voyage / OpenAI) with a single `.env` change, no code edits.
- 🧪 **Tested & typed** — 49 unit tests, type hints and docstrings throughout, fail-fast config validation.

---

## ✨ Features

- 📥 **Multi-PDF upload** from the sidebar.
- 🧹 **Text extraction + cleaning** — hyphenation repair, whitespace/unicode normalization, control-char stripping.
- ✂️ **Recursive, configurable chunking** with overlap (`RecursiveCharacterTextSplitter`).
- 🔢 **Pluggable embeddings** — local `sentence-transformers` by default (offline, no key), or Voyage AI / OpenAI.
- 🗄️ **Persistent vector store** — ChromaDB on disk; the index survives restarts (no re-embedding on launch).
- 🔍 **Top-K similarity search** with relevance scores.
- 💬 **Chat interface** with streamed answers and conversation history.
- 📎 **Source citations for every answer** — document, page, chunk ID, similarity %, and the exact snippet.
- 🚫 **Honest "not found"** when the answer isn't in your documents — no hallucination.
- ⏭️ **SHA-256 deduplication** — the same PDF is never indexed twice.
- ♻️ **Idempotent upserts** — re-indexing reuses stable chunk IDs; no duplicate vectors.
- 🔄 **Re-index** and 🗑️ **Clear database** controls in the UI.
- 🛡️ **Upload security** — extension allowlist, size cap, and magic-byte sniffing.
- 🔐 **Secrets only in `.env`** — never hardcoded, never sent to the frontend.

---

## 🏛️ Architecture

The app uses **clean, layered architecture** with a strict one-way dependency rule: `UI → Services → RAG → Config/Utils`. Lower layers never import upper ones, which keeps the RAG engine testable without the UI and the LLM swappable without touching the frontend.

```
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION   app/ui/        Streamlit: upload, chat, cite  │
├─────────────────────────────────────────────────────────────┤
│  SERVICE        app/services/  retrieve → build context →     │
│                                generate answer  (facade)      │
├─────────────────────────────────────────────────────────────┤
│  RAG PIPELINE   app/rag/       load → clean → chunk → embed   │
│                                → persist → search             │
├─────────────────────────────────────────────────────────────┤
│  CROSS-CUTTING  app/config/ (settings)   app/utils/ (logging, │
│                                           validation)         │
└─────────────────────────────────────────────────────────────┘
```

**Design principles applied throughout:** modular structure, type hints, docstrings, centralized logging, comprehensive error handling, typed configuration, and a clean facade (`rag_service`) that the UI depends on.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 / 3.11 |
| **UI** | Streamlit (chat, sidebar upload, status badges, source cards) |
| **LLM** | Llama 3.3 70B on **Groq** via `langchain-groq` (`ChatGroq`) |
| **Embeddings** | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) — pluggable to Voyage AI / OpenAI |
| **Vector DB** | **ChromaDB** (local, persistent, cosine similarity) |
| **Orchestration** | LangChain — `RecursiveCharacterTextSplitter`, `PyPDFLoader` |
| **PDF parsing** | `pypdf` |
| **Config & validation** | `pydantic` + `pydantic-settings` (typed, validated `.env`) |
| **Testing** | `pytest` (49 tests, fully offline) |

---

## 📂 Folder Structure

```
talk-to-your-data/
│
├── app/
│   ├── config/
│   │   └── settings.py          # Typed settings loaded & validated from .env
│   ├── utils/
│   │   ├── logger.py            # Centralized logging (console + rotating file)
│   │   └── validators.py        # Upload security + content-hash dedupe key
│   ├── rag/
│   │   ├── loader.py            # PDF → text + per-page metadata
│   │   ├── cleaner.py           # Deterministic text normalization
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter wrapper
│   │   ├── embeddings.py        # Pluggable embedding factory (local/voyage/openai)
│   │   ├── vector_store.py      # ChromaDB persistence, search, dedupe, clear
│   │   └── pipeline.py          # Ingestion orchestrator (load→…→store)
│   ├── services/
│   │   ├── retriever.py         # Top-K similarity search
│   │   ├── context_builder.py   # Numbered, citation-ready context + sources
│   │   ├── llm_client.py        # ChatGroq wrapper (Llama 3.3 70B on Groq)
│   │   └── rag_service.py       # Public facade: answer_question()
│   └── ui/
│       ├── streamlit_app.py     # Main UI: sidebar, chat, citations, controls
│       └── components.py        # Reusable render helpers
│
├── chroma_db/                   # Persistent vector store (gitignored)
├── documents/                   # Uploaded PDFs (gitignored)
├── tests/                       # Unit tests (pytest)
│
├── .env.example                 # Config template (safe to commit)
├── .gitignore
├── requirements.txt
├── main.py                      # Entry point → launches Streamlit
├── LICENSE
└── README.md
```

---

## ⚙️ Installation Guide

> **Prerequisites:** [Anaconda](https://www.anaconda.com/download) (or any Python ≥ 3.10) and a terminal. Tested on Windows with Python 3.10 / 3.11.

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
> The first run downloads the local embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) once, then caches it.

---

## 🔐 Environment Setup

Copy the template and add your key:

```powershell
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
```

Open `.env` and paste your Groq API key:

```ini
GROQ_API_KEY=gsk_your-real-key-here
```

> Get a **free** key at <https://console.groq.com/keys>.
> With the **default local embeddings**, the Groq key is the **only** secret you need.

**Key configuration options** (full annotated list in `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model for answer generation. |
| `EMBEDDING_BACKEND` | `local` | `local` / `voyage` / `openai`. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for the chosen backend. |
| `CHUNK_SIZE` | `1000` | Characters per chunk. |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks (~15%). |
| `TOP_K` | `4` | Chunks retrieved per question. |
| `MAX_FILE_SIZE_MB` | `25` | Per-file upload size cap. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## ▶️ Running the Application

```powershell
streamlit run main.py
```

or simply:

```powershell
python main.py
```

The app opens at <http://localhost:8501>.

---

## 📖 Usage Instructions

1. **Upload** one or more PDFs from the sidebar.
2. Click **📌 Index uploaded files** — the app validates, deduplicates, chunks, embeds, and stores them.
3. **Ask a question** in the chat box.
4. Read the **streamed answer**, then expand **📎 Sources** to see each cited passage with its `document · page · chunk · similarity %`.
5. Use **🔄 Re-index** to refresh, or **🗑️ Clear database** to start fresh.

> If a PDF is scanned/image-only, the app detects it and reports that no extractable text was found (OCR is on the roadmap).

---

## 🖼️ Screenshots

> Replace the placeholders below with real captures (drop the images in a `docs/` or `assets/` folder and update the paths).

| Upload & Indexing | Chat with Citations |
|---|---|
| ![Upload screen](docs/screenshot-upload.png) | ![Chat with cited answer](docs/screenshot-chat.png) |

| Source Citations Panel | Indexed Documents & Controls |
|---|---|
| ![Source citations](docs/screenshot-sources.png) | ![Sidebar controls](docs/screenshot-sidebar.png) |

---

## 🧠 RAG Pipeline Explanation

### A) Ingestion / Indexing
```
 User uploads PDF(s)  ──►  [validators]  size? type? magic bytes? → SHA-256 hash
                                  │ (reject invalid; skip already-indexed hash)
                                  ▼
                            [loader]   PDF → text + per-page metadata
                                  ▼
                            [cleaner]  normalize whitespace, repair hyphenation
                                  ▼
                            [chunker]  RecursiveCharacterTextSplitter
                                  │      → chunks + {source, page, chunk_id, hash}
                                  ▼
                          [embeddings] embed each chunk (local / voyage / openai)
                                  ▼
                         [vector_store] Chroma.upsert(ids, vectors, docs, metadata)
                                  ▼
                            Persisted to  chroma_db/   ✅ survives restarts
```

### B) Query / Answer
```
 User question  ──►  [retriever]   embed query → Chroma top-K → chunks + scores
                            ▼
                    [context_builder]  number + label sources  [Source N]
                            ▼
                      [llm_client]  Groq · Llama 3.3 70B
                            │   system prompt: answer ONLY from context,
                            │   cite [Source N], say "not found" otherwise
                            ▼
                      [rag_service]  → answer + structured citations
                            ▼
                 Streamlit:  streamed answer  +  history  +  📎 Sources
```

**Design choices that drive quality**
- **Chunking — `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150` (~15%):** large enough to hold a coherent idea, small enough to keep each embedding focused; overlap preserves facts split across boundaries. `RecursiveCharacterTextSplitter` prefers natural separators (paragraph → line → sentence → word).
- **Retrieval — `TOP_K=4`:** enough grounding without flooding the prompt with marginal text.
- **Citations:** the numbered `[Source N]` markers in the prompt are kept **1:1 in lockstep** with a structured `SourceCitation` list, so a `[Source 2]` reference in the answer maps back to a concrete document/page/chunk/score.
- **Cosine + normalized embeddings:** the Chroma collection uses cosine distance to match the normalized vectors the embedders produce; scores are reported as `1 − distance`.

---

## 💡 Example Questions

Once you've indexed a document, try questions like:

- *"Summarize this document in five bullet points."*
- *"What were the main findings or conclusions?"*
- *"What methodology was used, and on what data?"*
- *"List every limitation the authors mention."*
- *"What does the report say about [specific topic]?"*
- *"Which page discusses [term], and what does it say?"*
- *"Compare the recommendations in section 3 with section 5."*

> Ask something **not** covered by your documents and the assistant will respond *"I could not find this information in the provided documents"* — by design, it won't make things up.

---

## 🚀 Future Improvements

- 🔎 **Hybrid search** — combine dense vectors with BM25/keyword recall for exact terms.
- 🏷️ **Re-ranking** — a cross-encoder over the top-K to sharpen relevance ordering.
- 🖼️ **OCR support** — Tesseract / a vision model for scanned, image-only PDFs.
- 📚 **More formats** — DOCX, TXT, HTML, Markdown ingestion.
- 🧵 **Streaming citations** — highlight cited passages inline as the answer streams.
- 👥 **Multi-collection / per-user isolation** — separate namespaces per document set or user.
- 🧪 **Evaluation harness** — RAGAS-style metrics (faithfulness, answer relevance, context precision) in CI.
- 🗂️ **Metadata filtering** — restrict retrieval to specific documents, dates, or sections.
- 🌐 **Docker packaging** — one-command reproducible deployment.
- 💬 **Conversational memory** — follow-up questions via query rewriting / context carry-over.

---

## 🧪 Testing

```powershell
pytest -v
```

The suite (**49 tests**) covers validation, cleaning, chunking, the vector store (against a temporary Chroma instance), and the RAG service facade (with the retriever and LLM stubbed). It runs **fully offline** — no API key or network required.

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](./LICENSE). Review the licenses of bundled dependencies (Streamlit, ChromaDB, LangChain, langchain-groq, sentence-transformers) before redistribution.

---

<p align="center">
  <em>Built with clean architecture, type hints, docstrings, logging, error handling, and tests — designed to be read, extended, and trusted.</em>
</p>
```
