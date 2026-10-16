# Agentic Document Intelligence & Corrective RAG (CRAG) Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph_0.2+-orange.svg)
![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant-red.svg)
![Embeddings](https://img.shields.io/badge/Embeddings-FastEmbed_(BGE--small--en)-green.svg)
![Reranker](https://img.shields.io/badge/Reranker-FlashRank_(TinyBERT)-purple.svg)
![LLM](https://img.shields.io/badge/LLM-Groq_(GPT--OSS_120B)-yellow.svg)
![Streaming](https://img.shields.io/badge/Streaming-SSE_Token--by--Token-brightgreen.svg)
![Frontend](https://img.shields.io/badge/Frontend-React_18_+_MUI-61DAFB.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An **enterprise-grade Agentic Retrieval-Augmented Generation (RAG)** platform for multi-format document question answering. Built from scratch to go beyond naive RAG — implements a **self-corrective LangGraph CRAG workflow** with layout-aware parsing, local ONNX embeddings, cross-encoder reranking, relevance grading, iterative query rewriting, real-time SSE token streaming, and strict citation-grounded answers.

> 🔗 **GitHub:** [github.com/AkSankhyan/Document_QA_Agent](https://github.com/AkSankhyan/Document_QA_Agent)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM Inference** | Groq API (`openai/gpt-oss-120b`, `openai/gpt-oss-20b` fallback) |
| **Agent Orchestration** | LangGraph 0.2+ (Cyclic State Graph) |
| **Vector Database** | Qdrant (local on-disk, session-isolated collections) |
| **Embeddings** | FastEmbed — `BAAI/bge-small-en-v1.5` via ONNX (CPU, zero-GPU) |
| **Reranking** | FlashRank — `ms-marco-TinyBERT-L-2-v2` cross-encoder |
| **Document Parsing** | PyMuPDF4LLM, python-docx, python-pptx, openpyxl |
| **Backend API** | FastAPI + Uvicorn (async, SSE streaming) |
| **Frontend** | React 18, MUI v5, Framer Motion, ReactMarkdown |
| **Streaming** | Server-Sent Events (SSE) — token-by-token real-time output |

---

## 🏗️ Architecture & Agentic Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. INGESTION PIPELINE                           │
└────────────────────────────────────────────────────────────────────────┘
 [User Document] (PDF / DOCX / PPTX / XLSX / TXT / CSV / MD)
        │
        ▼
 [Layout & Table Parser] (PyMuPDF4LLM / Python-Docx / Python-PPTX)
        │ ── Preserves markdown tables, headers, and slide notes natively
        ▼
 [Recursive Token Splitter] ──► Chunks (600 tokens, 100 overlap)
        │                        Attaches metadata: {source, page, chunk_id}
        ▼
 [FastEmbed ONNX] ──► 384-dim dense vectors (BAAI/bge-small-en-v1.5)
        │
        ▼
 [Qdrant Vector DB] ──► Persistent, session-isolated vector collections


┌────────────────────────────────────────────────────────────────────────┐
│               2. CORRECTIVE RAG (CRAG) WORKFLOW (LANGGRAPH)            │
└────────────────────────────────────────────────────────────────────────┘
                      [User Question]
                             │
               ┌─── Intent Detection ───┐
               │  (Conversational?)     │
               ▼                        ▼
          [Instant reply]         [RAG Pipeline]
          (no vector search)            │
                                        ▼
                               ┌─────────────────┐
                               │  Retrieve Node  │ ◄── Top-12 Qdrant Cosine Search
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │   Rerank Node   │ ◄── FlashRank Cross-Encoder (→ Top-6)
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                          ┌───►│ Document Grader │ ◄── Fast LLM evaluates relevance
                          │    └────────┬────────┘
                          │             │
                   (Irrelevant)         ├── Relevant? ──► No, Retry<1 ──► [Rewrite Query]
                          │             │                                        │
                          │             ▼ Yes                                    │
                          │    ┌──────────────────┐                              │
                          │    │ Answer Generator │ ◄────────────────────────────┘
                          │    │  (SSE Streaming) │ ◄── Today's date injected (date math)
                          │    └────────┬─────────┘
                          │             ▼
                          │   [Streamed Tokens + Page Citations]
```

---

## ⚡ Key Engineering Features

### 1. Real-Time SSE Token Streaming
- **`POST /api/chat/stream`** delivers answer tokens as they are generated via Server-Sent Events.
- React frontend reads `ReadableStream` and updates the chat bubble **token-by-token**, eliminating perceived latency — identical UX to ChatGPT.

### 2. Conversational Intent Detection (Pre-RAG Guard)
- Regex classifier short-circuits greetings, thanks, and farewells **before** touching the vector store — instant reply, zero compute cost.

### 3. Date-Aware Reasoning
- Today's date is injected into every generation prompt — enables the LLM to calculate experience durations, ages, and time-elapsed directly from dates found in the document.

### 4. Layout & Table-Aware Parsing
- **PDF**: `PyMuPDF4LLM` preserves tables as Markdown, maintaining column relationships intact.
- **Office formats**: Pure-Python `python-docx`, `python-pptx`, `openpyxl` — zero Pandoc/OS binary dependencies.

### 5. Multi-Stage Retrieval & Local Reranking
- **Dense Vector Search**: Qdrant persistent local storage, Top-12 candidates retrieved.
- **Zero-API-Cost Embeddings**: `BAAI/bge-small-en-v1.5` via FastEmbed ONNX on CPU.
- **Cross-Encoder Reranking**: FlashRank rescores Top-12 → Top-6, improving Precision@K without GPU.

### 6. Self-Corrective Agent Loops (LangGraph)
- **Grader Node**: Rejects low-quality retrieved context, triggers query reformulation instead of hallucinating.
- **Query Rewriter Node**: Converts ambiguous natural-language questions into semantically precise search queries.

### 7. Automatic Model Fallback
- If primary LLM returns 404 / unavailable, agent silently retries `gpt-oss-20b` → `qwen3.8-27b` — zero user disruption.

---

## 📊 Naive RAG vs. This Agentic CRAG System

| Feature | Naive RAG | This System |
| :--- | :--- | :--- |
| **Parsing** | `PyPDF2` (corrupts tables) | **PyMuPDF4LLM** (Markdown tables preserved) |
| **Vector Storage** | None (text stuffed into prompt) | **Qdrant** session-isolated collections |
| **Reranking** | None | **FlashRank** cross-encoder (Top-12 → Top-6) |
| **Agent Logic** | Linear A → B | **Cyclic LangGraph** with Grader + Rewriter |
| **Streaming** | Full response at once | **SSE token streaming** (ChatGPT-like UX) |
| **Date Reasoning** | Not possible | **Today's date injected** — calculates durations |
| **Conversational** | Full RAG pipeline every time | **Intent detection** — instant no-RAG reply |
| **Citations** | None | **Exact page + snippet citations** |
| **Dependencies** | Requires OS Pandoc | **100% Native Python + ONNX** |

---

## 🚀 Quickstart

### 1. Clone & Set Up

```bash
git clone https://github.com/AkSankhyan/Document_QA_Agent.git
cd Document_QA_Agent

python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Set your Groq API key in .env:
# GROQ_API_KEY=gsk_your_groq_key_here
```

### 3. Run

```bash
# Backend
python main.py
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs

# Frontend (new terminal)
cd frontend
npm install && npm start
# UI: http://localhost:3000
```

---

## 💻 CLI Usage

```bash
python -m backend --file report.pdf --question "What was the Q3 revenue growth?"
```

```
📂 Parsing: report.pdf...
✅ 34 chunks across 12 pages indexed into Qdrant.
🤖 LangGraph CRAG Agent running...
[Retrieve] Top-12 candidates from Qdrant
[Rerank]   FlashRank → Top-6
[Grader]   Relevance: True
[Generator] Streaming answer...

ANSWER:
Q3 revenue grew 14.2% YoY → $4.85B [Source: report.pdf, Page 4]

CITATIONS:
[1] report.pdf, Page 4 (Score: 0.89)
    "Third quarter revenues reached $4,850M, a 14.2% increase..."
```

---

## 🎯 Resume & Interview Prep

### ✅ ATS-Optimized Resume Bullet Points

- **Built an end-to-end Agentic Corrective RAG (CRAG) platform** using LangGraph, Qdrant vector database, FastEmbed (ONNX), and FlashRank cross-encoder reranking for intelligent multi-format document Q&A (PDF, DOCX, PPTX, XLSX).

- **Implemented real-time Server-Sent Events (SSE) token streaming** with FastAPI and React 18, delivering character-by-character responses and reducing perceived latency to near-zero.

- **Designed a self-corrective LangGraph state graph** with document relevance grading, iterative query rewriting, and date-aware LLM reasoning — structurally preventing hallucinations through retrieval validation loops.

- **Architected layout-aware document ingestion** using PyMuPDF4LLM preserving markdown tables and complex schemas across 6 file formats with zero external binary dependencies (pure Python + ONNX).

- **Engineered conversational intent detection** to short-circuit non-document queries pre-retrieval and built automatic LLM fallback chains ensuring availability across Groq model deprecation events.

### 🔑 ATS Keywords
`RAG` · `Corrective RAG` · `LangGraph` · `LLM` · `Qdrant` · `Vector Database` · `FastAPI` · `React` · `Python` · `Embeddings` · `Reranking` · `NLP` · `Generative AI` · `Document AI` · `Groq` · `ONNX` · `Server-Sent Events` · `Streaming` · `FastEmbed` · `Agentic AI` · `Retrieval-Augmented Generation`

---

## 💬 Interview Q&A

**Q: Why FlashRank cross-encoder instead of pure cosine similarity?**
> Bi-encoder search compares query and document independently — great for recall. Cross-encoders evaluate them together at the token-interaction level, dramatically improving precision. FlashRank runs TinyBERT on CPU, making it production-viable without GPU.

**Q: How does your pipeline prevent hallucinations?**
> Three layers: (1) LangGraph Grader node validates chunk relevance before generation — poor context triggers query rewriting, not fabrication. (2) System prompts with `temperature=0.1` enforce citation-grounded answers. (3) LLM is explicitly instructed to say "I could not find..." rather than guessing.

**Q: Why Qdrant over Pinecone or FAISS?**
> Qdrant runs fully locally with persistent on-disk storage — zero API costs, no data egress. FAISS is in-memory with no metadata filtering. Pinecone adds managed SaaS latency and cost. For self-hosted document QA, Qdrant is the right tradeoff.

**Q: What is Corrective RAG vs. standard RAG?**
> Standard RAG always generates from whatever it retrieves — even irrelevant chunks. CRAG adds a grading step that evaluates context quality first. Poor context triggers query reformulation and a second retrieval pass. This feedback loop eliminates the garbage-in → garbage-out failure mode.

**Q: How does SSE streaming work end-to-end?**
> FastAPI `/api/chat/stream` returns a `StreamingResponse` with `text/event-stream` MIME type. LangChain `ChatGroq` runs with `streaming=True`, yielding each token as `data: {"token": "..."}` SSE events. React reads the `ReadableStream` via Fetch API, accumulates tokens, and updates the message bubble in-place via React state — no page refresh, no polling.

---

## 📄 License
MIT License. Free for personal and commercial use.
