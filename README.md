# SarvagyaAI (ResoAI)

> **An enterprise-grade RAG (Retrieval-Augmented Generation) agent** that answers questions about employees, projects, policies, holidays, and case studies using a hybrid retrieval pipeline — all running locally on CPU via [Ollama](https://ollama.com).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Running the Agent](#running-the-agent)
7. [Data Pipeline (How Data Flows)](#data-pipeline-how-data-flows)
8. [API Reference](#api-reference)
9. [Configuration Reference](#configuration-reference)
10. [MS Teams Integration](#ms-teams-integration)
11. [Utilities](#utilities)
12. [Troubleshooting](#troubleshooting)

---

## Overview

SarvagyaAI is an internal company assistant that can answer questions like:

- *"What is Anthony Young's salary?"*
- *"Which project involved migrating to a monorepo while maintaining PCI-DSS compliance?"*
- *"What is the notice period policy?"*
- *"List all projects that used GPU."*

It works by:

1. **Ingesting** company data (PDFs, employee records) into structured JSON files (zero-LLM, deterministic parsing).
2. **Embedding** those JSON records into a local ChromaDB vector database using Ollama's `nomic-embed-text` model.
3. **Retrieving** the most relevant chunks using a **hybrid approach** (BM25 keyword search + Vector semantic search + FlashRank re-ranking).
4. **Generating** a natural-language answer using a local LLM (e.g., `llama3.2` via Ollama).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                             │
│                                                                     │
│   MS Teams Bot (/api/messages)     HTTP API (/api/ask)             │
│         │                                │                          │
└─────────┼────────────────────────────────┼──────────────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (src/main.py)                    │
│                                                                     │
│  Endpoints:                                                         │
│    POST /api/ask                    → Ask a question                │
│    POST /api/debug                  → Debug retrieval (no LLM)      │
│    POST /api/consolidate-and-ingest → Full data refresh             │
│    POST /api/ingest                 → Re-index ChromaDB only        │
│    POST /api/messages               → MS Teams webhook              │
│    GET  /api/health                 → Health check                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RAG Engine  │  │ Consolidator │  │  Ingestion   │
│ (src/rag/)   │  │  Pipeline    │  │  Pipeline    │
│              │  │(consolidator)│  │(src/ingestion)│
│ • chain.py   │  │              │  │              │
│ • prompts.py │  │ • ingestors  │  │ • pipeline   │
│ • cache.py   │  │ • entities   │  │ • doc_builder│
│              │  │ • generator  │  │ • scheduler  │
│              │  │ • relations  │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Ollama     │  │  Raw PDFs    │  │  ChromaDB    │
│  (Local LLM) │  │  (Policies,  │  │ (Vector DB)  │
│              │  │  Case Studies)│  │              │
│ • llama3.2   │  │              │  │  data/       │
│ • nomic-     │  │  Employee API │  │  chromadb/   │
│   embed-text │  │  (Mock/Keka) │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Hybrid Retrieval Engine (the "Brain")

The RAG engine in `src/rag/chain.py` uses a **10-step pipeline** for every query:

| Step | Method | Purpose |
|------|--------|---------|
| 1 | Name Extraction | Detect person names (e.g., "Anthony") for employee lookup |
| 2 | BM25 Search | Statistical keyword matching across all chunks |
| 3 | Regex Keywords | ChromaDB `$contains` filter for exact term matches |
| 4 | Intent Classification | Route to employee vs. document vs. mixed intent |
| 5 | Vector Search | Semantic similarity via Ollama embeddings |
| 6 | Merge & Deduplicate | Combine all results, remove duplicates |
| 6b | Acronym Filter | Remove irrelevant chunks for tech-specific queries |
| 7 | FlashRank Re-ranking | Cross-encoder scoring for final relevance sort |
| 8 | Context Formatting | Label chunks with `[Source: ...]` tags |
| 9 | LLM Generation | Invoke Ollama with context + question |
| 10 | Cache & Return | Save to SQLite cache, return answer |

---

## Project Structure

```
SarvagyaAI/
├── data/                          # Source-of-truth JSON files + vector DB
│   ├── chromadb/                  # ChromaDB persistent vector storage (auto-generated)
│   ├── employees.json             # Employee records from Keka API
│   ├── projects.json              # Parsed case study projects (108 entries)
│   ├── policies.json              # Parsed company policy documents
│   ├── holidays.json              # Holiday calendar data
│   ├── misc_docs.json             # Miscellaneous documents
│   ├── master_data.md             # RAG-ready consolidated Markdown
│   ├── knowledge_graph.json       # Entity relationship graph
│   └── rag_cache.db               # SQLite response cache (auto-generated)
│
├── scripts/
│   └── mock_keka_server.py        # Mock Employee API for development
│
├── src/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Central Pydantic settings (all tuning knobs)
│   │
│   ├── bot/                       # MS Teams integration
│   │   ├── adapter.py             # Bot Framework adapter
│   │   └── teams_bot.py           # Teams message handler
│   │
│   ├── consolidator/              # Zero-LLM data preparation pipeline
│   │   ├── entities.py            # Dataclasses: Employee, Project, Policy, etc.
│   │   ├── ingestors.py           # PDF → JSON parsers (regex, no LLM)
│   │   ├── relationships.py       # Entity linker (employee↔project↔tool)
│   │   ├── generator.py           # JSON/Markdown output generators
│   │   └── run.py                 # Pipeline orchestrator
│   │
│   ├── ingestion/                 # JSON → ChromaDB vector ingestion
│   │   ├── pipeline.py            # Main ingestion pipeline (chunking + embedding)
│   │   ├── document_builder.py    # JSON → LangChain Document converters
│   │   ├── pdf_loader.py          # PDF text extraction utilities
│   │   └── scheduler.py           # APScheduler for periodic re-ingestion
│   │
│   ├── keka/                      # Keka HR API client
│   │   ├── client.py              # HTTP client for employee data
│   │   └── models.py              # Pydantic models for API responses
│   │
│   └── rag/                       # RAG query engine
│       ├── chain.py               # Hybrid retrieval + LLM generation (the core)
│       ├── prompts.py             # System prompt and template
│       └── cache.py               # SQLite-based response cache
│
├── tests/                         # Test suite
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (not in git)
└── .gitignore
```

---

## Prerequisites

Before running the agent, ensure you have the following installed on your system:

### 1. Python 3.12+

```bash
python3 --version   # Must be 3.12 or higher
```

### 2. Ollama (Local LLM Server)

Ollama runs the LLM and embedding models locally on your CPU/GPU.

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required models
ollama pull llama3.2           # LLM for generating answers (~2GB)
ollama pull nomic-embed-text   # Embedding model for vectorization (~270MB)

# Verify Ollama is running
ollama list
```

> **Note:** If you have a GPU, Ollama will automatically use it. On CPU-only machines, responses will take 1-3 minutes per query.

### 3. pdftotext (PDF parsing tool)

The consolidator uses `pdftotext` (from the `poppler-utils` package) for deterministic PDF text extraction.

```bash
# Ubuntu / Debian
sudo apt-get install -y poppler-utils

# macOS
brew install poppler

# Verify
pdftotext -v
```

### 4. System Packages (Optional)

```bash
# For SQLite (usually pre-installed)
sudo apt-get install -y sqlite3
```

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ashu199755tech/SarvagyaAI.git
cd SarvagyaAI
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create the `.env` File

Create a `.env` file in the project root. This file is **not committed to git** for security reasons.

```dotenv
# ─── Required ────────────────────────────────────────────────────
# Path to the folder containing your company PDFs (policies + case studies)
PDF_FOLDER=/path/to/your/Policies 2026

# ─── Optional: Ollama Settings (defaults shown) ─────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.1
OLLAMA_NUM_CTX=8192
OLLAMA_MAX_THREADS=12

# ─── Optional: Employee API ─────────────────────────────────────
# If using the mock server, set this to where the mock runs:
EMPLOYEE_API_BASE_URL=http://localhost:8001

# ─── Optional: MS Teams Bot (leave blank if not using Teams) ────
TEAMS_APP_ID=
TEAMS_APP_PASSWORD=

# ─── Optional: Retrieval Tuning ─────────────────────────────────
RETRIEVER_TOP_K=12
BM25_TOP_K=15
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

### Step 5: Prepare Your Data Folder

The system expects a folder (pointed to by `PDF_FOLDER`) with the following structure:

```
Policies 2026/
├── Case Studies @55 Website .pdf    # Case study PDF (contains 100+ projects)
├── Attendance Policy.pdf
├── Leave Policy.pdf
├── Travel Policy.pdf
├── Separation Policy.pdf
├── Holiday Policy/
│   └── Holiday Policy.pdf           # Contains the holiday calendar
└── ...other policy PDFs...
```

> **Important:** The case studies PDF filename must contain "case stud" (case-insensitive) for the parser to recognize it.

---

## Running the Agent

### Quick Start (3 Terminal Windows)

You need **3 terminals** to run the full system:

#### Terminal 1: Start the Mock Employee API

This is a lightweight local server that simulates the Keka HR API with 25 sample employees.

```bash
source .venv/bin/activate
python3 -m uvicorn scripts.mock_keka_server:app --port 8001
```

> **Note:** If `EMPLOYEE_API_BASE_URL` in `.env` is set to `http://localhost:8001`, the main agent will fetch employee data from this mock server.

#### Terminal 2: Run the Data Consolidation & Ingestion Pipeline

This is a **one-time setup** step that:
1. Parses all PDFs into structured JSON files (`data/projects.json`, `data/policies.json`, etc.)
2. Fetches employee data from the API and saves it to `data/employees.json`
3. Builds entity relationships (employee↔project↔tool links)
4. Generates `data/master_data.md` (RAG-ready document)
5. Chunks all JSON data and embeds it into ChromaDB

```bash
source .venv/bin/activate

# Step 1: Consolidate raw data into JSONs
python3 -m src.consolidator.run

# Step 2: Ingest JSONs into ChromaDB (creates embeddings)
python3 -m src.ingestion.pipeline
```

You should see output like:
```
SOT written: projects.json (108 records)
SOT written: policies.json (22 records)
...
Upserted batch 1/9 (50 chunks)
Upserted batch 2/9 (50 chunks)
...
Ingestion complete: {'documents_loaded': 179, 'total_chunks': 407}
```

#### Terminal 3: Start the Main Agent Server

```bash
source .venv/bin/activate
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The server is now running at `http://localhost:8000`.

### Verify It Works

```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the leave policy?"}'
```

### Expose to the Internet (Optional)

To make the agent accessible outside your local machine (e.g., for MS Teams or sharing with colleagues):

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000
```

This gives you a public URL like `https://xxxx.ngrok-free.dev` that forwards to your local server.

---

## Data Pipeline (How Data Flows)

The system has two distinct pipelines:

### Pipeline 1: Consolidation (PDF → JSON)

**File:** `src/consolidator/run.py` → calls `src/consolidator/ingestors.py`

This pipeline uses **zero LLM** — only `pdftotext` and regex — to ensure 100% fidelity.

```
Raw PDFs (Policies, Case Studies)
        │
        ▼
  pdftotext (text extraction)
        │
        ▼
  Regex-based parsing (ingestors.py)
        │
        ├──→ data/projects.json    (108 case study records)
        ├──→ data/policies.json    (22 policy documents)
        ├──→ data/holidays.json    (24 holidays)
        └──→ data/misc_docs.json   (9 miscellaneous docs)

Employee API (Keka / Mock)
        │
        ▼
  HTTP GET /employees
        │
        └──→ data/employees.json   (25 employee records)
```

**Trigger manually:**
```bash
python3 -m src.consolidator.run
```

### Pipeline 2: Ingestion (JSON → ChromaDB)

**File:** `src/ingestion/pipeline.py`

This pipeline reads the JSON files, converts them into LangChain `Document` objects, chunks them, and embeds them into ChromaDB.

```
data/employees.json ─┐
data/projects.json  ─┤
data/policies.json  ─┤    document_builder.py        pipeline.py
data/holidays.json  ─┘         │                         │
                               ▼                         ▼
                    LangChain Documents  →  Chunking  →  Ollama Embedding  →  ChromaDB
                    (with metadata:                     (nomic-embed-text)    (data/chromadb/)
                     record_type,
                     record_id, etc.)
```

**Trigger manually:**
```bash
python3 -m src.ingestion.pipeline
```

### Pipeline 3: Combined (One Command)

You can run both pipelines together via the API:

```bash
curl -X POST http://localhost:8000/api/consolidate-and-ingest
```

### Caching

- **What:** Every LLM-generated answer is cached in `data/rag_cache.db` (SQLite).
- **Key:** MD5 hash of the normalized question text.
- **Hit:** If the same question is asked again, the cached answer is returned in ~0.05s (no LLM invocation).
- **Invalidation:** The cache is automatically wiped every time you run the ingestion pipeline, ensuring no stale answers after data updates.

---

## API Reference

All endpoints are available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### `POST /api/ask`

Ask a question to the RAG agent.

```bash
curl -X POST http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which projects used GPU?"}'
```

**Response:**
```json
{
  "status": "success",
  "answer": "The following projects used GPU: ...",
  "time_elapsed_seconds": 137.5
}
```

### `POST /api/debug`

Returns retrieved chunks **without** generating an LLM answer. Use this to debug retrieval accuracy.

```bash
curl -X POST http://localhost:8000/api/debug \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which projects used GPU?"}'
```

**Response:**
```json
{
  "status": "success",
  "question": "Which projects used GPU?",
  "chunks_retrieved": 6,
  "chunks": [
    {
      "rank": 1,
      "source": "projects.json",
      "record_type": "project",
      "record_id": "PRJ:110_c1",
      "preview": "Project Name: Handsin Custom Split Payments..."
    }
  ]
}
```

### `POST /api/consolidate-and-ingest`

Run the full data refresh: consolidation (PDF → JSON) + ingestion (JSON → ChromaDB).

```bash
curl -X POST http://localhost:8000/api/consolidate-and-ingest
```

### `POST /api/ingest`

Re-index ChromaDB from existing JSON files (no PDF re-parsing). Truncates the vector store first.

```bash
curl -X POST http://localhost:8000/api/ingest
```

### `POST /api/ingest-pdfs`

Re-index ChromaDB from existing JSON files **without** truncating (additive).

```bash
curl -X POST http://localhost:8000/api/ingest-pdfs
```

### `GET /api/health`

Simple health check.

```bash
curl http://localhost:8000/api/health
# {"status": "ok", "service": "ResoAI"}
```

### `POST /api/messages`

MS Teams Bot Framework webhook endpoint. Not called directly — configured in Azure Bot Service.

---

## Configuration Reference

All settings are in `src/config.py` and can be overridden via the `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_FOLDER` | `/home/fifity-five/Downloads/Policies 2026` | Path to the folder containing company PDFs |
| `EMPLOYEE_API_BASE_URL` | `http://localhost:8005` | Base URL for the Employee API |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.2` | LLM model for answer generation |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for vectorization |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM creativity (0.0 = deterministic) |
| `OLLAMA_NUM_CTX` | `8192` | LLM context window size in tokens |
| `OLLAMA_MAX_THREADS` | `12` | Max CPU threads for Ollama |
| `RETRIEVER_TOP_K` | `12` | Number of chunks sent to LLM |
| `BM25_TOP_K` | `15` | BM25 candidate pool size |
| `CHUNK_SIZE` | `300` | Character length per chunk |
| `CHUNK_OVERLAP` | `50` | Character overlap between chunks |
| `INGESTION_INTERVAL_HOURS` | `6` | Auto re-ingestion frequency |
| `HOST` | `0.0.0.0` | FastAPI server bind address |
| `PORT` | `8001` | FastAPI server port |
| `TEAMS_APP_ID` | *(empty)* | Azure Bot App ID |
| `TEAMS_APP_PASSWORD` | *(empty)* | Azure Bot App Password |

---

## MS Teams Integration

To connect the agent to Microsoft Teams:

### 1. Register a Bot in Azure

1. Go to [Azure Portal](https://portal.azure.com) → Bot Services → Create.
2. Note the **App ID** and **App Password**.
3. Set the messaging endpoint to: `https://your-ngrok-url.ngrok-free.dev/api/messages`

### 2. Configure `.env`

```dotenv
TEAMS_APP_ID=your-azure-app-id
TEAMS_APP_PASSWORD=your-azure-app-password
```

### 3. Expose the Server

```bash
ngrok http 8000
```

### 4. Start the Server

```bash
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Users can now chat with the bot in MS Teams. The bot processes messages through `src/bot/teams_bot.py`, which calls the same `rag_ask()` function used by the HTTP API.

---

## Utilities

### PDF Table to JSON Converter

If you have a PDF containing tabular data (like an employee directory or a complex policy table), you can use the built-in utility to convert it into a structured JSON format.

**Command:**
```bash
./.venv/bin/python src/ingestion/pdf_table_to_json.py <path_to_pdf> --output <output_json_path> --pretty
```

- **Input:** Any PDF with 1 or more tables.
- **Output:** A JSON array where each object represents a row, and keys are automatically detected from the table headers.
- **Features:**
    - Automatically detects headers from the first row of each table.
    - Handles multi-page tables.
    - Cleans up whitespace and newlines within cells.

**Example:**
```bash
./.venv/bin/python src/ingestion/pdf_table_to_json.py "/path/to/my_table.pdf" --pretty
```

---

## Utilities

### PDF Table to JSON Converter

If you have a PDF containing tabular data (like an employee directory or a complex policy table), you can use the built-in utility to convert it into a structured JSON format.

**Command:**
```bash
./.venv/bin/python src/ingestion/pdf_table_to_json.py <path_to_pdf> --output <output_json_path> --pretty
```

- **Input:** Any PDF with 1 or more tables.
- **Output:** A JSON array where each object represents a row, and keys are automatically detected from the table headers.
- **Features:**
    - Automatically detects headers from the first row of each table.
    - Handles multi-page tables.
    - Cleans up whitespace and newlines within cells to ensure clean data for LLM consumption.

**Example:**
```bash
./.venv/bin/python src/ingestion/pdf_table_to_json.py "/path/to/my_table.pdf" --pretty
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pydantic_settings'"

You need to activate the virtual environment first:
```bash
source .venv/bin/activate
```

### "Connection refused" when querying Ollama

Ensure Ollama is running:
```bash
ollama serve   # Start Ollama server
ollama list    # Verify models are pulled
```

### "Collection does not exist" error from ChromaDB

The vector store needs to be rebuilt. Run:
```bash
python3 -m src.ingestion.pipeline
```

### Low retrieval accuracy or wrong project names

1. Check `data/projects.json` to verify the data was parsed correctly.
2. Use the `/api/debug` endpoint to see which chunks are being retrieved.
3. If the data looks wrong, re-run the full pipeline:
   ```bash
   python3 -m src.consolidator.run    # Re-parse PDFs
   python3 -m src.ingestion.pipeline  # Re-index ChromaDB
   ```

### Slow response times on CPU

This is expected. Local LLM inference on CPU takes 1-3 minutes per query. To speed things up:
- Reduce `OLLAMA_NUM_CTX` (e.g., from `8192` to `4096`)
- Reduce `RETRIEVER_TOP_K` (e.g., from `12` to `8`)
- Use a smaller model (e.g., `llama3.2:1b` instead of `llama3.2`)
- Or use a machine with a GPU

### Mock Employee API not returning data

Ensure the mock server is running on the port specified in `EMPLOYEE_API_BASE_URL`:
```bash
python3 -m uvicorn scripts.mock_keka_server:app --port 8001
```

---

## License

Internal use only — FiftyFive Technologies.
