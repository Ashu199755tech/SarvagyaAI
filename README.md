# ResoAI — Local RAG + MS Teams HR Chatbot (v1, current)

This repo implements an **Employee Data → RAG** pipeline:

1) **Ingest / index employee data** (offline) from a local HR API and PDF policy documents into ChromaDB vector embeddings. 2) **Expose a chatbot webhook** (online) via FastAPI for MS Teams. 3) Run a context-aware RAG query chain that: - retrieves relevant employee records and policy chunks from ChromaDB, - applies strict system prompts for concise answers, - runs local LLM inference via Ollama — writes no data outside the host machine.

This document matches the **current code** in `src/main.py`, `src/rag/chain.py`, `src/ingestion/pipeline.py`, and `src/bot/teams_bot.py`.

---

## 0) Repository Entry Points

**FastAPI server** (`src/main.py`)

- Webhook (Teams): `POST /api/messages`
- Direct query: `POST /api/ask`
- Manual ingest trigger: `POST /api/ingest`
- PDF-only ingest: `POST /api/ingest-pdfs`
- Health probe: `GET /api/health`
- Interactive docs: `GET /docs`

**RAG orchestrator** (`src/rag/chain.py`)

- Entry: `ask(question: str) -> dict`
  - Stages: `_extract_names` → `_text_search_employees` → `similarity_search` → `RAG_PROMPT | ChatOllama | StrOutputParser`

**Ingestion pipeline** (`src/ingestion/pipeline.py`)

- `IngestionPipeline.run()` — full sync (employees + PDFs)
- `IngestionPipeline.ingest_pdfs()` — PDF-only sync

**Background scheduler** (`src/ingestion/scheduler.py`)

- Runs `IngestionPipeline.run()` on a configurable interval (default every 6 hours)

---

## 1) Golden-Path Quickstart (Dev)

```bash
# Clone and enter the project
cd ResoAi

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

```bash
# Configure environment
cp .env.example .env
# Fill in your credentials:
# EMPLOYEE_API_BASE_URL=http://localhost:8000
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_LLM_MODEL=llama3.2
# OLLAMA_EMBEDDING_MODEL=nomic-embed-text
# TEAMS_APP_ID=<your-azure-app-id>
# TEAMS_APP_PASSWORD=<your-azure-app-password>
```

```bash
# Pull Ollama models (one-time)
ollama pull llama3.2
ollama pull nomic-embed-text

# Run one-time data ingestion
python3 -c "
import asyncio
from src.ingestion.pipeline import IngestionPipeline

async def ingest():
    p = IngestionPipeline()
    result = await p.run()
    print(result)
    await p.close()

asyncio.run(ingest())
"
```

```bash
# Start the FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

Then:
1. Start `ngrok http 8001` to get a public URL.
   - **Active tunnel URL:** `https://reniform-marx-nonmicroscopically.ngrok-free.dev`
2. Set your Azure Bot messaging endpoint to `https://reniform-marx-nonmicroscopically.ngrok-free.dev/api/messages`.
3. Sideload the Teams app and start asking questions.

**Test the RAG endpoint directly:**

```bash
curl -X POST https://reniform-marx-nonmicroscopically.ngrok-free.dev/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How many leave days is Anthony entitled to?"}'
```

---

## 2) Query & Answering Model

**Direct API query**

```bash
curl -X POST http://localhost:8001/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How many leave days is Anthony entitled to?"}'
```

**Response format**

```json
{ "status": "success", "answer": "24 leaves in a year" }
```

**Retrieval strategy** (`src/rag/chain.py`)

- Name extraction: regex over capitalized word-pairs → `_extract_names(question)`
- Text match: `ChromaDB $contains` search for exact employee name variants
- Vector search: `similarity_search` filtered by `record_type=employee` + separate `record_type=pdf`
- Merge: name-match docs take priority, then employee vectors, then PDF vectors

**System prompt rules** (`src/rag/prompts.py`)

- Answer ONLY from provided context
- On "how many" questions → output a number + short phrase only
- On multiple employee matches → list each separately (e.g. "Anthony West: 24 leaves.")
- Never reveal raw IDs or internal metadata

---

## 3) Ingestion Side (Offline)

**Employee data source** (`src/keka/client.py`)

- `GET /employees?skip=N&limit=100` — paginated fetch
- Each employee becomes one `Document` via `build_employee_document()` in `src/ingestion/document_builder.py`

**PDF data source** (`src/ingestion/pdf_loader.py`)

- Scans `PDF_FOLDER` (default: `/home/.../Policies 2026`) recursively for `*.pdf`
- Each page becomes a separate `Document` with `record_type=pdf` metadata

**Chunking settings** (configurable in `.env` or `src/config.py`)

- `CHUNK_SIZE=500`
- `CHUNK_OVERLAP=100`
- Splitter: `RecursiveCharacterTextSplitter` with `["\n\n", "\n", ". ", " ", ""]` separators

**Embedding** (`src/ingestion/pipeline.py` → `_get_embeddings()`)

- Model: `nomic-embed-text` (via `OllamaEmbeddings`)
- Each text chunk is converted into a high-dimensional vector using the local Ollama embedding model
- Threads: configurable via `OLLAMA_NUM_THREADS` (default: 6)

**ChromaDB upsert** (`src/ingestion/pipeline.py`)

- ID scheme: `{record_type}_{record_id}_chunk_{i}` — re-ingestion replaces stale chunks, no duplicates
- Collection name: `resoai_hr`
- Persist directory: `./data/chromadb`

---

## 4) MS Teams Integration

**Azure Bot Service setup**

1. Register a new Bot in Azure → note the App ID and App Password.
2. Set messaging endpoint: `https://<ngrok-url>/api/messages`.
3. Add `TEAMS_APP_ID` and `TEAMS_APP_PASSWORD` to your `.env` file.

**Adapter** (`src/bot/adapter.py`)

- Uses `CloudAdapter` from `botbuilder-integration-aiohttp`
- Validates the incoming JWT from Azure on every request

**Bot handler** (`src/bot/teams_bot.py`)

- Subclasses `ActivityHandler`
- `on_message_activity` → passes text to `rag.chain.ask()` → replies with `turn_context.send_activity()`

---

## 5) Configuration Reference (`src/config.py`)

| Variable | Default | Description |
|---|---|---|
| `EMPLOYEE_API_BASE_URL` | `http://localhost:8000` | Local HR data API |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Generation model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_MAX_THREADS` | `12` | Max total CPU threads |
| `CHROMA_PERSIST_DIR` | `./data/chromadb` | Vector DB path |
| `CHROMA_COLLECTION_NAME` | `resoai_hr` | ChromaDB collection |
| `PDF_FOLDER` | *(set in config)* | Policies folder path |
| `RETRIEVER_TOP_K` | `20` | Docs retrieved per query |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `INGESTION_INTERVAL_HOURS` | `6` | Auto-sync frequency |
| `TEAMS_APP_ID` | *(required)* | Azure Bot App ID |
| `TEAMS_APP_PASSWORD` | *(required)* | Azure Bot App Password |

---

## 6) Running Tests

```bash
python -m pytest tests/ -v
```

Test files:
- `tests/test_keka_client.py` — Employee API client unit tests
- `tests/test_ingestion.py` — Pipeline ingestion unit tests

---

## 7) Deployment (Production)

For production use, the local CPU/Ollama setup should be replaced. Options:

**Option A — Cloud LLM API (recommended)**
- Replace `ChatOllama` in `src/rag/chain.py` with `ChatOpenAI` or `AzureChatOpenAI`
- Add `OPENAI_API_KEY` to `.env`

**Option B — Cloud Ollama Host**
- Install Ollama on a GPU-enabled VPS
- Update `OLLAMA_BASE_URL` in `.env` to point to the remote host

**Option C — Full cloud deploy**
- Deploy FastAPI app to Azure App Service / AWS / Railway
- Use a cloud LLM (Option A)
- Replace ngrok with a static endpoint configured in Azure Bot
