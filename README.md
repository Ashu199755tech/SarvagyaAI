# ResoAI — MS Teams HR Chatbot

An AI-powered chatbot for **Microsoft Teams** that answers HR questions by ingesting data from the **Keka HR API** and serving answers through a **RAG (Retrieval-Augmented Generation) pipeline**.

## Features

- 🔄 **Automated Keka sync** — periodically fetches employees & projects
- 🧠 **RAG pipeline** — LangChain + ChromaDB for semantic search
- 💬 **MS Teams integration** — natural language Q&A inside Teams
- 🔒 **OAuth2 authentication** with Keka API
- ⚡ **FastAPI backend** — async, lightweight, easy to deploy

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Keka API credentials (client ID, secret, API key)
- OpenAI API key (or Azure OpenAI)
- MS Teams Bot registration (App ID + password)

### 2. Setup

```bash
# Clone & enter the project
cd ResoAi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### 3. Initial Data Ingestion

```bash
# Run one-time ingestion before starting the bot
python -c "
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

### 4. Run the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Connect to Teams

1. Set up **ngrok** or **dev tunnel** to expose port 8000
2. Configure your Bot's messaging endpoint to `https://<tunnel-url>/api/messages`
3. Sideload the Teams app and start chatting!

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/messages` | Teams bot webhook |
| `POST` | `/api/ingest` | Manual ingestion trigger |
| `GET` | `/api/health` | Health check |

## Project Structure

```
src/
├── main.py              # FastAPI entry point
├── config.py            # Settings from .env
├── keka/                # Keka HR API client
├── ingestion/           # RAG ingestion pipeline
├── rag/                 # Query chain + prompts
└── bot/                 # Teams bot handler
```

## Running Tests

```bash
python -m pytest tests/ -v
```
