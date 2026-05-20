# ResoAI: Agentic CRAG Pipeline

> **Enterprise-Grade internal assistant** providing 100% accurate quantitative answers and nuanced semantic retrieval. Built with an autonomous **Agentic CRAG (Conversational RAG)** architecture running locally on CPU via [Ollama](https://ollama.com).

---

## 🚀 Overview & New Architecture

ResoAI has evolved from a standard keyword-based RAG pipeline into a fully autonomous, **Self-Correcting Agentic CRAG System**. It uses a **Small Brain / Big Brain** tiered strategy to eliminate factual hallucinations, parse complex human language, and maintain ultra-low latency.

### 🧠 The "Small Brain": Query Decomposer
We have eliminated brittle, hardcoded intent lists (regex parsing) and replaced them with a lightweight, high-speed LLM (`llama3.2:1b`).
* It dynamically parses raw user questions into a structured JSON `QueryPlan`.
* Automatically infers **intent** (count, list, lookup, policy).
* Extracts exact **search_terms** and identifies the underlying **entity** (employee, project, holiday, policy).
* Features built-in noise-filtering that uses configurable stop-words (e.g., dynamically removing generic words like "department" from the LLM extraction).

### 🤖 The "Big Brain": Hybrid Retrieval & Synthesis
* Uses `llama3.2` for final answer generation with rigorous strict-fact prompting.
* Employs **Conversational Memory**: A stabilized 1-turn memory horizon prevents context contamination during sequential follow-up queries.
* **Hybrid Search Engine**: Combines Statistical (BM25) + Semantic (Vector Embeddings via `nomic-embed-text`) + Exact Substring matching.
* Uses **FlashRank Cross-Encoder Re-ranking** to filter candidate chunks down to the absolute most relevant context.

---

## ✨ Key Capabilities

1. **100% Count Accuracy**: Structured queries (e.g., *"How many people are in the HR department?"*) bypass the vector database completely and are routed to the deterministic **DataInterpreter**.
2. **Conversational Memory**: Users can pass a `session_id` to ask follow-up questions (e.g., Q1: *"Who manages Alpha?"* -> Q2: *"What is his email?"*) without losing context.
3. **Dynamic Acronym & Typo Resolution**: 
   - Typo-tolerance (`_fuzzy_expand_names`) automatically corrects misspelled names.
   - Acronyms (e.g., `nlp` -> `NLP`) and aliases (e.g., `HR` -> `Human Resource`) are automatically expanded to match unstructured PDFs.
4. **Universal Ingestion**: Automatically handles PDFs, Markdown, Excel, and API data.

---

## ⚙️ Installation & Setup (Docker Recommended)

We have containerized the entire application for "one-click" portability, preventing port conflicts and missing dependency issues.

### 🐳 Using Docker Compose (The Easy Way)
Ensure you have Docker and Docker Compose installed on your machine.

```bash
git clone https://github.com/Ashu199755tech/SarvagyaAI.git
cd SarvagyaAI

# Start the environment in detached mode
docker-compose up -d
```
**What this does:**
1. Spins up the FastAPI `resoai-app` container (available on port `8001`).
2. Spins up the `resoai-ollama` engine container.
3. Runs an automated setup script (`resoai-ollama-setup`) that waits for Ollama to boot, then automatically pulls the required `llama3.2` and `nomic-embed-text` models.

### 💻 Local/Native Setup
1. Install [Ollama](https://ollama.com).
2. Pull the required models:
   ```bash
   ollama pull llama3.2
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```
3. Setup Python environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   sudo apt install poppler-utils # Required for PDF parsing
   ```
4. Start the server (runs on `localhost:8000` locally):
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 🔌 Core API Endpoints

### 1. `/api/ask` (POST)
The primary endpoint for asking questions. Includes conversational memory support.
```bash
curl -X POST http://localhost:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Who is the lead for the ResoAI project?", "session_id": "user123"}'
```

### 2. `/api/debug` (POST)
Bypasses the final LLM generation step and returns exactly how the system routed the query (Decomposer JSON output, Retriever chunks, DataInterpreter matches). Crucial for debugging "Why did it give me this answer?".
```bash
curl -X POST http://localhost:8000/api/debug \
     -H "Content-Type: application/json" \
     -d '{"question": "How many employees are in HR?"}'
```

### 3. `/api/upload` (POST)
Incrementally upload any supported file (PDF, CSV, JSON, TXT). Converts the file to JSON and instantly ingests it into ChromaDB *without* requiring a full system re-index.
```bash
curl -X POST http://localhost:8000/api/upload -F 'file=@new_policy.pdf'
```

### 4. `/api/consolidate-and-ingest` (POST)
Triggers a full rebuild of the system. Consolidates fragmented raw data into single source-of-truth JSONs, truncates the ChromaDB database, and re-ingests everything from scratch.

---

## 📂 Project Structure

```text
├── src/
│   ├── main.py             # FastAPI App & Endpoints
│   ├── config.py           # Centralized Dynamic Settings
│   ├── rag/
│   │   ├── query_decomposer.py  # Small-brain LLM for intent parsing
│   │   ├── data_interpreter.py  # Deterministic JSON scanner
│   │   ├── chain.py             # Big-brain Hybrid Retrieval & Synthesis
│   │   ├── memory.py            # Conversational state & rewrite logic
│   ├── ingestion/          # Pipeline for PDFs & unstructured text
│   ├── consolidator/       # Unified data aggregation
├── data/
│   ├── directory.json      # Employee Source of Truth
│   ├── projects.json       # Case Studies SOT
│   ├── holidays.json       # Holiday SOT
│   ├── chromadb/           # Persistent Vector Store
├── docker-compose.yml      # Orchestration for app & ollama
└── README.md
```

---
*Developed by FiftyFive Technologies - Enterprise RAG Division*
