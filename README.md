# ResoAI: Precision Hybrid RAG Agent

> **Enterprise-Grade internal assistant** that provides 100% accurate quantitative answers and nuanced semantic retrieval — all running locally on CPU via [Ollama](https://ollama.com).

---

## 🚀 Overview

ResoAI (formerly SarvagyaAI) is designed to solve the "hallucination problem" inherent in standard RAG systems. It uses a **Hybrid Agentic Architecture** that dynamically routes queries:

1.  **Quantitative Queries** (e.g., *"How many people from Jaipur?"*): Routed to the **Zero-Shot Data Interpreter**, which performs deterministic scans on structured JSON sources for 100% accuracy in milliseconds.
2.  **Semantic Queries** (e.g., *"Explain the leave policy"*): Routed to the **RAG Engine**, which uses Vector Search + BM25 + Cross-Encoder Re-ranking to synthesize natural language answers from unstructured PDFs.

---

## 🏗️ High-Level Architecture

ResoAI follows a **Local-First** design pattern:
- **Data Sovereignty**: Employee records and company policies never leave your infrastructure.
- **Model Isolation**: LLM inference (`llama3.2`) and embeddings (`nomic-embed-text`) run entirely on-premises using **Ollama**.
- **Agentic Routing**: Automatically decides the best path (Data Scanning vs. Vector Search) based on user intent.

### Visual Architecture
See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed mermaid diagrams of the data and request flow.

---

## ✨ Key Features

- **100% Count Accuracy**: Direct data-interpreter path bypasses vector noise for aggregations.
- **Dynamic Configuration**: Zero hardcoding. Deploy for any client by updating `.env` in seconds.
- **Phrase-First Scrubbing**: Advanced query cleaning ensures irrelevant "human noise" doesn't confuse retrieval.
- **Hybrid Retrieval**: Combines statistical (BM25), semantic (Vector), and deterministic (Regex) filters.
- **Universal Ingestion**: Standardized pipeline for PDFs, Excel, and API data via a `UniversalConverter`.

---

## 🛠️ Retrieval Pipeline (The "Brain")

The engine in `src/rag/chain.py` uses a professional 10-step pipeline:

| Step | Action | Description |
| :--- | :--- | :--- |
| **1** | **Alias Expansion** | Expands shorthand (e.g., `@55` → `FiftyFive Technologies`) using dynamic config. |
| **2** | **Phrase Scrubbing** | Strips noise phrases using phrase-first boundary matching. |
| **3** | **Intent Routing** | Classifies query as **Quantitative** (scanning) or **Semantic** (RAG). |
| **4** | **Name Extraction** | Identifies person names to prevent cross-employee data leakage. |
| **5** | **Hybrid Search** | Concurrent search across BM25 and ChromaDB Vector Store. |
| **6** | **Regex Filtering** | Force-filtering via `$contains` for critical technical keywords. |
| **7** | **Merging** | Deduplicates and combines results from all retrieval paths. |
| **8** | **Re-ranking** | Cross-encoder scoring for final context sorting. |
| **9** | **Context Prep** | Formats context with clear `[Source: ...]` tags for the LLM. |
| **10** | **Local Generation** | Synthesizes final answer via Ollama with strict fact-only grounding. |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python 3.12+**
- **Ollama**: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
- **Poppler-Utils**: `sudo apt install poppler-utils` (for `pdftotext`)

### 2. Setup
```bash
git clone https://github.com/Ashu199755tech/SarvagyaAI.git
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Dynamic Configuration (`.env`)
The system is entirely configuration-driven. Create a `.env` in the root:

```ini
# Client Identity
COMPANY_NAME="FiftyFive Technologies"
COMPANY_ALIASES='["@55", "at 55", "55 tech"]'
DEPARTMENT_ALIASES='{"ai": "Artificial Intelligence", "hr": "Human Resource"}'

# Paths
PDF_FOLDER="/path/to/policies"
DATA_DIR="./data"

# LLM Tuning
OLLAMA_NUM_CTX=8192
OLLAMA_TEMPERATURE=0.1
RETRIEVER_TOP_K=12
```

---

## 📂 Project Structure

```text
SarvagyaAI/
├── src/
│   ├── main.py             # FastAPI App & Endpoints
│   ├── config.py           # Centralized Dynamic Settings
│   ├── rag/                # The Brain: Routing, Scanning, & RAG logic
│   ├── ingestion/          # The Lungs: Processing PDFs & JSONs
│   ├── consolidator/       # The Nervous System: Linking entities & generating Master Data
│   └── bot/                # The Voice: MS Teams & Adapter logic
├── data/
│   ├── converted/          # Clean JSON sources for Data Interpreter
│   ├── chromadb/           # Local Vector Store
│   └── master_data.md      # Consolidated Context for RAG
```

---

## ⚡ Running the System

1. **Refresh Data**: `python3 -m src.consolidator.run && python3 -m src.ingestion.pipeline`
2. **Start Server**: `uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`
3. **Verify**: `curl -X POST 'http://localhost:8000/api/ask' -d '{"question":"How many in AI team?"}'`

---
*Developed by FiftyFive Technologies - Enterprise RAG Division*
