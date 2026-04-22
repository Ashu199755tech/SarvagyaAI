# ResoAI High-Level Architecture

ResoAI is an **Enterprise-Grade RAG Agent** designed for data privacy and high-precision retrieval. It uses a **Hybrid Agentic Architecture** that dynamically routes queries between semantic vector search and deterministic data scanning to ensure 100% accuracy for quantitative questions.

---

## 🏗️ System Overview

The system follows a local-first pattern. All data, embeddings, and LLM logical processing remain on-premises (via **Ollama**), ensuring strict data sovereignty.

### Hybrid Routing Architecture

```mermaid
graph TD
    User([User Query]) --> PreProcess[Query Pre-Processing<br/>Alias Expansion & Cleaning]
    PreProcess --> Router{NLP Intent Router<br/>(spaCy Lemmatized)}
    
    %% Path A: Data Interpreter
    Router -- "Structured Intent<br/>(Counts/Lists/Who)" --> DI[Data Interpreter]
    DI --> JSON[(Unified JSON SOTs<br/>data/*.json)]
    JSON --> DI
    DI --> Response([Final Response])
    
    %% Path B: RAG Engine
    Router -- "Semantic Intent<br/>(Policies/Explanations)" --> RAG[Hybrid RAG Engine]
    RAG --> Vector[(ChromaDB<br/>Vector Store)]
    RAG --> BM25[BM25 Keyword Search]
    Vector --> Merge[Merge & Re-rank]
    BM25 --> Merge
    Merge --> LLM[Local LLM<br/>llama3.2]
    LLM --> Response
```

---

## 🛠️ Core Components

### 1. Data Ingestion & Consolidation
- **Consolidator Engine**: Automatically rebuilds Source-of-Truth (SOT) JSONs for employees, holidays, and projects directly from raw PDFs and APIs without LLM interference.
- **Universal Converter**: Handles long-form policy documents by converting them into chunkable JSON for the semantic RAG path.

### 2. The "Brain" (NLP Hybrid Orchestrator)
- **NLP Intent Router**: Uses **spaCy lemmatization** to identify user intent. It resolves verb forms (e.g., "praised" -> "praise") to match structured entities dynamically, bypassing the vector store for aggregations.
- **Data Interpreter**: Scans unified JSON sources in real-time. It uses fuzzy matching, plural-aware criteria, and phrase scrubbing to ensure sub-second accuracy for quantitative questions.
- **Vector RAG Engine**: The semantic fallback. Combines Vector Similarity with BM25 keyword matching and cross-encoder re-ranking for qualitative answers.

### 3. Local AI Infrastructure
- **Ollama**: The heart of the local stack. It hosts `llama3.2` for reasoning and `nomic-embed-text` for semantic mapping.
- **ChromaDB**: Persists document "embeddings" (mathematical fingerprints) locally.

---

## ⚙️ Dynamic Configuration

The system is **Client-Agnostic**. All hardcoded references have been removed from the source code.
- **config.py**: Centralizes all company names, aliases, file paths, and department mappings.
- **.env Overrides**: Deploy for any company by simply updating the environment variables—no coding required.

---
*Last Updated: April 2026*
