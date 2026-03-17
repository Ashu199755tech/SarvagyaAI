# ResoAI Agent Process Architecture

This document visualizes the internal "brain" and data flow process of the ResoAI agent.

## End-to-End Agent Process (Sequence Diagram)

This diagram shows how data travels through the system from the moment it's ingested to the moment a user receives an answer.

```mermaid
sequenceDiagram
    participant K as Keka API / PDFs
    participant I as Ingestion Pipeline
    participant DB as ChromaDB (Vector Store)
    participant U as User (Teams/API)
    participant A as ResoAI (LangChain)
    participant LLM as Ollama (Llama-3.2)

    Note over K, I: [Phase 1: Knowledge Acquisition]
    K->>I: Raw Data (JSON/PDF)
    I->>I: Chunking & Embedding
    I->>DB: Upsert Vectors + Metadata

    Note over U, LLM: [Phase 2: Query & Reasoning]
    U->>A: Ask: "How many leaves...?"
    A->>A: Name Extraction (Anthony)
    A->>DB: Semantic Search + Name Match
    DB-->>A: Relevant Context Chunks
    
    A->>LLM: Prompt (System Rules + Context + Query)
    LLM->>LLM: Local Inference
    LLM-->>A: Clean, Concise Answer
    A->>U: Final Response: "24 leaves"
```

## Functional Architecture

![Agent Process Workflow](/home/fifity-five/.gemini/antigravity/brain/ce90508a-4944-4b59-95c5-45f49de222bf/agent_process_flow_mockup_1773312622711.png)

### Core Pipeline Stages

1.  **Ingestion & Vectorization**: 
    - The agent fetches data from the **Employee API** (structured) and **Local PDFs** (unstructured).
    - It uses the **Nomic-Embed** model to convert text into multi-dimensional vectors.
2.  **Contextual Retrieval**:
    - When a query arrives, the agent performs a **Hybrid Search**. It looks for exact name matches (to avoid cross-employee confusion) and semantic matches (to find the right policy).
3.  **Prompt Orchestration**:
    - The agent wraps the retrieved data in a strict **System Prompt** that enforces brevity, numerical precision, and professional tone.
4.  **Local Inference**:
    - The **Ollama** engine runs the **Llama-3.2** model to synthesize the final answer using *only* the provided context, preventing hallucinations.

---
*Created by ResoAI Agent*
