# ResoAI Agent Build Process

This document outlines the step-by-step process I followed to build the ResoAI HR Chatbot agent. The goal was to create a local, privacy-first RAG (Retrieval-Augmented Generation) assistant that integrates directly into Microsoft Teams.

## 1. Project Initialization & Dependencies
I started by setting up a fresh Python environment to isolate the project dependencies.
- Created a virtual environment (`python3 -m venv .venv`).
- Installed the required libraries including `FastAPI` (for the server), `LangChain` and `ChromaDB` (for the RAG vector database), `Ollama` (for local LLM inference), and `botbuilder-core` (for MS Teams integration). 
- Configured environment variables inside a `.env` file to handle API URLs, ports, and MS Teams App credentials.

## 2. Setting Up the Local LLM (Ollama)
Instead of relying on a paid cloud API which could compromise sensitive HR data, I opted for a local, open-source model.
- Installed Ollama on my local machine.
- Pulled the `llama3.2` model for high-speed text generation and the `nomic-embed-text` model for generating vector embeddings.

## 3. Data Ingestion Pipeline
To make the agent useful, it needed context about the company. I built a dual-source ingestion pipeline:
- **Employee Data API:** I wrote a Python script (`src/keka/client.py`) to fetch mock employee profiles (names, departments, salaries, joining dates) from a local HR API endpoint.
- **PDF Documents:** I used LangChain's `PyPDFLoader` to parse company policy documents (e.g., Leave Policy, Travel Policy) from a local folder.
- **Vectorization:** I combined both data sources, broke them into manageable chunks using `RecursiveCharacterTextSplitter`, created vector embeddings via Ollama, and stored them persistently in a local `ChromaDB` database.

## 4. Building the RAG Chain
I created the core intelligence of the agent inside `src/rag/chain.py`.
- **Intelligent Retrieval:** When a user asks a question, the agent extracts names (if applicable) to do direct text-matching against employee records, while simultaneously running a semantic vector search against the policy PDFs.
- **System Prompts:** I engineered strict system instructions in `src/rag/prompts.py` to dictate how the AI should behave. For example, I instructed it to give extremely concise, numerical answers for "how many" questions, and to explicitly list out individual names if a query resulted in multiple employee matches.
- **Generation:** The retrieved documents and the user's question are passed to the `llama3.2` model to generate a professional, accurate response based *only* on the provided context.

## 5. Integrating with FastAPI
I needed a way for external services to communicate with the Python code.
- Built a `FastAPI` web server (`src/main.py`).
- Created a clean `/api/ask` POST endpoint using Pydantic `BaseModels` to define the request JSON schema, allowing the endpoint to be easily testable via FastAPI's auto-generated Swagger UI (`/docs`).

## 6. Microsoft Teams Integration
The final piece was bridging the gap between my local server and Microsoft Teams.
- Implemented the `TurnContext` and `CloudAdapter` logic from the Microsoft Bot Framework (`src/bot/adapter.py` and `src/bot/teams_bot.py`).
- Added a `/api/messages` POST webhook endpoint in FastAPI to receive traffic directly from Azure/MS Teams.
- Used `ngrok` to expose my local port 8001 to the public internet securely, allowing the Azure Bot Channel to route employee messages straight to my local RAG pipeline.

## 7. Visual Architecture Design
- **Goal:** Create a high-level visual representation of the entire system.
- **Actions:**
  - Designed a technical Mermaid flowchart mapping the request path from MS Teams to Ollama.
  - Generated a premium, high-fidelity visual architecture diagram to help stakeholders visualize the secure "Local-First" infrastructure.
  - **Reference:** [ARCHITECTURE.md](file:///home/fifity-five/projects/ResoAi/ARCHITECTURE.md)
