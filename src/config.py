"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for ResoAI, loaded from .env or environment."""

    # ── Employee API ─────────────────────────────────────
    employee_api_base_url: str = "http://localhost:8000"

    # ── PDF Ingestion ────────────────────────────────────
    pdf_folder: str = "/home/fifity-five/Downloads/Policies 2026"

    # ── Ollama (local LLM) ───────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_max_threads: int = 12  # Maximum CPU threads to allocate across all active requests
    ollama_num_threads: int = 6   # Default/fallback threads for background tasks

    # ── ChromaDB ─────────────────────────────────────────
    chroma_persist_dir: str = "./data/chromadb"
    chroma_collection_name: str = "resoai_hr"

    # ── MS Teams Bot ─────────────────────────────────────
    teams_app_id: str = ""
    teams_app_password: str = ""

    # ── Ingestion ────────────────────────────────────────
    ingestion_interval_hours: int = 6

    # ── RAG ──────────────────────────────────────────────
    retriever_top_k: int = 20
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ── Server ───────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8001

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton settings instance
settings = Settings()
