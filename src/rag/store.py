"""
store.py — Vector Store and Embedding Factories
===============================================

Centrally manages connections to ChromaDB and Ollama embeddings.
By isolating these factories, we avoid circular dependencies between
the ingestion pipeline and the RAG query chain.
"""

import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)

def _get_embeddings(num_thread: int | None = None) -> OllamaEmbeddings:
    """Return a fresh OllamaEmbeddings instance."""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        num_thread=num_thread or settings.ollama_num_threads,
    )

def get_vector_store(num_thread: int | None = None) -> Chroma:
    """Return a connection to the local ChromaDB vector store."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_get_embeddings(num_thread=num_thread),
        persist_directory=settings.chroma_persist_dir,
    )
