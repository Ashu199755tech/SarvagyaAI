"""
pipeline.py — Optimized JSON-First Ingestion Pipeline
======================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.rag.cache import clear_cache
from src.ingestion.document_builder import (
    build_employee_document,
    build_project_document,
    build_policy_document,
    build_holiday_document,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
SOT_PROJECTS  = DATA_DIR / "projects.json"
SOT_EMPLOYEES = DATA_DIR / "employees.json"
SOT_POLICIES  = DATA_DIR / "policies.json"
SOT_HOLIDAYS  = DATA_DIR / "holidays.json"


def _get_embeddings(num_thread: int | None = None) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        num_thread=num_thread or settings.ollama_num_threads,
    )


def get_vector_store(num_thread: int | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_get_embeddings(num_thread=num_thread),
        persist_directory=settings.chroma_persist_dir,
    )


class IngestionPipeline:
    def __init__(self, vector_store: Chroma | None = None) -> None:
        self.store = vector_store or get_vector_store()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Faster ingestion
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def truncate(self) -> None:
        logger.warning("Truncating ChromaDB collection: %s", settings.chroma_collection_name)
        try:
            self.store.delete_collection()
            self.store = get_vector_store()
            logger.info("Collection truncated successfully.")
        except Exception as e:
            logger.error("Failed to truncate collection: %s", e)

    def _load_json_docs(self) -> list[Document]:
        all_docs = []
        if SOT_EMPLOYEES.exists():
            records = json.loads(SOT_EMPLOYEES.read_text(encoding="utf-8"))
            all_docs.extend([build_employee_document(r) for r in records])
        if SOT_PROJECTS.exists():
            records = json.loads(SOT_PROJECTS.read_text(encoding="utf-8"))
            all_docs.extend([build_project_document(r) for r in records])
        if SOT_POLICIES.exists():
            records = json.loads(SOT_POLICIES.read_text(encoding="utf-8"))
            all_docs.extend([build_policy_document(r) for r in records])
        if SOT_HOLIDAYS.exists():
            records = json.loads(SOT_HOLIDAYS.read_text(encoding="utf-8"))
            all_docs.extend([build_holiday_document(r) for r in records])
        return all_docs

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            doc_chunks = self.splitter.split_documents([doc])
            base_id = doc.metadata.get("record_id", "doc")
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["record_id"] = f"{base_id}_c{i}"
                all_chunks.append(chunk)
        return all_chunks

    def _upsert_batches(self, chunks: list[Document], batch_size: int = 50) -> None:
        """Upsert in batches to avoid timeouts and memory issues."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [f"json_{c.metadata['record_id']}_{j}" for j, c in enumerate(batch)]
            self.store.add_documents(documents=batch, ids=ids)
            logger.info("Upserted batch %d/%d (%d chunks)", (i // batch_size) + 1, (len(chunks) // batch_size) + 1, len(batch))

    async def run(self, perform_truncate: bool = True) -> dict[str, int]:
        logger.info("Starting Optimized JSON-First Ingestion...")
        if perform_truncate:
            self.truncate()

        docs = self._load_json_docs()
        chunks = self._chunk_documents(docs)
        self._upsert_batches(chunks)
        
        # --- CACHE CLEANUP ---
        # Wipe the RAG cache after successful ingestion to prevent stale answers.
        clear_cache()
        # ---------------------

        return {"documents_loaded": len(docs), "total_chunks": len(chunks)}

    async def close(self) -> None:
        pass