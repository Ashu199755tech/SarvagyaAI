"""RAG ingestion pipeline: Employee API + PDFs → Documents → Chunks → ChromaDB."""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.document_builder import build_employee_document
from src.ingestion.pdf_loader import load_pdfs
from src.keka.client import KekaClient

logger = logging.getLogger(__name__)


def _get_embeddings(num_thread: int | None = None) -> OllamaEmbeddings:
    """Create the Ollama embedding model instance."""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        num_thread=num_thread or settings.ollama_num_threads,
    )


def get_vector_store(num_thread: int | None = None) -> Chroma:
    """Return a Chroma vector store instance with persistent storage."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_get_embeddings(num_thread=num_thread),
        persist_directory=settings.chroma_persist_dir,
    )


class IngestionPipeline:
    """Fetches data from Employee API and PDFs, builds documents, chunks, and upserts to ChromaDB."""

    def __init__(
        self,
        keka_client: KekaClient | None = None,
        vector_store: Chroma | None = None,
    ) -> None:
        self.keka = keka_client or KekaClient()
        self.store = vector_store or get_vector_store()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Fetch & Transform ────────────────────────────────

    async def _fetch_employee_docs(self) -> list[Document]:
        """Fetch all employees from the API and convert to Documents."""
        employees = await self.keka.get_all_employees()
        logger.info("Fetched %d employees from API", len(employees))
        return [build_employee_document(e) for e in employees]

    def _load_pdf_docs(self) -> list[Document]:
        """Load all PDFs from the configured folder."""
        docs = load_pdfs(settings.pdf_folder)
        logger.info("Loaded %d PDF pages from %s", len(docs), settings.pdf_folder)
        return docs

    # ── Chunk ────────────────────────────────────────────

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        """Split documents into smaller chunks for embedding and add unique IDs."""
        all_chunks = []
        for doc in docs:
            doc_chunks = self.splitter.split_documents([doc])
            base_id = doc.metadata.get("record_id", "doc")
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                # Make record_id unique per chunk for deduplication in RAG chain
                chunk.metadata["record_id"] = f"{base_id}_c{i}"
                all_chunks.append(chunk)
        
        logger.info("Split %d documents into %d chunks", len(docs), len(all_chunks))
        return all_chunks

    # ── Upsert ───────────────────────────────────────────

    def _upsert(self, chunks: list[Document]) -> None:
        """Add or replace document chunks in ChromaDB.

        Uses record_id + chunk index as the ChromaDB document ID
        so re-ingestion replaces stale data instead of duplicating.
        """
        ids: list[str] = []
        for i, chunk in enumerate(chunks):
            record_id = chunk.metadata.get("record_id", f"unknown_{i}")
            record_type = chunk.metadata.get("record_type", "doc")
            ids.append(f"{record_type}_{record_id}_chunk_{i}")

        if chunks:
            self.store.add_documents(documents=chunks, ids=ids)
            logger.info("Upserted %d chunks into ChromaDB", len(chunks))

    # ── Public API ───────────────────────────────────────

    async def run(self) -> dict[str, int]:
        """Execute the full ingestion pipeline (employees + PDFs).

        Returns a summary dict with counts.
        """
        logger.info("Starting ingestion pipeline …")

        # 1 – Fetch employees
        employee_docs = await self._fetch_employee_docs()

        # 2 – Load PDFs
        pdf_docs = self._load_pdf_docs()

        all_docs = employee_docs + pdf_docs

        # 3 – Chunk
        chunks = self._chunk_documents(all_docs)

        # 4 – Upsert
        self._upsert(chunks)

        summary = {
            "employees_fetched": len(employee_docs),
            "pdfs_loaded": len(pdf_docs),
            "total_chunks": len(chunks),
        }
        logger.info("Ingestion complete: %s", summary)
        return summary

    def ingest_pdfs(self) -> dict[str, int]:
        """Ingest only PDF files (no employee API call).

        Returns a summary dict with counts.
        """
        logger.info("Starting PDF-only ingestion …")

        pdf_docs = self._load_pdf_docs()
        chunks = self._chunk_documents(pdf_docs)
        self._upsert(chunks)

        summary = {
            "pdfs_loaded": len(pdf_docs),
            "total_chunks": len(chunks),
        }
        logger.info("PDF ingestion complete: %s", summary)
        return summary 

    async def close(self) -> None:
        """Clean up resources."""
        await self.keka.close()
