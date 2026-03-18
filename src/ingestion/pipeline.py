"""
pipeline.py — Data Ingestion Pipeline for ResoAI
=================================================

This module is responsible for the OFFLINE phase of ResoAI: pulling raw data
from two sources, converting it into text chunks, embedding those chunks as
vectors, and storing everything in ChromaDB.

Data sources:
  1. Keka HR API   — structured employee records (name, salary, department, etc.)
  2. Local PDFs    — unstructured company policy documents

Flow overview:
  Keka API  ──┐
               ├─► build Documents ─► SemanticChunker ─► ChromaDB (upsert)
  PDF files ──┘

The pipeline is designed to be IDEMPOTENT — running it multiple times on the
same data will UPDATE existing chunks, not create duplicates. This is achieved
by using a deterministic chunk ID scheme: `{record_type}_{record_id}_chunk_{i}`.

Chunking strategy:
  - Primary:  SemanticChunker splits text at MEANING boundaries (using the
              embedding model to detect topic shifts). This keeps policy rules
              and technical descriptions intact rather than cutting mid-sentence.
  - Fallback: RecursiveCharacterTextSplitter handles any semantic chunk that
              is still too large (> settings.semantic_chunk_max_size chars).
              It splits on paragraph → newline → sentence → word boundaries.

Usage:
  pipeline = IngestionPipeline()
  await pipeline.run()          # full sync: employees + PDFs
  pipeline.ingest_pdfs()        # PDF-only sync (faster)
  await pipeline.close()        # release HTTP client connections
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from src.config import settings
from src.ingestion.document_builder import build_employee_document
from src.ingestion.pdf_loader import load_pdfs
from src.keka.client import KekaClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _get_embeddings(num_thread: int | None = None) -> OllamaEmbeddings:
    """
    Create and return an OllamaEmbeddings instance.

    OllamaEmbeddings calls the local Ollama server to convert text into
    high-dimensional vectors (e.g. 768 dimensions for nomic-embed-text).
    These vectors capture the *semantic meaning* of text, allowing
    ChromaDB to find relevant chunks via cosine similarity.

    Args:
        num_thread: CPU threads to allocate for Ollama. If None, uses
                    settings.ollama_num_threads as the default. During
                    concurrent requests, the caller passes a reduced
                    thread count so multiple requests don't starve each other.

    Returns:
        A configured OllamaEmbeddings instance ready to embed text.

    Note:
        Do NOT change ollama_embedding_model after you have already ingested
        data. The stored vectors in ChromaDB will no longer match the new
        model's vector space, causing incorrect similarity search results.
        If you change the model, you must wipe ChromaDB and re-ingest everything.
    """
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        num_thread=num_thread or settings.ollama_num_threads,
    )


def get_vector_store(num_thread: int | None = None) -> Chroma:
    """
    Create and return a ChromaDB vector store instance with persistent storage.

    ChromaDB is the local vector database that stores all document chunks
    along with their embeddings. It supports:
      - Similarity search (find the k most semantically similar chunks)
      - Metadata filtering (e.g. only search employee records or only PDFs)
      - Text search via $contains (for exact keyword/name matching)
      - Persistent storage (data survives restarts via the persist_directory)

    This function is called:
      1. During ingestion — to write chunks into ChromaDB.
      2. At query time — to read chunks from ChromaDB (via chain.py).

    IMPORTANT: chain.py caches the returned store globally. Do NOT call this
    function on every request — use _get_or_create_store() in chain.py instead.

    Args:
        num_thread: Passed through to the embedding model (see _get_embeddings).

    Returns:
        A Chroma instance connected to the persistent local database.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_get_embeddings(num_thread=num_thread),
        persist_directory=settings.chroma_persist_dir,
    )


# ---------------------------------------------------------------------------
# IngestionPipeline class
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Orchestrates the full data ingestion process: fetch → build → chunk → upsert.

    This class ties together all ingestion components:
      - KekaClient: fetches employee records from the HR API
      - build_employee_document: converts Employee models → LangChain Documents
      - load_pdfs: loads and cleans PDF files → LangChain Documents
      - SemanticChunker: splits documents at meaning boundaries
      - RecursiveCharacterTextSplitter: fallback for oversized semantic chunks
      - ChromaDB: stores the final chunks as vectors

    Typical usage:
        pipeline = IngestionPipeline()
        result = await pipeline.run()
        # result = {"employees_fetched": 50, "pdfs_loaded": 8, "total_chunks": 312}
        await pipeline.close()

    The pipeline is safe to run repeatedly — it uses deterministic IDs to
    upsert (update-or-insert) chunks, so re-running never creates duplicates.
    """

    def __init__(
        self,
        keka_client: KekaClient | None = None,
        vector_store: Chroma | None = None,
    ) -> None:
        """
        Initialize the pipeline with optional dependency injection.

        Args:
            keka_client: An existing KekaClient instance. If None, a new one
                         is created using settings.employee_api_base_url.
                         Useful for testing — inject a mock client.
            vector_store: An existing Chroma instance. If None, a new one is
                          created using get_vector_store(). Useful for testing.
        """
        # HTTP client for the Keka HR API
        self.keka = keka_client or KekaClient()

        # ChromaDB vector store where all chunks are stored
        self.store = vector_store or get_vector_store()

        # ── Primary chunker: SemanticChunker ────────────────────────────────
        # SemanticChunker uses the embedding model to detect where the TOPIC
        # of a document changes, and splits there instead of at arbitrary
        # character counts. This keeps logically related sentences together.
        #
        # breakpoint_threshold_type="standard_deviation" means:
        #   - The chunker calculates the average similarity between consecutive
        #     sentences, then splits when similarity drops below (mean - 1 std dev).
        #   - This is the most conservative setting — it splits only at clear
        #     topic boundaries, producing larger but more coherent chunks.
        #
        # Trade-off: SemanticChunker is SLOWER than RecursiveCharacterTextSplitter
        # because it calls the embedding model during ingestion. This is acceptable
        # since ingestion runs offline, not during user queries.
        self.splitter = SemanticChunker(
            _get_embeddings(),
            breakpoint_threshold_type="standard_deviation",
        )

        # ── Fallback chunker: RecursiveCharacterTextSplitter ─────────────────
        # Used when a semantic chunk is still too large (> semantic_chunk_max_size).
        # This can happen with dense policy pages that discuss one topic for 3000+
        # characters without any clear semantic break.
        #
        # The splitter tries to split at the listed separators IN ORDER:
        #   "\n\n" → paragraph break (best split point)
        #   "\n"   → line break
        #   ". "   → sentence boundary
        #   " "    → word boundary
        #   ""     → character-level (last resort, avoids mid-word splits)
        #
        # chunk_size and chunk_overlap are now read from settings (previously
        # hardcoded as 2000 and 200 respectively — now 300 and 50).
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,        # max chars per chunk (default: 300)
            chunk_overlap=settings.chunk_overlap,  # overlap between chunks (default: 50)
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # -------------------------------------------------------------------------
    # Step 1: Fetch & Transform
    # -------------------------------------------------------------------------

    async def _fetch_employee_docs(self) -> list[Document]:
        """
        Fetch all employee records from the Keka API and convert to Documents.

        Calls KekaClient.get_all_employees() which handles pagination
        (fetching 100 employees per page until all are retrieved).

        Each Employee model is then passed to build_employee_document() which
        formats it into a human-readable text block like:
            "Employee: John Smith
             Department: Engineering
             Salary: 1200000.00
             ..."
        This text format is what gets embedded and stored in ChromaDB.

        Returns:
            List of LangChain Document objects, one per employee.
        """
        employees = await self.keka.get_all_employees()
        logger.info("Fetched %d employees from API", len(employees))
        return [build_employee_document(e) for e in employees]

    def _load_pdf_docs(self) -> list[Document]:
        """
        Load and clean all PDF files from the configured policies folder.

        Delegates to load_pdfs() in pdf_loader.py which:
          1. Recursively finds all *.pdf files under settings.pdf_folder
          2. Loads each PDF using PyPDFLoader
          3. Merges all pages of each PDF into a single Document
          4. Strips boilerplate headers (e.g. "Document Release Notice...")
          5. Flattens fragmented whitespace (common in policy PDFs)

        Returns:
            List of LangChain Documents, one per PDF file (not per page).
        """
        docs = load_pdfs(settings.pdf_folder)
        logger.info("Loaded %d PDF pages from %s", len(docs), settings.pdf_folder)
        return docs

    # -------------------------------------------------------------------------
    # Step 2: Chunk
    # -------------------------------------------------------------------------

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        """
        Split a list of Documents into smaller chunks suitable for embedding.

        Why chunking is necessary:
            Embedding models have a maximum token limit (nomic-embed-text: 8192).
            A full policy PDF might be 50,000 characters. If we embedded the
            whole thing as one vector, semantic search would match the document
            topic but NOT find the specific clause the user asked about.
            Chunking splits the document so each chunk represents ONE idea,
            making retrieval far more precise.

        Two-pass chunking strategy:
            Pass 1 — SemanticChunker:
                Splits the document at meaning boundaries using embedding
                similarity. Output chunks are coherent and topic-focused.

            Pass 2 — RecursiveCharacterTextSplitter (fallback):
                If any semantic chunk exceeds settings.semantic_chunk_max_size
                (default: 1500 chars), it is further split by the character
                splitter. This handles edge cases where one policy section is
                very long with no detectable topic change.

        Chunk ID scheme:
            Each chunk gets a deterministic record_id: "{base_id}_c{i}"
            This ensures re-ingestion REPLACES old chunks rather than
            creating duplicates. For example:
                "emp_EMP001_c0", "emp_EMP001_c1", "emp_EMP001_c2"
                "leave_policy_c0", "leave_policy_c1", ...

        Args:
            docs: List of full Documents (employees or PDFs) to be chunked.

        Returns:
            List of chunk Documents, each with updated metadata including
            chunk_index and a unique record_id.
        """
        all_chunks: list[Document] = []

        for doc in docs:
            # Pass 1: Semantic split — find meaning boundaries
            semantic_chunks = self.splitter.split_documents([doc])

            # Pass 2: Fallback split — enforce maximum chunk size
            final_doc_chunks: list[Document] = []
            for s_chunk in semantic_chunks:
                if len(s_chunk.page_content) > settings.semantic_chunk_max_size:
                    # Semantic chunk too large — split further by character count
                    logger.debug(
                        "Semantic chunk too large (%d chars), applying fallback split.",
                        len(s_chunk.page_content),
                    )
                    final_doc_chunks.extend(
                        self.fallback_splitter.split_documents([s_chunk])
                    )
                else:
                    final_doc_chunks.append(s_chunk)

            # Assign unique IDs to each chunk of this document
            base_id = doc.metadata.get("record_id", "doc")
            for i, chunk in enumerate(final_doc_chunks):
                # chunk_index helps debug retrieval — you can see which part
                # of the original document a chunk came from
                chunk.metadata["chunk_index"] = i
                # Unique record_id per chunk — used for ChromaDB deduplication
                chunk.metadata["record_id"] = f"{base_id}_c{i}"
                all_chunks.append(chunk)

        logger.info("Split %d documents into %d chunks", len(docs), len(all_chunks))
        return all_chunks

    # -------------------------------------------------------------------------
    # Step 3: Upsert into ChromaDB
    # -------------------------------------------------------------------------

    def _upsert(self, chunks: list[Document]) -> None:
        """
        Write document chunks into ChromaDB, replacing any existing data.

        ChromaDB's add_documents() with explicit IDs performs an UPSERT:
          - If a document with that ID already exists → it is REPLACED.
          - If it doesn't exist yet → it is INSERTED.

        This means running the ingestion pipeline twice on the same data
        is safe — you'll end up with exactly the same chunks in ChromaDB,
        not doubled data.

        ID format: "{record_type}_{record_id}_chunk_{i}"
        Examples:
            "employee_EMP001_c0_chunk_0"
            "pdf_leave_policy_c3_chunk_3"

        Args:
            chunks: List of Document chunks to store. Each must have
                    'record_id' and 'record_type' in its metadata.
        """
        ids: list[str] = []
        for i, chunk in enumerate(chunks):
            record_id   = chunk.metadata.get("record_id", f"unknown_{i}")
            record_type = chunk.metadata.get("record_type", "doc")
            # Build a globally unique, stable ID for this chunk
            ids.append(f"{record_type}_{record_id}_chunk_{i}")

        if chunks:
            self.store.add_documents(documents=chunks, ids=ids)
            logger.info("Upserted %d chunks into ChromaDB", len(chunks))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self) -> dict[str, int]:
        """
        Execute the FULL ingestion pipeline: employees + PDFs.

        This is the primary method to call for a complete data sync.
        It fetches all employee records from the API, loads all PDFs from
        the configured folder, chunks everything, and stores it in ChromaDB.

        Call this:
          - On first setup (to populate the vector database)
          - After HR data changes (new hires, terminations, salary updates)
          - After adding new policy PDFs to the folder
          - Automatically via the scheduler every settings.ingestion_interval_hours

        Returns:
            A summary dict with ingestion counts, e.g.:
            {
                "employees_fetched": 50,
                "pdfs_loaded": 8,
                "total_chunks": 312
            }
        """
        logger.info("Starting full ingestion pipeline (employees + PDFs)...")

        # Fetch from both sources in sequence
        employee_docs = await self._fetch_employee_docs()
        pdf_docs      = self._load_pdf_docs()

        # Combine and process
        all_docs = employee_docs + pdf_docs
        chunks   = self._chunk_documents(all_docs)
        self._upsert(chunks)

        summary = {
            "employees_fetched": len(employee_docs),
            "pdfs_loaded":       len(pdf_docs),
            "total_chunks":      len(chunks),
        }
        logger.info("Full ingestion complete: %s", summary)
        return summary

    def ingest_pdfs(self) -> dict[str, int]:
        """
        Execute a PDF-ONLY ingestion (skips the employee API call).

        Use this when you've added or updated policy PDF files but employee
        data hasn't changed. It's faster than the full pipeline since it
        skips the API call and only processes the PDF folder.

        Also called by the /api/ingest-pdfs endpoint when a PDF is uploaded
        directly via the FastAPI web interface.

        Returns:
            A summary dict with ingestion counts, e.g.:
            {
                "pdfs_loaded": 8,
                "total_chunks": 187
            }
        """
        logger.info("Starting PDF-only ingestion...")

        pdf_docs = self._load_pdf_docs()
        chunks   = self._chunk_documents(pdf_docs)
        self._upsert(chunks)

        summary = {
            "pdfs_loaded":  len(pdf_docs),
            "total_chunks": len(chunks),
        }
        logger.info("PDF ingestion complete: %s", summary)
        return summary

    async def close(self) -> None:
        """
        Release all resources held by the pipeline.

        This closes the underlying httpx.AsyncClient inside KekaClient,
        which releases the TCP connection pool. Always call this when done
        with the pipeline to avoid resource leaks, especially in long-running
        processes or tests.

        Example:
            pipeline = IngestionPipeline()
            try:
                await pipeline.run()
            finally:
                await pipeline.close()
        """
        await self.keka.close()