"""
pipeline.py — Dynamic Universal Ingestion Pipeline
===================================================

Flow:
  1. Convert any files in data/inbox/ → data/converted/  (via UniversalConverter)
  2. Scan data/converted/*.json + data/*.json for all JSON sources
  3. For each JSON file:
       - Check BUILDER_REGISTRY for a specialized builder
       - Otherwise use build_generic_document() for zero-config ingestion
  4. Chunk & upsert into ChromaDB
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
from src.rag.store import get_vector_store
from src.ingestion.document_builder import (
    BUILDER_REGISTRY,
    build_generic_document,
    build_misc_document,
)
from src.ingestion.universal_converter import convert_inbox

logger = logging.getLogger(__name__)

DATA_DIR      = Path(settings.data_dir)
INBOX_DIR     = Path(settings.inbox_dir)
CONVERTED_DIR = Path(settings.converted_dir)

# Legacy known JSON files (handled by specialized registry builders)
LEGACY_JSON_FILES = [
    DATA_DIR / "employees.json",
    DATA_DIR / "projects.json",
    DATA_DIR / "policies.json",
    DATA_DIR / "holidays.json",
    DATA_DIR / "misc_docs.json",
]

# Keywords to identify files already handled by specialized consolidator (SOT)
CORE_ASSET_KEYWORDS = {
    # Core Entities
    "holiday", "directory", "case studies", "employee", "project",
    # Policies
    "aup", "asset", "attendance", "business", "clear desk", "code of conduct", 
    "continual", "leave", "posh", "referral", "separation", "travel", "ethics",
    # Misc Docs
    "escalation", "culture", "benefits", "intern", "nonconformity", "performance", "probation"
}


def _is_core_asset(filepath: Path) -> bool:
    """Check if a file should be skipped by the generic converter because it's an SOT asset."""
    val = filepath.name.lower()
    return any(kw in val for kw in CORE_ASSET_KEYWORDS)





def _build_docs_from_json(json_path: Path) -> list[Document]:
    """Load a JSON file and build LangChain Documents using the registry or generic builder."""
    filename = json_path.name
    try:
        records = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read JSON: %s", json_path)
        return []

    if not isinstance(records, list):
        records = [records]

    # ── Specialized builder (registry) ───────────────────────────────────────
    if filename in BUILDER_REGISTRY:
        builder = BUILDER_REGISTRY[filename]
        docs: list[Document] = []
        for record in records:
            result = builder(record)
            if isinstance(result, list):
                docs.extend(result)
            else:
                docs.append(result)
        logger.info("Registry builder: %d docs from %s", len(docs), filename)
        return docs

    # ── Generic builder (auto-mode) ───────────────────────────────────────────
    docs = [
        build_generic_document(record, source_filename=filename, index=i)
        for i, record in enumerate(records)
    ]
    logger.info("Generic builder: %d docs from %s", len(docs), filename)
    return docs


class IngestionPipeline:
    def __init__(self, vector_store: Chroma | None = None) -> None:
        self.store = vector_store or get_vector_store()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion_chunk_size,
            chunk_overlap=settings.ingestion_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def truncate(self) -> None:
        logger.warning("Truncating ChromaDB collection: %s", settings.chroma_collection_name)
        try:
            from src.rag.chain import clear_vector_store_cache
            
            clear_cache()
            clear_vector_store_cache()
            
            self.store.delete_collection()
            self.store = get_vector_store()
            logger.info("Collection truncated successfully.")
        except Exception as e:
            logger.error("Failed to truncate collection: %s", e)

    def _convert_inbox(self) -> None:
        """Convert any non-JSON files in inbox/ → converted/ as JSON."""
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        
        # CLEANUP: Clear converted directory to remove stale redundant files
        logger.info("Cleaning up converted directory: %s", CONVERTED_DIR)
        for f in CONVERTED_DIR.glob("*.json"):
            f.unlink()

        from src.ingestion.universal_converter import CONVERTERS, convert_to_json
        
        new_files = []
        for f in sorted(INBOX_DIR.iterdir()):
            if f.suffix.lower() in CONVERTERS and f.suffix.lower() != ".json":
                # Skip core assets handled by SOT consolidator
                if _is_core_asset(f):
                    logger.debug("Skipping core asset during generic conversion: %s", f.name)
                    continue
                try:
                    out = convert_to_json(f, output_dir=CONVERTED_DIR)
                    new_files.append(out)
                except Exception:
                    logger.exception("Failed to convert %s", f.name)
        
        if new_files:
            logger.info("Converted %d new file(s) from inbox: %s", len(new_files), [f.name for f in new_files])

    def _collect_json_files(self) -> list[Path]:
        """Collect all JSON files to ingest: legacy + converted/."""
        files: list[Path] = []

        # 1. Legacy core JSON files
        for p in LEGACY_JSON_FILES:
            if p.exists():
                files.append(p)

        # 2. Auto-converted files from inbox (skip those already handled as legacy core)
        if CONVERTED_DIR.exists():
            for p in sorted(CONVERTED_DIR.glob("*.json")):
                if _is_core_asset(p):
                    continue
                files.append(p)

        logger.info("Collected %d JSON files to ingest", len(files))
        return files

    def _load_all_docs(self) -> list[Document]:
        """Convert inbox files then load all JSON files into Documents."""
        self._convert_inbox()
        json_files = self._collect_json_files()
        all_docs: list[Document] = []
        for jf in json_files:
            all_docs.extend(_build_docs_from_json(jf))
        logger.info("Total documents loaded: %d", len(all_docs))
        return all_docs

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        ATOMIC_TYPES = {"project", "holiday", "employee", "misc_table_row"}
        for doc in docs:
            record_type = doc.metadata.get("record_type", "")

            # Atomic records: never split them
            if record_type in ATOMIC_TYPES:
                chunk = doc.copy()
                chunk.metadata["chunk_index"] = 0
                chunk.metadata["parent_record_id"] = doc.metadata.get("record_id", "doc")
                chunk.metadata.setdefault("record_id", "doc")
                all_chunks.append(chunk)
                continue

            # Generic records from converted files: also keep atomic if small
            if len(doc.page_content) <= settings.ingestion_atomic_limit:
                chunk = doc.copy()
                chunk.metadata["chunk_index"] = 0
                chunk.metadata["parent_record_id"] = doc.metadata.get("record_id", "doc")
                chunk.metadata.setdefault("record_id", "doc")
                all_chunks.append(chunk)
                continue

            # Large text documents: split recursively
            doc_chunks = self.splitter.split_documents([doc])
            base_id = doc.metadata.get("record_id", "doc")
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["parent_record_id"] = base_id
                chunk.metadata["record_id"] = f"{base_id}_c{i}"
            all_chunks.extend(doc_chunks)

        return all_chunks

    def _upsert_batches(self, chunks: list[Document], batch_size: int = 50) -> None:
        """Upsert in batches to avoid timeouts and memory issues."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [f"json_{c.metadata['record_id']}_{j}" for j, c in enumerate(batch)]
            self.store.add_documents(documents=batch, ids=ids)
            logger.info(
                "Upserted batch %d/%d (%d chunks)",
                (i // batch_size) + 1,
                -(-len(chunks) // batch_size),
                len(batch),
            )

    async def run(self, perform_truncate: bool = True) -> dict[str, int]:
        logger.info("Starting Dynamic Universal Ingestion...")
        if perform_truncate:
            self.truncate()

        docs = self._load_all_docs()
        chunks = self._chunk_documents(docs)
        self._upsert_batches(chunks)
        clear_cache()

        return {"documents_loaded": len(docs), "total_chunks": len(chunks)}

    async def ingest_file(self, json_path: Path) -> dict[str, int]:
        """Incrementally ingest a single already-converted JSON file (no truncate)."""
        docs = _build_docs_from_json(json_path)
        chunks = self._chunk_documents(docs)
        self._upsert_batches(chunks)
        clear_cache()
        return {"documents_loaded": len(docs), "total_chunks": len(chunks)}

    async def close(self) -> None:
        pass


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    pipeline = IngestionPipeline()
    try:
        results = asyncio.run(pipeline.run())
        print(f"Ingestion complete: {results}")
    except Exception:
        logger.exception("Ingestion failed")