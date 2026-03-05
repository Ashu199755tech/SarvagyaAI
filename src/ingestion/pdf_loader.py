"""Load PDF files from a folder into LangChain Documents."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdfs(folder_path: str) -> list[Document]:
    """Recursively scan a folder for PDF files and load them as Documents.

    Each page becomes a separate Document with metadata including
    the source file path, filename, and page number.
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("PDF folder does not exist: %s", folder_path)
        return []

    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s", folder_path)
        return []

    all_docs: list[Document] = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Enrich metadata for each page
            for doc in pages:
                doc.metadata.update({
                    "record_type": "pdf",
                    "record_id": f"{pdf_path.stem}_p{doc.metadata.get('page', 0)}",
                    "filename": pdf_path.name,
                    "category": pdf_path.parent.name,
                    "source": str(pdf_path),
                })

            all_docs.extend(pages)
            logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)

        except Exception:
            logger.exception("Failed to load PDF: %s", pdf_path)

    logger.info("Total: loaded %d pages from %d PDFs", len(all_docs), len(pdf_files))
    return all_docs
