"""Load PDF files from a folder into LangChain Documents."""

from __future__ import annotations

import logging
import re
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

            # Merge all pages into a single document so chunks can span boundaries
            full_text = []
            for doc in pages:
                text = doc.page_content
                # Strip standard boilerplate header if present
                # "Document Release Notice This FiftyFive Technologies Policy on ..."
                boilerplate_pattern = r"(Document Release Notice.*?effect from \d{2} \w+ \d{4}\.)"
                text = re.sub(boilerplate_pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

                # The PDF is heavily fragmented with absolute positioned words causing
                # newlines between almost every word. Flatten ALL whitespace to a single space.
                text = re.sub(r'\s+', ' ', text)
                full_text.append(text.strip())
            
            # Since everything is flattened, join pages with a space
            merged_content = " ".join(full_text)
            
            merged_doc = Document(
                page_content=merged_content,
                metadata={
                    "record_type": "pdf",
                    "record_id": pdf_path.stem,
                    "filename": pdf_path.name,
                    "category": pdf_path.parent.name,
                    "source": str(pdf_path),
                }
            )

            all_docs.append(merged_doc)
            logger.info("Loaded %d pages and merged into 1 doc for %s", len(pages), pdf_path.name)

        except Exception:
            logger.exception("Failed to load PDF: %s", pdf_path)

    logger.info("Total: loaded %d pages from %d PDFs", len(all_docs), len(pdf_files))
    return all_docs
