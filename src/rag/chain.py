"""RAG query chain: retrieve context from ChromaDB and answer via Ollama."""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import settings
from src.ingestion.pipeline import get_vector_store
from src.rag.prompts import RAG_PROMPT

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM."""
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
        num_thread=settings.ollama_num_threads,
    )


def _format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _extract_names(question: str) -> list[str]:
    """Extract likely person names (consecutive capitalized word pairs) from the question.

    Returns full names like ["Clayton Walker"] rather than individual tokens.
    """
    common_words = {
        "What", "Who", "How", "When", "Where", "Which", "Why", "Can", "Tell",
        "Does", "List", "Show", "Find", "Give", "The", "And", "For", "About",
        "Many", "Much", "Please", "Employee", "Department", "Project", "Salary",
        "Leave", "Policy", "Travel", "Manager", "Engineer", "Lead", "Benefits",
        "Quote", "Exact", "Clause", "Per", "Year", "Month", "Day", "Sick",
        "Casual", "Privilege", "Annual", "Total", "All", "Any", "Some", "Not",
        "His", "Her", "Their", "Our", "This", "That", "From", "With",
    }

    # Find consecutive capitalized words that form a name
    # Normalize separators (commas, slashes, etc.) to spaces
    normalized = re.sub(r"[,/;]+", " ", question)
    words = normalized.split()
    names = []
    current_name = []

    for word in words:
        # Strip punctuation
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if clean and clean[0].isupper() and len(clean) >= 2 and clean not in common_words:
            current_name.append(clean)
        else:
            if len(current_name) >= 2:  # At least first + last name
                names.append(" ".join(current_name))
            elif len(current_name) == 1:
                names.append(current_name[0])  # Single name is okay too
            current_name = []

    # Don't forget the last group
    if len(current_name) >= 2:
        names.append(" ".join(current_name))
    elif len(current_name) == 1:
        names.append(current_name[0])

    return names


def _text_search_employees(store, names: list[str], k: int = 5) -> list[Document]:
    """Search for employees by name using ChromaDB text matching.

    Tries multiple case variations since ChromaDB $contains is case-sensitive.
    """
    if not names:
        return []

    all_docs = []
    seen_ids = set()

    for name in names:
        # Build search terms: full name variations + individual word variations
        search_terms = set()
        search_terms.update([name, name.lower(), name.title()])
        # Also search each word separately (handles mixed case like "shashank Pathak")
        for word in name.split():
            search_terms.update([word, word.lower(), word.title()])

        for variant in search_terms:
            try:
                collection = store._collection
                results = collection.get(
                    where_document={"$contains": variant},
                    limit=k,
                    include=["documents", "metadatas"],
                )

                for i, content in enumerate(results.get("documents") or []):
                    metadata = (results.get("metadatas") or [{}])[i] or {}
                    doc_id = metadata.get("record_id", f"unknown_{i}")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(Document(page_content=content, metadata=metadata))

                if results.get("documents"):
                    logger.info("Text search for '%s': found %d docs", variant, len(results["documents"]))

            except Exception:
                logger.exception("Text search failed for '%s'", variant)

    return all_docs


def _retrieve_mixed(store, question: str, k: int = 20) -> list[Document]:
    """Smart retrieval: text match for names + vector search for both types.

    1. Extract names from question → text-match in ChromaDB
    2. Vector search employees (for context)
    3. Vector search PDFs (for policies)
    4. Merge and deduplicate
    """
    half_k = max(k // 2, 5)

    # Step 1: Try to find specific employees by name (exact text match)
    names = _extract_names(question)
    name_docs = _text_search_employees(store, names, k=5) if names else []

    # Step 2: Vector search employees
    employee_docs = store.similarity_search(
        question,
        k=half_k,
        filter={"record_type": "employee"},
    )

    # Step 3: Vector search PDFs
    pdf_docs = store.similarity_search(
        question,
        k=half_k,
        filter={"record_type": "pdf"},
    )

    # Step 4: Merge — name matches first (highest priority), then others
    seen_ids = set()
    combined = []

    for doc in name_docs + employee_docs + pdf_docs:
        doc_id = doc.metadata.get("record_id", id(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined.append(doc)

    logger.info(
        "Retrieved %d name-match + %d employee + %d pdf = %d unique docs",
        len(name_docs),
        len(employee_docs),
        len(pdf_docs),
        len(combined),
    )
    return combined


async def ask(question: str) -> dict:
    """High-level helper: ask a question and get an answer + sources.

    Returns::

        {
            "answer": "...",
            "sources": [{"record_type": "...", "record_id": "...", ...}, ...]
        }
    """
    store = get_vector_store()
    llm = _get_llm()

    # Retrieve from both employees AND PDFs (with name matching)
    docs = _retrieve_mixed(store, question, k=settings.retriever_top_k)
    context = _format_docs(docs)

    # Build and invoke the chain
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": question})

    logger.info("RAG chain answered (top_k=%d)", settings.retriever_top_k)

    sources = []
    for doc in docs:
        sources.append(
            {
                "record_type": doc.metadata.get("record_type"),
                "record_id": doc.metadata.get("record_id"),
                "employee_name": doc.metadata.get("employee_name"),
                "project_name": doc.metadata.get("project_name"),
            }
        )

    return {
        "answer": answer or "Sorry, I couldn't find an answer.",
        "sources": sources,
    }
