"""RAG query chain: retrieve context from ChromaDB and answer via Ollama."""

from __future__ import annotations

import asyncio
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

_active_ask_requests = 0
_ask_lock = asyncio.Lock()


def _get_llm(num_thread: int | None = None) -> ChatOllama:
    """Instantiate the Ollama LLM."""
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_ctx=32768,
        num_thread=num_thread or settings.ollama_num_threads,
    )


def _format_docs(docs) -> str:
    """Format retrieved documents into a labelled context string.

    Each chunk is prefixed with its source type so the LLM can
    distinguish between employee records, PDF policies, case studies, etc.
    """
    parts = []
    for doc in docs:
        record_type = doc.metadata.get("record_type", "unknown")
        if record_type == "employee":
            label = "[Source: Employee Record]"
        elif record_type == "pdf":
            filename = doc.metadata.get("filename", "document")
            label = f"[Source: PDF — {filename}]"
        else:
            label = "[Source: Document]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


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


def _extract_keywords(question: str) -> list[str]:
    """Extract important keywords/acronyms from the question for text search.

    Picks out uppercase acronyms (e.g. OCR, NL2SQL, RAG) and significant
    multi-character words that aren't common stop words.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "but", "and", "or", "if", "while", "about", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "it",
        "its", "me", "my", "myself", "we", "our", "ours", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their",
        "give", "get", "got", "tell", "show", "find", "list", "many",
        "much", "also", "like", "make", "know", "take", "come", "see",
        "want", "look", "use", "day", "way", "any", "where", "was",
        "project", "projects", "used", "give", "tell", "show", "find",
    }
    # Preserve important bi-grams (2-word phrases)
    bi_grams = []
    phrase_candidates = ["notice period", "probation period", "notice pay", "notice days", "leave policy", "travel policy"]
    q_lower = question.lower()
    for phrase in phrase_candidates:
        if phrase in q_lower:
            bi_grams.append(phrase)
    
    keywords = []
    # If we found bi-grams, prioritize them and don't break them into single words if possible
    # to avoid noise (e.g. searching for 'period' by itself is very noisy).
    for word in question.split():
        clean = re.sub(r"[^a-zA-Z0-9]", "", word)
        if not clean or len(clean) < 2:
            continue
        
        # Always keep uppercase acronyms (OCR, RAG, etc.)
        if clean.isupper() and len(clean) >= 2:
            keywords.append(clean)
        # Keep significant words that aren't stop words
        elif clean.lower() not in stop_words and len(clean) >= 3:
            # If the word is part of a bi-gram we already found, skip it as a single token 
            # if it's a very generic word like 'period'.
            if clean.lower() == "period" and any("period" in b for b in bi_grams):
                continue
            keywords.append(clean)

    # Combine bi-grams into the keyword list
    final_keywords = bi_grams + keywords

    # Heuristic: If we have very specific technical keywords (uppercase acronyms),
    # then remove generic technical words like 'software' or 'system' to keep focus.
    acronyms = [k for k in final_keywords if k.isupper() and len(k) >= 2]
    if acronyms:
        generic_tech_words = {"software", "system", "platform", "development", "solution"}
        final_keywords = [k for k in final_keywords if k.lower() not in generic_tech_words or k in acronyms]

    return list(dict.fromkeys(final_keywords))  # Deduplicate while preserving order


def _text_search_keywords(store, keywords: list[str], k: int = 10) -> list[Document]:
    """Search for documents containing specific keywords via ChromaDB text match."""
    if not keywords:
        return []

    all_docs = []
    seen_ids = set()

    for keyword in keywords:
        # Prioritize exact uppercase matches for acronyms, or the phrase itself
        if keyword.isupper() or " " in keyword:
            variants = [keyword]
        else:
            variants = [keyword, keyword.lower(), keyword.title()]
            
        for variant in variants:
            try:
                collection = store._collection
                results = collection.get(
                    where_document={"$contains": variant},
                    limit=k,
                    include=["documents", "metadatas"],
                )
                for i, content in enumerate(results.get("documents") or []):
                    metadata = (results.get("metadatas") or [{}])[i] or {}
                    doc_id = metadata.get("record_id", f"kw_{i}")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc_obj = Document(page_content=content, metadata=metadata)
                        # Boost specific technical keywords to the front
                        if keyword.isupper():
                            all_docs.insert(0, doc_obj)
                        else:
                            all_docs.append(doc_obj)
                
                if results.get("documents"):
                    logger.info("Keyword search for '%s': found %d docs", variant, len(results["documents"]))
            except Exception:
                logger.exception("Keyword search failed for '%s'", variant)

    return all_docs


def _classify_query_intent(question: str, names: list[str]) -> str:
    """Classify query intent to dynamically adjust retrieval weights.

    Returns one of:
      'employee'  — question is about a specific person
      'document'  — question is about policies, technologies, case studies, etc.
      'mixed'     — question combines both (e.g. "what is John's leave policy?")
    """
    q = question.lower()

    # Signals that the question is about documents / PDFs / case studies
    doc_signals = {
        "policy", "case study", "case studies", "project", "technology",
        "tech stack", "framework", "built", "developed", "delivered",
        "client", "outcome", "solution", "architecture", "platform",
        "system", "application", "tool", "integration", "procedure",
        "guideline", "rule", "regulation", "benefit", "entitle",
        "probation", "notice period", "separation", "referral",
        "attendance", "holiday", "travel", "posh", "iso",
    }

    # Signals that the question is about specific employees
    emp_signals = {
        "salary", "joining date", "department", "designation",
        "email", "employee id", "emp id",
    }

    has_names = len(names) > 0
    has_doc_signal = any(s in q for s in doc_signals)
    has_emp_signal = any(s in q for s in emp_signals)

    if has_names and has_doc_signal:
        return "mixed"
    elif has_names or has_emp_signal:
        return "employee"
    else:
        return "document"


def _retrieve_mixed(store, question: str, k: int = 20) -> list[Document]:
    """Dynamic retrieval: classifies query intent, then retrieves accordingly.

    1. Extract names → text-match employees
    2. Extract keywords → text-match ALL docs
    3. Classify intent → adjust employee vs PDF retrieval weights
    4. Vector search employees + PDFs
    5. Merge and deduplicate
    """
    # Step 1: Try to find specific employees by name
    names = _extract_names(question)
    name_docs = _text_search_employees(store, names, k=5) if names else []

    # Step 2: Extract keywords and text-search across ALL docs
    keywords = _extract_keywords(question)
    keyword_docs = _text_search_keywords(store, keywords, k=10) if keywords else []
    logger.info("Keywords: %s → %d keyword-match docs", keywords, len(keyword_docs))

    # Step 3: Classify intent to decide retrieval balance
    intent = _classify_query_intent(question, names)
    logger.info("Query intent: '%s'", intent)

    if intent == "employee":
        emp_k = max(k * 2 // 3, 5)   # 67% employee, 33% PDF
        pdf_k = max(k // 3, 5)
    elif intent == "document":
        emp_k = max(k // 4, 3)        # 25% employee, 75% PDF
        pdf_k = max(k * 3 // 4, 5)
    else:  # mixed
        emp_k = max(k // 2, 5)        # 50/50
        pdf_k = max(k // 2, 5)

    # NOISE REDUCTION: If we have technical keyword matches, cut vector search 
    # budgets to prevent drowning out the precise technical info with generic docs.
    if any(k.isupper() and len(k) >= 2 for k in keywords):
        emp_k = 2 if intent != "employee" else emp_k // 2
        pdf_k = 5 if intent == "employee" else pdf_k // 2
        logger.info("Technical keyword detected. Reducing filler budget to emp_k=%d, pdf_k=%d", emp_k, pdf_k)

    # Step 4: Vector search
    employee_docs = store.similarity_search(
        question, k=emp_k, filter={"record_type": "employee"},
    )
    pdf_docs = store.similarity_search(
        question, k=pdf_k, filter={"record_type": "pdf"},
    )

    # Step 5: Merge — keyword text-matches first (most precise), then name
    # matches, then PDF vectors, then employee vectors
    seen_ids = set()
    combined = []

    for doc in keyword_docs + name_docs + pdf_docs + employee_docs:
        doc_id = doc.metadata.get("record_id", id(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined.append(doc)

    # Truncate to the requested top_k to avoid massive context windows.
    # Since keyword matches are pushed first, they will survive truncation.
    combined = combined[:k]

    logger.info(
        "[%s] Retrieved %d keyword + %d name + %d emp + %d pdf = %d total (capped to %d)",
        intent, len(keyword_docs), len(name_docs),
        len(employee_docs), len(pdf_docs), len(combined), k
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
    global _active_ask_requests
    async with _ask_lock:
        _active_ask_requests += 1
        current_requests = _active_ask_requests

    allocated_threads = max(1, settings.ollama_max_threads // current_requests)
    logger.info("[Dynamic Thread Allocator] Active requests: %d. Allocating %d threads for this request.", current_requests, allocated_threads)

    try:
        start_time = asyncio.get_event_loop().time()
        store = get_vector_store(num_thread=allocated_threads)
        llm = _get_llm(num_thread=allocated_threads)

        # Retrieve from both employees AND PDFs (with name matching)
        docs = _retrieve_mixed(store, question, k=settings.retriever_top_k)
        retrieval_time = asyncio.get_event_loop().time() - start_time
        logger.info("Docs retrieved in %.2fs", retrieval_time)
        context = _format_docs(docs)

        # Build and invoke the chain
        chain = RAG_PROMPT | llm | StrOutputParser()
        
        # Background task for live time logging
        async def log_elapsed_time():
            start = asyncio.get_event_loop().time()
            try:
                while True:
                    await asyncio.sleep(5)
                    elapsed = asyncio.get_event_loop().time() - start
                    logger.info("... Still generating (elapsed: %.1fs)", elapsed)
            except asyncio.CancelledError:
                pass

        timer_task = asyncio.create_task(log_elapsed_time())
        try:
            answer = await chain.ainvoke({"context": context, "question": question})
        finally:
            timer_task.cancel()

        total_time = asyncio.get_event_loop().time() - start_time
        logger.info("RAG chain answered (top_k=%d) in %.2fs", settings.retriever_top_k, total_time)

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
            "time_elapsed_seconds": round(total_time, 2),
        }
    finally:
        async with _ask_lock:
            _active_ask_requests -= 1
