"""
chain.py — RAG Query Chain for ResoAI
======================================

This is the "brain" of ResoAI. Every time a user asks a question — either
through MS Teams or the /api/ask endpoint — this module handles the full
pipeline from raw question to final answer.

High-level flow for each query:
  1. Extract person names from the question (e.g. "Anthony")
  2. BM25 statistical keyword search across all stored chunks
  3. Regex keyword search via ChromaDB $contains
  4. Classify intent (employee question / document question / mixed)
  5. Vector similarity search (semantic search via embeddings)
  6. Merge all retrieved chunks, deduplicate
  7. FlashRank re-ranking — re-score all candidates, keep the best k
  8. Format the top-k chunks as a labelled context string
  9. Invoke the LLM (Llama 3.2 via Ollama) with the context + question
 10. Return the answer + source metadata + elapsed time

Why hybrid retrieval (BM25 + vector + keyword)?
  - Vector search is great for semantic similarity ("what's the leave policy?"
    will find chunks about "annual leave entitlement") but misses exact matches.
  - BM25 is great for exact keyword matches but doesn't understand meaning.
  - Combining both gives the best of both worlds.
  - FlashRank re-ranking then acts as a final quality filter, using a
    cross-encoder model to score each chunk against the full question.

Performance design decisions:
  - _vector_store is cached globally (created once, not per request)
  - _bm25_index is built lazily on the first query and then reused
  - _ranker (FlashRank) is initialized once at module load
  - num_ctx is set to 8192 (not 32768) — fits all retrieved context with 4x speedup
  - retriever_top_k is 8 (not 20) — re-ranking makes fewer chunks more accurate
"""

from __future__ import annotations

import asyncio
import difflib  # stdlib — fuzzy name matching for typo-tolerant employee search
import logging
import re
import time  # Used for time.monotonic() — more reliable than event_loop.time()

from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Setup dedicated debug logger for chunks
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)
DEBUG_LOG_PATH = "logs/retrieval_debug.log"

def _log_retrieval(question: str, path_name: str, items: list, criteria: str = None):
    """Writes detailed retrieval results to logs/retrieval_debug.log (Overwrites on every query)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"QUESTION:  {question}\n")
        f.write(f"PATHWAY:   {path_name}\n")
        if criteria:
            f.write(f"CRITERIA:  {criteria}\n")
        f.write(f"{'-'*80}\n")
        
        for i, item in enumerate(items, 1):
            if hasattr(item, "page_content"): # LangChain Document
                source = item.metadata.get("source", "unknown")
                page = item.metadata.get("page", "?")
                score = item.metadata.get("relevance_score", "N/A")
                f.write(f"CHUNK {i} [Source: {source} | Page: {page} | Score: {score}]\n")
                f.write(f"CONTENT: {item.page_content[:500]}...\n\n")
            else: # Interpreter Result
                f.write(f"MATCH {i}: {item.get('name')} (from {item.get('file')})\n")
        f.write(f"{'='*80}\n")
        f.flush()

from src.config import settings
from src.rag.store import get_vector_store
from src.rag.prompts import RAG_PROMPT
from src.rag.cache import get_cached_answer, save_to_cache
from src.rag.data_interpreter import DataInterpreter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level DataInterpreter singleton — created ONCE, reused across all requests
# ---------------------------------------------------------------------------
# Previously, a new DataInterpreter() was created on every ask() call, causing
# all JSON files to be re-read from disk and re-parsed each time.
# Now it is created once at module load and its internal _record_cache persists.
_interpreter = DataInterpreter()

# ---------------------------------------------------------------------------
# Concurrency tracking — for dynamic thread allocation
# ---------------------------------------------------------------------------
# Tracks how many ask() calls are running simultaneously.
# When multiple users ask questions at the same time, each gets a fair share
# of Ollama's CPU threads (ollama_max_threads // active_requests).
_active_ask_requests: int = 0
_ask_lock = asyncio.Lock()  # Protects the counter from race conditions

# ---------------------------------------------------------------------------
# Module-level constants — word filter sets
# ---------------------------------------------------------------------------
# These were previously rebuilt inside functions on EVERY single request call,
# allocating and garbage-collecting a new set object each time.
# Moving them here means they are built ONCE at module load and shared
# across all requests. frozenset is used because:
#   1. It is immutable (safe to share between async tasks)
#   2. `x in frozenset` is O(1) — same speed as a regular set

# Words that look like names (capitalized) but are NOT person names.
# Base values come from config so the behavior can be tuned without code edits.
_COMMON_WORDS: frozenset[str] = frozenset(
    settings.rag_common_words
) | frozenset(settings.company_common_words)

# Common English words that carry no useful information for keyword search.
# Replaces 100+ previously hardcoded words with values from config.py.
_STOP_WORDS: frozenset[str] = frozenset(
    settings.rag_stop_words
) | frozenset(settings.company_stop_words)

# ---------------------------------------------------------------------------
# Global cached vector store — created ONCE, reused across all requests
# ---------------------------------------------------------------------------
# In the original code, get_vector_store() was called inside ask() on EVERY
# request. This reconstructed the ChromaDB client AND the OllamaEmbeddings
# object from scratch each time — wasting CPU and adding latency.
#
# Now it is created on the first request and cached here forever.
# The _store_lock ensures that if two requests arrive simultaneously before
# the store is initialized, only ONE of them creates it (the other waits).
_vector_store: Chroma | None = None
_store_lock = asyncio.Lock()


async def _get_or_create_store(num_thread: int | None = None) -> Chroma:
    """
    Return the globally cached ChromaDB vector store, creating it if needed.

    This function implements a thread-safe lazy singleton pattern:
      - First call: acquires the lock, creates the store, caches it, releases lock.
      - All subsequent calls: the `if _vector_store is None` check fails immediately
        without acquiring the lock, so there is zero overhead after initialization.

    This is safe for async concurrent use because asyncio.Lock() ensures
    only one coroutine runs the creation block at a time.

    Args:
        num_thread: CPU threads to pass to the embedding model on creation.
                    Only used during the first call (subsequent calls ignore it).

    Returns:
        The shared Chroma vector store instance.
    """
    global _vector_store
    async with _store_lock:
        if _vector_store is None:
            logger.info("Initializing vector store (first request — will be cached)...")
            _vector_store = get_vector_store(num_thread=num_thread)
    return _vector_store


def clear_vector_store_cache() -> None:
    """
    Clear the cached vector store instance and BM25 index.
    MUST be called when the underlying ChromaDB collection is truncated.
    """
    global _vector_store, _bm25_index, _bm25_documents
    _vector_store = None
    _bm25_index = None
    _bm25_documents = []
    logger.info("Vector store and BM25 cache cleared.")


# ---------------------------------------------------------------------------
# FlashRank re-ranker — initialized once at module load
# ---------------------------------------------------------------------------
# FlashRank is a lightweight cross-encoder re-ranker. Unlike the embedding
# model (which scores documents and queries independently), FlashRank scores
# each (question, chunk) PAIR together — making it much more accurate at
# judging relevance. It runs entirely on CPU and is fast enough for real-time use.
#
# The re-ranker is applied AFTER all retrieval methods have run, acting as a
# final quality filter that picks the truly most relevant chunks from the
# merged candidate pool.
#
# If FlashRank fails to initialize (missing model files, etc.), re-ranking is
# gracefully skipped and the pipeline falls back to priority-order truncation.
try:
    _ranker = Ranker()
    logger.info("FlashRank re-ranker initialized successfully.")
except Exception as e:
    logger.warning(
        "FlashRank failed to initialize: %s. Re-ranking will be skipped "
        "and results will be truncated by priority order instead.", e
    )
    _ranker = None

# ---------------------------------------------------------------------------
# BM25 index — lazy, initialized on the first query
# ---------------------------------------------------------------------------
# BM25 (Best Match 25) is a classic probabilistic keyword ranking algorithm
# used by search engines like Elasticsearch. It scores documents based on:
#   - Term frequency (how often the keyword appears in the chunk)
#   - Inverse document frequency (how rare the keyword is across all chunks)
#   - Document length normalization
#
# Unlike ChromaDB's $contains search (which does exact substring matching),
# BM25 scores ALL documents at once and returns the best-ranked ones.
# This catches relevant chunks even when the exact keyword isn't present
# (e.g. "leaves" will score documents containing "leave", "annual leave", etc.)
#
# The index is built LAZILY on the first query:
#   - At module load time, ChromaDB may not be populated yet
#   - Building it lazily ensures all documents are available when it runs
#   - After initialization, _bm25_index is not None, so subsequent calls
#     skip the lock and return immediately (O(1) check)
#
# The _bm25_lock prevents multiple simultaneous requests from each trying
# to build the index at the same time (double-initialization race condition).
_bm25_index: BM25Okapi | None = None
_bm25_documents: list[Document] = []  # parallel list — index i matches _bm25_index row i
_bm25_lock = asyncio.Lock()


async def _init_bm25(store: Chroma) -> None:
    """
    Build the BM25 search index from all documents stored in ChromaDB.

    This function is called at the start of every _retrieve_mixed() call,
    but only does real work on the very first call (lazy initialization).
    After that, the `if _bm25_index is not None` check exits immediately.

    Process:
      1. Fetch all document chunks from ChromaDB using store.get()
      2. Reconstruct LangChain Document objects from the raw data
      3. Tokenize each document's text (simple whitespace split)
      4. Build a BM25Okapi index from the tokenized corpus
      5. Store the index and the parallel document list as globals

    The index is stored as a module-level global so it persists across
    all future requests without needing to rebuild it.

    Args:
        store: The ChromaDB vector store to fetch documents from.
               Must already contain indexed documents (run ingestion first).

    Side effects:
        Sets _bm25_index and _bm25_documents global variables.
    """
    global _bm25_index, _bm25_documents
    async with _bm25_lock:
        # Early exit — already initialized (the common case after first call)
        if _bm25_index is not None:
            return

        logger.info("Initializing BM25 index — fetching all documents from ChromaDB...")
        try:
            # ChromaDB's .get() returns all stored chunks as raw dicts
            data = store.get()
            docs = [
                Document(
                    page_content=data["documents"][i],
                    metadata=data["metadatas"][i],
                )
                for i in range(len(data.get("ids", [])))
            ]

            if not docs:
                logger.warning(
                    "ChromaDB appears to be empty — BM25 index not built. "
                    "Run ingestion first (POST /api/ingest)."
                )
                return

            # Store documents so _bm25_search() can map indices back to Documents
            _bm25_documents = docs

            # Tokenize: lowercase + split on whitespace. Simple but effective.
            # More advanced tokenization (stemming, lemmatization) would improve
            # recall but is not needed for HR queries which use clear keywords.
            tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
            _bm25_index = BM25Okapi(tokenized_corpus)

            logger.info("BM25 index built successfully with %d documents.", len(docs))

        except Exception as e:
            logger.error(
                "Failed to build BM25 index: %s. "
                "BM25 search will be unavailable for this session.", e
            )


def _bm25_search(query: str, k: int = 10) -> list[Document]:
    """
    Search the BM25 index for the k most relevant documents to the query.

    BM25 scores every document in the corpus against the query tokens and
    returns the top-k by score. Only documents with score > 0 are returned
    (score of 0 means the document shares no tokens with the query).

    This is called before vector search because it is very fast (pure CPU,
    no network call) and catches exact keyword matches that vector search
    might miss if the embedding space doesn't perfectly capture the term.

    Args:
        query: The user's raw question string.
        k: Maximum number of documents to return.

    Returns:
        List of up to k Document objects, sorted by BM25 relevance (best first).
        Returns empty list if the BM25 index hasn't been initialized yet.
    """
    if _bm25_index is None or not _bm25_documents:
        # BM25 not ready — silently skip, other retrieval methods will still run
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Sort all document indices by score descending, take top k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Filter out zero-score documents (no keyword overlap at all)
    return [_bm25_documents[i] for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm(num_thread: int | None = None) -> ChatOllama:
    """
    Create and return a ChatOllama LLM instance for answer generation.

    ChatOllama wraps the local Ollama server, sending the formatted prompt
    (system instructions + retrieved context + user question) to the LLM
    and streaming back the response.

    Key parameters:
        temperature: Controls randomness in token selection.
                     0.1 = near-deterministic. Good for factual HR answers.
                     Higher values (0.7+) = more creative but less accurate.

        num_ctx:     The LLM's context window in tokens. This is the total
                     budget for system prompt + retrieved chunks + question.
                     Reduced from 32768 → 2048 in settings — this is the
                     single biggest speed improvement in the optimized version
                     because the LLM pre-allocates memory for the full window.

        num_thread:  CPU threads for this specific LLM call. Dynamically
                     reduced when multiple requests are active simultaneously
                     (see the thread allocator in ask()).

    Args:
        num_thread: CPU threads to allocate. None = use settings default.

    Returns:
        A configured ChatOllama instance ready to receive prompts.
    """
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,   # was hardcoded 0.1
        num_ctx=settings.ollama_num_ctx,           # was hardcoded 32768
        num_thread=num_thread or settings.ollama_num_threads,
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    """
    Format a list of retrieved Document chunks into a single context string.

    The LLM receives this string as part of its prompt. Each chunk is
    prefixed with a [Source: ...] label so the LLM knows where the information
    came from and can apply the correct source priority rules from the system prompt.

    Label format examples:
        [Source: Employee Record — Anthony Young]
        [Source: Project — Dubai Technologies]
        [Source: Policy — Leave Policy]
        [Source: PDF — filename.pdf] (fallback)

    Chunks are separated by "---" dividers to visually isolate them for the LLM.
    """
    parts = []
    for doc in docs:
        record_type = doc.metadata.get("record_type", "unknown")
        
        if record_type == "employee":
            name = doc.metadata.get("employee_name", "Unknown Employee")
            label = f"[Source: Employee Record — {name}]"
        elif record_type == "project":
            name = doc.metadata.get("project_name", "Unknown Project")
            label = f"[Source: Project Case Study PDF — {name}]"
        elif record_type == "policy":
            name = doc.metadata.get("policy_name", "Unknown Policy")
            label = f"[Source: Company Policy PDF — {name}]"
        elif record_type == "misc_table_row" or doc.metadata.get("source") == "directory.json":
            label = "[Source: OFFICIAL EMPLOYEE DIRECTORY]"
        elif record_type == "holiday":
            label = "[Source: Holiday Calendar PDF]"
        else:
            # For dynamically ingested files
            source_tag = doc.metadata.get("source_tag")
            if source_tag:
                label = f"[Source: {source_tag}]"
            else:
                filename = doc.metadata.get("source", doc.metadata.get("filename", "Document"))
                label = f"[Source: {filename}]"
            
        # Strip existing [Source: ...] or [DOCUMENT_NAME: ...] if it exists 
        # to ensure we use the consistent, canonical label from this formatter.
        content = doc.page_content
        content = re.sub(r"^\[(Source|DOCUMENT_NAME):.*?\]\n?", "", content, flags=re.IGNORECASE).strip()
        
        parts.append(f"{label}\n{content}")
        
    # Keep the final prompt bounded without hardcoding the cap in code.
    return "\n\n---\n\n".join(parts[: settings.rag_context_max_chunks])


# ---------------------------------------------------------------------------
# Debug logging helper
# ---------------------------------------------------------------------------

def _log_selected_chunks(docs: list[Document], stage: str) -> None:
    """
    Log a structured, human-readable summary of chunks at each retrieval stage.

    This function is called after every major retrieval step so you can see
    exactly WHICH chunks were selected and WHY. Each line shows:
        - Chunk rank within this stage (#01, #02, ...)
        - Source type and name (Employee | John Smith  or  PDF | Leave Policy.pdf)
        - First 120 characters of the chunk content as a preview

    Example terminal output:
        +-- [Step 2 -- BM25] 3 chunk(s) --
        | #01  Employee | John Smith                       >> Employee: John Smith | Dept: HR | Salary: 120000...
        | #02  PDF      | Leave Policy.pdf (p3)            >> Annual leave entitlement is 24 days per calendar year...
        | #03  Employee | Jane Doe                         >> Employee: Jane Doe | Dept: Engineering | Salary: 95000...
        +-- end [Step 2 -- BM25] --

    This output is invaluable for debugging retrieval quality issues — if the
    LLM gives a wrong answer, check which chunks were in the FINAL stage to
    see whether the right information was retrieved at all.

    Args:
        docs:  List of Document chunks to log.
        stage: Human-readable label for the retrieval stage (e.g. "Step 2 -- BM25").
    """
    if not docs:
        logger.info("  [%s] No chunks selected.", stage)
        return

    logger.info("  +-- [%s] %d chunk(s) --", stage, len(docs))
    for i, doc in enumerate(docs, 1):
        meta        = doc.metadata
        record_type = meta.get("record_type", "unknown")
        emp_name    = meta.get("employee_name", "")
        filename    = meta.get("filename", "")
        page        = meta.get("page", "")

        # Build a concise source description based on record type
        if record_type == "employee":
            source = f"Employee | {emp_name or meta.get('record_id', '?')}"
        elif record_type == "pdf":
            source = f"PDF      | {filename}" + (f" (p{page})" if page != "" else "")
        else:
            source = f"{record_type} | {meta.get('record_id', '?')}"

        # First 120 chars as a content preview (newlines flattened for readability)
        preview = doc.page_content.replace("\n", " ").strip()[:120]
        logger.info("  | #%02d  %-45s >> %s...", i, source, preview)

    logger.info("  +-- end [%s] --", stage)


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

def _is_acronym_token(word: str) -> bool:
    """
    Return True if a word is an acronym or tech abbreviation — NOT a person name.

    This helper catches false positives in name extraction like:
        "GPUs"  → GPU is the acronym, lowercase s is a plural suffix
        "APIs"  → same pattern
        "AI"    → pure acronym (all uppercase)
        "OCR"   → pure acronym

    A word is considered an acronym token if:
      - All characters are uppercase letters (e.g. "GPU", "API", "ML"), OR
      - It starts with 2+ CONSECUTIVE uppercase letters (e.g. "GPUs", "APIs")
        — an acronym with a lowercase plural/possessive suffix.

    IMPORTANT — CamelCase words are NOT acronyms:
      "FiftyFive"   → F...F scattered uppercase → NOT an acronym (it's a company name)
      "Technologies"→ T only at start → NOT an acronym
      "MacBook"     → Ma → only 1 uppercase at start → NOT an acronym
      "John"        → J only → NOT an acronym (it's a name)

    The consecutive-from-start check prevents CamelCase company/product names
    from being misidentified as acronyms, which previously caused "FiftyFive"
    to be stripped into "FF" during keyword extraction.

    Args:
        word: A cleaned alphabetic token (no punctuation).

    Returns:
        True if the token looks like an acronym, False if it could be a name.
    """
    if not word:
        return False
    # Pure acronym: every character is uppercase (e.g. "GPU", "AI", "NLP")
    if word.isupper():
        return True
    # Acronym with lowercase suffix: count CONSECUTIVE uppercase chars from start only
    # e.g. "GPUs" → G,P,U = 3 consecutive upper → True
    # e.g. "FiftyFive" → F = 1 consecutive upper (i is lowercase) → False
    consecutive_upper = 0
    for c in word:
        if c.isupper():
            consecutive_upper += 1
        else:
            break  # stop counting at first non-uppercase character
    return consecutive_upper >= 2


def _extract_names(question: str) -> list[str]:
    """
    Extract likely person names from a question string.

    Strategy: Find sequences of consecutive CAPITALIZED words that are NOT
    in the _COMMON_WORDS filter set AND are NOT acronym/tech tokens.

    Examples:
        "How many leaves does Anthony West have?" → ["Anthony West"]
        "What is the salary of John?" → ["John"]
        "Have we used GPUs in any projects?" → []  ← FIX: "GPUs" is an acronym
        "Tell me about AI projects at 55" → []     ← FIX: "AI" is an acronym
        "What is Priya's department?" → ["Priya"]
        "Find Clayton Walker's designation" → ["Clayton Walker"]

    Two-level filtering:
      Level 1 — _COMMON_WORDS: rejects known question/HR/domain words
      Level 2 — _is_acronym_token(): rejects uppercase acronyms and
                tech abbreviations like GPUs, APIs, ML, OCR, etc.

    Args:
        question: The raw user question string.

    Returns:
        List of extracted name strings (may be empty if no names found).
    """
    # Normalize separators (commas, slashes, semicolons → spaces) so that
    # "John, Jane" and "John/Jane" are both tokenized into ["John", "Jane"]
    normalized = re.sub(r"[,/;]+", " ", question)
    words = normalized.split()

    names: list[str] = []
    current_name: list[str] = []  # accumulator for multi-word names

    for word in words:
        # Strip possessive suffix first ("Kumar's" → "Kumar"),
        # then remove all remaining non-alphabetic characters.
        word_stripped = re.sub(r"['\'\u2019]s$", "", word, flags=re.IGNORECASE)
        clean = re.sub(r"[^a-zA-Z]", "", word_stripped)

        # A valid name token must:
        #   1. Be non-empty after stripping punctuation
        #   2. Start with an uppercase letter
        #   3. Be at least 2 characters long
        #   4. NOT be a known common/HR word
        #   5. NOT be an acronym or tech abbreviation (NEW — fixes "GPUs", "AI", etc.)
        is_name_token = (
            clean
            and clean[0].isupper()
            and len(clean) >= 2
            and clean not in _COMMON_WORDS
            and not _is_acronym_token(clean)   # ← BUG FIX
        )

        if is_name_token:
            current_name.append(clean)
        else:
            # Non-name token — flush whatever we've accumulated
            if len(current_name) >= 2:
                names.append(" ".join(current_name))
            elif len(current_name) == 1:
                names.append(current_name[0])
            current_name = []

    # Handle names at the very end of the question (no trailing non-name token)
    if len(current_name) >= 2:
        names.append(" ".join(current_name))
    elif len(current_name) == 1:
        names.append(current_name[0])

    return names


# ---------------------------------------------------------------------------
# Employee text search
# ---------------------------------------------------------------------------

def _fuzzy_expand_names(typed_names: list[str], store: Chroma) -> list[str]:
    """
    Expand typed names to include fuzzy-matched real names from the employee database.

    WHY THIS IS NEEDED:
        $contains is an exact substring match. A typo like "Anhtony" instead of
        "Anthony" returns zero results because the exact string isn't in the DB.
        This function uses Python's stdlib difflib.SequenceMatcher to find names
        in the database that are "close enough" to what the user typed.

    HOW IT WORKS:
        1. Fetch all unique employee_name values stored in ChromaDB metadata.
        2. For each typed name, compute similarity ratio against every stored name
           and every individual word (first/last name).
        3. If similarity >= FUZZY_THRESHOLD (0.78), treat it as a match and add
           the CORRECT spelling to the search terms.
        4. Return the original typed names PLUS any fuzzy-matched correct names.
           This way if the user typed correctly, the exact match still fires.

    Threshold choice (0.78):
        - "Anhtony" vs "Anthony" → ratio ≈ 0.87 → match ✓
        - "Antony"  vs "Anthony" → ratio ≈ 0.87 → match ✓
        - "Priya"   vs "Priya"   → ratio = 1.00 → match ✓
        - "John"    vs "Jane"    → ratio ≈ 0.67 → no match ✓ (different people)
        - "Ravi"    vs "Rabi"    → ratio ≈ 0.75 → borderline — kept as no match

    Args:
        typed_names: Names as extracted from the user query (may contain typos).
        store:       ChromaDB vector store to fetch real employee names from.

    Returns:
        Deduplicated list of name strings including both typed names and any
        fuzzy-matched correct spellings found in the database.
    """
    fuzzy_threshold = settings.rag_fuzzy_name_threshold

    # Fetch all stored employee names from ChromaDB metadata (one-time per call)
    try:
        data = store._collection.get(
            where={"record_type": "employee"},
            include=["metadatas"],
        )
        stored_names: list[str] = []
        seen_stored: set[str] = set()
        for meta in (data.get("metadatas") or []):
            emp_name = (meta or {}).get("employee_name", "")
            if emp_name and emp_name not in seen_stored:
                seen_stored.add(emp_name)
                stored_names.append(emp_name)
    except Exception:
        logger.exception("Failed to fetch employee names for fuzzy matching")
        return typed_names  # fallback: just use the typed names as-is

    if not stored_names:
        return typed_names

    # Build a flat list of (full_name, word) pairs for matching
    # e.g. "Anthony West" → [("Anthony West", "Anthony West"),
    #                         ("Anthony West", "Anthony"),
    #                         ("Anthony West", "West")]
    name_word_pairs: list[tuple[str, str]] = []
    for full_name in stored_names:
        name_word_pairs.append((full_name, full_name))
        for word in full_name.split():
            name_word_pairs.append((full_name, word))

    # For each typed name, find the best fuzzy match
    expanded: list[str] = list(typed_names)  # start with what user typed
    for typed in typed_names:
        best_match_name: str | None = None
        best_ratio: float = 0.0

        for full_name, candidate in name_word_pairs:
            ratio = difflib.SequenceMatcher(
                None, typed.lower(), candidate.lower()
            ).ratio()
            if ratio >= fuzzy_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_match_name = full_name

        if best_match_name and best_match_name not in expanded:
            logger.info(
                "Fuzzy name match: %r → %r (similarity=%.2f)",
                typed, best_match_name, best_ratio,
            )
            expanded.append(best_match_name)

    return expanded


def _text_search_employees(store: Chroma, names: list[str], k: int = 5) -> list[Document]:
    """
    Search ChromaDB for employee records matching the extracted names.

    Two-phase search for maximum recall:

    Phase 1 — Fuzzy name expansion:
        Before searching, _fuzzy_expand_names() compares each typed name against
        all stored employee names using difflib similarity. If the user typed
        "Anhtony" (typo), it finds "Anthony West" (ratio=0.87 >= 0.78 threshold)
        and adds the correct spelling to the search terms automatically.
        This makes name search resilient to typos without any extra libraries.

    Phase 2 — $contains search with case variants:
        For each name (original + fuzzy-expanded), build search variants:
            - Full name:          "Anthony West"
            - Lowercase:          "anthony west"
            - Title case:         "Anthony West"
            - Each word separate: "Anthony", "anthony", "West", "west"
        Tries all variants because ChromaDB $contains is case-sensitive.

    Args:
        store: The ChromaDB vector store.
        names: Names extracted from the question (may contain typos).
        k:     Max results per $contains search variant.

    Returns:
        Deduplicated list of employee Documents (chunks).
    """
    if not names:
        return []

    # Phase 1: Fuzzy-expand names to handle typos
    # e.g. ["Anhtony West"] → ["Anhtony West", "Anthony West"]
    expanded_names = _fuzzy_expand_names(names, store)
    if expanded_names != names:
        logger.info("Name expansion: %s → %s", names, expanded_names)

    all_docs: list[Document] = []
    seen_ids: set[str] = set()

    for name in expanded_names:
        # Phase 2: Build case variants for $contains search
        search_terms: set[str] = set()
        search_terms.update([name, name.lower(), name.title()])
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
                    logger.info("Name search %r: %d docs found", variant, len(results["documents"]))
            except Exception:
                logger.exception("Text search failed for name variant %r", variant)

    return all_docs


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _normalise_company_aliases(question: str) -> str:
    """
    Expand common shorthand references to the company into searchable terms.

    Users often refer to the company as "55", "@55", "at 55", "as 55" instead
    of "FiftyFive" or "FiftyFive Technologies". None of those short forms will
    match the text stored in case study PDFs, which use the full company name.

    This function rewrites those aliases BEFORE keyword extraction so that
    downstream search methods find the correct chunks.

    Replacements performed (case-insensitive):
        "@55"         -> "FiftyFive Technologies"
        "at 55"       -> "FiftyFive Technologies"
        "as 55"       -> "FiftyFive Technologies"   (common typo for "at 55")
        "55 tech"     -> "FiftyFive Technologies"
        "fiftyfive"   -> "FiftyFive Technologies"   (normalise casing)

    Args:
        question: The raw user question string.

    Returns:
        The question with company aliases replaced by the canonical name.
    """
    # Build regex patterns dynamically from config — no hardcoded company names
    canonical = settings.company_name  # e.g. "FiftyFive Technologies"
    replacements = []
    for alias in settings.company_aliases:
        # Escape special regex chars in the alias, then allow flexible whitespace
        pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        replacements.append((pattern, canonical))

    result = question
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Collapse accidental double last-word duplication (e.g. "Technologies Technologies")
    last_word = canonical.split()[-1]
    result = re.sub(rf"\b{last_word}\s+{last_word}\b", last_word, result)
    return result


def _normalise_tech_acronyms(question: str) -> str:
    """
    Normalise known technology acronyms to uppercase regardless of how the user typed them.

    WHY THIS IS NEEDED:
        ChromaDB $contains is case-sensitive. If a PDF stores "OCR" (uppercase)
        and the user types "ocr" (lowercase), the $contains search for "ocr"
        will NOT match "OCR" — causing the entire chunk to be missed.

        Additionally, _is_acronym_token() only recognises uppercase tokens, so
        "ocr" typed lowercase would NOT trigger acronym-based noise reduction
        (which raises pdf_k and lowers emp_k for better retrieval balance).

    HOW IT WORKS:
        Uses whole-word regex ( boundaries) to replace known lowercase/mixed
        acronyms with their canonical uppercase form. For example:
            "tell me all projects where ocr is used"
         →  "tell me all projects where OCR is used"

        The word-boundary () prevents false replacements inside longer words,
        e.g. "macro" should NOT become "maCRO".

    ADDING NEW ACRONYMS:
        Add entries to settings.rag_known_acronyms in config.py.
        Each entry is a string like "ocr", "gpu", "nlp" — the lowercase form.
        The function auto-uppercases them for the replacement.

    Args:
        question: The raw user question string.

    Returns:
        Question with known acronyms replaced by their uppercase canonical form.
    """
    result = question
    for acronym in settings.rag_known_acronyms:
        # Canonical form: strip trailing 'S' from plural acronyms so the
        # normalised question contains "GPU" not "GPUS", "API" not "APIS".
        # This ensures $contains searches match PDF text which uses singular form.
        # e.g. "gpus" → canonical "GPU", "apis" → "API", "gpu" → "GPU"
        canonical = acronym.upper().rstrip("S") if acronym.endswith("s") and len(acronym) > 2 else acronym.upper()
        pattern = r"\b" + re.escape(acronym) + r"\b"
        result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)
    return result


def _extract_keywords(question: str) -> list[str]:
    """
    Extract meaningful search keywords from the question for ChromaDB text search.

    Pre-processing:
        Before extraction, the question is passed through _normalise_company_aliases()
        to expand shorthand like "@55", "at 55", "as 55" into "FiftyFive Technologies".
        This is critical for case study queries — the PDFs use the full company name
        and would never match the numeric shorthand.

    Three types of keywords are extracted:
      1. Bi-gram phrases (2-word HR/tech terms from settings.rag_phrase_candidates)
         Example: "notice period" → treated as ONE unit, not "notice" + "period"

      2. Acronym tokens (all-uppercase words or acronym+suffix like "GPUs")
         Example: "GPU", "GPUs", "OCR", "AI", "ML" → high-value exact match signals
         These are boosted to the FRONT of search results in _text_search_keywords.
         BUG FIX: "GPUs" was previously also extracted as a NAME — this is now
         correctly handled by _is_acronym_token() in _extract_names().

      3. Single-word content words (not in _STOP_WORDS, length >= 3)
         Example: "leaves", "GPU", "project" → useful signals

    Acronym noise reduction:
        If the question contains a specific acronym (e.g. "OCR"), generic tech words
        like "system", "platform", "software" are removed from the keyword list
        to avoid diluting results with too-common matches.

    Args:
        question: The raw user question string.

    Returns:
        Deduplicated list of keyword strings, ordered as:
        [bi-gram phrases, acronyms/tech terms, content words] with duplicates removed.
    """
    # ── Pre-processing ────────────────────────────────────────────────────
    # Step A: Expand company aliases ("as 55" → "FiftyFive Technologies")
    question = _normalise_company_aliases(question)
    # Step B: Uppercase known tech acronyms ("ocr" → "OCR", "gpu" → "GPU")
    # This ensures $contains search hits uppercase text in PDFs, and activates
    # acronym-based noise reduction (which boosts PDF budget over employee budget).
    question = _normalise_tech_acronyms(question)
    q_lower = question.lower()

    # ── Step 1: Bi-gram phrase detection ─────────────────────────────────
    # Loaded from settings — add new HR/tech phrases in config.py
    bi_grams = [phrase for phrase in settings.rag_phrase_candidates if phrase in q_lower]

    # ── Step 2: Single-word keyword extraction ────────────────────────────
    keywords: list[str] = []
    for word in question.split():
        # Strip non-alphanumeric characters (punctuation, apostrophes, etc.)
        clean = re.sub(r"[^a-zA-Z0-9]", "", word)
        if not clean or len(clean) < 2:
            continue

        # Always keep acronyms (all uppercase, e.g. GPU, OCR, AI, ML)
        if clean.isupper() and len(clean) >= 2:
            keywords.append(clean)
        # Keep acronym+suffix tokens (e.g. "GPUs", "APIs") — strip the suffix
        # so the raw acronym is what gets searched (more reliable match)
        elif _is_acronym_token(clean) and len(clean) >= 3:
            # Extract just the uppercase prefix — e.g. "GPUs" → "GPU"
            acronym_part = "".join(c for c in clean if c.isupper())
            if len(acronym_part) >= 2:
                keywords.append(acronym_part)
        # Keep regular content words (not stop words, 3+ chars)
        elif clean.lower() not in _STOP_WORDS and len(clean) >= 3:
            # Skip "period" if already captured in a bi-gram to avoid noise
            if clean.lower() == "period" and any("period" in b for b in bi_grams):
                continue
            keywords.append(clean)

    # ── Step 3: Combine ───────────────────────────────────────────────────
    # Bi-grams first (most specific), then single words
    final_keywords = bi_grams + keywords

    # ── Step 4: Acronym-based noise suppression ───────────────────────────
    # If a specific acronym is present, remove generic tech words that would
    # match too broadly (e.g. "system", "platform", "software")
    acronyms = [kw for kw in final_keywords if kw.isupper() and len(kw) >= 2]
    if acronyms:
        generic_tech_words = set(settings.rag_generic_tech_words)
        final_keywords = [
            kw for kw in final_keywords
            if kw.lower() not in generic_tech_words or kw in acronyms
        ]

    # dict.fromkeys() deduplicates while preserving insertion order
    return list(dict.fromkeys(final_keywords))


# ---------------------------------------------------------------------------
# Keyword-based ChromaDB text search
# ---------------------------------------------------------------------------

def _text_search_keywords(store: Chroma, keywords: list[str], k: int = 10) -> list[Document]:
    """
    Search ChromaDB for documents containing specific keywords.

    Similar to _text_search_employees() but searches ALL document types
    (both employee records AND PDF policy chunks), not just employees.

    For each keyword:
      - If it's an acronym (all uppercase) or a phrase (contains spaces):
        Search for the EXACT string only (no case variants) — precise match.
      - Otherwise: Try the original, lowercase, and title-case variants.

    Acronym boosting:
        Documents matching an uppercase acronym (e.g. "OCR") are inserted at
        the FRONT of the results list instead of appended to the end.
        This ensures they survive the later deduplication step (first-seen wins).

    Args:
        store:    The ChromaDB vector store.
        keywords: Extracted keywords from _extract_keywords().
        k:        Maximum results per keyword variant search.

    Returns:
        Deduplicated list of Documents, with acronym matches at the front.
    """
    if not keywords:
        return []

    all_docs: list[Document] = []
    seen_ids: set[str] = set()

    for keyword in keywords:
        # Acronyms and phrases: exact match only (case variants would be too noisy)
        # Other words: try original + lowercase + title case
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
                        if keyword.isupper():
                            # Boost acronym matches to the front — they are
                            # the most specific signal and should survive truncation
                            all_docs.insert(0, doc_obj)
                        else:
                            all_docs.append(doc_obj)

                if results.get("documents"):
                    logger.info(
                        "Keyword search '%s': %d docs found", variant, len(results["documents"])
                    )
            except Exception:
                logger.exception("Keyword search failed for '%s'", variant)

    return all_docs


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Context Expansion (Sliding Window / Neighbor Fetching)
# ---------------------------------------------------------------------------

def _get_neighbors(store: Chroma, parent_id: str, current_index: int, window: int = 1) -> list[Document]:
    """
    Fetch the neighbor chunks for a given parent record by their index.
    
    Args:
        store: Chroma vector store.
        parent_id: The base ID of the document.
        current_index: The chunk index of the 'hit' chunk.
        window: How many neighbors to fetch in each direction.
    """
    if not parent_id or current_index is None:
        return []

    target_indices = []
    for offset in range(-window, window + 1):
        if offset == 0: continue
        idx = current_index + offset
        if idx >= 0:
            target_indices.append(idx)

    if not target_indices:
        return []

    try:
        # Use metadata filtering to find all siblings within the window
        # Note: We fetch ALL siblings in one go if possible, or iterate
        all_neighbors = []
        for idx in target_indices:
            results = store._collection.get(
                where={
                    "$and": [
                        {"parent_record_id": parent_id},
                        {"chunk_index": idx}
                    ]
                },
                include=["documents", "metadatas"],
                limit=1
            )
            for i, content in enumerate(results.get("documents") or []):
                meta = (results.get("metadatas") or [{}])[i] or {}
                all_neighbors.append(Document(page_content=content, metadata=meta))
        return all_neighbors
    except Exception:
        logger.debug("Failed to fetch neighbors for %s index %d", parent_id, current_index)
        return []


def _expand_and_stitch_context(store: Chroma, docs: list[Document], window: int = 1) -> list[Document]:
    """
    For each search result, fetch its neighbors and stitch contiguous chunks together.
    
    This turns fragmentation into flow and ensures no info is hidden across boundaries.
    """
    if not docs:
        return []

    # 1. Expand the set with neighbors
    expanded_pool = list(docs)
    seen_ids = {doc.metadata.get("record_id") for doc in docs if doc.metadata.get("record_id")}

    for doc in docs:
        parent_id = doc.metadata.get("parent_record_id")
        current_idx = doc.metadata.get("chunk_index")
        
        if parent_id and current_idx is not None:
            neighbors = _get_neighbors(store, parent_id, current_idx, window=window)
            for n in neighbors:
                n_id = n.metadata.get("record_id")
                if n_id and n_id not in seen_ids:
                    seen_ids.add(n_id)
                    expanded_pool.append(n)

    # 2. Group by parent document and stitch
    # We want to keep the original retrieval order but improve the content of each 'hit'
    stitched_docs: list[Document] = []
    
    # Sort pool by parent and index to make stitching easy
    # But we want to preserve the relative 'relevancy' order from the original search
    # So we'll iterate through the ORIGINAL hits and just replace them with their stitched context
    
    handled_parents = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_record_id")
        if not parent_id:
            stitched_docs.append(doc)
            continue
            
        if parent_id in handled_parents:
            continue
            
        handled_parents.add(parent_id)
        
        # Find all expanded chunks for THIS parent
        family = [d for d in expanded_pool if d.metadata.get("parent_record_id") == parent_id]
        # Sort by index
        family.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Stitch them together
        full_content = ""
        last_idx = -1
        for member in family:
            curr_idx = member.metadata.get("chunk_index", 0)
            if last_idx != -1 and curr_idx > last_idx + 1:
                full_content += "\n\n... [gap in document] ...\n\n"
            
            full_content += member.page_content + " "
            last_idx = curr_idx
            
        # Create the new super-doc
        stitched_docs.append(Document(
            page_content=full_content.strip(),
            metadata=doc.metadata # Keep the metadata of the highest scoring hit
        ))

    return stitched_docs


def _classify_query_intent(question: str, names: list[str]) -> str:
    """
    Classify the user's query into one of three intent categories.

    The intent is used to dynamically adjust how many chunks to retrieve
    from employee records vs PDF policy documents in the vector search step.

    Intent categories:
        "employee"  — Question is about a specific person's data
                      e.g. "What is Priya's salary?" or "When did John join?"
                      → Allocate more vector budget to employee record chunks

        "document"  — Question is about a company policy, technology, or project
                      e.g. "What is the notice period policy?" or "What tech did we use?"
                      → Allocate more vector budget to PDF policy chunks

        "mixed"     — Question combines both a person AND a policy topic
                      e.g. "How many leaves is Anthony entitled to?"
                      → Split vector budget 50/50 between employees and PDFs

    Signal-based classification:
        - If names are detected AND doc_signals are present → "mixed"
        - If names are detected OR emp_signals are present → "employee"
        - Otherwise → "document"

    The signal word lists (rag_doc_signals, rag_emp_signals) are loaded from
    settings so new HR terminology can be added without changing this code.

    Args:
        question: The raw user question string.
        names:    List of person names extracted by _extract_names().

    Returns:
        One of: "employee", "document", or "mixed"
    """
    # Normalise company aliases first so "as 55" / "@55" don't confuse intent
    q = _normalise_company_aliases(question).lower()
    # Load signal sets from config (previously hardcoded inline — now configurable)
    doc_signals = set(settings.rag_doc_signals)
    emp_signals = set(settings.rag_emp_signals)

    has_names      = len(names) > 0
    has_doc_signal = any(s in q for s in doc_signals)
    has_emp_signal = any(s in q for s in emp_signals)

    if has_names and has_doc_signal:
        return "mixed"
    elif has_names or has_emp_signal:
        return "employee"
    else:
        return "document"



# ---------------------------------------------------------------------------
# Listing query detection
# ---------------------------------------------------------------------------

def _is_listing_query(question: str) -> bool:
    """
    Detect whether the user wants a COMPLETE LIST of all matching items.

    This distinction matters critically for vector search budget allocation:

    PRECISION query — user wants the single best answer:
        "What OCR system did we use in Dubai?"
        -> Noise reduction ON: cut pdf_k to focus on the most relevant chunk.

    RECALL / LISTING query — user wants ALL matching items:
        "Give me ALL projects where OCR was used"
        "List every project that used machine learning"
        "Which projects have we used GPU in?"
        "as 55, have we used GPUs and in which prjects?"  ← typo handled
        -> Noise reduction OFF: keep pdf_k wide so no project is missed.
        -> pdf_k gets maximum budget (k - 2) so every relevant PDF is covered.

    Detection strategy:
        1. Exact phrase matching for clean queries
        2. Fuzzy regex matching for typo-resilient detection:
           - "which" + word starting with "proj" catches "which prjects",
             "which projs", "which projectss" etc.
           - "in which" pattern is a strong signal regardless of the
             next word spelling
        3. Question words "which" / "what" combined with a document-type
           query are treated as listing intent

    Args:
        question: The (already normalised) question string.

    Returns:
        True if the query is asking for a complete list, False otherwise.
    """
    q = question.lower()

    # ── Exact phrase triggers ─────────────────────────────────────────────
    listing_triggers = settings.rag_listing_triggers

    if any(trigger in q for trigger in listing_triggers):
        return True

    # ── Fuzzy / typo-resilient detection ─────────────────────────────────
    # Broader pattern for "in which <word resembling projects>":
    # Uses r"in\s+which\s+\w*pr\w*" which matches "in which prjects",
    # "in which projetcs", "in which projects" etc.
    if re.search(r"in\s+which\s+\w*pr\w*", q):
        return True

    # "which <word resembling projects>" — catches "which prjects used GPU"
    if re.search(r"which\s+\w*pr\w*", q):
        return True

    # "and in which" — strong signal for enumeration follow-up
    # e.g. "have we used GPU and in which prjects"
    if re.search(r"and\s+in\s+which", q):
        return True

    return False


# ---------------------------------------------------------------------------
# Main hybrid retrieval function
# ---------------------------------------------------------------------------

async def _retrieve_mixed(store: Chroma, question: str, k: int = 20) -> list[Document]:
    """
    Run the full hybrid retrieval pipeline and return the top-k relevant chunks.

    This is the core retrieval logic of ResoAI. It runs multiple retrieval
    strategies in parallel conceptually, merges their results, then applies
    FlashRank re-ranking to select the best final set for the LLM.

    Detailed pipeline steps:

    Step 0 — BM25 init (lazy, runs only on the very first query):
        Fetches all ChromaDB documents and builds the BM25 index.
        All subsequent calls skip this step instantly.

    Step 1 — Name Match:
        Extracts person names from the question and searches ChromaDB using
        exact text matching. This is the most reliable way to retrieve a
        specific employee's record and avoids mixing up similarly-named people.

    Step 2 — BM25 Search:
        Scores all indexed documents against the query using BM25 and returns
        the top bm25_top_k candidates. Great for questions with clear keywords
        like "notice period" or "probation" that appear verbatim in policy text.

    Step 3 — Regex Keyword Search:
        Extracts meaningful keywords from the question and runs ChromaDB
        $contains searches. Handles bi-gram phrases and acronym boosting.

    Step 4 — Intent Classification + Vector Budget:
        Determines whether the question is about an employee, a document, or both.
        Then sets emp_k and pdf_k: how many chunks to request from each category
        in the upcoming vector search. Adjusts further if acronyms are detected.

    Step 5 — Vector Search:
        Runs semantic similarity_search using the embedding model. Finds chunks
        that are semantically RELATED to the question even if they don't share
        exact keywords. Filtered by record_type to control the emp/PDF balance.

    Step 6 — Merge & Deduplicate:
        Combines all candidates from all retrieval methods using a priority order:
          BM25 > Regex Keywords > Name Matches > PDF Vectors > Employee Vectors
        Deduplication uses record_id — if the same chunk appears in multiple
        retrieval results, only the highest-priority occurrence is kept.

    Step 7 — FlashRank Re-ranking:
        Sends all merged candidates to the FlashRank cross-encoder. It scores
        each (question, chunk) pair holistically and returns them re-ordered by
        true relevance. The top `k` chunks are kept for the LLM.
        Falls back to priority-order truncation if FlashRank is unavailable.

    Args:
        store:    The ChromaDB vector store (globally cached).
        question: The raw user question string.
        k:        Maximum number of chunks to return (settings.retriever_top_k).

    Returns:
        List of the top-k most relevant Document chunks for this question.
    """
    logger.info("=" * 72)
    logger.info("CHUNK SELECTION -- question: %r", question)
    logger.info("=" * 72)

    # ── Step 0: Pre-process question ──────────────────────────────────────
    # Normalise company aliases and tech acronyms ONCE here so that all
    # downstream steps (name extraction, BM25, intent classification,
    # keyword search) all work on the same cleaned question string.
    question = _normalise_company_aliases(question)
    question = _normalise_tech_acronyms(question)
    logger.info(">> Step 0 | Normalised question: %r", question)

    # ── Step 0b: BM25 initialization (no-op after first call) ─────────────
    await _init_bm25(store)

    # ── Step 1: Name match ─────────────────────────────────────────────────
    names = _extract_names(question)
    logger.info(">> Step 1 | Extracted names: %s", names if names else "none")
    name_docs = (
        _text_search_employees(store, names, k=settings.rag_name_search_top_k)
        if names else []
    )
    _log_selected_chunks(name_docs, "Step 1 -- Name Match")

    # ── Step 2: BM25 keyword search ────────────────────────────────────────
    bm25_docs = _bm25_search(question, k=settings.bm25_top_k)
    _log_selected_chunks(bm25_docs, "Step 2 -- BM25")

    # ── Step 3: Regex keyword search ───────────────────────────────────────
    keywords = _extract_keywords(question)
    logger.info(">> Step 3 | Extracted keywords: %s", keywords if keywords else "none")
    keyword_docs = (
        _text_search_keywords(store, keywords, k=settings.rag_keyword_search_top_k)
        if keywords else []
    )
    _log_selected_chunks(keyword_docs, "Step 3 -- Regex Keywords")

    # ── Step 4: Intent classification + vector budget ──────────────────────
    intent = _classify_query_intent(question, names)
    # Detect if this is a "list all" / "give me all" type query.
    # For these queries we want RECALL (find every matching project),
    # not PRECISION (find the single best matching chunk).
    is_listing_query = _is_listing_query(question)
    logger.info(">> Step 4 | Query intent: '%s' | Listing query: %s", intent, is_listing_query)

    # Assign vector search budgets based on intent.
    # For listing queries, give PDF vector search the full remaining budget
    # so we have the best chance of finding ALL matching projects.
    if intent == "employee":
        emp_k = max(k // 2, 5)
        pdf_k = max(k // 3, 5)
    elif intent == "document":
        if is_listing_query:
            # Listing queries need maximum PDF coverage — don't limit PDF slots
            emp_k = 2
            pdf_k = k - 2        # give almost everything to PDFs
        else:
            emp_k = max(k // 4, 3)
            pdf_k = max(k * 2 // 3, 5)
    else:  # mixed — equal split
        emp_k = max(k // 2, 5)
        pdf_k = max(k // 2, 5)

    # Noise reduction: if a specific acronym is detected AND this is NOT a
    # listing query, tighten vector budgets to boost precision.
    # We skip noise reduction for listing queries because it cuts pdf_k in
    # half — exactly the wrong thing when we need to find ALL projects.
    # Example:
    #   "what OCR system was used in Dubai?" → precision query → reduce budget ✓
    #   "give me ALL projects where OCR was used" → recall query → keep budget ✓
    has_acronym = any(kw.isupper() and len(kw) >= 2 for kw in keywords)
    if has_acronym and not is_listing_query:
        emp_k = 2 if intent != "employee" else emp_k // 2
        pdf_k = 5 if intent == "employee" else pdf_k // 2
        logger.info(
            ">> Step 4 | Acronym + precision query — tightening budget: emp_k=%d, pdf_k=%d",
            emp_k, pdf_k,
        )
    elif has_acronym and is_listing_query:
        logger.info(
            ">> Step 4 | Acronym detected but listing query — keeping wide budget for recall",
        )

    logger.info(">> Step 4 | Vector budget: emp_k=%d, pdf_k=%d", emp_k, pdf_k)

    # ── Step 5: Vector (semantic) search ──────────────────────────────────
    # similarity_search converts the question to a vector, then finds the
    # k most similar chunks using cosine similarity in ChromaDB's index.
    employee_docs = store.similarity_search(
        question, k=emp_k, filter={"record_type": "employee"}
    )
    _log_selected_chunks(employee_docs, "Step 5 -- Vector (Employee)")

    # Global search (projects, policies, etc.)
    pdf_docs = store.similarity_search(
        question, k=pdf_k
    )
    _log_selected_chunks(pdf_docs, "Step 5 -- Vector (Global)")

    # ── Step 5b: Surgical Directory Sweep (Sledgehammer 2.0) ──────────────
    # For directory listing queries, vector search often fails to rank
    # minority roles (like Data Scientist) in the top slots.
    # We manually sweep the directory records for the query keywords.
    # ── Step 5b: Semantic Directory Sweep (Sledgehammer 3.0) ──────────────
    # Transitioned from strict keyword match to SEMANTIC SEARCH restricted
    # to the directory. This allows the system to recognize synonyms like
    # 'Developer' vs 'Engineer'.
    directory_sweep_docs = []
    if intent in ("employee", "mixed") and (is_listing_query or "directory" in question.lower()):
        try:
            # We use a high k (30) to capture all matching people.
            # Filtering by 'misc_table_row' ensures we only get directory records.
            directory_sweep_docs = store.similarity_search(
                question,
                k=settings.rag_directory_sweep_top_k,
                filter={"record_type": "misc_table_row"}
            )
            logger.info(
                ">> Step 5b | Semantic Directory Sweep: Found %d matches",
                len(directory_sweep_docs)
            )
        except Exception as e:
            logger.error("Semantic Directory Sweep failed: %s", e)

    # ── Step 6: Merge & deduplicate ────────────────────────────────────────
    # For directory listing queries, if we found matches via the surgical sweep,
    # we enter 'PURE DIRECTORY MODE'. We discard everything else to avoid noise
    # from project case studies that might confuse the LLM.
    if directory_sweep_docs and intent in ("employee", "mixed") and (is_listing_query or "directory" in question.lower()):
        combined = directory_sweep_docs[:k]
        logger.info(">> Step 6 | PURE DIRECTORY MODE: Using %d sweep matches, discarding %d other candidates", len(combined), len(bm25_docs + keyword_docs + pdf_docs))
    else:
        # Normal priority order: BM25 → regex keywords → name matches → Global vectors → employee vectors.
        seen_ids: set = set()
        combined = []
        for doc in bm25_docs + keyword_docs + name_docs + pdf_docs + employee_docs:
            doc_id = doc.metadata.get("record_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append(doc)

    # ── Step 6b: Acronym relevance filter ──────────────────────────────────
    # When the query contains a specific acronym (GPU, OCR, etc.), vector search
    # often returns chunks from popular unrelated projects (e.g. Dubai Technologies)
    # that are semantically similar but do NOT actually mention the acronym.
    # This causes the LLM to HALLUCINATE — it sees a Dubai chunk in context and
    # incorrectly attributes GPU/OCR usage to Dubai.
    #
    # IMPORTANT: We must apply this filter BEFORE passing to FlashRank (Step 7).
    # If we only demote chunks, FlashRank re-scores everything and moves them
    # back to the top — completely undoing the demotion.
    #
    # For listing queries with acronyms: EXCLUDE chunks that don't contain the
    # acronym from the FlashRank input entirely. They are stored separately and
    # only added back if we don't have enough relevant chunks to fill top_k.
    #
    # EXTENDED: also handles mixed-case technology names (Redis, Kafka, Flutter etc.)
    # which are NOT all-uppercase but are still specific tech terms.
    # e.g. "which projects used Redis?" — "Redis" is not an acronym but should
    # still trigger the filter so only Redis-containing chunks reach FlashRank.
    acronym_keywords = [kw for kw in keywords if kw.isupper() and len(kw) >= 2]

    # Detect mixed-case tech terms from the query by checking against config list
    known_tech = set(settings.rag_known_tech_terms)
    tech_term_keywords = [
        kw for kw in keywords
        if kw.lower() in known_tech and kw not in acronym_keywords
    ]

    # Combine: all-uppercase acronyms + known mixed-case tech terms
    filter_keywords = acronym_keywords + tech_term_keywords
    if filter_keywords:
        logger.info(">> Step 6b | Filter keywords: %s", filter_keywords)

    excluded_by_acronym: list[Document] = []
    if filter_keywords and is_listing_query:
        relevant = []
        excluded = []
        for doc in combined:
            text_lower = doc.page_content.lower()
            if any(fkw.lower() in text_lower for fkw in filter_keywords):
                relevant.append(doc)
            else:
                excluded.append(doc)
        if relevant:
            # Only pass matching chunks to FlashRank.
            # Non-matching chunks are excluded so FlashRank cannot rescue them.
            excluded_by_acronym = excluded
            combined = relevant
            logger.info(
                ">> Step 6b | Tech filter '%s': %d relevant chunks → FlashRank | %d excluded",
                filter_keywords, len(relevant), len(excluded),
            )
        else:
            logger.info(
                ">> Step 6b | Tech filter '%s': no matching chunks — skipping filter",
                filter_keywords,
            )

    # ── Step 7: FlashRank re-ranking ───────────────────────────────────────
    # Re-rank all merged candidates using a cross-encoder model.
    # Unlike bi-encoder (embedding) models which score query and doc separately,
    # cross-encoders score them TOGETHER, giving much better relevance judgement.
    if _ranker and len(combined) > 2:
        try:
            # Build the passage list FlashRank expects
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(combined)
            ]
            rerank_request = RerankRequest(query=question, passages=passages)
            results = _ranker.rerank(rerank_request)

            # ── Step 7b: Keyword Boosting (Post-Rerank) ──────────────────────
            # Even with cross-encoders, occasionally a semantically matched 
            # but keyword-empty chunk (e.g. Bereavement for a Marriage query)
            # can outrank a keyword-rich chunk. We apply a manual boost here.
            boosted_results = []
            # Use keywords extracted in Step 3 for boosting
            boost_kws = [kw.lower() for kw in keywords]
            
            for res in results:
                score = res["score"]
                text_lower = res["text"].lower()
                
                # Count how many unique query keywords are present in this chunk
                matches = sum(1 for kw in boost_kws if kw in text_lower)
                if matches > 0:
                    score += min(
                        matches * settings.rag_keyword_boost_per_match,
                        settings.rag_keyword_boost_cap,
                    )
                
                res["boosted_score"] = score
                boosted_results.append(res)

            # Re-sort based on boosted score
            boosted_results.sort(key=lambda x: x["boosted_score"], reverse=True)

            # Reconstruct documents in re-ranked order, taking only top k
            combined = [combined[res["id"]] for res in boosted_results[:k]]
            logger.info(
                ">> Step 7 | Re-ranked %d candidates → applied keyword boost → kept top %d",
                len(passages), len(combined),
            )
        except Exception as e:
            logger.error(
                "Re-ranking failed: %s. Falling back to priority-order truncation.", e
            )
            combined = combined[:k]
    else:
        # FlashRank unavailable or too few candidates — just truncate
        combined = combined[:k]

    # ── Step 8: Adaptive Context Expansion ─────────────────────────────────
    # Automatically fetch neighbor chunks and stitch contiguous pieces.
    # This ensures no content is hidden across chunk boundaries.
    logger.info(">> Step 8 | Expanding and stitching context (window=%d)...", settings.rag_retriever_look_ahead)
    combined = _expand_and_stitch_context(
        store, 
        combined, 
        window=settings.rag_retriever_look_ahead
    )

    # ── Final summary log ──────────────────────────────────────────────────
    _log_selected_chunks(combined, "FINAL -- context sent to LLM")
    logger.info(
        "SUMMARY [intent=%s] bm25=%d  regex=%d  name=%d  emp_vec=%d  pdf_vec=%d  final=%d",
        intent, len(bm25_docs), len(keyword_docs), len(name_docs),
        len(employee_docs), len(pdf_docs), len(combined),
    )
    logger.info("=" * 72)
    return combined


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ask(question: str) -> dict:
    """
    Handle a user question end-to-end: retrieve context, generate answer, return result.

    This is the ONLY public function in this module. It is called by:
      - FastAPI's /api/ask endpoint (direct HTTP queries)
      - MS Teams bot's on_message_activity handler (Teams messages)

    Full execution flow:
      1. Increment active request counter and calculate allocated threads
      2. Get (or create) the cached vector store
      3. Run _retrieve_mixed() to get the top-k relevant chunks
      4. Format the chunks into a context string via _format_docs()
      5. Build the LangChain chain: RAG_PROMPT | ChatOllama | StrOutputParser
      6. Invoke the chain with {context, question} → get the answer string
      7. Build the sources list for the API response
      8. Decrement the active request counter

    Thread allocation (dynamic concurrency control):
        Ollama is CPU-bound. When 3 users ask simultaneously, each gets
        1/3 of the total thread budget (ollama_max_threads // 3).
        This prevents any single request from monopolizing all CPU cores.
        The counter is tracked with an async lock to avoid race conditions.

    Background timer:
        A background asyncio task logs "Still generating..." every 5 seconds
        during LLM inference, so you can see in the terminal that the server
        is alive and working (not hung) on slow CPU machines.

    Error handling:
        If any exception occurs during retrieval or generation, it propagates
        up to the caller (FastAPI or the Teams bot), which catches it and
        returns an appropriate error response.
        The finally block ALWAYS decrements the active request counter to
        prevent the counter from drifting out of sync on errors.

    Args:
        question: The raw user question string (from Teams or the API).

    Returns:
        A dict with three keys:
            "answer"               — the LLM's response string
            "sources"              — list of dicts with metadata for each used chunk
            "time_elapsed_seconds" — total wall clock time for the full request
    """
    global _active_ask_requests

    # Atomically increment the counter — track this concurrent request
    async with _ask_lock:
        _active_ask_requests += 1
        current_requests = _active_ask_requests

    # Fair thread allocation: divide Ollama's CPU budget equally among active requests
    allocated_threads = max(1, settings.ollama_max_threads // current_requests)
    logger.info(
        "[Thread Allocator] Active requests: %d | Threads allocated for this request: %d",
        current_requests, allocated_threads,
    )

    try:
        start_time = time.monotonic()  # monotonic clock — immune to system clock changes

        # --- CACHE CHECK ---
        cached_res = get_cached_answer(question)
        if cached_res:
            cached_res["time_elapsed_seconds"] = settings.rag_cache_hit_elapsed_seconds
            return cached_res
        # ------------------


        # --- DYNAMIC DATA INTERPRETER (ADVANCED AGENTIC ROUTING) ---
        # Detect if this is a 'How many' or 'List all' query that requires
        # 100% precision from structured files.
        interpreter = _interpreter
        itp_result = interpreter.interpret(question)
        if itp_result:
            entity_type = itp_result["entity"]
            criteria = itp_result["criteria"]
            operation = itp_result.get("operation", "count")
            logger.info(
                "[Interpreter] Structured query detected: operation=%s | entity=%s | criteria=%s",
                operation, entity_type, criteria,
            )

            count = itp_result["count"]
            matches = itp_result["matches"]

            if count == 0:
                qualifier = "" if criteria == "__all__" else f" matching '{criteria}'"
                answer = (
                    f"Based on the live data scan, there are **0** {entity_type}(s)"
                    f"{qualifier}. No matching records were found."
                )
            else:
                # OPTIMIZATION: If we have exactly 1 match, we should show the full details 
                # (Email, Phone, etc.) regardless of the inferred operation.
                if count == 1:
                    match = matches[0]
                    record = match.get("data", {})
                    detail_lines = [f"I found one match for **{match['name']}**:"]
                    
                    # Ignore internal / redundant keys
                    skip_keys = {"record_id", "record_type", "is_tabular", "id", "name", "full_text"}
                    for key, val in record.items():
                        if key.lower() not in skip_keys and val:
                            label = key.replace("_", " ").title()
                            detail_lines.append(f"- **{label}**: {val}")
                    
                    answer = "\n".join(detail_lines)
                else:
                    preview_limit = min(settings.rag_context_max_chunks, len(matches))
                    match_listing = ", ".join([m["name"] for m in matches[:preview_limit]])
                    if len(matches) > preview_limit:
                        match_listing += f", and {len(matches)-preview_limit} others"
    
                    if operation == "list":
                        answer = (
                            f"Based on the live data scan, I found **{count}** {entity_type}(s). "
                            f"The results include: {match_listing}."
                        )
                    else:
                        qualifier = " in the company" if criteria == "__all__" else f" matching '{criteria}'"
                        answer = (
                            f"Based on the live data scan, there are **{count}** {entity_type}(s)"
                            f"{qualifier}. The results include: {match_listing}."
                        )


            # Dynamic Logging for Interpreter
            _log_retrieval(question, "DataInterpreter", matches, criteria=criteria)

            total_time = time.monotonic() - start_time
            sources = [{
                "record_type": "data_interpreter",
                "count_found": count,
                "criteria": criteria,
                "entity": entity_type,
                "operation": operation,
                "filename": settings.interpreter_file_map.get(entity_type, "universal_json_scan"),
            }]

            # Save interpreter results to cache so repeated questions get instant hits
            save_to_cache(question, answer, sources)

            return {
                "answer": answer,
                "sources": sources,
                "time_elapsed_seconds": total_time
            }
        # ------------------------------------------------------------

        # Get the cached global vector store (no-op after first request)
        store = await _get_or_create_store(num_thread=allocated_threads)

        # Create an LLM instance with the allocated thread budget
        llm = _get_llm(num_thread=allocated_threads)

        # Run the full hybrid retrieval pipeline
        docs = await _retrieve_mixed(store, question, k=settings.retriever_top_k)
        retrieval_time = time.monotonic() - start_time

        # Dynamic Logging for Hybrid RAG
        _log_retrieval(question, "HybridRAG", docs)

        # Format retrieved chunks into the context string for the LLM
        context = _format_docs(docs)

        # Build the LangChain LCEL chain:
        #   RAG_PROMPT  → formats {context} and {question} into a chat prompt
        #   llm         → sends the prompt to Ollama and gets the answer
        #   StrOutputParser → extracts the answer string from the ChatMessage
        chain = RAG_PROMPT | llm | StrOutputParser()

        # Background task: log progress every 5 seconds during LLM generation.
        # This is useful on slow CPU machines where generation takes 30+ seconds.
        # The task is cancelled in the finally block once the answer arrives.
        async def log_elapsed_time() -> None:
            gen_start = time.monotonic()
            try:
                while True:
                    await asyncio.sleep(settings.rag_generation_log_interval_seconds)
                    elapsed = time.monotonic() - gen_start
                    logger.info("... LLM still generating (%.1fs elapsed)", elapsed)
            except asyncio.CancelledError:
                pass  # Normal exit — cancelled when answer is ready

        timer_task = asyncio.create_task(log_elapsed_time())
        try:
            # Invoke the chain asynchronously — waits for the full answer
            logger.info(">> Step 10 | Generation starting... (context: %d chars)", len(context))
            answer = await chain.ainvoke({"context": context, "question": question})
        finally:
            # Always cancel the timer, whether answer succeeded or raised an exception
            timer_task.cancel()

        total_time = time.monotonic() - start_time
        logger.info(
            "Answer ready in %.2fs (top_k=%d, context_chars=%d)",
            total_time, settings.retriever_top_k, len(context),
        )

        # Build the sources list — tells the caller which chunks were used
        # (surfaced in the API response but not shown in the Teams chat)
        sources = [
            {
                "record_type":   doc.metadata.get("record_type"),
                "record_id":     doc.metadata.get("record_id"),
                "employee_name": doc.metadata.get("employee_name"),
                "project_name":  doc.metadata.get("project_name"),
            }
            for doc in docs
        ]

        # --- SAVE TO CACHE ---
        save_to_cache(question, answer, sources)
        # --------------------

        return {
            "answer":               answer or settings.rag_fallback_answer,
            "sources":              sources,
            "time_elapsed_seconds": round(total_time, 2),
        }

    finally:
        # ALWAYS decrement the counter, even if an exception was raised above.
        # Without this, a failed request would permanently reduce thread budget
        # for all future requests.
        async with _ask_lock:
            _active_ask_requests -= 1
