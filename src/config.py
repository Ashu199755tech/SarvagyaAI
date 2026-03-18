"""
config.py — Central Application Configuration for ResoAI
=========================================================

All runtime settings are defined here as a single Pydantic BaseSettings class.
Every value can be overridden via a `.env` file or environment variable without
touching source code — making it safe to tune the system in production.

How it works:
  - On startup, `settings = Settings()` is called once.
  - Pydantic reads each field from the environment (or .env file).
  - If a variable is not set, the default value defined below is used.
  - All other modules import `settings` from here — there is only ONE instance.

To override any setting, add it to your .env file, e.g.:
    RETRIEVER_TOP_K=12
    OLLAMA_NUM_CTX=16384
    CHUNK_SIZE=400
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration class for ResoAI.

    All fields map 1-to-1 to environment variables (case-insensitive).
    For example, `ollama_llm_model` reads from the env var `OLLAMA_LLM_MODEL`.
    """

    # -------------------------------------------------------------------------
    # Employee API
    # -------------------------------------------------------------------------
    # Base URL for the local Keka HR API that provides employee records.
    # The ingestion pipeline calls GET /employees on this URL.
    employee_api_base_url: str = "http://localhost:8000"

    # -------------------------------------------------------------------------
    # PDF Ingestion
    # -------------------------------------------------------------------------
    # Absolute path to the folder containing company policy PDFs.
    # The ingestion pipeline recursively scans this folder for *.pdf files.
    # Change this path if your policies folder is in a different location.
    pdf_folder: str = "/home/fifity-five/Downloads/Policies 2026"

    # -------------------------------------------------------------------------
    # Ollama — Local LLM Server
    # -------------------------------------------------------------------------
    # URL where Ollama is running. Default is localhost.
    # For remote GPU servers, set this to e.g. http://192.168.1.100:11434
    ollama_base_url: str = "http://localhost:11434"

    # The Ollama model used for generating answers (the "brain").
    # Pull it first with: ollama pull llama3.2
    ollama_llm_model: str = "llama3.2"

    # The Ollama model used for creating vector embeddings during ingestion.
    # Pull it first with: ollama pull nomic-embed-text
    # Do NOT change this after ingesting data — the vectors won't match.
    ollama_embedding_model: str = "nomic-embed-text"

    # Total CPU threads available to Ollama across ALL concurrent requests.
    # When 2 requests arrive simultaneously, each gets max_threads // 2 threads.
    # Set this to your CPU's physical core count for best performance.
    ollama_max_threads: int = 12

    # Default thread count for background tasks (ingestion scheduler, etc.)
    # Should be lower than ollama_max_threads to leave headroom for queries.
    ollama_num_threads: int = 6

    # -------------------------------------------------------------------------
    # Ollama LLM Tuning
    # -------------------------------------------------------------------------
    # temperature controls how "creative" vs "deterministic" the LLM is.
    # 0.0 = fully deterministic (same question always gives same answer).
    # 1.0 = highly creative (may hallucinate). For HR facts, keep this LOW.
    # Previously hardcoded as 0.1 inside _get_llm() in chain.py — now configurable.
    ollama_temperature: float = 0.1

    # num_ctx is the LLM's context window size in tokens.
    # This is how much text the LLM can "see" at once (retrieved chunks + question).
    # PERFORMANCE NOTE: Reducing this from 32768 → 8192 is the single biggest
    # speed improvement. With top_k=8 chunks of ~300 chars each, the total context
    # is ~2400 tokens — well within 8192. This cuts LLM processing time ~4x.
    # Increase this only if answers are getting cut off or missing context.
    ollama_num_ctx: int = 8192

    # -------------------------------------------------------------------------
    # ChromaDB — Vector Database
    # -------------------------------------------------------------------------
    # Directory where ChromaDB persists its vector data to disk.
    # This folder is created automatically if it doesn't exist.
    chroma_persist_dir: str = "./data/chromadb"

    # Name of the ChromaDB collection that stores all document chunks.
    # Think of this like a table name in a SQL database.
    chroma_collection_name: str = "resoai_hr"

    # -------------------------------------------------------------------------
    # MS Teams Bot
    # -------------------------------------------------------------------------
    # Azure Bot App ID — obtained when registering the bot in Azure Portal.
    # Required for the bot to authenticate with Microsoft's Bot Framework.
    teams_app_id: str = ""

    # Azure Bot App Password (client secret) — paired with the App ID above.
    # Keep this value secret and never commit it to source control.
    teams_app_password: str = ""

    # -------------------------------------------------------------------------
    # Ingestion Scheduler
    # -------------------------------------------------------------------------
    # How often the background scheduler re-syncs employee data and PDFs.
    # Default is every 6 hours. Reduce if your HR data changes frequently.
    # The scheduler calls IngestionPipeline.run() on this interval.
    ingestion_interval_hours: int = 6

    # -------------------------------------------------------------------------
    # RAG Retrieval Settings
    # -------------------------------------------------------------------------
    # Maximum number of document chunks passed to the LLM as context.
    # PERFORMANCE NOTE: Reduced from 20 → 8. With FlashRank re-ranking,
    # 8 high-quality chunks consistently outperform 20 mixed-quality chunks,
    # AND the LLM processes ~4x less text, making responses significantly faster.
    # Increase to 12-15 only if answers seem to be missing relevant context.
    # Increased from 8 → 12 to ensure multi-project listing queries (e.g.
    # "which projects used GPU") retrieve enough chunks to cover all projects.
    # With 3 GPU projects × 2 chunks each = 6 GPU chunks needed minimum,
    # plus 2 slots for context — 12 gives safe headroom for FlashRank to work.
    retriever_top_k: int = 12

    # How many candidate documents BM25 search returns before re-ranking.
    # BM25 casts a wide net; FlashRank then picks the best `retriever_top_k`.
    # Higher values = more candidates for re-ranking but slightly slower retrieval.
    bm25_top_k: int = 15

    # -------------------------------------------------------------------------
    # Chunking Settings (used by ingestion/pipeline.py)
    # -------------------------------------------------------------------------
    # NOTE: These settings were previously defined in config but IGNORED by
    # pipeline.py (which had its own hardcoded values). They are now actually
    # read and used by the IngestionPipeline. Re-ingest after changing these.

    # Maximum character length of each text chunk stored in ChromaDB.
    # This is the fallback splitter's chunk size when semantic chunks are too large.
    # PERFORMANCE NOTE: Reduced from 500 → 300. Smaller chunks mean:
    #   - More precise retrieval (each chunk covers one topic, not three)
    #   - Less text sent to the LLM per chunk → faster generation
    #   - Better BM25 scoring (less noise per chunk)
    # If you find answers are incomplete, try increasing to 400-500.
    chunk_size: int = 300

    # Number of characters that overlap between consecutive chunks.
    # Overlap ensures that sentences split across chunk boundaries are still
    # fully represented in at least one chunk. Too high = redundant context.
    chunk_overlap: int = 50

    # Maximum character size a semantic chunk is allowed to be before the
    # fallback RecursiveCharacterTextSplitter further splits it.
    # SemanticChunker can produce very large chunks for dense policy pages —
    # this cap prevents those from consuming too much of the LLM's context window.
    # Reduced from 3000 → 1500 to keep chunks tighter.
    semantic_chunk_max_size: int = 1500

    # -------------------------------------------------------------------------
    # RAG Keyword & Intent Tuning
    # -------------------------------------------------------------------------
    # All lists below were previously hardcoded inside chain.py functions,
    # meaning every code change required editing Python. They are now here so
    # non-developers can extend them without touching source code.

    # Two-word HR phrases that should be treated as a single search unit.
    # Example: "notice period" should NOT be split into "notice" + "period"
    # because searching for "period" alone is very noisy. Add new HR terms here
    # as your policy vocabulary grows.
    rag_phrase_candidates: list[str] = [
        # ── HR policy bi-grams ─────────────────────────────────────────────
        "notice period",
        "probation period",
        "notice pay",
        "notice days",
        "leave policy",
        "travel policy",
        "maternity leave",
        "paternity leave",
        "gratuity policy",
        "appraisal cycle",
        "work from home",
        "expense reimbursement",
        # ── Technology / case study bi-grams ───────────────────────────────
        # BUG FIX: these phrases appear in case study PDFs and should be
        # searched as a unit, not split into individual words.
        "deep learning",
        "machine learning",
        "neural network",
        "computer vision",
        "natural language",
        "tech stack",
        "case study",
        "case studies",
        # ── Company name variants ──────────────────────────────────────────
        # BUG FIX: users write "at 55", "as 55", "@55" — these are expanded
        # to "FiftyFive Technologies" by _normalise_company_aliases() in chain.py
        # before keyword extraction. The canonical form is listed here so it
        # is treated as a single search unit (not split into two words).
        "fiftyfive technologies",
    ]

    # Generic technology words that are suppressed from keyword search when
    # a specific acronym (e.g. OCR, RAG, NLP) is already found in the query.
    # Example: "What OCR system did we build?" → "system" is noise, "OCR" is signal.
    rag_generic_tech_words: list[str] = [
        "software", "system", "platform", "development", "solution",
    ]

    # Words that indicate the user is asking about a DOCUMENT, POLICY, or PROJECT
    # rather than a specific employee. Used by _classify_query_intent() in chain.py
    # to decide how to split the vector search budget (more PDF chunks vs employee chunks).
    # Add new signals here when users start asking about new topics not yet covered.
    rag_doc_signals: list[str] = [
        # Policy / HR document signals
        "policy", "case study", "case studies", "project", "technology",
        "tech stack", "framework", "built", "developed", "delivered",
        "client", "outcome", "solution", "architecture", "platform",
        "system", "application", "tool", "integration", "procedure",
        "guideline", "rule", "regulation", "benefit", "entitle",
        "probation", "notice period", "separation", "referral",
        "attendance", "holiday", "travel", "posh", "iso",
        # Technology / hardware signals — BUG FIX: queries like
        # "have we used GPUs?" were classified as "employee" intent
        # because "GPUs" was mistakenly extracted as a person name.
        # These signals ensure GPU/ML/AI queries get document intent.
        "gpu", "gpus", "ai", "ml", "nlp", "ocr", "llm", "deep learning",
        "machine learning", "neural network", "computer vision",
        "fiftyfive", "fiftyfive technologies", "55 technologies",
        "used", "stack", "infrastructure",
    ]

    # Words that indicate the user is asking about a SPECIFIC EMPLOYEE's data
    # (as opposed to a general company policy). These trigger "employee" intent,
    # which allocates more vector budget to employee records than PDF chunks.
    rag_emp_signals: list[str] = [
        "salary", "joining date", "department", "designation",
        "email", "employee id", "emp id",
    ]

    # -------------------------------------------------------------------------
    # Known Tech Acronyms — for case-insensitive normalisation
    # -------------------------------------------------------------------------
    # WHY THIS EXISTS:
    #   ChromaDB $contains is case-sensitive. If a user types "ocr" (lowercase)
    #   but PDFs store "OCR" (uppercase), the search returns zero results.
    #   chain.py's _normalise_tech_acronyms() reads this list and rewrites
    #   any matching word in the question to uppercase BEFORE searching,
    #   so "tell me projects where ocr is used" becomes
    #      "tell me projects where OCR is used" → correctly hits PDF chunks.
    #
    # HOW TO ADD NEW ACRONYMS:
    #   Just append the lowercase form here. chain.py auto-uppercases them.
    #   For plurals (e.g. "apis"), the trailing 's' is stripped automatically
    #   so "apis" → "API" (not "APIS") to match how PDFs write them.
    #
    # EXAMPLES OF QUERIES THIS FIXES:
    #   "give me all projects where ocr was used"   → OCR
    #   "have we used gpus?"                        → GPU
    #   "tell me about nlp work"                    → NLP
    #   "which projects used ai"                    → AI
    rag_known_acronyms: list[str] = [
        "ocr",   # Optical Character Recognition
        "gpu",   # Graphics Processing Unit
        "gpus",  # plural — normalised to "GPU"
        "ai",    # Artificial Intelligence
        "ml",    # Machine Learning
        "nlp",   # Natural Language Processing
        "llm",   # Large Language Model
        "llms",  # plural — normalised to "LLM"
        "rag",   # Retrieval Augmented Generation
        "api",   # Application Programming Interface
        "apis",  # plural — normalised to "API"
        "ui",    # User Interface
        "ux",    # User Experience
        "erp",   # Enterprise Resource Planning
        "crm",   # Customer Relationship Management
        "bi",    # Business Intelligence
        "etl",   # Extract Transform Load
        "cv",    # Computer Vision
        "rpa",   # Robotic Process Automation
        "iot",   # Internet of Things
        "aws",   # Amazon Web Services
        "gcp",   # Google Cloud Platform
        "ci",    # Continuous Integration
        "cd",    # Continuous Delivery / Deployment
    ]

    # -------------------------------------------------------------------------
    # Known Technology Terms (mixed-case) — for tech-name listing queries
    # -------------------------------------------------------------------------
    # These are technology names that are NOT all-uppercase acronyms but
    # should still trigger the relevance filter in chain.py Step 6b.
    # When a user asks "which projects used Redis?", the filter needs to know
    # "Redis" is a specific tech term — just like "GPU" or "OCR".
    # chain.py checks this list to decide if Step 6b should filter chunks.
    # Add any technology name here that users might ask about by name.
    rag_known_tech_terms: list[str] = [
        # Databases / caches
        "redis", "kafka", "mongodb", "postgres", "postgresql", "elasticsearch",
        "chromadb", "firebase", "dynamodb", "cassandra", "sqlite",
        "mariadb", "mysql",
        # Frameworks / runtimes
        "flutter", "react", "django", "fastapi", "laravel", "nestjs",
        "nextjs", "pytorch", "tensorflow", "langchain",
        # Cloud / infra
        "kubernetes", "docker", "terraform", "airflow", "jenkins",
        # Other tech
        "stripe", "twilio", "openai", "ollama",
    ]

    # -------------------------------------------------------------------------
    # FastAPI Server
    # -------------------------------------------------------------------------
    # Host and port for the FastAPI web server.
    # 0.0.0.0 means "listen on all network interfaces" (required for ngrok/docker).
    host: str = "0.0.0.0"
    port: int = 8001

    # Pydantic configuration — tells it to read from a .env file.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# ---------------------------------------------------------------------------
# Singleton — imported by all other modules
# ---------------------------------------------------------------------------
# This is the single shared settings instance used throughout the application.
# Import pattern in other files:
#   from src.config import settings
#   print(settings.ollama_llm_model)
settings = Settings()