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
    employee_api_base_url: str = "http://localhost:8005"

    # -------------------------------------------------------------------------
    # Client Identity — Company-specific values
    # -------------------------------------------------------------------------
    # The canonical company name used in responses and alias expansion.
    company_name: str = "FiftyFive Technologies"

    # Short aliases users type instead of the full company name.
    # Each alias is expanded to company_name during query pre-processing.
    # Override in .env as a JSON list: COMPANY_ALIASES='["@55","at 55"]'
    company_aliases: list[str] = [
        "@55", "at 55", "as 55", "55 tech", "55 technologies",
    ]

    # Company-specific words to add to the keyword stop-word filter.
    # These appear in nearly every chunk and add noise to $contains search.
    company_stop_words: list[str] = ["fiftyfive", "technologies"]

    # Company-specific words to add to the name-extraction exclusion list.
    # Prevents the system from treating "FiftyFive" as a person's name.
    company_common_words: list[str] = ["Technologies", "FiftyFive"]

    # -------------------------------------------------------------------------
    # Data Paths — Centralized directory locations
    # -------------------------------------------------------------------------
    # Root directory for all data files (JSON, ChromaDB, inbox, converted).
    data_dir: str = "./data"

    # Directory where uploaded files are stored before conversion.
    inbox_dir: str = "./data/inbox"

    # Directory where converted JSON files are stored after ingestion.
    converted_dir: str = "./data/converted"

    # -------------------------------------------------------------------------
    # Data Interpreter — Entity File Map
    # -------------------------------------------------------------------------
    # Maps entity types to their JSON filenames in the converted directory.
    # The DataInterpreter uses this to know which file to scan for each type.
    # Override in .env as JSON: INTERPRETER_FILE_MAP='{"employee":"staff.json"}'
    interpreter_file_map: dict = {
        "employee": "fiftyfive_employee_directory.json",
        "project": "Case Studies @55 Website .json",
        "holiday": "holidays.json",
        "praise": "praise_report.json",
    }

    # -------------------------------------------------------------------------
    # Department Alias Bridge
    # -------------------------------------------------------------------------
    # Maps short user phrases to actual department names in the data.
    # Used by the DataInterpreter to resolve abbreviations like "AI" → full name.
    department_aliases: dict = {
        "ai": "Artificial Intelligence",
        "hr": "Human Resource",
        "ml": "Machine Learning",
    }

    # -------------------------------------------------------------------------
    # PDF Ingestion
    # -------------------------------------------------------------------------
    # Relative path to the folder containing company policy PDFs.
    # The ingestion pipeline recursively scans this folder for *.pdf files.
    # Users can place their PDF files here for scanning.
    pdf_folder: str = "./data/inbox"

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

    # PERFORMANCE NOTE: Reducing this from 32768 → 16384 supports large 
    # context expansion (neighbor fetching) while remaining much faster 
    # than the full 32k window.
    ollama_num_ctx: int = 16384

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
    retriever_top_k: int = 20

    # How many candidate documents BM25 search returns before re-ranking.
    # BM25 casts a wide net; FlashRank then picks the best `retriever_top_k`.
    # Higher values = more candidates for re-ranking but slightly slower retrieval.
    bm25_top_k: int = 30

    # How many neighbor chunks to fetch in each direction during retrieval expansion.
    # Higher = more context but handles larger context window.
    rag_retriever_look_ahead: int = 1

    # Maximum number of chunks formatted into the final LLM context.
    rag_context_max_chunks: int = 10

    # Max docs fetched during exact name search per search variant.
    rag_name_search_top_k: int = 5

    # Max docs fetched during keyword text search per search variant.
    rag_keyword_search_top_k: int = 10

    # Max directory rows fetched for directory-specific listing queries.
    rag_directory_sweep_top_k: int = 30

    # Manual keyword boost tuning applied after FlashRank re-ranking.
    rag_keyword_boost_per_match: float = 0.05
    rag_keyword_boost_cap: float = 0.2

    # Progress logging cadence during slow LLM generations.
    rag_generation_log_interval_seconds: int = 5

    # Reported elapsed time for cache hits.
    rag_cache_hit_elapsed_seconds: float = 0.05

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

    # Capitalized tokens that should never be treated as person names.
    rag_common_words: list[str] = [
        "What", "Who", "How", "When", "Where", "Which", "Why", "Can", "Tell",
        "Does", "List", "Show", "Find", "Give", "The", "And", "For", "About",
        "Many", "Much", "Please", "His", "Her", "Their", "Our", "This", "That",
        "From", "With", "Not", "All", "Any", "Some",
        "Employee", "Department", "Project", "Salary", "Leave", "Policy",
        "Travel", "Manager", "Engineer", "Lead", "Benefits", "Quote", "Exact",
        "Clause", "Per", "Year", "Month", "Day", "Sick", "Casual", "Privilege",
        "Annual", "Total", "Case", "Study", "Studies", "Client", "Team",
        "Have", "Has", "Had", "Do", "Did", "Is", "Are", "Was", "Were", "Be",
        "Been", "Will", "Would", "Could", "Should", "May", "Might", "Shall",
        "Use", "Used", "In", "On", "At", "By", "To", "Of", "Up",
        "Company", "Organisation", "Organization",
        "Institute", "Corporation", "Services", "Solutions",
    ]

    # Listing / enumeration triggers for project and directory style queries.
    rag_listing_triggers: list[str] = [
        "give me all",
        "list all",
        "show all",
        "tell me all",
        "all projects",
        "all cases",
        "every project",
        "which projects",
        "what projects",
        "in which projects",
        "how many projects",
        "list of projects",
        "all the projects",
        "projects where",
        "projects that",
        "projects using",
        "directory",
        "list some",
    ]

    # Similarity threshold for typo-tolerant employee-name expansion.
    rag_fuzzy_name_threshold: float = 0.78

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

    # Structured query interpreter vocabulary.
    data_interpreter_universal_noise: list[str] = [
        "the", "a", "an", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "who", "what", "which", "my", "your", "our", "their",
        "its", "whose", "whom",
        "in", "on", "at", "from", "to", "of", "by", "with", "for", "about",
        "between", "through", "during", "into", "onto", "upon", "under",
        "over", "after", "before", "since", "until", "within", "without",
        "and", "or", "but", "so", "yet", "nor", "if", "then", "than",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "done",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "how", "where", "when", "why",
        "get", "got", "getting", "give", "gave", "given", "giving",
        "take", "took", "taken", "taking",
        "tell", "told", "telling", "say", "said", "saying",
        "know", "knew", "known", "knowing",
        "go", "went", "gone", "going", "come", "came", "coming",
        "make", "made", "making", "see", "seen", "seeing",
        "find", "found", "finding", "show", "shown", "showing",
        "praise", "praises", "praised", "mentioned", "awarded", "appreciated", "recognized",
        "used", "using", "involved", "included", "including",
        "discussed", "discussing", "referred", "referring",
        "talked", "talking", "described", "describing",
        "covered", "covering", "addressed", "addressing",
        "listed", "listing", "received", "receiving",
        "appeared", "appearing", "featured", "featuring",
        "worked", "working", "doing", "completed", "completing",
        "delivered", "delivering", "contributed", "contributing",
        "participated", "participating", "related", "relating",
        "associated", "belonging",
        "there", "here", "also", "just", "only", "very", "much",
        "many", "some", "any", "all", "each", "every", "other",
        "more", "most", "such", "too", "quite", "really", "still",
        "even", "already", "always", "never", "often", "ever",
        "total", "count", "number", "sum", "times", "time",
        "people", "person", "persons", "employee", "employees",
        "member", "members", "team", "teams", "staff",
        "project", "projects", "policy", "policies",
        "document", "documents", "record", "records",
        "entry", "entries", "file", "files", "report", "reports",
        "holiday", "holidays", "leave", "leaves",
        "rule", "rules", "benefits", "benefit", "item", "items",
        "category", "categories", "type", "types", "details", "detail",
        "department", "departments", "dept", "location", "locations", "office", "offices",
        "please", "want", "need", "like", "regarding", "concerning",
    ]

    data_interpreter_aggregation_triggers: list[str] = [
        "how many times", "how many", "how often", "how much",
        "total count of", "total count", "total number of", "total number",
        "count all", "count of", "list all", "number of",
    ]

    data_interpreter_data_starters: list[str] = [
        "who", "what", "which", "how", "list", "total", "show", "count", "name", "give",
    ]

    data_interpreter_employee_keywords: list[str] = [
        "people", "employee", "person", "member", "team", "staff", "intern", "engineer",
    ]

    data_interpreter_project_keywords: list[str] = [
        "project", "case study", "case studies", "used",
    ]

    data_interpreter_policy_keywords: list[str] = [
        "policy", "rule", "benefits", "guideline",
    ]

    data_interpreter_holiday_keywords: list[str] = [
        "holiday", "holidays", "leave",
    ]

    data_interpreter_entity_noise: list[str] = [
        "people", "employee", "employees", "person", "persons",
        "project", "projects", "policy", "policies", "holiday", "holidays",
    ]

    data_interpreter_all_query_words: list[str] = [
        "total", "all", "every", "count", "list",
    ]

    data_interpreter_count_tokens: list[str] = [
        "many", "count", "number", "total", "headcount",
    ]

    data_interpreter_list_tokens: list[str] = [
        "which", "list", "show", "who", "name",
    ]

    data_interpreter_selective_term_max_ratio: float = 0.6
    data_interpreter_schema_sample_size: int = 50
    data_interpreter_entity_value_sample_size: int = 75
    data_interpreter_entity_value_max_length: int = 3
    data_interpreter_enable_nlp_fallback: bool = True
    data_interpreter_nlp_model: str = "en_core_web_sm"

    rag_fallback_answer: str = "Sorry, I couldn't find an answer."

    # -------------------------------------------------------------------------
    # FastAPI Server
    # -------------------------------------------------------------------------
    # Host and port for the FastAPI web server.
    # 0.0.0.0 means "listen on all network interfaces" (required for ngrok/docker).
    host: str = "0.0.0.0"
    port: int = 8001

    # Pydantic configuration — tells it to read from a .env file.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Singleton — imported by all other modules
# ---------------------------------------------------------------------------
# This is the single shared settings instance used throughout the application.
# Import pattern in other files:
#   from src.config import settings
#   print(settings.ollama_llm_model)
settings = Settings()
