"""FastAPI application — entry point for ResoAI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from botbuilder.core import TurnContext
from botbuilder.integration.aiohttp import CloudAdapter  # noqa: F401
from botbuilder.schema import Activity
from fastapi import FastAPI, File, Request, Response, UploadFile

from src.bot.adapter import create_adapter
from src.bot.teams_bot import ResoAIBot
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.scheduler import start_scheduler
from src.rag.chain import ask as rag_ask
from src.rag.chain import _retrieve_mixed, _get_or_create_store, _interpreter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals initialised at startup ───────────────────────
adapter = create_adapter()
bot = ResoAIBot()
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI app."""
    global scheduler

    logger.info("ResoAI starting up …")

    # Start periodic ingestion scheduler
    scheduler = start_scheduler()

    yield

    # Shutdown
    logger.info("ResoAI shutting down …")
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="ResoAI",
    description="MS Teams chatbot with Keka HR RAG pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Bot Webhook Endpoint ─────────────────────────────────


@app.post("/api/messages")
async def messages(request: Request) -> Response:
    """Receive activities from the Bot Framework channel."""
    body = await request.json()
    activity = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    async def turn_callback(turn_context: TurnContext):
        await bot.on_turn(turn_context)

    await adapter.process_activity(activity, auth_header, turn_callback)
    return Response(status_code=200)


# ── Manual Ingestion Trigger ─────────────────────────────


@app.post("/api/ingest")
async def trigger_ingestion():
    """Trigger full ingestion (employees + PDFs)."""
    pipeline = IngestionPipeline()
    try:
        result = await pipeline.run()
        return {"status": "success", **result}
    except Exception as exc:
        logger.exception("Manual ingestion failed")
        return {"status": "error", "detail": str(exc)}
    finally:
        await pipeline.close()


@app.post("/api/consolidate-and-ingest")
async def trigger_full_refresh():
    """Consolidate raw data into JSONs, then ingest JSONs into ChromaDB."""
    from src.consolidator.run import run_consolidation
    
    try:
        # Step 1: Consolidate
        logger.info("Starting consolidation...")
        c_result = await run_consolidation()
        
        # Step 2: Ingest
        logger.info("Starting JSON-first ingestion...")
        pipeline = IngestionPipeline()
        i_result = await pipeline.run(perform_truncate=True)
        
        return {
            "status": "success",
            "consolidation": c_result,
            "ingestion": i_result
        }
    except Exception as exc:
        logger.exception("Full refresh failed")
        return {"status": "error", "detail": str(exc)}


@app.post("/api/ingest-pdfs")
async def trigger_pdf_ingestion():
    """Re-load JSON files into vector store (no truncate)."""
    pipeline = IngestionPipeline()
    try:
        # In the new JSON-first world, we just run the pipeline
        # without truncating to refresh from existing JSONs.
        result = await pipeline.run(perform_truncate=False)
        return {"status": "success", **result}
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        return {"status": "error", "detail": str(exc)}


@app.post("/api/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """
    Upload any file (PDF, Excel, Word, CSV, TXT) and ingest it into the RAG pipeline.

    The file is saved to data/inbox/, converted to JSON, then incrementally
    ingested into ChromaDB — no full re-index needed.

    Usage:
        curl -X POST http://localhost:8000/api/upload -F 'file=@report.xlsx'
    """
    import shutil
    from src.ingestion.universal_converter import convert_to_json, CONVERTERS
    from pathlib import Path as _Path

    INBOX_DIR     = _Path(settings.inbox_dir)
    CONVERTED_DIR = _Path(settings.converted_dir)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    suffix = _Path(file.filename).suffix.lower()
    if suffix not in CONVERTERS:
        return {
            "status": "error",
            "detail": f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(CONVERTERS))}",
        }

    inbox_path = INBOX_DIR / file.filename
    with open(inbox_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("Uploaded file saved: %s", inbox_path)

    try:
        if suffix == ".json":
            import shutil as _sh
            json_path = CONVERTED_DIR / file.filename
            _sh.copy(inbox_path, json_path)
        else:
            json_path = convert_to_json(inbox_path, output_dir=CONVERTED_DIR)

        pipeline = IngestionPipeline()
        result = await pipeline.ingest_file(json_path)

        return {
            "status": "success",
            "filename": file.filename,
            "json_output": str(json_path),
            **result,
        }
    except Exception as exc:
        logger.exception("Upload ingestion failed for %s", file.filename)
        return {"status": "error", "detail": str(exc)}


# ── Ask Endpoint ─────────────────────────────────────────

from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(request_data: AskRequest):
    """Ask a question to the RAG chatbot."""
    question = request_data.question
    if not question:
        return {"status": "error", "detail": "No question provided"}

    try:
        result = await rag_ask(question)
        return {
            "status": "success", 
            "answer": result["answer"],
            "time_elapsed_seconds": result.get("time_elapsed_seconds")
        }
    except Exception as exc:
        logger.exception("Ask failed")
        return {"status": "error", "detail": str(exc)}


@app.post("/api/debug")
async def debug_retrieval(request_data: AskRequest):
    """
    Debug endpoint — returns retrieved chunks WITHOUT generating an answer.

    Use this to check if retrieval is working correctly for any question.
    Much faster than /api/ask because it skips the LLM generation step.

    Usage:
        curl -X POST http://localhost:8001/api/debug
             -H "Content-Type: application/json"
             -d '{"question": "which projects used GPU?"}'

    Returns each chunk with:
        - source: which PDF or employee record it came from
        - preview: first 300 chars of the chunk text
        - record_type: "pdf" or "employee"
        - record_id: the chunk identifier
    """
    question = request_data.question
    if not question:
        return {"status": "error", "detail": "No question provided"}
    try:
        # Use the LLM Query Decomposer (mirrors ask() logic)
        from src.rag.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        query_plan = decomposer.decompose(question)

        if query_plan.is_structured_query:
            # Route through DataInterpreter using decomposer output
            itp_result = _interpreter.query(query_plan.entity, query_plan.criteria)
            matches = itp_result["matches"][:20] if itp_result else []
            count = itp_result["count"] if itp_result else 0
            return {
                "status": "success",
                "question": question,
                "pathway": "DataInterpreter (via Decomposer)",
                "decomposer": {
                    "intent": query_plan.intent,
                    "entity": query_plan.entity,
                    "search_terms": query_plan.search_terms,
                    "attribute": query_plan.attribute,
                    "is_listing": query_plan.is_listing,
                    "decompose_time": round(query_plan.decompose_time, 2),
                },
                "entity": query_plan.entity,
                "operation": query_plan.intent,
                "criteria": query_plan.criteria,
                "count": count,
                "matches": matches,
            }

        # Fall through to RAG retrieval
        from src.config import settings
        store = await _get_or_create_store()
        chunks = await _retrieve_mixed(store, question, k=settings.retriever_top_k)
        return {
            "status": "success",
            "question": question,
            "pathway": "HybridRAG (via Decomposer)",
            "decomposer": {
                "intent": query_plan.intent,
                "entity": query_plan.entity,
                "search_terms": query_plan.search_terms,
                "attribute": query_plan.attribute,
                "is_listing": query_plan.is_listing,
                "decompose_time": round(query_plan.decompose_time, 2),
            },
            "chunks_retrieved": len(chunks),
            "chunks": [
                {
                    "rank": i + 1,
                    "source": doc.metadata.get("filename", doc.metadata.get("employee_name", "unknown")),
                    "record_type": doc.metadata.get("record_type", "unknown"),
                    "record_id": doc.metadata.get("record_id", "unknown"),
                    "preview": doc.page_content[:300].replace("\n", " "),
                    "contains_keyword": None,
                }
                for i, doc in enumerate(chunks)
            ],
        }
    except Exception as exc:
        logger.exception("Debug retrieval failed")
        return {"status": "error", "detail": str(exc)}


# ── Health Check ─────────────────────────────────────────


@app.get("/api/health")
async def health():
    """Simple health probe."""
    return {"status": "ok", "service": "ResoAI"}