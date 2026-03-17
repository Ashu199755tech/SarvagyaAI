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


# ── PDF-Only Ingestion ───────────────────────────────────


@app.post("/api/ingest-pdfs")
async def trigger_pdf_ingestion(file: UploadFile = File(None)):
    """Trigger PDF-only ingestion.

    - With a file: saves the uploaded PDF to the configured PDF folder, then ingests it.
    - Without a file: re-scans the entire PDF folder and re-ingests all PDFs.
    """
    import shutil
    from pathlib import Path

    pipeline = IngestionPipeline()
    try:
        if file is not None:
            # Save uploaded file to the PDF folder
            pdf_folder = Path(pipeline.store._collection.metadata.get("pdf_folder", "")) or Path("data/pdfs")
            save_path = Path(settings.pdf_folder) / file.filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            logger.info("Saved uploaded PDF to %s", save_path)

        result = pipeline.ingest_pdfs()
        return {"status": "success", **result}
    except Exception as exc:
        logger.exception("PDF ingestion failed")
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


# ── Health Check ─────────────────────────────────────────


@app.get("/api/health")
async def health():
    """Simple health probe."""
    return {"status": "ok", "service": "ResoAI"}
