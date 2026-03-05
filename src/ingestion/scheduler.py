"""Scheduled periodic ingestion of Keka data into ChromaDB."""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import settings
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def _run_ingestion_sync() -> None:
    """Synchronous wrapper so APScheduler can trigger the async pipeline."""
    loop = asyncio.new_event_loop()
    try:
        pipeline = IngestionPipeline()
        result = loop.run_until_complete(pipeline.run())
        logger.info("Scheduled ingestion finished: %s", result)
    except Exception:
        logger.exception("Scheduled ingestion failed")
    finally:
        loop.run_until_complete(pipeline.close())
        loop.close()


def start_scheduler() -> BackgroundScheduler:
    """Start a background scheduler that runs ingestion periodically.

    Returns the scheduler instance so the caller can shut it down later.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _run_ingestion_sync,
        "interval",
        hours=settings.ingestion_interval_hours,
        id="keka_ingestion",
        name="Keka HR data ingestion",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Ingestion scheduler started — runs every %d hour(s)",
        settings.ingestion_interval_hours,
    )
    return scheduler
