"""
NEW consolidator/run.py
=======================
Adds refresh_all_sources() call so /api/consolidate also rebuilds
the source-of-truth JSON files without using LLM.

The run order is now:
  1. refresh_all_sources()    → rebuild 4 JSON files from raw PDFs (no LLM)
  2. ingest_employees()       → Keka API → employees.json
  3. build_knowledge_base()   → link entities
  4. generate_markdown()      → master_data.md for RAG
  5. generate_knowledge_graph() → knowledge_graph.json for auditing
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.consolidator.ingestors import (
    ingest_employees_from_api,
    ingest_pdfs,
    refresh_all_sources,          # NEW
)
from src.consolidator.relationships import build_knowledge_base
from src.consolidator.generator import generate_markdown, generate_knowledge_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from src.config import settings
OUTPUT_DIR = settings.data_dir


async def run_consolidation() -> dict:
    """Execute the full consolidation pipeline — zero LLM in data preparation."""
    start = time.monotonic()
    logger.info("=" * 60)
    logger.info("STARTING DATA CONSOLIDATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Rebuild source-of-truth JSON files from raw PDFs (no LLM)
    logger.info("Step 1/5: Rebuilding source-of-truth JSON files...")
    sot_counts = refresh_all_sources()
    logger.info("  SOT: %s", sot_counts)

    # Step 2: Ingest employees from Keka API (writes employees.json as side-effect)
    logger.info("Step 2/5: Ingesting employees from Keka API...")
    employees = await ingest_employees_from_api()

    # Step 3: Load structured data from source-of-truth JSONs (no LLM)
    logger.info("Step 3/5: Loading policies + projects + holidays from source-of-truth JSONs...")
    policies, projects, tools, holidays = ingest_pdfs()

    logger.info(
        "  Loaded: %d employees, %d policies, %d projects, %d tools, %d holidays",
        len(employees), len(policies), len(projects), len(tools), len(holidays),
    )

    # Step 4: Build entity relationships
    logger.info("Step 4/5: Building entity relationships...")
    kb = build_knowledge_base(employees, policies, projects, tools, holidays)
    logger.info("  Knowledge base: %s", kb.stats)

    # Step 5: Generate outputs
    logger.info("Step 5/5: Generating Markdown + knowledge graph...")
    md_path   = generate_markdown(kb,       f"{OUTPUT_DIR}/master_data.md")
    json_path = generate_knowledge_graph(kb, f"{OUTPUT_DIR}/knowledge_graph.json")

    elapsed = time.monotonic() - start
    logger.info("=" * 60)
    logger.info("CONSOLIDATION COMPLETE in %.2fs", elapsed)
    logger.info("  master_data.md:        %s", md_path)
    logger.info("  knowledge_graph.json:  %s", json_path)
    logger.info("  employees.json:        data/employees.json")
    logger.info("  projects.json:         data/projects.json")
    logger.info("  policies.json:         data/policies.json")
    logger.info("  holidays.json:         data/holidays.json")
    logger.info("  misc_docs.json:        data/misc_docs.json")
    logger.info("=" * 60)

    return {
        "status":           "success",
        "stats":            kb.stats,
        "sot_refreshed":    sot_counts,
        "markdown_path":    md_path,
        "json_path":        json_path,
        "elapsed_seconds":  round(elapsed, 2),
    }


if __name__ == "__main__":
    asyncio.run(run_consolidation())