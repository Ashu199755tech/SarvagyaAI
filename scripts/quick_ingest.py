import asyncio
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline

async def run_full_reingestion():
    print("Initializing pipeline...")
    pipeline = IngestionPipeline()
    print("Starting full re-ingestion (converting inbox + indexing all files)...")
    await pipeline.run(perform_truncate=True)
    print("Full re-ingestion complete!")

if __name__ == "__main__":
    asyncio.run(run_full_reingestion())
