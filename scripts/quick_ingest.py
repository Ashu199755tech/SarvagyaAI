import asyncio
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline

async def reingest_employees():
    print("Initializing pipeline...")
    pipeline = IngestionPipeline()
    print("Ingesting employees.json incrementally...")
    result = await pipeline.ingest_file(Path("data/employees.json"))
    print(f"Incremental ingestion complete! Result: {result}")

if __name__ == "__main__":
    asyncio.run(reingest_employees())
