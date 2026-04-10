"""
cache.py - Persistent SQLite-based cache for RAG responses.
=========================================================

This module provides a simple key-value store to cache LLM answers for 
repeated questions. This is crucial for local CPU performance.

Features:
- Thread-safe (using sqlite3)
- Persistent across server restarts (stable MD5 hashing)
- Automatic table creation
"""

import json
import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

# Path to the cache database
CACHE_DB_PATH = Path(settings.chroma_persist_dir).parent / "rag_cache.db"

def _get_connection():
    """Return a connection to the SQLite cache database."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    # Return as a dictionary for easier access
    conn.row_factory = sqlite3.Row
    return conn

def _get_stable_hash(text: str) -> str:
    """Generate a stable MD5 hash for the given text."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def init_cache():
    """Initialize the cache table if it doesn't exist."""
    try:
        with _get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_cache (
                    question_hash TEXT PRIMARY KEY,
                    question_text TEXT,
                    answer TEXT,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Index for faster lookup (though id is primary key)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON rag_cache(question_hash)")
        logger.info(f"RAG cache initialized at {CACHE_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG cache: {e}")

def get_cached_answer(question: str) -> dict[str, Any] | None:
    """
    Retrieve a cached answer for the given question.
    Returns None if not found or on error.
    """
    q_hash = _get_stable_hash(question)
    
    try:
        init_cache() # Ensure table exists
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT answer, sources FROM rag_cache WHERE question_hash = ?",
                (q_hash,)
            ).fetchone()
            
            if row:
                logger.info(f"Cache HIT for question: {question[:50]}...")
                return {
                    "answer": row["answer"],
                    "sources": json.loads(row["sources"]),
                    "is_cached": True
                }
    except Exception as e:
        logger.error(f"Cache lookup failed: {e}")
    
    return None

def save_to_cache(question: str, answer: str, sources: list[dict]):
    """Save a successful RAG response to the cache."""
    q_hash = _get_stable_hash(question)
    
    try:
        init_cache()
        with _get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rag_cache (question_hash, question_text, answer, sources)
                VALUES (?, ?, ?, ?)
            """, (q_hash, question, answer, json.dumps(sources)))
        logger.info(f"Saved response to cache for question: {question[:50]}...")
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}")

def clear_cache():
    """Wipe the entire cache. Called during data ingestion."""
    try:
        if CACHE_DB_PATH.exists():
            with _get_connection() as conn:
                conn.execute("DELETE FROM rag_cache")
            logger.info("RAG cache cleared successfully.")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
