import json
import logging
import sqlite3
import httpx
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# Same location as rag_cache.db
MEMORY_DB_PATH = Path(settings.chroma_persist_dir).parent / "rag_memory.db"

REWRITE_PROMPT = """You are an AI assistant that rewrites follow-up questions to be standalone, using context from the conversation history.
If the current question is already standalone (does not contain pronouns like 'he', 'she', 'it', 'they', 'this', 'that' referring to previous turns), return it exactly as is.
DO NOT answer the question. ONLY return the rewritten question.

Conversation History:
{history}

Current Question: {question}
Standalone Question:"""

def _get_connection():
    """Return a connection to the SQLite memory database."""
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_memory():
    """Initialize the memory table if it doesn't exist."""
    try:
        with _get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id)")
    except Exception as e:
        logger.error(f"Failed to initialize RAG memory: {e}")

def add_message(session_id: str, role: str, content: str):
    """Add a message to the session's chat history."""
    if not session_id or not content:
        return
    try:
        init_memory()
        with _get_connection() as conn:
            conn.execute("""
                INSERT INTO chat_history (session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, role, content))
    except Exception as e:
        logger.error(f"Failed to save message to memory: {e}")

def get_history(session_id: str, limit: int = 6) -> list[dict]:
    """Retrieve the last N messages for a session (default 6 = 3 turns)."""
    if not session_id:
        return []
    try:
        init_memory()
        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT role, content FROM chat_history 
                WHERE session_id = ? 
                ORDER BY id DESC LIMIT ?
            """, (session_id, limit)).fetchall()
            
            # Rows are retrieved in descending order, so reverse to chronological
            history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
            return history
    except Exception as e:
        logger.error(f"Failed to retrieve memory: {e}")
        return []

async def rewrite_query(session_id: str, current_question: str) -> str:
    """Rewrite the current question based on session history."""
    if not session_id:
        return current_question

    history = get_history(session_id, limit=6)
    if not history:
        return current_question

    # Format history into a string
    history_str = ""
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "AI"
        history_str += f"{role_label}: {msg['content']}\n"

    prompt = REWRITE_PROMPT.format(history=history_str.strip(), question=current_question)
    
    model = settings.decomposer_model
    base_url = settings.ollama_base_url.rstrip("/")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 1024,
                        "temperature": 0.0,
                        "num_predict": 100,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            resp_json = response.json()
            rewritten_text = resp_json.get("response", "").strip()

            # Clean up potential markdown formatting or quotes from the LLM
            rewritten_text = rewritten_text.strip('"\'')
            if rewritten_text.lower().startswith("standalone question:"):
                rewritten_text = rewritten_text[len("standalone question:"):].strip()

            if rewritten_text and len(rewritten_text) > 3:
                logger.info(f"[Memory] Rewrote query: '{current_question}' -> '{rewritten_text}'")
                return rewritten_text
    except Exception as e:
        logger.exception(f"[Memory] Failed to rewrite query: {e}")
    
    # Fallback to original question on failure
    return current_question
