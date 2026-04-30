"""
query_decomposer.py — LLM-Based Query Understanding for ResoAI
================================================================

This module replaces ALL hardcoded word lists (noise words, intent signals,
listing triggers, count tokens, etc.) with a single LLM call that
semantically parses any user question into structured intent.

Architecture:
  User Query → Llama 3.2:1b (tiny, fast) → Structured JSON
  {
    "intent":       "lookup" | "list" | "count" | "policy" | "general",
    "entity":       "employee" | "project" | "policy" | "holiday" | "directory" | null,
    "search_terms": ["yash pancholi"],     ← actual values to search for
    "attribute":    "department",           ← what info is being requested
    "is_listing":   false                  ← does user want ALL matches?
  }

This single JSON output replaces:
  - data_interpreter_universal_noise   (~200 words)
  - data_interpreter_count_tokens      (5 words)
  - data_interpreter_list_tokens       (5 words)
  - rag_doc_signals                    (~40 words)
  - rag_emp_signals                    (~8 words)
  - rag_listing_triggers               (~17 phrases)
  - rag_common_words                   (~80 words)

Performance:
  - Uses llama3.2:1b (1.2GB) — the smallest available model
  - num_ctx=256 — minimal context window for fast inference
  - Typical latency: 4-8 seconds on CPU
  - Results are cached to avoid repeated decomposition
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decomposition result
# ---------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """Structured output from the LLM query decomposer."""
    intent: str           # lookup, list, count, policy, general
    entity: Optional[str] # employee, project, policy, holiday, directory
    search_terms: list[str] = field(default_factory=list)
    attribute: Optional[str] = None
    is_listing: bool = False
    raw_question: str = ""
    decompose_time: float = 0.0

    @property
    def is_structured_query(self) -> bool:
        """Can this be answered by the DataInterpreter (structured JSON scan)?"""
        return (
            self.intent in ("lookup", "list", "count")
            and self.entity in ("employee", "project", "holiday", "directory")
        )

    @property
    def criteria(self) -> str:
        """Search criteria string for DataInterpreter."""
        return " ".join(self.search_terms).strip() if self.search_terms else "__all__"

    @property
    def rag_intent(self) -> str:
        """Map decomposer intent to chain.py's intent format."""
        if self.entity == "employee" or self.intent == "lookup":
            return "employee"
        elif self.entity in ("project", "policy", "holiday"):
            return "document"
        else:
            return "document"


# ---------------------------------------------------------------------------
# The decomposition prompt
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are a query parser for an HR knowledge system. Parse the question into JSON. Do not include any explanations.

Rules:
- intent: one of "lookup" (specific fact), "list" (enumerate items), "count" (how many), "policy" (company rule/guideline), "general" (greeting/other)
- entity: one of "employee", "project", "policy", "holiday", "directory", or null
- search_terms: ONLY specific names, values, or identifiers to search for. Do NOT include generic words like "department", "salary", "employee", "project", "belongs", "works", "policy", "notice".
- attribute: what specific information is being requested (e.g., "department", "salary", "notice_period")
- is_listing: true ONLY if the user wants ALL/every matching items listed, false otherwise

Question: "{question}"
JSON:
"""


# ---------------------------------------------------------------------------
# In-memory cache for decomposed queries
# ---------------------------------------------------------------------------

_decompose_cache: dict[str, QueryPlan] = {}
_MAX_CACHE_SIZE = 200


# ---------------------------------------------------------------------------
# Main decomposer class
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """
    Uses a small LLM (llama3.2:1b) to parse natural language queries
    into structured intent, replacing all hardcoded word lists.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        num_ctx: int = 512,
        temperature: float = 0.0,
    ):
        self.model = model or settings.decomposer_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        logger.info(
            "[Decomposer] Initialized with model=%s, num_ctx=%d",
            self.model, self.num_ctx,
        )

    def decompose(self, question: str) -> QueryPlan:
        """
        Parse a natural language question into a structured QueryPlan.

        Uses the LLM to extract intent, entity, search terms, and attribute.
        Falls back to a safe default if the LLM fails or returns invalid JSON.

        Args:
            question: Raw user question string.

        Returns:
            QueryPlan with structured intent information.
        """
        # Normalize for cache lookup
        cache_key = question.strip().lower()

        # Check cache first
        if cache_key in _decompose_cache:
            cached = _decompose_cache[cache_key]
            logger.info(
                "[Decomposer] Cache hit: intent=%s, entity=%s, terms=%s",
                cached.intent, cached.entity, cached.search_terms,
            )
            return cached

        start = time.monotonic()

        try:
            plan = self._call_llm(question)
        except Exception as e:
            logger.error("[Decomposer] LLM call failed: %s — using fallback", e)
            plan = self._fallback_parse(question)

        plan.raw_question = question
        plan.decompose_time = time.monotonic() - start

        # Cache the result
        if len(_decompose_cache) >= _MAX_CACHE_SIZE:
            # Evict oldest entries (simple FIFO)
            oldest_key = next(iter(_decompose_cache))
            del _decompose_cache[oldest_key]
        _decompose_cache[cache_key] = plan

        logger.info(
            "[Decomposer] Parsed in %.1fs: intent=%s, entity=%s, "
            "search_terms=%s, attribute=%s, is_listing=%s",
            plan.decompose_time, plan.intent, plan.entity,
            plan.search_terms, plan.attribute, plan.is_listing,
        )

        return plan

    def _call_llm(self, question: str) -> QueryPlan:
        """Make a synchronous call to the Ollama generate API."""
        prompt = DECOMPOSE_PROMPT.format(question=question)

        logger.info("[Decomposer] Calling LLM with model=%s, num_ctx=%d", self.model, self.num_ctx)

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                    "num_predict": 100,  # Limit output tokens
                },
            },
            timeout=120.0,
        )
        response.raise_for_status()

        resp_json = response.json()
        raw_text = resp_json.get("response", "").strip()

        if not raw_text:
            logger.warning("[Decomposer] Empty response from LLM")
            return self._default_plan()

        logger.info("[Decomposer] Raw LLM output (%d chars): %r", len(raw_text), raw_text[:300])
        return self._parse_json_response(raw_text)

    def _parse_json_response(self, raw_text: str) -> QueryPlan:
        """
        Extract and validate the JSON from the LLM's raw text response.

        The LLM may wrap the JSON in markdown fences or add explanation text.
        This method is resilient to those variations.
        """
        # Try to extract JSON from markdown code fences
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find the first {...} block (greedy to capture full JSON)
            json_match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("[Decomposer] No JSON found in response: %r", raw_text[:200])
                return self._default_plan()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("[Decomposer] Invalid JSON: %s — raw: %r", e, json_str[:200])
            return self._default_plan()

        # Validate and normalize fields
        intent = data.get("intent", "general")
        if intent not in ("lookup", "list", "count", "policy", "general"):
            intent = "general"

        entity = data.get("entity")
        if entity not in ("employee", "project", "policy", "holiday", "directory", None):
            entity = None

        search_terms = data.get("search_terms", [])
        if isinstance(search_terms, str):
            search_terms = [search_terms]
        # Clean search terms — remove empty strings and very short terms
        search_terms = [t.strip().lower() for t in search_terms if t.strip() and len(t.strip()) > 1]

        attribute = data.get("attribute")
        is_listing = bool(data.get("is_listing", False))

        return QueryPlan(
            intent=intent,
            entity=entity,
            search_terms=search_terms,
            attribute=attribute,
            is_listing=is_listing,
        )

    def _fallback_parse(self, question: str) -> QueryPlan:
        """
        Ultra-simple regex fallback when the LLM is unavailable.

        This is intentionally basic — it handles the most common patterns
        and defaults to 'general' intent for everything else, which routes
        the query to HybridRAG where the full LLM handles it.
        """
        q = question.lower().strip()

        # Detect intent
        intent = "general"
        if any(w in q for w in ("how many", "count", "total", "number of")):
            intent = "count"
        elif any(w in q for w in ("list", "show all", "give me all", "which")):
            intent = "list"
        elif any(w in q for w in ("what is", "what's", "tell me about", "who")):
            intent = "lookup"
        elif any(w in q for w in ("policy", "procedure", "guideline", "rule")):
            intent = "policy"

        # Detect entity
        entity = None
        if any(w in q for w in ("employee", "salary", "department", "designation")):
            entity = "employee"
        elif "project" in q:
            entity = "project"
        elif any(w in q for w in ("policy", "notice period", "leave")):
            entity = "policy"
        elif "holiday" in q:
            entity = "holiday"

        is_listing = any(w in q for w in ("list", "all", "every", "which"))

        return QueryPlan(
            intent=intent,
            entity=entity,
            search_terms=[],
            attribute=None,
            is_listing=is_listing,
        )

    def _default_plan(self) -> QueryPlan:
        """Return a safe default that routes to HybridRAG."""
        return QueryPlan(intent="general", entity=None)


def clear_decompose_cache() -> None:
    """Clear the in-memory decomposition cache."""
    _decompose_cache.clear()
    logger.info("[Decomposer] Cache cleared.")
