"""Tests for the RAG query chain (LLM calls are mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.documents import Document


@pytest.mark.asyncio
async def test_ask_returns_answer_and_sources():
    """ask() returns an answer string and source metadata."""
    import src.rag.chain as chain_module

    mock_docs = [
        Document(
            page_content="Employee: Alice Smith ...",
            metadata={
                "record_type": "employee",
                "record_id": "emp-123",
                "employee_name": "Alice Smith",
                "project_name": None,
            },
        )
    ]

    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Alice Smith is a Backend Engineer in Engineering."

    with (
        patch.object(
            chain_module,
            "_retrieve_mixed",
            return_value=mock_docs,
        ),
        patch.object(chain_module, "RAG_PROMPT") as mock_prompt,
        patch.object(chain_module, "_get_llm"),
    ):
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)
        result = await chain_module.ask("Who is Alice Smith?")

    assert "Alice Smith" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["record_type"] == "employee"
    assert result["sources"][0]["employee_name"] == "Alice Smith"


@pytest.mark.asyncio
async def test_ask_handles_empty_context():
    """ask() returns a fallback when no context is found."""
    import src.rag.chain as chain_module

    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = ""

    with (
        patch.object(
            chain_module,
            "_retrieve_mixed",
            return_value=[],
        ),
        patch.object(chain_module, "RAG_PROMPT") as mock_prompt,
        patch.object(chain_module, "_get_llm"),
    ):
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)
        result = await chain_module.ask("What is the meaning of life?")

    assert result["answer"] == "Sorry, I couldn't find an answer."
    assert result["sources"] == []
