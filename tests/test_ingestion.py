"""Tests for the ingestion document builder and pipeline logic."""

from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.document_builder import build_employee_document
from src.keka.models import Employee


# ── Employee Document Tests ──────────────────────────────


def test_build_employee_document_full():
    """Employee document contains all fields and correct metadata."""
    emp = Employee(
        employee_id="EMP0001",
        first_name="Alice",
        last_name="Smith",
        email="alice@example.com",
        department="Engineering",
        designation="Senior Developer",
        joining_date=date(2023, 3, 15),
        salary=1200000.00,
        project_name="Phoenix",
        project_start_date=date(2024, 1, 1),
        project_end_date=date(2024, 12, 31),
        client_name="Accenture",
    )

    doc = build_employee_document(emp)

    assert "Alice Smith" in doc.page_content
    assert "EMP0001" in doc.page_content
    assert "Engineering" in doc.page_content
    assert "Senior Developer" in doc.page_content
    assert "2023-03-15" in doc.page_content
    assert "1200000" in doc.page_content
    assert "Phoenix" in doc.page_content
    assert "Accenture" in doc.page_content

    assert doc.metadata["record_type"] == "employee"
    assert doc.metadata["record_id"] == "EMP0001"
    assert doc.metadata["department"] == "Engineering"
    assert doc.metadata["project_name"] == "Phoenix"
    assert doc.metadata["client_name"] == "Accenture"


def test_build_employee_document_minimal():
    """Employee document handles missing optional fields."""
    emp = Employee(employee_id="EMP0456", first_name="Bob")

    doc = build_employee_document(emp)

    assert "Bob" in doc.page_content
    assert "N/A" in doc.page_content  # missing fields show N/A
    assert doc.metadata["record_id"] == "EMP0456"


# ── Chunking Tests ───────────────────────────────────────


def test_chunking_produces_expected_chunks():
    """RecursiveCharacterTextSplitter produces multiple chunks for large docs."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20
    )
    long_text = "This is a sentence about an employee. " * 20

    from langchain_core.documents import Document

    docs = [Document(page_content=long_text, metadata={"id": "test"})]
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 1
    # Each chunk should be ≤ chunk_size (roughly)
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # allow small overshoot
