"""Tests for the ingestion document builder and pipeline logic."""

from __future__ import annotations

from datetime import date
import json
from unittest.mock import patch

import pytest

from src.ingestion.document_builder import build_employee_document
from src.keka.models import Employee
from src.rag.data_interpreter import _calendar_month_terms
from src.rag.data_interpreter import _criteria_matches_record
from src.rag.data_interpreter import DataInterpreter


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


def test_data_interpreter_matches_criteria_by_terms_not_only_exact_substring():
    """Criteria matching succeeds when all terms exist but formatting differs."""
    record = {
        "name": "Alice Smith",
        "department": "Artificial Intelligence",
        "designation": "Senior Developer",
    }

    assert _criteria_matches_record(record, "artificial intelligence")
    assert _criteria_matches_record(record, "senior artificial")
    assert not _criteria_matches_record(record, "finance manager")


def test_data_interpreter_detect_intent_uses_configured_data_starters():
    """Intent detection should respect configurable opening-word triggers."""
    interpreter = DataInterpreter()

    with patch(
        "src.rag.data_interpreter.settings.data_interpreter_data_starters",
        ["show"],
    ):
        entity, criteria = interpreter.detect_intent("show AI employees")

    assert entity == "employee"
    assert criteria == "Artificial Intelligence"


def test_data_interpreter_interprets_company_employee_total(tmp_path):
    """Structured queries should count all employees without hardcoded phrases."""
    employee_path = tmp_path / "employees.json"
    employee_path.write_text(json.dumps([
        {"Name": "Alice Smith", "Department": "Engineering"},
        {"Name": "Bob Jones", "Department": "Artificial Intelligence"},
    ]), encoding="utf-8")

    with patch("src.rag.data_interpreter.settings.interpreter_file_map", {"employee": "employees.json"}):
        interpreter = DataInterpreter(data_dir=str(tmp_path))
        result = interpreter.interpret("total number of employees in the company")

    assert result is not None
    assert result["operation"] == "count"
    assert result["entity"] == "employee"
    assert result["criteria"] == "__all__"
    assert result["count"] == 2


def test_data_interpreter_interprets_gurgaon_office_people_query(tmp_path):
    """Semantic entity inference should map generic workforce wording to employees."""
    employee_path = tmp_path / "employees.json"
    employee_path.write_text(json.dumps([
        {"Name": "Alice Smith", "Department": "Engineering", "Location": "Gurgaon"},
        {"Name": "Bob Jones", "Department": "Artificial Intelligence", "Location": "Gurgaon"},
        {"Name": "Carol White", "Department": "Engineering", "Location": "Jaipur"},
    ]), encoding="utf-8")

    with patch("src.rag.data_interpreter.settings.interpreter_file_map", {"employee": "employees.json"}):
        interpreter = DataInterpreter(data_dir=str(tmp_path))
        result = interpreter.interpret("Total number of people in gurgaon office?")

    assert result is not None
    assert result["operation"] == "count"
    assert result["entity"] == "employee"
    assert "gurgaon" in result["criteria_terms"]
    assert result["count"] == 2


def test_data_interpreter_interprets_project_listing_with_selective_term(tmp_path):
    """Structured queries should keep only selective filter terms such as OCR."""
    project_path = tmp_path / "projects.json"
    project_path.write_text(json.dumps([
        {"project_name": "Alpha", "technology_used": "OCR, Python", "industry": "Finance"},
        {"project_name": "Beta", "technology_used": "React, FastAPI", "industry": "Retail"},
        {"project_name": "Gamma", "technology_used": "OCR, OpenCV", "industry": "Healthcare"},
    ]), encoding="utf-8")

    with patch("src.rag.data_interpreter.settings.interpreter_file_map", {"project": "projects.json"}):
        interpreter = DataInterpreter(data_dir=str(tmp_path))
        result = interpreter.interpret("which projects used ocr?")

    assert result is not None
    assert result["operation"] == "list"
    assert result["entity"] == "project"
    assert "ocr" in result["criteria_terms"]
    assert result["count"] == 2


def test_calendar_month_terms_normalize_abbreviations():
    """Month extraction should work without hardcoded month lists."""
    assert _calendar_month_terms("holidays in May") == {"may"}
    assert _calendar_month_terms("events in Sep") == {"september"}


def test_data_interpreter_uses_month_signal_for_holiday_queries(tmp_path):
    """Hybrid parsing should keep month filters even if generic stop words drop them."""
    holiday_path = tmp_path / "holidays.json"
    holiday_path.write_text(json.dumps([
        {"Holiday": "Labour Day", "text": "1 May 2026", "source_file": "holidays.json"},
        {"Holiday": "Independence Day", "text": "15 August 2026", "source_file": "holidays.json"},
    ]), encoding="utf-8")

    with patch("src.rag.data_interpreter.settings.interpreter_file_map", {"holiday": "holidays.json"}):
        interpreter = DataInterpreter(data_dir=str(tmp_path))
        result = interpreter.interpret("How many holidays are in May?")

    assert result is not None
    assert result["entity"] == "holiday"
    assert result["operation"] == "count"
    assert "may" in result["criteria_terms"]
    assert result["count"] == 1
