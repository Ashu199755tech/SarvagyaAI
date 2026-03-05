"""Convert API records into LangChain Documents for RAG ingestion."""

from __future__ import annotations

from langchain_core.documents import Document

from src.keka.models import Employee


def build_employee_document(emp: Employee) -> Document:
    """Turn an Employee model into a LangChain Document.

    The page_content is a human-readable summary suitable for embedding,
    while metadata carries structured fields for filtered retrieval.
    """
    name = f"{emp.first_name} {emp.last_name}".strip() or "Unknown"
    doj = emp.joining_date.isoformat() if emp.joining_date else "N/A"
    proj_start = emp.project_start_date.isoformat() if emp.project_start_date else "N/A"
    proj_end = emp.project_end_date.isoformat() if emp.project_end_date else "N/A"

    content = (
        f"Employee: {name}\n"
        f"Employee ID: {emp.employee_id}\n"
        f"Email: {emp.email or 'N/A'}\n"
        f"Department: {emp.department or 'N/A'}\n"
        f"Designation: {emp.designation or 'N/A'}\n"
        f"Date of Joining: {doj}\n"
        f"Salary: {emp.salary:.2f}\n"
        f"Project: {emp.project_name or 'N/A'}\n"
        f"Project Start Date: {proj_start}\n"
        f"Project End Date: {proj_end}\n"
        f"Client: {emp.client_name or 'N/A'}\n"
    )

    metadata = {
        "source": "employee_api",
        "record_type": "employee",
        "record_id": emp.employee_id,
        "employee_name": name,
        "department": emp.department,
        "designation": emp.designation,
        "project_name": emp.project_name,
        "client_name": emp.client_name,
    }

    return Document(page_content=content, metadata=metadata)
