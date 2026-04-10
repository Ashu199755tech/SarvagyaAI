"""Convert JSON source records into LangChain Documents for RAG ingestion."""

from __future__ import annotations

import json
from langchain_core.documents import Document


def build_employee_document(emp: dict) -> Document:
    """Turn an employee dict (from raw API) into a LangChain Document.
    
    Fields from Keka API: employee_id, first_name, last_name, email, 
    department, designation, joining_date, salary, project_name, etc.
    """
    name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip() or "Unknown"
    
    content = (
        f"Employee: {name}\n"
        f"Employee ID: {emp.get('employee_id', 'N/A')}\n"
        f"Email: {emp.get('email', 'N/A')}\n"
        f"Department: {emp.get('department', 'N/A')}\n"
        f"Designation: {emp.get('designation', 'N/A')}\n"
        f"Date of Joining: {emp.get('joining_date', 'N/A')}\n"
        f"Salary: {emp.get('salary', 'N/A')}\n"
        f"Project: {emp.get('project_name', 'N/A')}\n"
        f"Client: {emp.get('client_name', 'N/A')}\n"
    )

    metadata = {
        "source": "employees.json",
        "record_type": "employee",
        "record_id": emp.get("employee_id"),
        "employee_name": name,
        "department": emp.get("department"),
        "project_name": emp.get("project_name"),
    }

    return Document(page_content=content, metadata=metadata)


def build_project_document(prj: dict) -> Document:
    """Turn a project dict (from projects.json) into a LangChain Document."""
    
    # We combine key fields into a descriptive block
    content = (
        f"Project Name: {prj.get('Project name', 'N/A')}\n"
        f"Industry: {prj.get('Industry ', 'N/A')}\n"
        f"Project Brief: {prj.get('Project brief', 'N/A')}\n"
        f"Project Description: {prj.get('Project description', 'N/A')}\n"
        f"About the Client: {prj.get('About the Client', 'N/A')}\n"
        f"The Story: {prj.get('The Story', 'N/A')}\n"
        f"Challenges: {', '.join(prj.get('Challenges ', []))}\n"
        f"The Solution: {', '.join(prj.get('The Solution ', []))}\n"
        f"The Outcome: {prj.get('The Outcome ', 'N/A')}\n"
        f"Technologies Used: {', '.join(prj.get('Technologies used', []))}\n"
    )

    metadata = {
        "source": "projects.json",
        "record_type": "project",
        "record_id": f"PRJ:{prj.get('project id', 'unknown')}",
        "project_name": prj.get("Project name"),
        "industry": prj.get("Industry "),
    }

    return Document(page_content=content, metadata=metadata)


def build_policy_document(pol: dict) -> Document:
    """Turn a policy dict (from policies.json) into a LangChain Document."""
    
    content = (
        f"Policy: {pol.get('policy_name', 'N/A')}\n"
        f"Category: {pol.get('category', 'N/A')}\n"
        f"Summary: {pol.get('summary', 'N/A')}\n"
        f"Key Rules:\n- " + "\n- ".join(pol.get("key_rules", [])) + "\n"
        f"Key Numbers: {', '.join(pol.get('key_numbers', []))}\n"
    )

    metadata = {
        "source": "policies.json",
        "record_type": "policy",
        "record_id": pol.get("policy_id"),
        "policy_name": pol.get("policy_name"),
    }

    return Document(page_content=content, metadata=metadata)


def build_holiday_document(hol: dict) -> Document:
    """Turn a holiday dict into a LangChain Document."""
    
    content = (
        f"Holiday: {hol.get('name', 'N/A')}\n"
        f"Date: {hol.get('date', 'N/A')}\n"
        f"Day: {hol.get('day', 'N/A')}\n"
        f"Type: {hol.get('type', 'N/A')}\n"
    )

    metadata = {
        "source": "holidays.json",
        "record_type": "holiday",
        "record_id": hol.get("id"),
        "holiday_name": hol.get("name"),
        "holiday_date": hol.get("date"),
    }

    return Document(page_content=content, metadata=metadata)
