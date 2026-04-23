"""Convert JSON source records into LangChain Documents for RAG ingestion.

Architecture:
  - BUILDER_REGISTRY: maps specific filenames to specialized builders that
    handle custom field merging (e.g. first_name + last_name → Name).
  - build_generic_document(): handles ANY JSON record from ANY file
    by iterating all keys dynamically. Used for all files not in the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from langchain_core.documents import Document


# ── Specialized builders (kept for backwards compatibility) ───────────────────

def build_employee_document(emp: dict) -> Document:
    """Turn an employee dict into a High-Contrast LangChain Document."""
    name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip() or "Unknown"
    
    content = (
        f"Name: {name}\n"
        f"Role: {emp.get('designation', 'N/A')}\n"
        f"Dept: {emp.get('department', 'N/A')}\n"
        f"Email: {emp.get('email', 'N/A')}\n"
        f"Project: {emp.get('project_name', 'N/A')}\n"
        f"Source: Employee API\n"
    )

    metadata = {
        "source": "employees.json",
        "record_type": "employee",
        "record_id": emp.get("employee_id"),
        "employee_name": name,
        "designation": emp.get("designation"),
    }

    return Document(page_content=content, metadata=metadata)


def build_project_document(prj: dict) -> Document:
    """Turn a project dict (from projects.json) into a LangChain Document."""
    project_name = prj.get('Project name') or prj.get('project_name', 'Unknown Project')
    industry = prj.get('Industry ') or prj.get('industry', 'N/A')
    
    content = (
        f"[PROJECT: {project_name}]\n"
        f"Industry: {industry}\n"
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
        f"Month: {hol.get('month', 'N/A')}\n"
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
        "month": hol.get("month", ""),
    }

    return Document(page_content=content, metadata=metadata)
 
 
def build_directory_document(record: dict) -> Document:
    """Turn a directory.json record into a High-Contrast Document."""
    name = record.get("Name") or record.get("name", "Unknown")
    role = record.get("Role") or record.get("role", "N/A")
    dept = record.get("Department") or record.get("department", "N/A")
    loc = record.get("Location") or record.get("location", "N/A")
    
    # Use extremely explicit, high-contrast labels for the LLM
    content = (
        f"EMPLOYEE_NAME: {name}\n"
        f"OFFICIAL_ROLE: {role}\n"
        f"OFFICIAL_DEPARTMENT: {dept}\n"
        f"OFFICIAL_LOCATION: {loc}\n"
    )
    # Add email and phone at the top for better attention
    if record.get("Email"):
        content += f"OFFICIAL_EMAIL: {record.get('Email')}\n"
    if record.get("Phone"):
        content += f"OFFICIAL_PHONE: {record.get('Phone')}\n"

    # Add any other present fields
    for k, v in record.items():
        if k.lower() not in ["name", "role", "department", "location", "email", "phone", "record_id", "record_type"]:
            content += f"{k}: {v}\n"
 
    metadata = {
        "source": "directory.json",
        "record_type": "employee",
        "record_id": record.get("record_id") or f"DIR_{name.replace(' ', '_')}",
        "name": name,
        "role": role,
        "department": dept,
        "location": loc,
    }
    return Document(page_content=content, metadata=metadata)


def build_misc_document(doc: dict) -> Document | list[Document]:
    """Turn a miscellaneous doc dict into one or more High-Contrast Documents."""
    
    # CASE 1: Structured Tabular Data (e.g. Employee Directory)
    if doc.get("is_tabular") and doc.get("table_data"):
        rows = doc.get("table_data", [])
        documents = []
        for i, row in enumerate(rows):
            name = row.get("Name", row.get("Full Name", "Unknown"))
            role = row.get("Role", row.get("Designation", "N/A"))
            
            content = (
                f"Name: {name}\n"
                f"Role: {role}\n"
                f"Source: {doc.get('doc_name')}\n"
            )
            # Add remaining fields
            for k, v in row.items():
                if k not in ["Name", "Full Name", "Role", "Designation"]:
                    content += f"{k}: {v}\n"
            
            metadata = {
                "source": doc.get("filename", "misc.pdf"),
                "record_type": "misc_table_row",
                "record_id": f"{doc.get('doc_id')}_row{i}",
                "doc_name": doc.get("doc_name"),
                "is_tabular": True
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    # CASE 2: Standard Text Document
    content = (
        f"[DOCUMENT_NAME]: {doc.get('doc_name', 'N/A')}\n\n"
        f"{doc.get('full_text', 'No content available.')}\n"
    )

    metadata = {
        "source": doc.get("filename", "misc.pdf"),
        "record_type": "misc_text",
        "record_id": doc.get("doc_id"),
        "doc_name": doc.get("doc_name"),
        "is_tabular": False
    }

    return Document(page_content=content, metadata=metadata)


# ── Generic builder (NEW) ─────────────────────────────────────────────────────

def _flatten_value(val: object) -> str:
    """Convert any value type to a readable string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v is not None)
    if isinstance(val, dict):
        return "; ".join(f"{k}: {v}" for k, v in val.items())
    return str(val).strip()


def build_generic_document(record: dict, source_filename: str, index: int) -> Document:
    """
    Dynamically build a Document from ANY JSON record.
    Prioritizes 'header' and 'text' for readability.
    """
    stem = Path(source_filename).stem
    source_tag = stem.replace("_", " ").title()
    
    header = record.get("header") or record.get("title") or ""
    text = record.get("text") or record.get("content") or ""
    
    # 1. Build Page Content
    content_lines = [f"[Source: {source_tag}]"]
    if header:
        content_lines.append(f"Header: {header}")
    
    # Add other fields (excluding header, text, and internal metadata)
    for key, val in record.items():
        if key.lower() in ["header", "text", "title", "content", "record_type", "source_file", "doc_name"] or key.startswith("_"):
            continue
        content_lines.append(f"{key}: {_flatten_value(val)}")
    
    # Final text always at the bottom for better attention
    if text:
        content_lines.append(f"\n{text}")
        
    content = "\n".join(content_lines)

    # 2. Build Metadata
    metadata = {
        "source": source_filename,
        "record_type": record.get("record_type") or stem,
        "record_id": record.get("record_id") or f"{stem}_{index}",
        "source_tag": source_tag,
    }
    # Promote top-level string fields to metadata for filtering
    for key, val in record.items():
        if isinstance(val, (str, int, float)) and not key.startswith("_"):
            metadata[key.lower().replace(" ", "_")] = str(val)[:400]

    return Document(page_content=content, metadata=metadata)


# ── Builder Registry ──────────────────────────────────────────────────────────
# Maps specific JSON filenames to their specialized builder functions.
# Files NOT in this registry are handled by build_generic_document().

from pathlib import Path  # noqa: E402 (imported here for registry clarity)

BUILDER_REGISTRY: dict[str, object] = {
    "employees.json":    build_employee_document,
    "projects.json":     build_project_document,
    "policies.json":     build_policy_document,
    "holidays.json":     build_holiday_document,
    "directory.json":    build_directory_document,
    "praise_report.json": lambda rec: build_generic_document(rec, "praise_report.json", 0),
    "misc_docs.json":     build_misc_document,
}
