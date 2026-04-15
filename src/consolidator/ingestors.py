"""
NEW ingestors.py — Zero-LLM source-of-truth pipeline
=====================================================

Key changes vs original:
  - ingest_employees_from_api()  → Saves raw API response to employees.json for fidelity.
  - ingest_holidays_from_pdf()   → NEW: Extracts structured holidays from PDF.
  - refresh_all_sources()        → Rebuilds all 5 JSON files from raw data.
  - ingest_pdfs()                → Reads from JSONs and returns (Policies, Projects, Tools, Holidays).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from src.config import settings
from src.consolidator.entities import (
    EmployeeEntity,
    PolicyEntity,
    ProjectEntity,
    HolidayEntity,
    ToolEntity,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(settings.data_dir)
SOT_EMPLOYEES = DATA_DIR / "employees.json"
SOT_POLICIES  = DATA_DIR / "policies.json"
SOT_PROJECTS  = DATA_DIR / "projects.json"
SOT_HOLIDAYS  = DATA_DIR / "holidays.json"
SOT_MISC      = DATA_DIR / "misc_docs.json"


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').replace('\u200b', '')).strip()


def _pdftotext(pdf_path: Path) -> str:
    """Extract text from a PDF using pdftotext (no LLM)."""
    if shutil.which("pdftotext"):
        try:
            r = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True, check=True,
            )
            return r.stdout
        except Exception as e:
            logger.warning("pdftotext failed for %s: %s", pdf_path.name, e)

    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error("pypdf also failed for %s: %s", pdf_path.name, e)
        return ""


def _to_bullets(text: str) -> list[str]:
    items = []
    # Support both round bullets and numeric lists
    for line in text.replace('●', '\n-').replace('•', '\n-').splitlines():
        line = re.sub(r'^[\s\-•​]+', '', line).strip()
        if len(line) > 3:
            items.append(line)
    return items


def _save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("SOT written: %s (%d records)", path.name, len(data))


def _extract_table_data(pdf_path: Path) -> list[dict]:
    """Extract list of dicts from PDF tables using pdfplumber."""
    if not pdfplumber:
        logger.warning("pdfplumber not installed, skipping table extraction for %s", pdf_path.name)
        return []

    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    
                    # First row is headers. Clean them.
                    headers = [str(h or "").strip().replace("\n", " ") for h in table[0]]
                    if not any(headers):
                        headers = [f"col_{i}" for i in range(len(table[0]))]

                    # Remaining rows
                    for row in table[1:]:
                        if not any(row): continue
                        obj = {}
                        for i, cell in enumerate(row):
                            key = headers[i] if i < len(headers) else f"col_{i}"
                            obj[key] = str(cell or "").strip().replace("\n", " ")
                        results.append(obj)
    except Exception as e:
        logger.error("pdfplumber failed for %s: %s", pdf_path, e)
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMPLOYEES — Keka API → employees.json
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_employees_from_api() -> list[EmployeeEntity]:
    """Fetch employees from Keka API and persist to employees.json (no LLM)."""
    from src.keka.client import KekaClient

    client = KekaClient()
    try:
        raw_employees = await client.get_all_employees()
        employees = []
        raw_records = []

        for emp in raw_employees:
            name = f"{emp.first_name} {emp.last_name}".strip() or "Unknown"
            entity = EmployeeEntity(
                id=emp.employee_id,
                name=name,
                email=emp.email or "",
                department=emp.department or "",
                designation=emp.designation or "",
                salary=emp.salary,
                joining_date=emp.joining_date,
                project_name=emp.project_name or "",
                project_start_date=emp.project_start_date,
                project_end_date=emp.project_end_date,
                client_name=emp.client_name or "",
            )
            employees.append(entity)
            # FIX: Save raw dictionary from model for 1:1 fidelity with Keka API
            raw_records.append(emp.model_dump(mode="json"))

        # Persist as source-of-truth (no LLM involved)
        _save_json(raw_records, SOT_EMPLOYEES)
        logger.info("employees.json updated with %d raw records", len(raw_records))
        return employees

    finally:
        await client.close()


def ingest_holidays_from_pdf() -> list[HolidayEntity]:
    """Extract holiday list from the Holiday Policy PDF using pdfplumber."""
    
    # Locate the holiday policy PDF
    pdf_dir = Path(settings.pdf_folder)
    holiday_pdf = None
    for p in pdf_dir.rglob("*.pdf"):
        if "Holiday" in p.name:
            holiday_pdf = p
            break
            
    if not holiday_pdf:
        logger.warning("No Holiday Policy PDF found in %s", pdf_dir)
        return []

    logger.info("Extracting holidays from %s using pdfplumber", holiday_pdf)
    rows = _extract_table_data(holiday_pdf)
    
    holidays = []
    records = []
    
    for row in rows:
        # Map dynamic headers to HolidayEntity fields
        # Note: headers vary by page/table, so we check for common keys
        idx  = row.get("Sr. No") or row.get("Index") or ""
        name = row.get("Holiday") or row.get("Name") or ""
        mon  = row.get("Month") or ""
        dstr = row.get("Date") or ""
        day  = row.get("Day") or ""
        rem  = row.get("Remarks") or row.get("Type") or ""
        
        if not name: continue  # Skip rows without names

        try:
            # Parse date e.g. "4-Sep-26" or "21-Mar-26" -> 2026-09-04
            clean_date = None
            if dstr:
                dt = datetime.strptime(dstr, "%d-%b-%y")
                clean_date = dt.date()
        except ValueError:
            clean_date = None
            
        entity = HolidayEntity(
            id=f"HOL_{idx}" if idx else f"HOL_{name[:3].upper()}",
            name=name.strip(),
            date=clean_date,
            day=day.strip(),
            month=mon.strip(),
            type=rem.strip()
        )
        holidays.append(entity)
        
        # Format for JSON
        record = asdict(entity)
        if record["date"]:
            record["date"] = record["date"].isoformat()
        records.append(record)
        
    if records:
        _save_json(records, SOT_HOLIDAYS)
    
    return holidays


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROJECTS — case studies PDF → projects.json
# ─────────────────────────────────────────────────────────────────────────────

def _parse_case_study_pdf(pdf_path: Path) -> list[dict]:
    """Parse the case studies PDF into structured project dicts. No LLM."""
    raw = _pdftotext(pdf_path).replace('\x0c', '\n')
    
    # Split into per-case-study blocks by the leading project number (supports space or dash, 1-3 digits)
    blocks = re.split(r'\n(?=\d{1,3}[-– ]\s*[A-Z])', raw)
    projects = []

    for block in blocks:
        block = block.strip()
        if not block or not re.match(r'^\d{1,3}[-– ]\s*', block):
            continue

        lines = block.splitlines()
        first_line = lines[0].strip()
        # Match ID followed by space, dash, or en-dash (1-3 digits)
        m = re.match(r'^(\d+)[-– ]\s*(.*)', first_line)
        if not m:
            continue

        proj_id  = m.group(1)
        raw_name = m.group(2).strip()

        # Collect name continuation lines
        name_parts = [raw_name]
        brief = ''
        for line in lines[1:6]:
            line = line.strip()
            if re.match(r'^(Industry|Services|FiftyFive|About|The Story|Challenges)', line, re.I):
                break
            if line and len(line) > 3:
                if re.search(r'(built|replaced|designed|developed|delivered|enabled|improved|moderniz|automated)\b', line, re.I):
                    brief = _clean(line)
                else:
                    name_parts.append(line)

        project_name = _clean(' '.join(name_parts))

        # Extraction logic using targeted patterns
        ind_m = re.search(r'\nIndustry\s*[-–:]?\s*(.*?)(?=\nServices|\nAbout|\nProject Description|\nThe Story)', block, re.DOTALL | re.I)
        industry = _clean(ind_m.group(1)) if ind_m else ''

        svc_m = re.search(r'\nServices\s*[-–:]?\s*(.*?)(?=\nAbout|\nThe Story|\nProject Description|\nTechnologies)', block, re.DOTALL | re.I)
        services = _to_bullets(svc_m.group(1)) if svc_m else []

        desc_m = re.search(r'\nProject Description\s*[:\s]*\n(.*?)(?=\nIndustry|\nServices|\nAbout)', block, re.DOTALL | re.I)
        description = _clean(desc_m.group(1)) if desc_m else ""

        about_m = re.search(r'\nAbout (?:the |Our )?Client\s*[:\s]*\n(.*?)(?=\nThe Story|\nChallenges|\nProject Description)', block, re.DOTALL | re.I)
        about_client = _clean(about_m.group(1)) if about_m else ''

        story_m = re.search(r'\n(?:The )?Story\s*[:\s]*\n(.*?)(?=\nChallenges|\nThe Challenges|\nSolution|\nThe Solution)', block, re.DOTALL | re.I)
        story = _clean(story_m.group(1)) if story_m else ''

        tech_m = re.search(r'\n(?:Technologies used|Tech stack|Technologies)\s*[:\s]*\n(.*?)(?=\nChallenges|\nSolution|\nThe Solution|\nThe Outcome|\nAbout the Client)', block, re.DOTALL | re.I)
        technologies = _to_bullets(tech_m.group(1)) if tech_m else []

        # Fallback for technologies inside "The Solution"
        if not technologies and "The Solution" in block:
            ts_m = re.search(r'Tech stack[:\s]+(.*?)(?=\.|\n|$)', block, re.I)
            if ts_m:
                technologies = [t.strip() for t in ts_m.group(1).split(',')]

        chal_m = re.search(r'\n(?:The )?Challenges\s*[:\s]*\n(.*?)(?=\nSolution|\nThe Solution|\nOutcome|\nThe Outcome)', block, re.DOTALL | re.I)
        challenges = _to_bullets(chal_m.group(1)) if chal_m else []

        sol_m = re.search(r'\n(?:The )?Solution\s*[:\s]*\n(.*?)(?=\nOutcome|\nThe Outcome|\nTechnologies used)', block, re.DOTALL | re.I)
        solution = _to_bullets(sol_m.group(1)) if sol_m else []

        out_m = re.search(r'\n(?:The )?Outcome\s*[:\s]*\n(.*?)$', block, re.DOTALL | re.I)
        outcome = _clean(out_m.group(1)) if out_m else ''

        if not description and brief:
            description = brief

        projects.append({
            "project id":   proj_id,
            "Project name": project_name,
            "Project brief": brief,
            "Project description": description,
            "Industry ":     industry,
            "Services ":     services,
            "About the Client": about_client,
            "The Story":        story,
            "Technologies used": technologies,
            "Challenges ":   challenges,
            "The Solution ":     solution,
            "The Outcome ":      outcome,
        })

    projects.sort(key=lambda x: int(x['project id']) if x['project id'].isdigit() else 0)
    return projects


def refresh_projects(pdf_folder: str | None = None) -> list[dict]:
    """Rebuild projects.json from the case studies PDF. No LLM."""
    folder = Path(pdf_folder or settings.pdf_folder)
    projects = []

    for pdf_path in sorted(folder.rglob("*.pdf")):
        if "case stud" in pdf_path.name.lower() or "case_stud" in pdf_path.name.lower():
            logger.info("Parsing case studies PDF: %s", pdf_path.name)
            projects.extend(_parse_case_study_pdf(pdf_path))

    _save_json(projects, SOT_PROJECTS)
    return projects


# ─────────────────────────────────────────────────────────────────────────────
# 3. POLICIES — policy PDFs → policies.json
# ─────────────────────────────────────────────────────────────────────────────

def _parse_policy_pdf(pdf_path: Path) -> dict:
    """Parse one policy PDF into a structured dict. No LLM."""
    text = _pdftotext(pdf_path).replace('\x0c', '\n')

    # Remove boilerplate headers
    text = re.sub(r'Document Release Notice.*?effect from \d{2} \w+ \d{4}\.', '', text,
                  flags=re.IGNORECASE | re.DOTALL)

    # Summary = first substantive paragraph (>80 chars)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 80]
    summary = _clean(paragraphs[0]) if paragraphs else ''

    # Key rules = bullet/numbered lines
    rules = []
    for line in text.splitlines():
        stripped = re.sub(r'^[\s\-●•▪❖\d]+[.)]\s*', '', line).strip()
        if len(stripped) > 30 and not re.match(r'^(page|document|release|version|confidential)', stripped, re.I):
            rules.append(_clean(stripped))
        if len(rules) >= 20:
            break

    # Key numbers (e.g. "24 days", "3 months")
    numbers = re.findall(r'\b(\d+)\s+(day|days|week|weeks|month|months|year|years|hour|hours)\b', text, re.I)
    key_numbers = list(dict.fromkeys([f"{n} {u.lower()}" for n, u in numbers]))[:10]

    return {
        "policy_id":   f"POL:{pdf_path.stem}",
        "policy_name": pdf_path.stem.replace("_", " ").replace("-", " ").title(),
        "filename":    pdf_path.name,
        "category":    pdf_path.parent.name,
        "summary":     summary[:800],
        "key_rules":   rules,
        "key_numbers": key_numbers,
        "full_text":   text.strip(),
    }


def refresh_policies(pdf_folder: str | None = None) -> list[dict]:
    """Rebuild policies.json from all non-case-study PDFs."""
    folder = Path(pdf_folder or settings.pdf_folder)
    policies = []

    for pdf_path in sorted(folder.rglob("*.pdf")):
        name_lower = pdf_path.name.lower()
        if "case stud" in name_lower or "case_stud" in name_lower:
            continue
        logger.info("Parsing policy PDF: %s", pdf_path.name)
        policies.append(_parse_policy_pdf(pdf_path))

    _save_json(policies, SOT_POLICIES)
    return policies


# ─────────────────────────────────────────────────────────────────────────────
# 4. MISC DOCS — other PDFs → misc_docs.json
# ─────────────────────────────────────────────────────────────────────────────

def refresh_misc_docs(pdf_folder: str | None = None,
                      policy_keywords: list[str] | None = None) -> list[dict]:
    """Collect PDFs that don't match case study or policy patterns."""
    folder = Path(pdf_folder or settings.pdf_folder)
    policy_kws = policy_keywords or ["polic", "leave", "travel", "attendance",
                                     "hr", "gratuity", "referral", "separation"]
    misc = []

    for pdf_path in sorted(folder.rglob("*.pdf")):
        name_lower = pdf_path.name.lower()
        if "case stud" in name_lower or "case_stud" in name_lower:
            continue
        if any(kw in name_lower for kw in policy_kws):
            continue
            
        logger.info("Parsing misc PDF: %s", pdf_path.name)
        
        # Check if it's a directory - use structured table extract
        if "directory" in name_lower:
            logger.info("Structured table extraction for directory: %s", pdf_path.name)
            table_data = _extract_table_data(pdf_path)
            # Store as formatted text for now so RAG can still read it 
            # OR store the list and let RAG deal with it.
            # Best is to store the list in a specific 'table_data' field
            full_text = json.dumps(table_data, indent=2)
        else:
            table_data = []
            full_text = _pdftotext(pdf_path).replace('\x0c', '\n')

        misc.append({
            "doc_id":      f"DOC:{pdf_path.stem}",
            "doc_name":    pdf_path.stem.replace("_", " ").replace("-", " ").title(),
            "filename":    pdf_path.name,
            "category":    pdf_path.parent.name,
            "full_text":   full_text.strip(),
            "table_data":  table_data,
            "is_tabular":  len(table_data) > 0
        })

    _save_json(misc, SOT_MISC)
    return misc


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC API — called by consolidator/run.py
# ─────────────────────────────────────────────────────────────────────────────

def refresh_all_sources(pdf_folder: str | None = None) -> dict:
    """Rebuild all 5 source-of-truth JSON files from raw data."""
    folder = pdf_folder or settings.pdf_folder
    projects    = refresh_projects(folder)
    policies    = refresh_policies(folder)
    misc_docs   = refresh_misc_docs(folder)
    holidays    = ingest_holidays_from_pdf()

    return {
        "projects":  len(projects),
        "policies":  len(policies),
        "misc_docs": len(misc_docs),
        "holidays":  len(holidays),
    }


def ingest_pdfs(folder: str | None = None) -> tuple[list[PolicyEntity], list[ProjectEntity], list[ToolEntity], list[HolidayEntity]]:
    """Reads from source-of-truth JSON files and returns Entity lists."""
    folder = folder or settings.pdf_folder

    if not SOT_PROJECTS.exists() or not SOT_POLICIES.exists():
        logger.info("Source-of-truth files not found — building now...")
        refresh_all_sources(folder)

    # 1. Projects
    projects: list[ProjectEntity] = []
    tools_dict: dict[str, ToolEntity] = {}
    known_tech = set(settings.rag_known_tech_terms) | {
        a.upper() for a in settings.rag_known_acronyms
    }

    if SOT_PROJECTS.exists():
        raw_projects = json.loads(SOT_PROJECTS.read_text(encoding="utf-8"))
        for p in raw_projects:
            project_id = f"PRJ:{p.get('project id', 'unknown')}"
            
            full_text = " ".join([
                p.get("The Story", ""),
                p.get("The Outcome ", ""),
                " ".join(p.get("The Solution ", [])),
            ]).lower()
            
            found_tools = []
            for tech in known_tech:
                if tech.lower() in full_text:
                    tool_name = tech.upper() if len(tech) <= 4 else tech.title()
                    found_tools.append(tool_name)
                    tool_id = f"TOOL:{tool_name}"
                    if tool_id not in tools_dict:
                        tools_dict[tool_id] = ToolEntity(id=tool_id, name=tool_name)
                    tools_dict[tool_id].project_ids.append(project_id)

            projects.append(ProjectEntity(
                id=project_id,
                name=p.get("Project name", ""),
                brief=p.get("Project brief", ""),
                description=p.get("Project description", ""),
                industry=p.get("Industry ", ""),
                services=p.get("Services ", []),
                about_client=p.get("About the Client", ""),
                the_story=p.get("The Story", ""),
                technologies_used=p.get("Technologies used", []),
                challenges=p.get("Challenges ", []),
                the_solution=p.get("The Solution ", []),
                the_outcome=p.get("The Outcome ", ""),
                tools=found_tools,
            ))

    # 2. Policies
    policies: list[PolicyEntity] = []
    if SOT_POLICIES.exists():
        raw_policies = json.loads(SOT_POLICIES.read_text(encoding="utf-8"))
        for p in raw_policies:
            policies.append(PolicyEntity(
                id=p.get("policy_id", ""),
                name=p.get("policy_name", ""),
                filename=p.get("filename", ""),
                category=p.get("category", ""),
                summary=p.get("summary", ""),
                key_rules=p.get("key_rules", []),
            ))

    # 3. Holidays
    holidays: list[HolidayEntity] = []
    if SOT_HOLIDAYS.exists():
        raw_holidays = json.loads(SOT_HOLIDAYS.read_text(encoding="utf-8"))
        for h in raw_holidays:
            from datetime import date
            clean_date = date.fromisoformat(h["date"]) if h.get("date") else None
            
            holidays.append(HolidayEntity(
                id=h.get("id", ""),
                name=h.get("name", ""),
                date=clean_date,
                day=h.get("day", ""),
                month=h.get("month", ""),
                type=h.get("type", ""),
            ))

    logger.info(
        "Loaded from SOT: %d projects, %d policies, %d tools, %d holidays",
        len(projects), len(policies), len(tools_dict), len(holidays),
    )
    return policies, projects, list(tools_dict.values()), holidays