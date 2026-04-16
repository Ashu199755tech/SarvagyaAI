"""
universal_converter.py — Convert any supported file format to a normalized JSON array.

Supported input formats:
  .pdf   → pdfplumber (tables → rows, plain pages → text chunks)
  .xlsx  → openpyxl   (each sheet: row 1 = headers, rest = data)
  .xls   → openpyxl   (same as xlsx)
  .csv   → stdlib csv (row 1 = headers, rest = data)
  .docx  → python-docx (tables → rows, paragraphs → text chunks)
  .txt   → stdlib      (blank-line-delimited sections)
  .json  → pass-through (returned as-is)

All tabular output follows:
  [{"Column A": "value", "Column B": "value"}, ...]

All prose output follows:
  [{"source_file": "x.pdf", "section": 1, "text": "..."}, ...]
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(val: Any) -> str:
    """Coerce a cell value to a clean string."""
    if val is None:
        return ""
    return str(val).strip().replace("\n", " ")


def _row_to_dict(headers: list[str], row: list[Any]) -> dict:
    """Zip a header list with a data row into a dict, handling length mismatches."""
    return {
        headers[i] if i < len(headers) else f"col_{i+1}": _clean(row[i])
        for i in range(max(len(headers), len(row)))
    }


# ── Per-format converters ─────────────────────────────────────────────────────

def _convert_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _clean(v) for k, v in row.items()})
    logger.info("CSV: extracted %d rows from %s", len(rows), path.name)
    return rows


def _convert_xlsx(path: Path) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    all_rows: list[dict] = []
    for sheet in wb.worksheets:
        data = list(sheet.iter_rows(values_only=True))
        if len(data) < 2:
            continue
        headers = [_clean(h) or f"col_{i+1}" for i, h in enumerate(data[0])]
        for raw_row in data[1:]:
            d = _row_to_dict(headers, list(raw_row))
            # skip entirely empty rows
            if any(v for v in d.values()):
                d["_sheet"] = sheet.title
                all_rows.append(d)
    wb.close()
    logger.info("XLSX: extracted %d rows from %s", len(all_rows), path.name)
    return all_rows


def _convert_docx(path: Path) -> list[dict]:
    from docx import Document as DocxDocument
    doc = DocxDocument(str(path))
    rows: list[dict] = []

    # 1. Tables → structured rows
    for t_idx, table in enumerate(doc.tables):
        if len(table.rows) < 2:
            continue
        headers = [_clean(cell.text) or f"col_{i+1}" for i, cell in enumerate(table.rows[0].cells)]
        for row in table.rows[1:]:
            cells = [_clean(cell.text) for cell in row.cells]
            d = _row_to_dict(headers, cells)
            if any(v for v in d.values()):
                d["_table_index"] = t_idx
                rows.append(d)

    # 2. Paragraphs → text chunks (group by blank lines)
    if not rows:
        section_texts: list[str] = []
        current: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                current.append(text)
            else:
                if current:
                    section_texts.append(" ".join(current))
                    current = []
        if current:
            section_texts.append(" ".join(current))
        rows = [
            {"source_file": path.name, "section": i + 1, "text": t}
            for i, t in enumerate(section_texts)
        ]

    logger.info("DOCX: extracted %d records from %s", len(rows), path.name)
    return rows


def _convert_pdf(path: Path) -> list[dict]:
    import pdfplumber
    from src.ingestion.semantic_parser import SemanticPDFParser
    
    rows: list[dict] = []
    
    # 1. Try structured TABLE extraction first
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                page_table_rows = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    headers = [_clean(h) or f"col_{i+1}" for i, h in enumerate(table[0])]
                    for raw_row in table[1:]:
                        d = _row_to_dict(headers, [_clean(c) for c in raw_row])
                        if any(v for v in d.values()):
                            d["_page"] = page_num
                            page_table_rows.append(d)
                
                if page_table_rows:
                    rows.extend(page_table_rows)
    except Exception as e:
        logger.warning(f"Table extraction failed for {path.name}, falling back: {e}")

    if rows:
        logger.info("PDF: TABLE extraction successful for %s (%d rows)", path.name, len(rows))
        return rows

    # 2. Try Semantic Parsing (for Projects/Policies)
    with pdfplumber.open(str(path)) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    
    semantic_data = SemanticPDFParser.parse(full_text, path.name)
    if semantic_data:
        logger.info("PDF: SEMANTIC extraction successful for %s", path.name)
        return semantic_data

    # 3. Universal Fallback: Plain text extraction
    with pdfplumber.open(str(path)) as pdf:
        fallback_rows = []
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                fallback_rows.append({"source_file": path.name, "page": page_num, "text": text})
    
    logger.info("PDF: extracted %d records from %s (generic fallback)", len(fallback_rows), path.name)
    return fallback_rows


def _convert_txt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8", errors="replace")
    sections = [s.strip() for s in content.split("\n\n") if s.strip()]
    rows = [{"source_file": path.name, "section": i + 1, "text": s} for i, s in enumerate(sections)]
    logger.info("TXT: extracted %d sections from %s", len(rows), path.name)
    return rows


def _passthrough_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return [{"value": str(data)}]


# ── Public API ─────────────────────────────────────────────────────────────────

CONVERTERS = {
    ".csv":  _convert_csv,
    ".xlsx": _convert_xlsx,
    ".xls":  _convert_xlsx,
    ".docx": _convert_docx,
    ".doc":  _convert_docx,
    ".pdf":  _convert_pdf,
    ".txt":  _convert_txt,
    ".json": _passthrough_json,
}


def convert_to_json(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """
    Convert any supported file to a JSON array and save it.

    Args:
        input_path: Path to the source file (.pdf, .xlsx, .docx, .csv, .txt, .json)
        output_dir: Where to write the output JSON. Defaults to same directory as input.

    Returns:
        Path to the written JSON file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    src = Path(input_path)
    ext = src.suffix.lower()

    if ext not in CONVERTERS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(CONVERTERS))}"
        )

    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (src.stem + ".json")

    logger.info("Converting %s → %s", src.name, out_path)
    records = CONVERTERS[ext](src)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d records to %s", len(records), out_path)
    return out_path


def convert_inbox(inbox_dir: str | Path, converted_dir: str | Path) -> list[Path]:
    """
    Scan inbox_dir for any supported files, convert them all, and 
    save JSONs to converted_dir.

    Returns list of output JSON paths.
    """
    inbox = Path(inbox_dir)
    out_dir = Path(converted_dir)

    if not inbox.exists():
        logger.warning("Inbox does not exist: %s", inbox)
        return []

    outputs: list[Path] = []
    for f in sorted(inbox.iterdir()):
        if f.suffix.lower() in CONVERTERS and f.suffix.lower() != ".json":
            try:
                out = convert_to_json(f, output_dir=out_dir)
                outputs.append(out)
            except Exception:
                logger.exception("Failed to convert %s", f.name)
    return outputs
