"""
Utility script to convert PDF tables into structured JSON format.
This script extracts tables from each page of a PDF and converts rows into JSON objects
with automatically detected column headers as keys.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pdfplumber
except ImportError:
    print("Error: 'pdfplumber' is not installed. Run 'pip install pdfplumber'.")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("pdf_table_to_json")


def clean_text(text: Any) -> str:
    """Clean and strip whitespace from cell text."""
    if text is None:
        return ""
    return str(text).strip().replace("\n", " ")


def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extracts all tables from a PDF and converts them into a list of dictionaries.
    Assumes the first row of each table contains the column headers.
    """
    all_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    logger.debug(f"No tables found on page {page_num}")
                    continue
                
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                        
                    # Extract headers from the first row
                    headers = [clean_text(h) for h in table[0]]
                    
                    # If headers are empty, generate dummy headers
                    if not any(headers):
                        headers = [f"column_{i+1}" for i in range(len(table[0]))]
                        logger.warning(f"Empty headers detected on page {page_num}, table {table_idx}. Using fallback names.")
                    
                    # Process subsequent rows
                    for row in table[1:]:
                        if not any(row):  # Skip empty rows
                            continue
                            
                        row_dict = {}
                        for i, cell in enumerate(row):
                            # Handle cases where row might be longer than headers (rare with pdfplumber extract_tables)
                            header = headers[i] if i < len(headers) else f"extra_col_{i+1}"
                            row_dict[header] = clean_text(cell)
                        
                        all_data.append(row_dict)
                
                logger.info(f"Processed page {page_num}, extracted {len(all_data)} rows so far.")
                
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        raise

    return all_data


def main():
    parser = argparse.ArgumentParser(description="Convert PDF tables to JSON objects.")
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument("--output", "-o", help="Path to the output JSON file (default: same as input with .json)")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty-print the output JSON")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")
    
    try:
        logger.info(f"Starting extraction from: {input_path}")
        extracted_data = extract_tables_from_pdf(str(input_path))
        
        with open(output_path, "w", encoding="utf-8") as f:
            indent = 4 if args.pretty else None
            json.dump(extracted_data, f, ensure_ascii=False, indent=indent)
            
        logger.info(f"Successfully saved {len(extracted_data)} objects to {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")


if __name__ == "__main__":
    main()
