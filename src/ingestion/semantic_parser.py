import re
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SemanticPDFParser:
    """
    Intelligently parses non-tabular PDF text into structured JSON schemas
    based on identified headers and content patterns.
    """

    @staticmethod
    def parse(text: str, filename: str) -> Optional[List[Dict[str, Any]]]:
        # 1. Detect if it is a Project Case Study
        if "Industry -" in text or "About the Client" in text:
            return SemanticPDFParser.parse_projects(text)
        
        # 2. Detect if it is a Company Policy
        if "Document Release Notice" in text or "Approved By:" in text:
            return SemanticPDFParser.parse_policy(text, filename)

        # No high-confidence specialized format matched
        return None


    @staticmethod
    def parse_generic_chunks(text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Generic parser that splits text into logical blocks using structural cues.
        """
        stem = Path(filename).stem.replace('_', ' ').title()
        segments = []
        
        # Split by blocks (double newlines) or Entry markers
        # Pattern detects "Entry X", "Section X", "Part X" or just single-line Title Case headers
        pattern = re.compile(r'\n\s*(\d{1,3}\.|[A-Z][\w\s]{2,40}|Entry\s+\d+|Section\s+\d+)\s*\n')
        
        # Find all potential breaks
        parts = pattern.split("\n" + text)
        
        # if no clear headers found, just split by double line breaks
        if len(parts) < 3:
            parts = re.split(r'\n\n+', text)
            for i, chunk in enumerate(parts):
                if chunk.strip():
                    segments.append({
                        "header": f"{stem} Segment {i+1}",
                        "text": chunk.strip().replace('\n', ' '),
                        "page": (i // 2) + 1  # rough estimate
                    })
        else:
            # First part is intro text before first header
            if parts[0].strip():
                segments.append({
                    "header": f"{stem} Intro",
                    "text": parts[0].strip().replace('\n', ' '),
                    "page": 1
                })
            
            # parts contains [header, content, header, content...]
            for i in range(1, len(parts), 2):
                header = parts[i].strip()
                content = parts[i+1].strip()
                if content:
                    segments.append({
                        "header": header,
                        "text": content.replace('\n', ' '),
                        "page": (i // 4) + 1 # rough estimate
                    })

        # Add global metadata
        for seg in segments:
            seg["source_file"] = filename
            seg["record_type"] = "universal_segment"
            seg["doc_name"] = stem

        return segments

    @staticmethod
    def parse_projects(text: str) -> List[Dict[str, Any]]:
        """
        Parses the master Case Study PDF which contains multiple projects.
        Each project starts with a 3-digit ID like '133' or '132'.
        """
        projects = []
        
        # Split by Project ID (usually 3 digits at the start of a line)
        # Pattern looks for a number followed by the Project Name (which matches the style in the PDF)
        parts = re.split(r'\n(\d{3})\s+', "\n" + text)
        
        # re.split with groups returns [prefix, ID, content, ID, content...]
        for i in range(1, len(parts), 2):
            proj_id = parts[i]
            content = parts[i+1]
            
            project_data = {
                "project_id": proj_id,
                "project_name": content.split('\n')[0].strip(),
                "project_brief": "",
                "project_description": "",
                "industry": "",
                "services": "",
                "technology_used": "",
                "about_client": "",
                "story": "",
                "challenges": "",
                "solution": "",
                "outcome": ""
            }

            # Extract fields using Regex
            # Industry
            m = re.search(r'Industry\s*-\s*([^\n]+)', content)
            if m: project_data["industry"] = m.group(1).strip()

            # Services
            # Services usually spans until "About the Client"
            m = re.search(r'Services\s*-\s*(.*?)(?=About the Client|$)', content, re.DOTALL)
            if m: project_data["services"] = m.group(1).strip().replace('\n', ' ')

            # About the Client
            m = re.search(r'About the Client\s*\n(.*?)(?=The Story|$)', content, re.DOTALL)
            if m: project_data["about_client"] = m.group(1).strip()

            # The Story
            m = re.search(r'The Story\s*\n(.*?)(?=Challenges|$)', content, re.DOTALL)
            if m: project_data["story"] = m.group(1).strip()

            # Challenges
            m = re.search(r'Challenges\s*\n(.*?)(?=The Solution|$)', content, re.DOTALL)
            if m: project_data["challenges"] = m.group(1).strip()

            # The Solution
            m = re.search(r'The Solution\s*\n(.*?)(?=The Outcome|$)', content, re.DOTALL)
            if m: project_data["solution"] = m.group(1).strip()

            # The Outcome
            m = re.search(r'The Outcome\s*\n(.*?)(?=\n\d{3}\s+|$)', content, re.DOTALL)
            if m: project_data["outcome"] = m.group(1).strip()

            # Technology Used (Extracting from Outcome or Solution if possible)
            # Or by looking for tech keywords in Solution
            tech_keywords = ["AWS", "Ollama", "Python", "React", "Typescript", "Node", "PostgreSQL", "Gemini", "Vertex AI", "OCR", "GPU"]
            found_tech = [t for t in tech_keywords if t.lower() in project_data["solution"].lower()]
            project_data["technology_used"] = ", ".join(found_tech)

            # Description (First paragraph after name)
            lines = content.split('\n')
            if len(lines) > 2:
                # Brief is usually the first paragraph before Industry
                brief_lines = []
                for line in lines[1:]:
                    if "Industry -" in line: break
                    brief_lines.append(line)
                project_data["project_brief"] = " ".join(brief_lines).strip()
                project_data["project_description"] = project_data["project_brief"]

            projects.append(project_data)

        return projects

    @staticmethod
    def parse_policy(text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Parses a single Policy PDF into granular sections/rules.
        """
        policy_name = Path(filename).stem.replace('_', ' ').title()
        segments = []

        # 1. Extract Summary/Intro as its own segment
        m = re.search(r'This FiftyFive Technologies.*?(?=Approved By|$)', text, re.DOTALL)
        if m:
            segments.append({
                "header": f"{policy_name} Summary",
                "text": m.group(0).strip().replace('\n', ' '),
                "policy_name": policy_name,
                "record_type": "policy_segment"
            })

        # 2. Extract Key Rules as individual segments
        rule_pattern = re.compile(r'(?:\n|^)\s*(?:[a-z]\)|[0-9]+\.|[\u25CF\u2022\u25CB\u25AA*])\s*(.*?)(?=\n\s*(?:[a-z]\)|[0-9]+\.|[\u25CF\u2022\u25CB\u25AA*])|$)', re.DOTALL)
        rules = rule_pattern.findall(text)
        
        for i, r in enumerate(rules):
            clean = r.strip().replace('\n', ' ')
            if len(clean) > 20 and "Approved By" not in clean:
                segments.append({
                    "header": f"{policy_name} Rule {i+1}",
                    "text": clean,
                    "policy_name": policy_name,
                    "record_type": "policy_segment"
                })

        # Add global metadata to all
        for seg in segments:
            seg["source_file"] = filename
            seg["doc_name"] = policy_name

        return segments
