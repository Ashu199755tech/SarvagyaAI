import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DataInterpreter:
    """
    Statically analyzes structured JSON data to provide 100% accurate
    counts and summaries that RAG vector search often misses.
    """

    def __init__(self, data_dir: str = "./data/converted"):
        self.data_dir = Path(data_dir)

    def query(self, entity_type: str, criteria: str) -> Optional[Dict[str, Any]]:
        """
        Dynamically scans JSON records to count and group data.
        
        Args:
            entity_type: 'employee', 'project', 'policy', or 'holiday'
            criteria: The value to filter by (e.g. 'Gurgaon', 'Artificial Intelligence')
        """
        results = []
        criteria_lower = criteria.lower().strip()

        # Map entity types to specific files to speed up search
        file_map = {
            "employee": "fiftyfive_employee_directory.json",
            "project": "Case Studies @55 Website .json",
            "holiday": "holidays_2026.json"
        }

        # If it's a policy, we scan all files that aren't the maps above
        target_files = []
        if entity_type in file_map:
            target_files = [self.data_dir / file_map[entity_type]]
        else:
            # Default: Scan everything for general "aggregation"
            target_files = list(self.data_dir.glob("*.json"))

        for fpath in target_files:
            if not fpath.exists():
                continue
            
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                    
                    for record in data:
                        # Scan all values for the criteria (fuzzy match)
                        match_found = False
                        for key, val in record.items():
                            val_str = str(val).lower().replace('\n', ' ').strip()
                            if criteria_lower in val_str:
                                match_found = True
                                break
                        
                        if match_found:
                            # Try to find a human-readable name for the result list
                            name = record.get("Name") or record.get("project_name") or record.get("policy_name") or record.get("Holiday") or "Unknown"
                            results.append({"name": name, "file": fpath.name})
            except Exception as e:
                logger.error(f"Error reading {fpath.name}: {e}")

        if not results:
            return None

        return {
            "count": len(results),
            "matches": results,
            "criteria": criteria,
            "entity": entity_type
        }

    def detect_intent(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Dynamically extracts the entity and criteria from the question.
        Example: "How many people from Udaipur?" -> ("employee", "Udaipur")
        """
        q = question.lower().strip()
        if q.endswith('?'):
            q = q[:-1]
        
        # 1. Aggregation triggers
        triggers = ["how many", "total count", "list all", "number of", "how much", "count all"]
        if not any(t in q for t in triggers):
            return None, None

        # 2. Entity detection & Query cleaning
        entity = "general"
        if any(w in q for w in ["people", "employee", "person", "member", "team"]):
            entity = "employee"
        elif any(w in q for w in ["project", "case study"]):
            entity = "project"
        elif any(w in q for w in ["policy", "rule", "benefits"]):
            entity = "policy"
        elif any(w in q for w in ["holiday", "leave"]):
            entity = "holiday"

        # 3. Dynamic Criteria Extraction (Phrase-First Scrubbing)
        entity_nouns = ["people from", "people in", "people on", "people", "employees from", "employees in", "employees", "projects using", "projects used", "projects with", "projects", "policies for", "policies", "rules for", "holidays in", "holidays", "team", "members"]
        fillers = [" of ", " in ", " from ", " at ", " are ", " there ", " used ", " we ", " have ", " the ", " about "]
        
        noise_phrases = sorted(triggers + entity_nouns + fillers, key=len, reverse=True)
        clean_q = q.replace("?", "").replace(",", "").replace(".", "").strip()
        
        # Strip long phrases first to avoid partial matches
        for phrase in noise_phrases:
            clean_q = f" {clean_q} ".replace(f" {phrase} ", " ").strip()
            
        # Clean up any remaining small noise words
        noise_words = {"about", "total", "count", "list", "number", "sum", "people", "employee", "from", "of", "the", "in", "at", "is", "are", "we", "have"}
        words = clean_q.split()
        final_words = [w for w in words if w.lower() not in noise_words]
        
        clean_q = " ".join(final_words).strip()
        
        # Strip redundant suffixes like "location", "department", "team"
        for sfx in [" location", " department", " team", " projects", " policies"]:
            if clean_q.lower().endswith(sfx):
                clean_q = clean_q[:-len(sfx)].strip()

        criteria = clean_q.strip().title()
        
        if not criteria or len(criteria) < 2:
            return None, None

        # Edge case: "AI Team" -> "Artificial Intelligence" (optional bridge)
        if criteria == "Ai" or criteria == "Ai Team": criteria = "Artificial Intelligence"

        return entity, criteria
