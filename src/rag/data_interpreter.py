import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universal Noise Vocabulary — English function words (closed class)
# These NEVER change regardless of what PDFs you upload.
# ---------------------------------------------------------------------------
UNIVERSAL_NOISE = {
    # Determiners & Articles
    "the", "a", "an", "this", "that", "these", "those",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "who", "what", "which", "my", "your", "our", "their",
    "its", "whose", "whom",
    # Prepositions
    "in", "on", "at", "from", "to", "of", "by", "with", "for", "about",
    "between", "through", "during", "into", "onto", "upon", "under",
    "over", "after", "before", "since", "until", "within", "without",
    # Conjunctions
    "and", "or", "but", "so", "yet", "nor", "if", "then", "than",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # Common question verbs & participles
    "get", "got", "getting", "give", "gave", "given", "giving",
    "take", "took", "taken", "taking",
    "tell", "told", "telling", "say", "said", "saying",
    "know", "knew", "known", "knowing",
    "go", "went", "gone", "going", "come", "came", "coming",
    "make", "made", "making", "see", "seen", "seeing",
    "find", "found", "finding", "show", "shown", "showing",
    # Action verbs commonly in aggregation queries
    "praised", "mentioned", "awarded", "appreciated", "recognized",
    "used", "using", "involved", "included", "including",
    "discussed", "discussing", "referred", "referring",
    "talked", "talking", "described", "describing",
    "covered", "covering", "addressed", "addressing",
    "listed", "listing", "received", "receiving",
    "appeared", "appearing", "featured", "featuring",
    "worked", "working", "done", "doing",
    "completed", "completing", "delivered", "delivering",
    "contributed", "contributing", "participated", "participating",
    "related", "relating", "associated", "belonging",
    # General filler words
    "there", "here", "also", "just", "only", "very", "much",
    "many", "some", "any", "all", "each", "every", "other",
    "more", "most", "such", "too", "quite", "really", "still",
    "even", "already", "always", "never", "often", "ever",
    # Question-specific noise (entity nouns)
    "total", "count", "number", "sum", "times", "time", "often",
    "people", "person", "persons", "employee", "employees",
    "member", "members", "team", "teams", "staff",
    "project", "projects", "policy", "policies",
    "document", "documents", "record", "records",
    "entry", "entries", "file", "files", "report", "reports",
    "holiday", "holidays", "leave", "leaves",
    "rule", "rules", "benefits", "benefit", "item", "items",
    "category", "categories", "type", "types",
    # Polite/conversational noise
    "please", "can", "want", "need", "like", "regarding", "concerning",
}

# Aggregation trigger phrases — sorted longest-first for greedy matching
AGGREGATION_TRIGGERS = [
    "how many times", "how many", "how often", "how much",
    "total count of", "total count", "total number of", "total number",
    "count all", "count of", "list all", "number of",
]


class DataInterpreter:
    """
    Universal data scanner — works for ANY JSON file without code changes.

    Strategy: "Aggressive Subtraction"
    Instead of trying to guess what the user is asking about, we strip ALL
    known English function words from the question. Whatever survives IS
    the search criteria. This works because English function words are a
    closed class (~200 words) that never changes regardless of data.
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.converted_dir)

    def query(self, entity_type: str, criteria: str) -> Optional[Dict[str, Any]]:
        """
        Dynamically scans JSON records to count and group data.

        Args:
            entity_type: 'employee', 'project', 'policy', 'holiday', or 'general'
            criteria: The value to filter by (e.g. 'Gurgaon', 'Vipin Rai')
        """
        results = []
        criteria_lower = criteria.lower().strip()

        # Map entity types to specific files — driven by config, not hardcoded
        file_map = settings.interpreter_file_map

        # Determine which files to scan
        target_files = []
        if entity_type in file_map:
            target_files = [self.data_dir / file_map[entity_type]]
        else:
            # "general" → Scan EVERYTHING (any new PDF auto-included)
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
                            # Smart label: try structured keys, then source_file, then filename
                            name = (
                                record.get("Name")
                                or record.get("project_name")
                                or record.get("policy_name")
                                or record.get("Holiday")
                                or record.get("source_file", "").replace(".pdf", "").replace(".json", "").replace("_", " ").title()
                                or fpath.stem.replace("_", " ").title()
                            )
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
        Universal intent detection using Aggressive Subtraction.

        1. Check if the question contains aggregation triggers
        2. Detect entity type (employee/project/policy/holiday/general)
        3. Strip ALL function words → whatever remains IS the search criteria
        """
        q = question.lower().strip()
        if q.endswith('?'):
            q = q[:-1].strip()

        # 1. Aggregation trigger check (longest-first greedy matching)
        triggered = False
        for trigger in AGGREGATION_TRIGGERS:
            if trigger in q:
                triggered = True
                break

        if not triggered:
            return None, None

        # 2. Entity detection
        entity = "general"
        if any(w in q for w in ["people", "employee", "person", "member", "team", "staff"]):
            entity = "employee"
        elif any(w in q for w in ["project", "case study", "case studies"]):
            entity = "project"
        elif any(w in q for w in ["policy", "rule", "benefits"]):
            entity = "policy"
        elif any(w in q for w in ["holiday", "holidays"]):
            entity = "holiday"

        # 3. Aggressive Subtraction
        clean = q

        # Strip trigger phrases first (longest-first to avoid partial matches)
        for trigger in sorted(AGGREGATION_TRIGGERS, key=len, reverse=True):
            clean = clean.replace(trigger, " ")

        # Split into words and strip ALL function/noise words
        words = clean.split()
        criteria_words = [w for w in words if w.lower() not in UNIVERSAL_NOISE and len(w) > 1]

        # Rejoin and clean up
        criteria = " ".join(criteria_words).strip().title()

        # Strip redundant suffixes (e.g. "Gurgaon Location" → "Gurgaon")
        for sfx in [" Location", " Department", " Office", " City", " Region", " Area"]:
            if criteria.endswith(sfx):
                criteria = criteria[:-len(sfx)].strip()

        if not criteria or len(criteria) < 2:
            return None, None

        # 4. Department Alias Bridge — driven by config, not hardcoded
        dept_aliases = settings.department_aliases
        criteria_key = criteria.lower().strip()
        if criteria_key in dept_aliases:
            criteria = dept_aliases[criteria_key]

        return entity, criteria
