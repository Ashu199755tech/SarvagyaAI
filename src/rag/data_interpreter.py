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
    "have", "has", "had", "do", "does", "did", "does", "done",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # Question words
    "who", "what", "which", "how", "where", "when", "why",
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
    "category", "categories", "type", "types", "details", "detail",
    "department", "departments", "dept", "location", "locations", "office", "offices",
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
        """
        results = []
        criteria_lower = criteria.lower().strip()
        seen_names = set()

        # Map entity types to specific files
        file_map = settings.interpreter_file_map
        target_files = []
        if entity_type in file_map:
            target_files = [self.data_dir / file_map[entity_type]]
        else:
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
                        match_found = False
                        if criteria == "__all__":
                            match_found = True
                        else:
                            for val in record.values():
                                val_str = " ".join(str(val).lower().split())
                                if criteria_lower in val_str:
                                    match_found = True
                                    break

                        if match_found:
                            name = (
                                record.get("Name")
                                or record.get("name")
                                or record.get("project_name")
                                or record.get("policy_name")
                                or record.get("Holiday")
                                or record.get("source_file", "").replace(".pdf", "").replace(".json", "").replace("_", " ").title()
                                or fpath.stem.replace("_", " ").title()
                            )
                            if name not in seen_names:
                                results.append({"name": name, "file": fpath.name})
                                seen_names.add(name)
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
        Dynamic intent detection using Subtraction-First logic.
        
        1. Identifies if the question is asking for a count/list (Intent).
        2. Specifically identifies the target entity (Employee/Project/etc).
        3. Strips away all noise (English function words).
        4. Whatever survives is the criteria.
        """
        q = question.lower().strip()
        if q.endswith('?'):
            q = q[:-1].strip()

        # 1. Intent Detection: Is this an aggregation/listing question?
        # We look for "Data Starters" (Who, What, Which, How) or "Quantity" words
        DATA_STARTERS = {"who", "what", "which", "how", "list", "total", "show", "count", "name", "give"}
        words = q.split()
        
        has_intent = False
        if any(w in DATA_STARTERS for w in words[:3]): # Check first few words for intent
            has_intent = True
        
        if not has_intent:
            # Check for explicit triggers anywhere in the question
            for trigger in AGGREGATION_TRIGGERS:
                if trigger in q:
                    has_intent = True
                    break

        if not has_intent:
            return None, None

        # 2. Entity Detection (from the RAW question to be safe)
        entity = "general"
        if any(w in q for w in ["people", "employee", "person", "member", "team", "staff", "intern", "engineer"]):
            entity = "employee"
        elif any(w in q for w in ["project", "case study", "case studies", "used"]):
            entity = "project"
        elif any(w in q for w in ["policy", "rule", "benefits", "guideline"]):
            entity = "policy"
        elif any(w in q for w in ["holiday", "holidays", "leave"]):
            entity = "holiday"

        # 3. Aggressive Subtraction
        # We strip ALL UNIVERSAL_NOISE tokens
        criteria_words = [w for w in words if w not in UNIVERSAL_NOISE and len(w) > 1]
        
        # Also strip any leftover entity words used for detection
        entity_noise = {"people", "employee", "employees", "person", "persons", "project", "projects", "policy", "policies", "holiday", "holidays"}
        final_words = [w for w in criteria_words if w not in entity_noise]

        criteria = " ".join(final_words).strip().title()

        # Handle "Total" questions with no specific filters
        if not criteria:
            if any(w in words for w in ["total", "all", "every", "count", "list"]):
                return entity, "__all__"
            return None, None

        # 4. Department Alias Bridge
        dept_aliases = settings.department_aliases
        criteria_key = criteria.lower().strip()
        if criteria_key in dept_aliases:
            criteria = dept_aliases[criteria_key]

        return entity, criteria
