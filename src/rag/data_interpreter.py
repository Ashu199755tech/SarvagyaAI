import json
import logging
import re
from calendar import month_abbr, month_name
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings

logger = logging.getLogger(__name__)
_NLP_MODEL = None
_NLP_LOAD_ATTEMPTED = False
_WORDNET = None
_WORDNET_LOAD_ATTEMPTED = False


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary values into a comparable lowercase string."""
    return " ".join(str(value).lower().split())


def _tokenize_text(value: Any) -> list[str]:
    """Tokenize text into lowercase word-like units."""
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _token_variants(token: str) -> set[str]:
    """Return a small set of useful token variants for fuzzy field matching."""
    variants = {token}
    # Collective nouns and direct aliases
    aliases = {
        "staff": "employee",
        "employees": "employee",
        "projects": "project",
        "holidays": "holiday",
    }
    if token in aliases:
        variants.add(aliases[token])
        
    if token.endswith("ies") and len(token) > 3:
        variants.add(token[:-3] + "y")
    elif token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return variants


def _calendar_month_terms(text: str) -> set[str]:
    """Extract month mentions using stdlib calendar names, not hardcoded lists."""
    candidates = {
        name.lower() for name in month_name[1:] if name
    } | {
        name.lower() for name in month_abbr[1:] if name
    }
    text_tokens = set(_tokenize_text(text))
    matched = text_tokens & candidates

    # Normalize abbreviations such as "sep" to their full month name when possible.
    full_month_by_abbr = {
        abbr.lower(): full.lower()
        for abbr, full in zip(month_abbr[1:], month_name[1:], strict=False)
        if abbr and full
    }
    return {
        full_month_by_abbr.get(term, term)
        for term in matched
    }


def _load_optional_nlp_model():
    """Load spaCy lazily if the optional NLP fallback is enabled and installed."""
    global _NLP_MODEL, _NLP_LOAD_ATTEMPTED
    if _NLP_LOAD_ATTEMPTED:
        return _NLP_MODEL

    _NLP_LOAD_ATTEMPTED = True
    if not settings.data_interpreter_enable_nlp_fallback:
        return None

    try:
        import spacy  # type: ignore

        _NLP_MODEL = spacy.load(settings.data_interpreter_nlp_model)
        logger.info(
            "DataInterpreter NLP fallback enabled with spaCy model %s",
            settings.data_interpreter_nlp_model,
        )
    except Exception as exc:
        logger.info("DataInterpreter NLP fallback unavailable: %s", exc)
        _NLP_MODEL = None

    return _NLP_MODEL


def _load_optional_wordnet():
    """Load NLTK WordNet lazily if it is installed and the corpus is available."""
    global _WORDNET, _WORDNET_LOAD_ATTEMPTED
    if _WORDNET_LOAD_ATTEMPTED:
        return _WORDNET

    _WORDNET_LOAD_ATTEMPTED = True
    try:
        from nltk.corpus import wordnet as wn  # type: ignore

        # Touch the corpus to ensure it is actually downloaded.
        wn.synsets("employee")
        _WORDNET = wn
    except Exception as exc:
        logger.info("DataInterpreter WordNet expansion unavailable: %s", exc)
        _WORDNET = None

    return _WORDNET


def _wordnet_terms(token: str) -> set[str]:
    """Expand a token using optional WordNet lexical relations."""
    wn = _load_optional_wordnet()
    if wn is None or len(token) < 3:
        return set()

    expanded = set()
    for synset in wn.synsets(token):
        for lemma in synset.lemmas():
            expanded.update(_tokenize_text(lemma.name().replace("_", " ")))
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                expanded.update(_tokenize_text(lemma.name().replace("_", " ")))
    return {term for term in expanded if len(term) > 2}


def _semantic_token_family(token: str) -> set[str]:
    """Build a reusable semantic family for a token."""
    family = set()
    for variant in _token_variants(token):
        family.add(variant)
        family.update(_wordnet_terms(variant))
    return family


def _term_matches(term: str, text: str) -> bool:
    """Check if a term (or its basic stem) appears in text."""
    if term in text:
        return True
    # Try basic de-pluralisation: "leads" -> "lead", "engineers" -> "engineer"
    if term.endswith("ies") and len(term) > 3:
        return term[:-3] + "y" in text
    if term.endswith("es") and len(term) > 3:
        return term[:-2] in text
    if term.endswith("s") and len(term) > 2:
        return term[:-1] in text
    return False


def _criteria_matches_record(record: Dict[str, Any], criteria: str | list[str]) -> bool:
    """
    Return True when the full phrase or all criteria terms appear in a record.

    This keeps matching robust when the JSON formatting differs slightly from
    how the user phrases the query.
    """
    if isinstance(criteria, list):
        criteria_terms = [_normalize_text(term) for term in criteria if _normalize_text(term)]
        if not criteria_terms:
            return False
        combined_text = " ".join(
            part for part in (_normalize_text(val) for val in record.values()) if part
        )
        return all(_term_matches(term, combined_text) for term in criteria_terms)

    criteria_text = _normalize_text(criteria)
    if not criteria_text:
        return False

    combined_text = " ".join(
        part for part in (_normalize_text(val) for val in record.values()) if part
    )
    if criteria_text in combined_text:
        return True

    criteria_terms = [term for term in criteria_text.split() if len(term) > 1]
    return bool(criteria_terms) and all(_term_matches(term, combined_text) for term in criteria_terms)


@dataclass
class StructuredQuery:
    operation: str
    entity_type: str
    criteria_terms: list[str]
    raw_question: str

    @property
    def criteria(self) -> str:
        return " ".join(self.criteria_terms).strip() if self.criteria_terms else "__all__"


class DataInterpreter:
    """
    Schema-aware structured query executor for aggregation and listing questions.

    Instead of relying on brittle phrase lists for every business query, this
    interpreter:
      1. infers the target entity from available JSON schemas and filenames
      2. infers whether the user wants a count or a list
      3. extracts only selective filter terms based on the actual data
      4. executes the result deterministically against JSON records
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.data_dir)
        self._record_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_entity_records(self, entity_type: str) -> list[dict[str, Any]]:
        """Load records for a configured entity from the converted JSON directory."""
        if entity_type in self._record_cache:
            return self._record_cache[entity_type]

        file_map = settings.interpreter_file_map
        target_name = file_map.get(entity_type)
        if not target_name:
            self._record_cache[entity_type] = []
            return []

        path = self.data_dir / target_name
        if not path.exists():
            # Fallback: check the converted sub-directory (some files like
            # praise_report.json live there instead of the root data dir).
            fallback = Path(settings.converted_dir) / target_name
            if fallback.exists():
                path = fallback
            else:
                self._record_cache[entity_type] = []
                return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Error reading %s: %s", path.name, exc)
            self._record_cache[entity_type] = []
            return []

        if isinstance(data, list):
            records = [record for record in data if isinstance(record, dict)]
        elif isinstance(data, dict):
            records = [data]
        else:
            records = []

        # Filter out alphabet section headers — single-character "Name" entries
        # with empty Role/Department that are not real records.
        # e.g. {"Name": "A", "Role": "", "Department": "", ...}
        # NOTE: We check both "Name" (directory) and "name" (holidays/projects)
        # to avoid accidentally filtering out valid records that use lowercase keys.
        records = [
            r for r in records
            if not (
                # Only apply the "header" filter to records that don't have long-form text
                "text" not in r
                and not r.get("Project name")
                and len(str(r.get("Name", r.get("name", ""))).strip()) <= 1
                and not str(r.get("Role", r.get("role", ""))).strip()
                and not str(r.get("Department", r.get("department", ""))).strip()
            )
        ]

        self._record_cache[entity_type] = records
        return records

    def _entity_signature_tokens(
        self,
        entity_type: str,
        records: list[dict[str, Any]],
    ) -> set[str]:
        """Build a schema-aware token signature for an entity."""
        signature = set()
        file_name = settings.interpreter_file_map.get(entity_type, "")
        signature.update(_tokenize_text(entity_type))
        signature.update(_tokenize_text(file_name.replace(".json", "")))

        sample_size = settings.data_interpreter_schema_sample_size
        for record in records[:sample_size]:
            signature.update(
                token
                for key in record.keys()
                for token in _tokenize_text(key)
                if len(token) > 2
            )

        value_sample_size = settings.data_interpreter_entity_value_sample_size
        max_value_length = settings.data_interpreter_entity_value_max_length
        for record in records[:value_sample_size]:
            for value in record.values():
                value_tokens = [
                    token for token in _tokenize_text(value)
                    if len(token) > 2
                ]
                if 0 < len(value_tokens) <= max_value_length:
                    signature.update(value_tokens)
        return signature

    def _infer_entity(
        self,
        tokens: list[str],
        nlp_signals: Optional[dict[str, list[str]]] = None,
    ) -> Optional[str]:
        """Infer the most likely entity by comparing query terms to dataset schemas."""
        if not tokens:
            return None

        token_pool = set()
        for token in tokens:
            token_pool.update(_semantic_token_family(token))
        if nlp_signals:
            for token in nlp_signals.get("lemma_terms", []):
                token_pool.update(_semantic_token_family(token))
            for group in ("noun_chunk_terms", "person_terms", "date_terms", "month_terms"):
                for phrase in nlp_signals.get(group, []):
                    for token in _tokenize_text(phrase):
                        token_pool.update(_semantic_token_family(token))

        # ── DYNAMIC IDENTITY SCORING ──────────────────────────────────────────
        # Instead of a hardcoded identity map, dynamically match query tokens
        # (and their lemmas) against the entity names from config. This makes
        # routing automatically work for any new entity added to the file map.
        #
        # A small alias dict handles non-obvious synonyms that lemmatisation
        # alone cannot resolve (e.g. "vacation" → "holiday").
        # ─────────────────────────────────────────────────────────────────────
        entity_names = set(settings.interpreter_file_map.keys())
        # Semantic aliases for concepts that lemmatisation can't bridge
        semantic_aliases: dict[str, str] = {
            "vacation": "holiday",
            "staff": "employee",
            "personnel": "employee",
            "shoutout": "praise",
            "appreciation": "praise",
        }

        # Build the full set of tokens to check: raw tokens + their variants
        # + spaCy lemmas (which dynamically resolve verb forms like "praised" → "praise")
        check_tokens: set[str] = set()
        for token in tokens:
            check_tokens.update(_token_variants(token))
        if nlp_signals:
            for lemma in nlp_signals.get("lemma_terms", []):
                check_tokens.update(_token_variants(lemma))

        # Score every entity by how many signals point to it
        identity_scores: dict[str, int] = {}
        for check_token in check_tokens:
            # Direct match against entity name from config
            if check_token in entity_names:
                identity_scores[check_token] = identity_scores.get(check_token, 0) + 10
            # Alias match for synonyms
            if check_token in semantic_aliases:
                target = semantic_aliases[check_token]
                identity_scores[target] = identity_scores.get(target, 0) + 10

        if identity_scores:
            best_identity = max(identity_scores, key=identity_scores.get)
            logger.info(
                "[Interpreter] Dynamic identity scores: %s → winner: %s",
                identity_scores, best_identity,
            )
            return best_identity
        # ─────────────────────────────────────────────────────────────────────


        best_entity: Optional[str] = None
        best_score = 0

        for entity_type in settings.interpreter_file_map:
            records = self._load_entity_records(entity_type)
            if not records:
                continue

            signature = self._entity_signature_tokens(entity_type, records)
            signature_family = set()
            for token in signature:
                signature_family.update(_semantic_token_family(token))

            overlap = len(token_pool & signature_family)

            # STRATEGIC BOOST: Reward direct entity-name matches (e.g. "holiday" or "holidays")
            # We check the token variants of the entity name itself against the query pool.
            direct_overlap = 0
            for ent_token in _tokenize_text(entity_type):
                ent_family = _semantic_token_family(ent_token)
                if token_pool & ent_family:
                    direct_overlap += 1
            
            # Massive score multiplier for identity matches to discourage falling back
            # to semantic search when we have a structured entity match.
            score = (overlap * 1) + (direct_overlap * 20)

            if score > best_score:
                best_score = score
                best_entity = entity_type

        return best_entity if best_score > 0 else None

    def _extract_nlp_signals(self, question: str) -> dict[str, list[str]]:
        """
        Extract lightweight NLP signals, then enrich them with spaCy if available.

        This keeps the fast schema parser as the default path while allowing
        date/person/noun-chunk hints to rescue ambiguous aggregation queries.
        """
        signals = {
            "lemma_terms": [],
            "noun_chunk_terms": [],
            "person_terms": [],
            "date_terms": [],
            "month_terms": sorted(_calendar_month_terms(question)),
        }

        nlp = _load_optional_nlp_model()
        if nlp is None:
            return signals

        doc = nlp(question)
        signals["lemma_terms"] = [
            token.lemma_.lower()
            for token in doc
            if token.lemma_ and token.lemma_ != "-PRON-"
        ]
        try:
            signals["noun_chunk_terms"] = [
                _normalize_text(chunk.text)
                for chunk in doc.noun_chunks
                if _normalize_text(chunk.text)
            ]
        except Exception:
            signals["noun_chunk_terms"] = []

        for ent in doc.ents:
            ent_text = _normalize_text(ent.text)
            if not ent_text:
                continue
            if ent.label_ == "PERSON":
                signals["person_terms"].append(ent_text)
            elif ent.label_ == "DATE":
                signals["date_terms"].append(ent_text)
                signals["month_terms"].extend(sorted(_calendar_month_terms(ent.text)))

        # Deduplicate while preserving order.
        for key, values in signals.items():
            signals[key] = list(dict.fromkeys(values))

        return signals

    def _infer_operation(
        self,
        tokens: list[str],
        nlp_signals: Optional[dict[str, list[str]]] = None,
    ) -> Optional[str]:
        """Infer whether the user wants a count or a list."""
        token_set = set(tokens)
        if nlp_signals:
            token_set.update(nlp_signals.get("lemma_terms", []))
        count_tokens = set(settings.data_interpreter_count_tokens)
        list_tokens = set(settings.data_interpreter_list_tokens)

        if token_set & count_tokens:
            return "count"
        if token_set & list_tokens:
            return "list"
        return None

    def _extract_selective_terms(
        self,
        question_tokens: list[str],
        entity_type: str,
        records: list[dict[str, Any]],
        operation: str,
        nlp_signals: Optional[dict[str, list[str]]] = None,
    ) -> list[str]:
        """
        Keep only terms that are selective for the chosen dataset.

        This avoids treating broad tokens like "company" or "employee" as filters
        just because they survived generic stop-word removal.
        """
        ignored_tokens = set(settings.data_interpreter_universal_noise)
        ignored_tokens.update(_tokenize_text(entity_type))
        ignored_tokens.update(_tokenize_text(settings.interpreter_file_map.get(entity_type, "")))
        ignored_tokens.update(_tokenize_text(settings.company_name))
        for alias in settings.company_aliases:
            ignored_tokens.update(_tokenize_text(alias))

        if operation == "count":
            ignored_tokens.update(settings.data_interpreter_count_tokens)
        elif operation == "list":
            ignored_tokens.update(settings.data_interpreter_list_tokens)

        candidate_terms = [
            token for token in question_tokens
            if len(token) > 1 and token not in ignored_tokens
        ]
        if nlp_signals:
            enriched_terms = []
            for group in ("month_terms", "person_terms", "date_terms"):
                for value in nlp_signals.get(group, []):
                    enriched_terms.extend(
                        token for token in _tokenize_text(value)
                        if token not in ignored_tokens
                    )
            candidate_terms.extend(enriched_terms)
            # Deduplicate while preserving order
            candidate_terms = list(dict.fromkeys(candidate_terms))
        if not candidate_terms:
            # Fallback: if no tokens were selective but we have specific NLP signals
            # (like a Month), use those instead of returning an empty list.
            if nlp_signals and nlp_signals.get("month_terms"):
                return nlp_signals["month_terms"]
            return []

        sample_records = records[: settings.data_interpreter_schema_sample_size] or records
        selective_terms: list[str] = []
        max_ratio = settings.data_interpreter_selective_term_max_ratio

        for token in candidate_terms:
            ratio = sum(
                token in _normalize_text(record)
                for record in sample_records
            ) / max(len(sample_records), 1)
            if ratio == 0 or ratio <= max_ratio:
                selective_terms.append(token)

        # Allow department aliases to expand after selectivity pruning.
        joined = " ".join(selective_terms).strip()
        alias = settings.department_aliases.get(joined.lower())
        if alias:
            return _tokenize_text(alias)
        return selective_terms

    def plan(self, question: str) -> Optional[StructuredQuery]:
        """Parse a question into a deterministic structured query plan."""
        normalized_question = _normalize_text(question.rstrip("?"))
        question_tokens = _tokenize_text(normalized_question)
        if not question_tokens:
            return None

        nlp_signals = self._extract_nlp_signals(question)

        operation = self._infer_operation(question_tokens, nlp_signals=nlp_signals)
        if not operation:
            return None

        entity_type = self._infer_entity(question_tokens, nlp_signals=nlp_signals)
        if not entity_type:
            return None

        records = self._load_entity_records(entity_type)
        if not records:
            return None

        criteria_terms = self._extract_selective_terms(
            question_tokens=question_tokens,
            entity_type=entity_type,
            records=records,
            operation=operation,
            nlp_signals=nlp_signals,
        )

        return StructuredQuery(
            operation=operation,
            entity_type=entity_type,
            criteria_terms=criteria_terms,
            raw_question=question,
        )

    def query(self, entity_type: str, criteria: str | list[str]) -> Optional[Dict[str, Any]]:
        """Execute a count/list style query against entity records."""
        records = self._load_entity_records(entity_type)
        if not records:
            return None

        results = []
        seen_names = set()
        file_name = settings.interpreter_file_map.get(entity_type, "unknown.json")

        for record in records:
            if criteria == "__all__":
                match_found = True
            else:
                match_found = _criteria_matches_record(record, criteria)

            if not match_found:
                continue

            name = (
                record.get("Name")
                or record.get("name")
                or record.get("Project name")
                or record.get("project_name")
                or record.get("policy_name")
                or record.get("Holiday")
                or record.get("source_file", "").replace(".pdf", "").replace(".json", "").replace("_", " ").title()
                or Path(file_name).stem.replace("_", " ").title()
            )
            # Prevent aggressive deduplication: if name already seen, suffix it
            # so that employees with the same name are all counted.
            if name in seen_names:
                name = f"{name} (Record {len(results)+1})"

            if name:
                results.append({"name": name, "file": file_name})
                seen_names.add(name)

        if not results:
            return None

        criteria_text = criteria if isinstance(criteria, str) else " ".join(criteria)
        return {
            "count": len(results),
            "matches": results,
            "criteria": criteria_text,
            "entity": entity_type,
        }

    def interpret(self, question: str) -> Optional[Dict[str, Any]]:
        """Build a plan from the question and execute it if it looks structured."""
        plan = self.plan(question)
        if not plan:
            return None

        result = self.query(plan.entity_type, plan.criteria if plan.criteria != "__all__" else "__all__")
        if result:
            result["operation"] = plan.operation
            result["criteria_terms"] = plan.criteria_terms
            return result

        # The plan was valid (entity + operation recognized) but zero records
        # matched the criteria. Return a structured "0 found" response instead
        # of None, which would wastefully fall through to HybridRAG.
        return {
            "count": 0,
            "matches": [],
            "criteria": plan.criteria,
            "entity": plan.entity_type,
            "operation": plan.operation,
            "criteria_terms": plan.criteria_terms,
        }
