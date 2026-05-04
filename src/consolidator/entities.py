"""Entity dataclasses for the consolidation pipeline.

These represent the extracted, typed entities from all data sources.
Each entity carries a unique ID for cross-referencing in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ---------------------------------------------------------------------------
# Entity Types
# ---------------------------------------------------------------------------

@dataclass
class EmployeeEntity:
    """An employee extracted from the Keka HR API."""

    id: str                           # e.g. "EMP0003"
    name: str                         # e.g. "Jennifer Hernandez"
    email: str = ""
    department: str = ""
    designation: str = ""
    salary: float = 0.0
    joining_date: date | None = None
    project_name: str = ""
    project_start_date: date | None = None
    project_end_date: date | None = None
    client_name: str = ""
    location: str = ""
    phone: str = ""


@dataclass
class ProjectEntity:
    """A project extracted from case studies or employee assignments."""

    id: str                            # e.g. "PRJ:Main Compliance"
    name: str
    brief: str = ""
    description: str = ""
    industry: str = ""
    services: list[str] = field(default_factory=list)
    about_client: str = ""
    the_story: str = ""
    technologies_used: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)
    the_solution: list[str] = field(default_factory=list)
    the_outcome: str = ""
    employee_ids: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


@dataclass
class PolicyEntity:
    """A company policy extracted from PDF documents."""

    id: str                            # e.g. "POL:Leave Policy"
    name: str                          # e.g. "Leave Policy"
    filename: str = ""
    category: str = ""
    summary: str = ""
    key_rules: list[str] = field(default_factory=list)
    applicable_to: str = "All employees"


@dataclass
class ToolEntity:
    """A technology or tool mentioned in projects or case studies."""

    id: str                            # e.g. "TOOL:GPU"
    name: str
    project_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    """A directed relationship between two entities."""

    from_id: str
    to_id: str
    relation_type: str                 # e.g. "works_on", "uses_tech", "governed_by"
    description: str = ""


# ---------------------------------------------------------------------------
# Knowledge Base — the container for everything
# ---------------------------------------------------------------------------

@dataclass
class HolidayEntity:
    """A holiday entry from the company holiday list."""

    id: str                            # e.g. "HOL_2026_11"
    name: str
    date: date | None = None
    day: str = ""
    month: str = ""
    type: str = ""                     # e.g. "Public Holiday", "Floater Leave"


@dataclass
class KnowledgeBase:
    """Container holding all extracted entities and their relationships."""

    employees: dict[str, EmployeeEntity] = field(default_factory=dict)
    projects: dict[str, ProjectEntity] = field(default_factory=dict)
    policies: dict[str, PolicyEntity] = field(default_factory=dict)
    holidays: dict[str, HolidayEntity] = field(default_factory=dict)
    tools: dict[str, ToolEntity] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "employees": len(self.employees),
            "projects": len(self.projects),
            "policies": len(self.policies),
            "holidays": len(self.holidays),
            "tools": len(self.tools),
            "relationships": len(self.relationships),
        }
