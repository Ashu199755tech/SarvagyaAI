"""Pydantic models for the Employee API responses."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


# ── Employee Model ───────────────────────────────────────────


class Employee(BaseModel):
    """Flat employee record matching the API at /employees."""

    employee_id: str = ""
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    department: str = ""
    designation: str = ""
    joining_date: date | None = None
    salary: float = 0
    project_name: str | None = None
    project_start_date: date | None = None
    project_end_date: date | None = None
    client_name: str | None = None


# ── API Pagination Wrapper ───────────────────────────────────


class PaginatedResponse(BaseModel):
    """Wrapper for paginated API responses (skip/limit style)."""

    total: int = 0
    skip: int = 0
    limit: int = 100
    data: list[dict] = Field(default_factory=list)
