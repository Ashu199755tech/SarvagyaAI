"""Tests for the Employee API client — all HTTP calls are mocked."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from src.keka.client import KekaClient, KekaAPIError


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def keka_client() -> KekaClient:
    return KekaClient(base_url="http://test-api:8000")


# ── Employee Tests ───────────────────────────────────────

EMPLOYEE_PAGE = {
    "total": 2,
    "skip": 0,
    "limit": 100,
    "data": [
        {
            "employee_id": "EMP0001",
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice@example.com",
            "department": "Engineering",
            "designation": "Senior Developer",
            "joining_date": "2023-01-15",
            "salary": 1200000.00,
            "project_name": "Phoenix",
            "project_start_date": "2024-01-01",
            "project_end_date": "2024-12-31",
            "client_name": "Accenture",
        },
        {
            "employee_id": "EMP0002",
            "first_name": "Bob",
            "last_name": "Jones",
            "email": "bob@example.com",
            "department": "Sales",
            "designation": "Manager",
            "joining_date": "2022-06-01",
            "salary": 900000.00,
            "project_name": "Atlas",
            "project_start_date": "2023-03-01",
            "project_end_date": "2023-12-31",
            "client_name": "TCS",
        },
    ],
}


@respx.mock
@pytest.mark.asyncio
async def test_get_employees(keka_client: KekaClient):
    """Fetches employees, deserialises into Employee models."""
    respx.get("http://test-api:8000/employees").mock(
        return_value=Response(200, json=EMPLOYEE_PAGE)
    )

    employees = await keka_client.get_employees()
    assert len(employees) == 2
    assert employees[0].first_name == "Alice"
    assert employees[0].last_name == "Smith"
    assert employees[0].department == "Engineering"
    assert employees[1].employee_id == "EMP0002"
    assert employees[1].salary == 900000.00


@respx.mock
@pytest.mark.asyncio
async def test_get_all_employees_pagination(keka_client: KekaClient):
    """Auto-pagination fetches multiple pages via skip/limit."""
    page1 = {
        "total": 2,
        "skip": 0,
        "limit": 1,
        "data": [
            {
                "employee_id": "EMP0001",
                "first_name": "Alice",
                "last_name": "Smith",
                "email": "alice@example.com",
                "department": "Engineering",
                "designation": "Lead",
                "joining_date": "2023-01-15",
                "salary": 1000000.00,
                "project_name": "Phoenix",
                "project_start_date": "2024-01-01",
                "project_end_date": "2024-12-31",
                "client_name": "Accenture",
            }
        ],
    }
    page2 = {
        "total": 2,
        "skip": 1,
        "limit": 1,
        "data": [
            {
                "employee_id": "EMP0002",
                "first_name": "Bob",
                "last_name": "Jones",
                "email": "bob@example.com",
                "department": "Sales",
                "designation": "Manager",
                "joining_date": "2022-06-01",
                "salary": 900000.00,
                "project_name": "Atlas",
                "project_start_date": "2023-03-01",
                "project_end_date": "2023-12-31",
                "client_name": "TCS",
            }
        ],
    }

    route = respx.get("http://test-api:8000/employees")
    route.side_effect = [
        Response(200, json=page1),
        Response(200, json=page2),
    ]

    employees = await keka_client.get_all_employees(page_size=1)
    assert len(employees) == 2
    assert employees[0].first_name == "Alice"
    assert employees[1].first_name == "Bob"


# ── Error Handling ───────────────────────────────────────


@respx.mock
@pytest.mark.asyncio
async def test_api_error_raises(keka_client: KekaClient):
    """Non-2xx API response raises KekaAPIError."""
    respx.get("http://test-api:8000/employees").mock(
        return_value=Response(500, text="Internal Server Error")
    )

    with pytest.raises(KekaAPIError):
        await keka_client.get_employees()
