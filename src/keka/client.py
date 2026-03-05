"""Async HTTP client for the Employee API with skip/limit pagination."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import settings
from src.keka.models import Employee, PaginatedResponse

logger = logging.getLogger(__name__)


class KekaAPIError(Exception):
    """Raised on non-2xx API responses."""


class KekaClient:
    """Async client for the Employee API.

    Handles paginated fetching of employee records via skip/limit.
    No authentication required for the local API.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.employee_api_base_url).rstrip("/")
        self._http: httpx.AsyncClient | None = None

    # ── Lifecycle ────────────────────────────────────────

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── Low-level request ────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a request to the API."""
        http = await self._get_http()
        url = f"{self.base_url}/{path.lstrip('/')}"

        resp = await http.request(method, url, params=params)

        if resp.status_code >= 400:
            raise KekaAPIError(
                f"API error {resp.status_code} on {method} {path}: {resp.text}"
            )

        return resp.json()

    # ── Public API: Employees ────────────────────────────

    async def get_employees(
        self, skip: int = 0, limit: int = 100
    ) -> list[Employee]:
        """Fetch a single page of employees."""
        raw = await self._request(
            "GET",
            "/employees",
            params={"skip": skip, "limit": limit},
        )
        paginated = PaginatedResponse.model_validate(raw)
        return [Employee.model_validate(r) for r in paginated.data]

    async def get_all_employees(self, page_size: int = 100) -> list[Employee]:
        """Fetch every employee across all pages."""
        all_records: list[dict[str, Any]] = []
        skip = 0
        limit = page_size

        while True:
            raw = await self._request(
                "GET",
                "/employees",
                params={"skip": skip, "limit": limit},
            )
            paginated = PaginatedResponse.model_validate(raw)
            all_records.extend(paginated.data)

            logger.info(
                "Fetched %d/%d employees (skip=%d)",
                len(all_records),
                paginated.total,
                skip,
            )

            if skip + limit >= paginated.total:
                break
            skip += limit

        return [Employee.model_validate(r) for r in all_records]
