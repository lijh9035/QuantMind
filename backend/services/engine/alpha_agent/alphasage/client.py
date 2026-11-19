"""Authenticated HTTP client for the isolated AlphaSAGE GPU worker."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx


class AlphaSageWorkerError(RuntimeError):
    """The GPU worker rejected a request or returned an invalid receipt."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AlphaSageWorkerNotFound(AlphaSageWorkerError):
    """The GPU worker has no durable record for this task ID."""


class AlphaSageWorkerUnavailable(AlphaSageWorkerError):
    """The GPU worker cannot currently accept work."""


_TASK_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MAX_RESULT_BYTES = 12 * 1024 * 1024


class AlphaSageClient:
    """Small fixed-route client; callers never provide worker URLs or file paths."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("QM_ALPHASAGE_URL", "http://alphasage-gpu:8090")
        ).rstrip("/")
        self.token = token or os.getenv("ALPHASAGE_INTERNAL_TOKEN", "").strip()
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise AlphaSageWorkerUnavailable(
                "AlphaSAGE GPU worker token is not configured"
            )
        return {"Authorization": f"Bearer {self.token}"}

    @staticmethod
    def _task_path(task_id: str) -> str:
        if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
            raise ValueError("Invalid AlphaSAGE task ID")
        return task_id

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout or self.timeout_seconds,
                    connect=min(5.0, timeout or self.timeout_seconds),
                ),
                trust_env=False,
            ) as client:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    json=json_body,
                )
        except (httpx.HTTPError, OSError) as exc:
            raise AlphaSageWorkerUnavailable(
                f"AlphaSAGE GPU worker is unreachable ({type(exc).__name__})"
            ) from exc
        if response.status_code >= 500:
            raise AlphaSageWorkerUnavailable(
                f"AlphaSAGE GPU worker returned HTTP {response.status_code}",
                status_code=response.status_code,
            )
        if response.status_code == 404:
            raise AlphaSageWorkerNotFound(
                "AlphaSAGE GPU worker has no record for this task",
                status_code=404,
            )
        if response.is_error:
            raise AlphaSageWorkerError(
                f"AlphaSAGE GPU worker rejected request with HTTP {response.status_code}",
                status_code=response.status_code,
            )
        if len(response.content) > _MAX_RESULT_BYTES:
            raise AlphaSageWorkerError("AlphaSAGE GPU worker response is too large")
        try:
            payload = response.json()
        except ValueError as exc:
            raise AlphaSageWorkerError(
                "AlphaSAGE GPU worker returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise AlphaSageWorkerError(
                "AlphaSAGE GPU worker response must be an object"
            )
        return payload

    async def health(self) -> dict[str, Any]:
        payload = await self._request("GET", "/health", timeout=8.0)
        status = str(payload.get("status") or "").lower()
        if status not in {"ok", "ready", "healthy"}:
            raise AlphaSageWorkerUnavailable(
                "AlphaSAGE GPU worker is not ready"
            )
        if payload.get("gpu_available") is not True:
            raise AlphaSageWorkerUnavailable(
                "AlphaSAGE GPU worker has no ready CUDA device"
            )
        return payload

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = self._task_path(payload.get("task_id"))
        fingerprint = payload.get("execution_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("AlphaSAGE worker requests need an execution fingerprint")
        return await self._request("POST", "/jobs", json_body={**payload, "task_id": task_id})

    async def status(self, task_id: str) -> dict[str, Any]:
        safe_id = self._task_path(task_id)
        return await self._request("GET", f"/jobs/{safe_id}")

    async def result(self, task_id: str) -> dict[str, Any]:
        safe_id = self._task_path(task_id)
        return await self._request(
            "GET", f"/jobs/{safe_id}/result", timeout=60.0
        )

    async def cancel(self, task_id: str) -> dict[str, Any]:
        safe_id = self._task_path(task_id)
        return await self._request("DELETE", f"/jobs/{safe_id}", timeout=30.0)
