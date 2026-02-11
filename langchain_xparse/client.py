"""xParse Pipeline API client: auth, request/response handling, sync and async."""

from __future__ import annotations

import json
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.textin.com"
PIPELINE_PATH = "/api/xparse/pipeline"
DEFAULT_STAGES = [{"type": "parse", "config": {"provider": "textin"}}]


class XParseAPIError(Exception):
    """Raised when the Pipeline API returns code != 200."""

    def __init__(self, code: int, message: str, *args: Any, **kwargs: Any) -> None:
        self.code = code
        self.message = message
        super().__init__(code, message, *args, **kwargs)


class PipelineClient:
    """Client for xParse Pipeline API (sync and async)."""

    def __init__(
        self,
        app_id: str,
        secret_code: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.app_id = app_id
        self.secret_code = secret_code
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "x-ti-app-id": app_id,
            "x-ti-secret-code": secret_code,
        }

    def _url(self) -> str:
        return f"{self.base_url}{PIPELINE_PATH}"

    def _parse_response(self, response: httpx.Response) -> list[dict[str, Any]]:
        try:
            data = response.json()
        except Exception as e:
            raise XParseAPIError(
                response.status_code,
                f"Invalid JSON response: {e}",
            ) from e
        code = data.get("code", response.status_code)
        msg = data.get("message", "")
        if code != 200:
            raise XParseAPIError(code, msg or f"HTTP {response.status_code}")
        # API may return elements at top level or nested in 'data' field
        elements = data.get("elements")
        if elements is None and "data" in data and isinstance(data["data"], dict):
            elements = data["data"].get("elements")
        return elements or []

    def run_pipeline(
        self,
        file_content: bytes,
        filename: str,
        stages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the pipeline (sync). Returns list of elements."""
        stages = stages or DEFAULT_STAGES
        url = self._url()
        files = {"file": (filename, file_content)}
        data = {"stages": json.dumps(stages)}
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=self._headers, files=files, data=data)
        resp.raise_for_status()
        return self._parse_response(resp)

    async def arun_pipeline(
        self,
        file_content: bytes,
        filename: str,
        stages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the pipeline (async). Returns list of elements."""
        stages = stages or DEFAULT_STAGES
        url = self._url()
        files = {"file": (filename, file_content)}
        data = {"stages": json.dumps(stages)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=self._headers, files=files, data=data)
        resp.raise_for_status()
        return self._parse_response(resp)
