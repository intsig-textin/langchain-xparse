"""xParse Parse Sync API client: auth, request/response handling, sync and async."""

from __future__ import annotations

import json
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.textin.com"
PARSE_SYNC_PATH = "/api/v1/xparse/parse/sync"
DEFAULT_CONFIG: dict[str, Any] = {
    "capabilities": {
        "include_hierarchy": True,
    }
}


class XParseAPIError(Exception):
    """Raised when the Parse API returns code != 200."""

    def __init__(self, code: int, message: str, *args: Any, **kwargs: Any) -> None:
        self.code = code
        self.message = message
        super().__init__(code, message, *args, **kwargs)


class ParseClient:
    """Client for xParse Parse Sync API (sync and async)."""

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
        return f"{self.base_url}{PARSE_SYNC_PATH}"

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
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
        # Return the data object containing elements, markdown, etc.
        return data.get("data", {})

    def parse(
        self,
        file_content: bytes,
        filename: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute document parsing (sync). Returns parsed data with elements."""
        config = config or DEFAULT_CONFIG
        url = self._url()
        files = {"file": (filename, file_content)}
        data = {"config": json.dumps(config)}
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=self._headers, files=files, data=data)
        resp.raise_for_status()
        return self._parse_response(resp)

    async def aparse(
        self,
        file_content: bytes,
        filename: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute document parsing (async). Returns parsed data with elements."""
        config = config or DEFAULT_CONFIG
        url = self._url()
        files = {"file": (filename, file_content)}
        data = {"config": json.dumps(config)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=self._headers, files=files, data=data)
        resp.raise_for_status()
        return self._parse_response(resp)
