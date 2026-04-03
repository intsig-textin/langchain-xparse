"""Unit tests for ParseClient and XParseAPIError."""

from unittest.mock import Mock, patch

import httpx
import pytest

from langchain_xparse.client import (
    DEFAULT_BASE_URL,
    DEFAULT_CONFIG,
    PARSE_SYNC_PATH,
    ParseClient,
    XParseAPIError,
)


def test_xparse_api_error_attributes() -> None:
    err = XParseAPIError(40101, "auth empty")
    assert err.code == 40101
    assert err.message == "auth empty"


def test_client_default_url() -> None:
    c = ParseClient(app_id="a", secret_code="b")
    assert c.base_url == DEFAULT_BASE_URL
    assert c._url() == f"{DEFAULT_BASE_URL}{PARSE_SYNC_PATH}"
    assert c._headers["x-ti-app-id"] == "a"
    assert c._headers["x-ti-secret-code"] == "b"


def test_client_custom_base_url() -> None:
    c = ParseClient(app_id="a", secret_code="b", base_url="https://custom.example.com")
    assert c._url() == "https://custom.example.com/api/v1/xparse/parse/sync"


def test_parse_response_success_returns_data() -> None:
    c = ParseClient(app_id="a", secret_code="b")
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {
        "code": 200,
        "message": "success",
        "data": {
            "elements": [{"element_id": "1", "type": "Text", "metadata": {}, "text": "hi"}],
            "metadata": {"filename": "test.pdf"},
        },
    }
    out = c._parse_response(resp)
    assert "elements" in out
    assert len(out["elements"]) == 1
    assert out["elements"][0]["text"] == "hi"
    assert out["metadata"]["filename"] == "test.pdf"


def test_parse_response_non_200_raises() -> None:
    c = ParseClient(app_id="a", secret_code="b")
    resp = Mock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"code": 40102, "message": "invalid auth"}
    with pytest.raises(XParseAPIError) as exc_info:
        c._parse_response(resp)
    assert exc_info.value.code == 40102
    assert "invalid auth" in exc_info.value.message


def test_parse_response_empty_data_returns_dict() -> None:
    c = ParseClient(app_id="a", secret_code="b")
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {"code": 200, "message": "success"}
    out = c._parse_response(resp)
    assert out == {}


@patch("langchain_xparse.client.httpx.Client")
def test_parse_sends_multipart(mock_client_class: Mock) -> None:
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "code": 200,
        "message": "ok",
        "data": {"elements": [], "metadata": {}},
    }
    mock_resp.raise_for_status = Mock()
    mock_client = Mock()
    mock_client.post.return_value = mock_resp
    mock_client_class.return_value.__enter__.return_value = mock_client

    c = ParseClient(app_id="app", secret_code="secret")
    c.parse(b"file bytes", "doc.pdf", config=DEFAULT_CONFIG)

    mock_client.post.assert_called_once()
    call_kw = mock_client.post.call_args[1]
    assert call_kw["headers"]["x-ti-app-id"] == "app"
    assert call_kw["headers"]["x-ti-secret-code"] == "secret"
    assert call_kw["files"]["file"][0] == "doc.pdf"
    assert call_kw["files"]["file"][1] == b"file bytes"
    assert "config" in call_kw["data"]
    import json

    assert json.loads(call_kw["data"]["config"]) == DEFAULT_CONFIG
