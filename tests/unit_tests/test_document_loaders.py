"""Unit tests for XParseLoader and _SingleDocumentLoader."""

from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from langchain_xparse.client import ParseClient
from langchain_xparse.document_loaders import XParseLoader, _SingleDocumentLoader


# --- XParseLoader init ---


def test_loader_initializes_with_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPARSE_APP_ID", raising=False)
    monkeypatch.delenv("XPARSE_SECRET_CODE", raising=False)
    loader = XParseLoader(file_path="dummy.pdf")
    assert loader.file_path == "dummy.pdf"
    assert loader.file is None
    assert loader._app_id == ""
    assert loader._secret_code == ""
    assert "capabilities" in loader._config


def test_loader_initializes_with_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPARSE_APP_ID", "FAKE_APP_ID")
    monkeypatch.setenv("XPARSE_SECRET_CODE", "FAKE_SECRET")
    loader = XParseLoader(file_path="dummy.pdf")
    assert loader._app_id == "FAKE_APP_ID"
    assert loader._secret_code == "FAKE_SECRET"


def test_loader_raises_when_file_path_and_file_both_set() -> None:
    with pytest.raises(ValueError) as e:
        XParseLoader(file_path="a.pdf", file=Mock())
    assert "file_path and file" in str(e.value)


def test_loader_raises_when_file_without_metadata_filename() -> None:
    with pytest.raises(ValueError) as e:
        XParseLoader(file=Mock(), metadata_filename=None)
    assert "metadata_filename" in str(e.value)


def test_loader_uses_config_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPARSE_APP_ID", "id")
    monkeypatch.setenv("XPARSE_SECRET_CODE", "secret")
    custom = {
        "capabilities": {
            "include_hierarchy": True,
            "include_table_structure": True,
        }
    }
    loader = XParseLoader(file_path="a.pdf", config=custom)
    assert loader._config == custom


# --- _SingleDocumentLoader _file_content ---


def test_single_loader_gets_content_from_file() -> None:
    mock_file = Mock()
    mock_file.read.return_value = b"file content"
    client = Mock(spec=ParseClient)
    loader = _SingleDocumentLoader(
        client=client,
        file=mock_file,
        config={"capabilities": {"include_hierarchy": True}},
        metadata_filename="fake.txt",
    )
    assert loader._file_content() == b"file content"
    mock_file.read.assert_called_once()


@patch("builtins.open", create=True, new_callable=MagicMock)
def test_single_loader_gets_content_from_file_path(mock_open_fn: MagicMock) -> None:
    mock_open_fn.return_value.__enter__.return_value.read.return_value = b"path content"
    client = Mock(spec=ParseClient)
    loader = _SingleDocumentLoader(
        client=client,
        file_path="dummy.pdf",
        config={"capabilities": {"include_hierarchy": True}},
    )
    assert loader._file_content() == b"path content"
    mock_open_fn.assert_called_once_with("dummy.pdf", "rb")


def test_single_loader_raises_without_file_or_path() -> None:
    client = Mock(spec=ParseClient)
    loader = _SingleDocumentLoader(
        client=client,
        config={"capabilities": {"include_hierarchy": True}},
    )
    with pytest.raises(ValueError) as e:
        loader._file_content()
    assert "file or file_path" in str(e.value)


# --- lazy_load Document output ---


@pytest.fixture
def sample_elements() -> list[dict[str, Any]]:
    return [
        {
            "element_id": "elem_1",
            "type": "NarrativeText",
            "metadata": {"filename": "test.pdf", "page_number": 1},
            "text": "First paragraph.",
        },
        {
            "element_id": "elem_2",
            "type": "Title",
            "metadata": {"filename": "test.pdf", "page_number": 2},
            "text": "Second.",
        },
    ]


@patch("builtins.open", create=True, new_callable=MagicMock)
def test_lazy_load_yields_documents_with_metadata(
    mock_open_fn: MagicMock,
    sample_elements: list[dict[str, Any]],
) -> None:
    mock_open_fn.return_value.__enter__.return_value.read.return_value = b"path content"
    client = Mock(spec=ParseClient)
    client.parse.return_value = {
        "elements": sample_elements,
        "metadata": {"filename": "test.pdf", "page_count": 2},
    }
    loader = _SingleDocumentLoader(
        client=client,
        file_path="/path/to/doc.pdf",
        config={"capabilities": {"include_hierarchy": True}},
    )
    docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].page_content == "First paragraph."
    assert docs[0].metadata["source"] == "/path/to/doc.pdf"
    assert docs[0].metadata["category"] == "NarrativeText"
    assert docs[0].metadata["element_id"] == "elem_1"
    assert docs[0].metadata["filename"] == "test.pdf"
    assert docs[0].metadata["page_number"] == 1
    assert docs[1].page_content == "Second."
    assert docs[1].metadata["category"] == "Title"
    client.parse.assert_called_once()
    call_args = client.parse.call_args
    assert call_args[0][0] == b"path content"
    assert call_args[0][1] == "doc.pdf"
    assert call_args[0][2] == {"capabilities": {"include_hierarchy": True}}


def test_lazy_load_applies_post_processors(sample_elements: list[dict[str, Any]]) -> None:
    def suffix(t: str) -> str:
        return t + "!"

    client = Mock(spec=ParseClient)
    client.parse.return_value = {
        "elements": sample_elements,
        "metadata": {"filename": "test.pdf"},
    }
    loader = _SingleDocumentLoader(
        client=client,
        file_path="doc.pdf",
        config={"capabilities": {"include_hierarchy": True}},
        post_processors=[suffix],
    )
    with patch("builtins.open", create=True, new_callable=MagicMock) as m:
        m.return_value.__enter__.return_value.read.return_value = b"x"
        docs = list(loader.lazy_load())
    assert docs[0].page_content == "First paragraph.!"
    assert docs[1].page_content == "Second.!"


@patch("builtins.open", create=True, new_callable=MagicMock)
def test_xparse_loader_lazy_load_single_file(
    mock_open: MagicMock,
    sample_elements: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_open.return_value.__enter__.return_value.read.return_value = b"x"
    monkeypatch.setenv("XPARSE_APP_ID", "id")
    monkeypatch.setenv("XPARSE_SECRET_CODE", "secret")
    mock_result = {"elements": sample_elements, "metadata": {"filename": "test.pdf"}}
    with patch.object(ParseClient, "parse", return_value=mock_result):
        loader = XParseLoader(file_path="single.pdf")
        docs = list(loader.lazy_load())
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "single.pdf"


@patch("builtins.open", create=True, new_callable=MagicMock)
def test_xparse_loader_lazy_load_multiple_files(
    mock_open: MagicMock,
    sample_elements: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_open.return_value.__enter__.return_value.read.return_value = b"x"
    monkeypatch.setenv("XPARSE_APP_ID", "id")
    monkeypatch.setenv("XPARSE_SECRET_CODE", "secret")
    mock_result = {"elements": sample_elements, "metadata": {"filename": "test.pdf"}}
    with patch.object(ParseClient, "parse", return_value=mock_result):
        loader = XParseLoader(file_path=["a.pdf", "b.pdf"])
        docs = list(loader.lazy_load())
    assert len(docs) == 4  # 2 elements per file
    assert docs[0].metadata["source"] == "a.pdf"
    assert docs[2].metadata["source"] == "b.pdf"
