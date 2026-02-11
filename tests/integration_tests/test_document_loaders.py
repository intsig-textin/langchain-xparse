"""Integration tests: require XPARSE_APP_ID and XPARSE_SECRET_CODE."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from langchain_xparse import XParseLoader

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("XPARSE_APP_ID") or not os.getenv("XPARSE_SECRET_CODE"),
    reason="XPARSE_APP_ID and XPARSE_SECRET_CODE required",
)
def test_loader_load_real_api() -> None:
    """Call real Pipeline API if credentials are set. Use a small PDF if available."""
    example_dir = Path(__file__).parent.parent.parent / "example_docs"
    pdf_path = example_dir / "layout-parser-paper.pdf"
    if not pdf_path.exists():
        pytest.skip("example_docs/layout-parser-paper.pdf not found")
    loader = XParseLoader(file_path=str(pdf_path))
    docs = loader.load()
    assert len(docs) >= 1
    for doc in docs:
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")
        if doc.page_content:
            assert "source" in doc.metadata or "filename" in doc.metadata
