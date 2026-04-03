"""xParse document loader: XParseLoader and _SingleDocumentLoader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_xparse.client import DEFAULT_CONFIG, ParseClient


class _SingleDocumentLoader(BaseLoader):
    """Loads a single file via xParse Parse API into LangChain Documents."""

    def __init__(
        self,
        *,
        client: ParseClient,
        file_path: str | Path | None = None,
        file: Any = None,
        config: dict[str, Any],
        post_processors: list[Callable[[str], str]] | None = None,
        metadata_filename: str | None = None,
    ) -> None:
        self.client = client
        self.file_path = str(file_path) if isinstance(file_path, Path) else file_path
        self.file = file
        self.config = config
        self.post_processors = post_processors or []
        self.metadata_filename = metadata_filename

    def _file_content(self) -> bytes:
        if self.file is not None:
            return self.file.read()
        if self.file_path:
            with open(self.file_path, "rb") as f:
                return f.read()
        raise ValueError("file or file_path must be defined.")

    def _filename(self) -> str:
        if self.file_path:
            return Path(self.file_path).name
        if self.metadata_filename:
            return self.metadata_filename
        return "unknown"

    def _source(self) -> str:
        return self.file_path or self.metadata_filename or ""

    def _element_to_document(self, element: dict[str, Any], file_metadata: dict[str, Any]) -> Document:
        text = element.get("text") or ""
        for fn in self.post_processors:
            text = fn(text)
        meta: dict[str, Any] = {
            "source": self._source(),
            "category": element.get("type"),
            "element_id": element.get("element_id"),
            "filename": file_metadata.get("filename", self._filename()),
        }
        # Add page_number if available
        if "page_number" in element:
            meta["page_number"] = element["page_number"]
        # Add element metadata
        if element.get("metadata"):
            meta.update(element["metadata"])
        return Document(page_content=text, metadata=meta)

    def lazy_load(self) -> Iterator[Document]:
        content = self._file_content()
        filename = self._filename()
        result = self.client.parse(content, filename, self.config)
        elements = result.get("elements", [])
        file_metadata = result.get("metadata", {})
        for el in elements:
            yield self._element_to_document(el, file_metadata)

    async def alazy_load(self) -> AsyncIterator[Document]:
        content = self._file_content()
        filename = self._filename()
        result = await self.client.aparse(content, filename, self.config)
        elements = result.get("elements", [])
        file_metadata = result.get("metadata", {})
        for el in elements:
            yield self._element_to_document(el, file_metadata)


class XParseLoader(BaseLoader):
    """Load documents via xParse Parse API for intelligent document parsing.

    Setup:
        Set environment variables or pass credentials:

        ```bash
        export XPARSE_APP_ID="your-app-id"
        export XPARSE_SECRET_CODE="your-secret-code"
        ```

    Example:
        ```python
        from langchain_xparse import XParseLoader

        # Basic usage (parse only)
        loader = XParseLoader(file_path="example.pdf")
        docs = loader.load()
        print(docs[0].page_content[:200])
        print(docs[0].metadata)  # source, category, element_id, filename, page_number

        # With custom config
        loader = XParseLoader(
            file_path="doc.pdf",
            config={
                "document": {"password": "pdf-password"},
                "capabilities": {
                    "include_hierarchy": True,
                    "include_table_structure": True,
                    "title_tree": True,
                },
                "scope": {"page_range": "1-10"},
            },
        )
        for doc in loader.lazy_load():
            print(doc.page_content[:100], doc.metadata)
        ```
    """

    def __init__(
        self,
        file_path: str | Path | list[str] | list[Path] | None = None,
        *,
        file: Any = None,
        app_id: str | None = None,
        secret_code: str | None = None,
        base_url: str | None = None,
        config: dict[str, Any] | None = None,
        post_processors: list[Callable[[str], str]] | None = None,
        metadata_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        if file_path is not None and file is not None:
            raise ValueError("file_path and file cannot be defined simultaneously.")
        if file is not None and metadata_filename is None:
            raise ValueError(
                "When using file (file-like), metadata_filename must be specified."
            )

        self._app_id = app_id or os.getenv("XPARSE_APP_ID") or ""
        self._secret_code = secret_code or os.getenv("XPARSE_SECRET_CODE") or ""
        self._base_url = base_url or "https://api.textin.com"
        self.file_path = file_path
        self.file = file
        self.post_processors = post_processors or []
        self.metadata_filename = metadata_filename
        self._config = config or DEFAULT_CONFIG

        self._client = ParseClient(
            app_id=self._app_id,
            secret_code=self._secret_code,
            base_url=self._base_url,
        )

    def _load_one(
        self,
        f_path: str | Path | None = None,
        f: Any = None,
        meta_filename: str | None = None,
    ) -> Iterator[Document]:
        single = _SingleDocumentLoader(
            client=self._client,
            file_path=str(f_path) if f_path is not None else None,
            file=f,
            config=self._config,
            post_processors=self.post_processors,
            metadata_filename=meta_filename,
        )
        yield from single.lazy_load()

    async def _aload_one(
        self,
        f_path: str | Path | None = None,
        f: Any = None,
        meta_filename: str | None = None,
    ) -> AsyncIterator[Document]:
        single = _SingleDocumentLoader(
            client=self._client,
            file_path=str(f_path) if f_path is not None else None,
            file=f,
            config=self._config,
            post_processors=self.post_processors,
            metadata_filename=meta_filename,
        )
        async for doc in single.alazy_load():
            yield doc

    def lazy_load(self) -> Iterator[Document]:
        if isinstance(self.file, list):
            for f in self.file:
                yield from self._load_one(f=f, meta_filename=self.metadata_filename)
            return
        if isinstance(self.file_path, list):
            for p in self.file_path:
                yield from self._load_one(f_path=p)
            return
        yield from self._load_one(
            f_path=self.file_path,
            f=self.file,
            meta_filename=self.metadata_filename,
        )

    async def alazy_load(self) -> AsyncIterator[Document]:
        if isinstance(self.file, list):
            for f in self.file:
                async for doc in self._aload_one(f=f, meta_filename=self.metadata_filename):
                    yield doc
            return
        if isinstance(self.file_path, list):
            for p in self.file_path:
                async for doc in self._aload_one(f_path=p):
                    yield doc
            return
        async for doc in self._aload_one(
            f_path=self.file_path,
            f=self.file,
            meta_filename=self.metadata_filename,
        ):
            yield doc
