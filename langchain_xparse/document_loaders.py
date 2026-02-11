"""xParse Pipeline document loader: XParseLoader and _SingleDocumentLoader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_xparse.client import DEFAULT_STAGES, PipelineClient


def _build_stages(
    *,
    parse_provider: str = "textin",
    chunk_strategy: str | None = None,
    chunk_max_characters: int | None = None,
    chunk_overlap: int | None = None,
    chunk_include_orig_elements: bool = False,
    chunk_new_after_n_chars: int | None = None,
    embed_provider: str | None = None,
    embed_model_name: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Build Pipeline stages from convenience parameters (parse/chunk/embed only)."""
    stages: list[dict[str, Any]] = [
        {"type": "parse", "config": {"provider": parse_provider}}
    ]
    if chunk_strategy is not None:
        chunk_config: dict[str, Any] = {"strategy": chunk_strategy}
        if chunk_max_characters is not None:
            chunk_config["max_characters"] = chunk_max_characters
        if chunk_overlap is not None:
            chunk_config["overlap"] = chunk_overlap
        if chunk_include_orig_elements:
            chunk_config["include_orig_elements"] = True
        if chunk_new_after_n_chars is not None:
            chunk_config["new_after_n_chars"] = chunk_new_after_n_chars
        stages.append({"type": "chunk", "config": chunk_config})
    if embed_provider is not None and embed_model_name is not None:
        stages.append(
            {
                "type": "embed",
                "config": {
                    "provider": embed_provider,
                    "model_name": embed_model_name,
                },
            }
        )
    return stages


class _SingleDocumentLoader(BaseLoader):
    """Loads a single file via xParse Pipeline API into LangChain Documents."""

    def __init__(
        self,
        *,
        client: PipelineClient,
        file_path: str | Path | None = None,
        file: Any = None,
        stages: list[dict[str, Any]],
        post_processors: list[Callable[[str], str]] | None = None,
        metadata_filename: str | None = None,
    ) -> None:
        self.client = client
        self.file_path = str(file_path) if isinstance(file_path, Path) else file_path
        self.file = file
        self.stages = stages
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

    def _element_to_document(self, element: dict[str, Any]) -> Document:
        text = element.get("text") or ""
        for fn in self.post_processors:
            text = fn(text)
        meta: dict[str, Any] = {
            "source": self._source(),
            "category": element.get("type"),
            "element_id": element.get("element_id"),
        }
        if element.get("metadata"):
            meta.update(element["metadata"])
        if element.get("embeddings") is not None:
            meta["embeddings"] = element["embeddings"]
        return Document(page_content=text, metadata=meta)

    def lazy_load(self) -> Iterator[Document]:
        content = self._file_content()
        filename = self._filename()
        elements = self.client.run_pipeline(content, filename, self.stages)
        for el in elements:
            yield self._element_to_document(el)

    async def alazy_load(self) -> AsyncIterator[Document]:
        content = self._file_content()
        filename = self._filename()
        elements = await self.client.arun_pipeline(content, filename, self.stages)
        for el in elements:
            yield self._element_to_document(el)


class XParseLoader(BaseLoader):
    """Load documents via xParse Pipeline API (parse/chunk/embed; no extract).

    Setup:
        Set environment variables or pass credentials:

        ```bash
        export XPARSE_APP_ID="your-app-id"
        export XPARSE_SECRET_CODE="your-secret-code"
        ```

    Example:
        ```python
        from langchain_xparse import XParseLoader

        loader = XParseLoader(file_path="example.pdf")
        docs = loader.load()

        # With convenience params (parse + chunk):
        loader = XParseLoader(
            file_path="doc.pdf",
            parse_provider="textin",
            chunk_strategy="by_title",
            chunk_max_characters=500,
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
        stages: list[dict[str, Any]] | None = None,
        post_processors: list[Callable[[str], str]] | None = None,
        metadata_filename: str | None = None,
        parse_provider: str = "textin",
        chunk_strategy: str | None = None,
        chunk_max_characters: int | None = None,
        chunk_overlap: int | None = None,
        chunk_include_orig_elements: bool = False,
        chunk_new_after_n_chars: int | None = None,
        embed_provider: str | None = None,
        embed_model_name: str | None = None,
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

        if stages is not None:
            self._stages = stages
        else:
            self._stages = _build_stages(
                parse_provider=parse_provider,
                chunk_strategy=chunk_strategy,
                chunk_max_characters=chunk_max_characters,
                chunk_overlap=chunk_overlap,
                chunk_include_orig_elements=chunk_include_orig_elements,
                chunk_new_after_n_chars=chunk_new_after_n_chars,
                embed_provider=embed_provider,
                embed_model_name=embed_model_name,
                **kwargs,
            )

        self._client = PipelineClient(
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
            stages=self._stages,
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
            stages=self._stages,
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
