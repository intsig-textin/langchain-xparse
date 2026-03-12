# langchain-xparse

LangChain integration with [xParse Pipeline API](https://docs.textin.com/api-reference/endpoint/pipeline) for document parsing, chunking and embedding. Supports parse / chunk / embed stages only (extract is not supported in this loader).

## Installation

From PyPI:

```bash
pip install langchain-xparse
```

Local editable install:

```bash
pip install -e .
```

## Configuration

Set your TextIn credentials (from [Textin Workspace](https://www.textin.com/console/dashboard/setting) ):

```bash
export XPARSE_APP_ID="your-app-id"
export XPARSE_SECRET_CODE="your-secret-code"
```

Or pass them when creating the loader:

```python
loader = XParseLoader(
    file_path="doc.pdf",
    app_id="your-app-id",
    secret_code="your-secret-code",
)
```

## Usage

### Basic (parse only)

```python
from langchain_xparse import XParseLoader

loader = XParseLoader(file_path="example.pdf")
docs = loader.load()
print(docs[0].page_content[:200])
print(docs[0].metadata)  # source, category, element_id, filename, page_number, ...
```

### Lazy load

```python
for doc in loader.lazy_load():
    # process(doc)
```

### Async

```python
async for doc in loader.alazy_load():
    # process(doc)
```

### Convenience params (parse + chunk, or parse + chunk + embed)

```python
loader = XParseLoader(
    file_path="doc.pdf",
    parse_provider="textin",
    chunk_strategy="by_title",
    chunk_max_characters=500,
    chunk_overlap=50,
)
# Or with embed:
loader = XParseLoader(
    file_path="doc.pdf",
    parse_provider="textin",
    chunk_strategy="basic",
    chunk_max_characters=1000,
    embed_provider="qwen",
    embed_model_name="text-embedding-v4",
)
docs = loader.load()
```

### Custom stages (advanced)

```python
loader = XParseLoader(
    file_path="doc.pdf",
    stages=[
        {"type": "parse", "config": {"provider": "textin"}},
        {"type": "chunk", "config": {"strategy": "by_page", "max_characters": 800}},
    ],
)
```

### Multiple files

```python
loader = XParseLoader(file_path=["a.pdf", "b.pdf"])
for doc in loader.lazy_load():
    print(doc.metadata.get("source"), doc.page_content[:50])
```

### File-like object

When passing a file-like object instead of a path, you must set `metadata_filename`:

```python
with open("doc.pdf", "rb") as f:
    loader = XParseLoader(file=f, metadata_filename="doc.pdf")
    docs = loader.load()
```

## References

- [xParse overview](https://docs.textin.com/pipeline/overview)
- [Pipeline API](https://docs.textin.com/api-reference/endpoint/pipeline)
