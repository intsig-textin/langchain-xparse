# langchain-xparse

LangChain integration with [xParse Parse API](https://docs.textin.com/api-reference/endpoint/xparse/v1/parse-sync) for intelligent document parsing. Converts unstructured documents (PDF, images, Word, Excel, PPT, etc.) into AI-friendly structured data (JSON, Markdown) with rich metadata.

## Installation

From PyPI:

```bash
pip install langchain-xparse
```

## Configuration

Set your TextIn credentials (from [Textin Workspace](https://www.textin.com/console/dashboard/setting)):

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

### Basic Usage

```python
from langchain_xparse import XParseLoader

loader = XParseLoader(file_path="example.pdf")
docs = loader.load()
print(docs[0].page_content[:200])
print(docs[0].metadata)  # source, category, element_id, filename, page_number
```

### Lazy Load

```python
for doc in loader.lazy_load():
    # process each document
    print(doc.page_content[:100])
```

### Async Load

```python
async for doc in loader.alazy_load():
    # process each document asynchronously
    print(doc.page_content[:100])
```

### Custom Parse Configuration

Customize parsing behavior using the `config` parameter. See [Parse Config Documentation](https://docs.textin.com/xparse/v1/parse-config) for details.

```python
loader = XParseLoader(
    file_path="doc.pdf",
    config={
        "document": {
            "password": "pdf-password"  # For encrypted PDFs
        },
        "capabilities": {
            "include_hierarchy": True,         # Include parent-child relationships
            "include_inline_objects": True,    # Extract formulas, handwriting, etc.
            "include_table_structure": True,   # Detailed table structure
            "include_char_details": True,      # Character-level details
            "include_image_data": True,        # Image URLs and data
            "pages": True,                     # Page metadata
            "title_tree": True,                # Document outline/TOC
            "table_view": "html"               # Table format: "html" or "markdown"
        },
        "scope": {
            "page_range": "1-10"               # Process specific pages
        },
        "config": {
            "force_engine": "textin",          # Engine selection (expert mode)
            "engine_params": {
                "formula_level": 0,
                "image_output_type": "url"
            }
        }
    }
)
docs = loader.load()
```

### Multiple Files

```python
loader = XParseLoader(file_path=["a.pdf", "b.pdf", "c.docx"])
for doc in loader.lazy_load():
    print(f"{doc.metadata.get('source')}: {doc.page_content[:50]}")
```

### File-like Object

When passing a file-like object instead of a path, you must set `metadata_filename`:

```python
with open("doc.pdf", "rb") as f:
    loader = XParseLoader(file=f, metadata_filename="doc.pdf")
    docs = loader.load()
```

## Document Metadata

Each loaded document includes rich metadata:

- `source`: File path or filename
- `category`: Element type (Title, NarrativeText, Table, Image, Formula, etc.)
- `element_id`: Unique element identifier
- `filename`: Original filename
- `page_number`: Page number (if available)
- `parent_id`: Parent element ID (with `include_hierarchy`)
- `children_ids`: Child element IDs (with `include_hierarchy`)
- Additional element-specific metadata

## References

- [xParse Parse API](https://docs.textin.com/api-reference/endpoint/xparse/v1/parse-sync) - API endpoint documentation
- [Parse Config](https://docs.textin.com/xparse/v1/parse-config) - Configuration parameters
- [Parse Response](https://docs.textin.com/xparse/v1/parse-response) - Response structure and fields
