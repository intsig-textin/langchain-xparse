"""LangChain integration with xParse Pipeline API."""

from importlib import metadata

from langchain_xparse.client import XParseAPIError
from langchain_xparse.document_loaders import XParseLoader

try:
    __version__ = metadata.version("langchain-xparse")
except metadata.PackageNotFoundError:
    try:
        __version__ = metadata.version(__package__ or "langchain_xparse")
    except metadata.PackageNotFoundError:
        __version__ = ""

__all__ = [
    "XParseAPIError",
    "XParseLoader",
    "__version__",
]
