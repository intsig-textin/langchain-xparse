"""LangChain integration with xParse Parse API."""

from importlib import metadata

from langchain_xparse.client import ParseClient, XParseAPIError
from langchain_xparse.document_loaders import XParseLoader

try:
    __version__ = metadata.version("langchain-xparse")
except metadata.PackageNotFoundError:
    try:
        __version__ = metadata.version(__package__ or "langchain_xparse")
    except metadata.PackageNotFoundError:
        __version__ = ""

__all__ = [
    "ParseClient",
    "XParseAPIError",
    "XParseLoader",
    "__version__",
]
