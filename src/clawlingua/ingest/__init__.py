"""Input ingestion helpers."""

from .file_reader import read_text_file
from .normalizer import normalize_text, strip_markdown_to_text

__all__ = ["normalize_text", "strip_markdown_to_text", "read_text_file"]
