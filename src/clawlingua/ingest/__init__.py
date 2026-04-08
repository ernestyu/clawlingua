"""Input ingestion helpers."""

from .file_reader import read_text_file
from .main_content import extract_main_content
from .normalizer import normalize_text
from .url_fetcher import fetch_url

__all__ = ["fetch_url", "extract_main_content", "normalize_text", "read_text_file"]

