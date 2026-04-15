"""Read `.pdf` content and convert it to plain text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ClawLearnError, build_error
from ..exit_codes import ExitCode
from .normalizer import NormalizeOptions, normalize_text


@dataclass(frozen=True)
class PdfReadResult:
    title: str | None
    text: str


def _import_pdf_reader() -> Any:
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            return PdfReader
        except Exception as exc:
            raise build_error(
                error_code="INPUT_PDF_DEPENDENCY_MISSING",
                cause="PDF dependency is missing.",
                detail=str(exc),
                next_steps=["Install `pypdf` (recommended) or `PyPDF2`"],
                exit_code=ExitCode.INPUT_ERROR,
            ) from exc


def _read_pdf_text(path: Path) -> tuple[str | None, list[str]]:
    reader_cls = _import_pdf_reader()
    reader = reader_cls(str(path))
    if getattr(reader, "is_encrypted", False):
        raise build_error(
            error_code="INPUT_PDF_ENCRYPTED",
            cause="PDF is encrypted and cannot be read.",
            detail=f"path={path}",
            next_steps=["Use an unencrypted PDF file"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    title = None
    metadata = getattr(reader, "metadata", None)
    if metadata is not None:
        title_value = getattr(metadata, "title", None)
        if title_value is None and isinstance(metadata, dict):
            title_value = metadata.get("/Title") or metadata.get("Title")
        if title_value:
            title = str(title_value).strip() or None

    pages: list[str] = []
    for page in getattr(reader, "pages", []):
        text = ""
        try:
            extracted = page.extract_text()
            text = str(extracted or "")
        except Exception:
            text = ""
        if text.strip():
            pages.append(text.strip())
    return title, pages


def read_pdf_file(path: Path) -> PdfReadResult:
    if not path.exists() or not path.is_file():
        raise build_error(
            error_code="INPUT_FILE_NOT_FOUND",
            cause="Input file does not exist.",
            detail=f"path={path}",
            next_steps=["Check the file path"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    try:
        title, pages = _read_pdf_text(path)
    except ClawLearnError:
        raise
    except Exception as exc:
        raise build_error(
            error_code="INPUT_PDF_READ_FAILED",
            cause="PDF parsing failed.",
            detail=f"path={path}, reason={exc}",
            next_steps=["Check whether the PDF is valid and readable"],
            exit_code=ExitCode.INPUT_ERROR,
        ) from exc

    merged = "\n\n".join(pages).strip()
    cleaned = normalize_text(
        merged,
        options=NormalizeOptions(short_line_max_words=0, material_profile="prose_article"),
    ) if merged else ""
    if not cleaned:
        raise build_error(
            error_code="INPUT_PDF_EMPTY",
            cause="PDF contains no readable text content.",
            detail=f"path={path}",
            next_steps=["Try another PDF file"],
            exit_code=ExitCode.INPUT_ERROR,
        )
    return PdfReadResult(title=title, text=cleaned)
