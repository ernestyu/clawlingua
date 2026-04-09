"""Read `.pdf` content and convert it to plain text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ClawLinguaError, build_error
from ..exit_codes import ExitCode


@dataclass(frozen=True)
class PdfReadResult:
    title: str | None
    text: str


def _import_pdf_reader() -> Any:
    try:
        from pypdf import PdfReader

        return PdfReader
    except Exception as exc:
        raise build_error(
            error_code="DEPENDENCY_PDF_MISSING",
            cause="PDF input requires `pypdf`.",
            detail=str(exc),
            next_steps=["Install dependencies with `pip install -r requirements.txt`"],
            exit_code=ExitCode.CONFIG_ERROR,
        ) from exc


def _extract_pdf_title(metadata: Any) -> str | None:
    if metadata is None:
        return None

    title = getattr(metadata, "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()

    if isinstance(metadata, dict):
        raw = metadata.get("/Title")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def read_pdf_file(path: Path) -> PdfReadResult:
    if not path.exists() or not path.is_file():
        raise build_error(
            error_code="INPUT_FILE_NOT_FOUND",
            cause="Input file does not exist.",
            detail=f"path={path}",
            next_steps=["Check the file path"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    PdfReader = _import_pdf_reader()
    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            # Empty password decrypt attempt handles "encrypted but no password" PDFs.
            decrypt_status = reader.decrypt("")
            if not decrypt_status and getattr(reader, "is_encrypted", False):
                raise build_error(
                    error_code="INPUT_PDF_ENCRYPTED",
                    cause="PDF is encrypted and cannot be read without a password.",
                    detail=f"path={path}",
                    next_steps=["Use a non-encrypted PDF or remove PDF password protection first"],
                    exit_code=ExitCode.INPUT_ERROR,
                )
    except ClawLinguaError:
        raise
    except Exception as exc:
        raise build_error(
            error_code="INPUT_PDF_READ_FAILED",
            cause="PDF parsing failed.",
            detail=f"path={path}, reason={exc}",
            next_steps=["Check whether the PDF file is valid and readable"],
            exit_code=ExitCode.INPUT_ERROR,
        ) from exc

    pages = getattr(reader, "pages", [])
    chunks: list[str] = []
    for index, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise build_error(
                error_code="INPUT_PDF_PAGE_READ_FAILED",
                cause="Failed to extract text from PDF page.",
                detail=f"path={path}, page={index}, reason={exc}",
                next_steps=["Try another PDF or pre-convert it to plain text"],
                exit_code=ExitCode.INPUT_ERROR,
            ) from exc
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if normalized:
            chunks.append(normalized)

    merged = "\n\n".join(chunks).strip()
    if not merged:
        raise build_error(
            error_code="INPUT_PDF_EMPTY",
            cause="PDF contains no readable text content.",
            detail=f"path={path}",
            next_steps=["Try another PDF file or OCR the document first"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    title = _extract_pdf_title(getattr(reader, "metadata", None))
    return PdfReadResult(title=title, text=merged)
