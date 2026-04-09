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
        import pymupdf as fitz  # type: ignore[import-not-found]

        return fitz
    except Exception as exc:
        try:
            import fitz  # type: ignore[import-not-found]

            return fitz
        except Exception:
            raise build_error(
                error_code="DEPENDENCY_PDF_MISSING",
                cause="PDF input requires `PyMuPDF`.",
                detail=str(exc),
                next_steps=["Install dependencies with `pip install -r requirements.txt`"],
                exit_code=ExitCode.CONFIG_ERROR,
            ) from exc


def _extract_pdf_title(metadata: Any) -> str | None:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        raw = metadata.get("title") or metadata.get("/Title")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    title = getattr(metadata, "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _extract_page_text(page: Any) -> str:
    try:
        blocks = page.get_text("blocks", sort=True) or []
    except TypeError:
        blocks = page.get_text("blocks") or []

    ordered_blocks: list[tuple[float, float, str]] = []
    for block in blocks:
        if not isinstance(block, (tuple, list)) or len(block) < 5:
            continue
        text = str(block[4] or "").strip()
        if not text:
            continue
        try:
            y0 = float(block[1])
            x0 = float(block[0])
        except Exception:
            y0 = 0.0
            x0 = 0.0
        ordered_blocks.append((y0, x0, text))

    if not ordered_blocks:
        plain = page.get_text("text") or ""
        return plain.strip()

    ordered_blocks.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(item[2] for item in ordered_blocks).strip()


def _read_with_legacy_reader(reader_cls: Any, path: Path) -> PdfReadResult:
    reader = reader_cls(str(path))
    if getattr(reader, "is_encrypted", False):
        decrypt = getattr(reader, "decrypt", None)
        decrypt_ok = bool(decrypt("")) if callable(decrypt) else False
        if not decrypt_ok and getattr(reader, "is_encrypted", False):
            raise build_error(
                error_code="INPUT_PDF_ENCRYPTED",
                cause="PDF is encrypted and cannot be read without a password.",
                detail=f"path={path}",
                next_steps=["Use a non-encrypted PDF or remove PDF password protection first"],
                exit_code=ExitCode.INPUT_ERROR,
            )
    pages = getattr(reader, "pages", [])
    chunks: list[str] = []
    for index, page in enumerate(pages, start=1):
        extractor = getattr(page, "extract_text", None)
        if not callable(extractor):
            continue
        try:
            text = extractor() or ""
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


def read_pdf_file(path: Path) -> PdfReadResult:
    if not path.exists() or not path.is_file():
        raise build_error(
            error_code="INPUT_FILE_NOT_FOUND",
            cause="Input file does not exist.",
            detail=f"path={path}",
            next_steps=["Check the file path"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    fitz = _import_pdf_reader()
    try:
        if not hasattr(fitz, "open"):
            return _read_with_legacy_reader(fitz, path)

        with fitz.open(str(path)) as doc:
            if getattr(doc, "needs_pass", False):
                auth_ok = bool(doc.authenticate(""))
                if not auth_ok and getattr(doc, "needs_pass", False):
                    raise build_error(
                        error_code="INPUT_PDF_ENCRYPTED",
                        cause="PDF is encrypted and cannot be read without a password.",
                        detail=f"path={path}",
                        next_steps=["Use a non-encrypted PDF or remove PDF password protection first"],
                        exit_code=ExitCode.INPUT_ERROR,
                    )

            chunks: list[str] = []
            for index in range(doc.page_count):
                page_number = index + 1
                page = doc.load_page(index)
                try:
                    text = _extract_page_text(page)
                except Exception as exc:
                    raise build_error(
                        error_code="INPUT_PDF_PAGE_READ_FAILED",
                        cause="Failed to extract text from PDF page.",
                        detail=f"path={path}, page={page_number}, reason={exc}",
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
            title = _extract_pdf_title(getattr(doc, "metadata", None))
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

    return PdfReadResult(title=title, text=merged)
