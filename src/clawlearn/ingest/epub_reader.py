"""Read `.epub` content and convert it to plain text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

from ..errors import ClawLearnError, build_error
from ..exit_codes import ExitCode
from .normalizer import strip_html_to_text


@dataclass(frozen=True)
class EpubReadResult:
    title: str | None
    text: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _attr(element: ET.Element, name: str) -> str | None:
    if name in element.attrib:
        return element.attrib[name]
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1] == name:
            return value
    return None


def _read_zip_text(zf: zipfile.ZipFile, inner_path: str) -> str:
    raw = zf.read(inner_path)
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _resolve_inner_path(base_file: str, relative_path: str) -> str:
    base_dir = PurePosixPath(base_file).parent
    return str((base_dir / relative_path).as_posix())


def _discover_opf_path(zf: zipfile.ZipFile) -> str:
    try:
        container_xml = _read_zip_text(zf, "META-INF/container.xml")
        root = ET.fromstring(container_xml)
        for elem in root.iter():
            if _local_name(elem.tag) == "rootfile":
                full_path = _attr(elem, "full-path")
                if full_path and full_path in zf.namelist():
                    return full_path
    except Exception:
        pass

    for name in sorted(zf.namelist()):
        if name.lower().endswith(".opf"):
            return name

    raise build_error(
        error_code="INPUT_EPUB_INVALID",
        cause="EPUB parsing failed.",
        detail="content.opf not found in EPUB archive.",
        next_steps=["Check whether the .epub file is valid"],
        exit_code=ExitCode.INPUT_ERROR,
    )


def _extract_title(opf_root: ET.Element) -> str | None:
    for elem in opf_root.iter():
        if _local_name(elem.tag) == "title":
            value = (elem.text or "").strip()
            if value:
                return value
    return None


def _extract_manifest(opf_root: ET.Element) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for elem in opf_root.iter():
        if _local_name(elem.tag) != "item":
            continue
        item_id = (_attr(elem, "id") or "").strip()
        href = (_attr(elem, "href") or "").strip()
        media_type = (_attr(elem, "media-type") or "").strip().lower()
        if not item_id or not href:
            continue
        # Keep textual XHTML/HTML resources only.
        if media_type and "html" not in media_type and "xhtml" not in media_type:
            if not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
        manifest[item_id] = href
    return manifest


def _extract_spine_ids(opf_root: ET.Element) -> list[str]:
    ids: list[str] = []
    for elem in opf_root.iter():
        if _local_name(elem.tag) != "itemref":
            continue
        idref = (_attr(elem, "idref") or "").strip()
        if idref:
            ids.append(idref)
    return ids


def read_epub_file(path: Path) -> EpubReadResult:
    if not path.exists() or not path.is_file():
        raise build_error(
            error_code="INPUT_FILE_NOT_FOUND",
            cause="Input file does not exist.",
            detail=f"path={path}",
            next_steps=["Check the file path"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    try:
        with zipfile.ZipFile(path, "r") as zf:
            opf_path = _discover_opf_path(zf)
            opf_text = _read_zip_text(zf, opf_path)
            opf_root = ET.fromstring(opf_text)

            title = _extract_title(opf_root)
            manifest = _extract_manifest(opf_root)
            spine_ids = _extract_spine_ids(opf_root)

            sections: list[str] = []
            for item_id in spine_ids:
                href = manifest.get(item_id)
                if not href:
                    continue
                inner_path = _resolve_inner_path(opf_path, href)
                if inner_path not in zf.namelist():
                    continue
                chapter_text = _read_zip_text(zf, inner_path)
                plain = strip_html_to_text(chapter_text)
                if plain:
                    sections.append(plain)

            if not sections:
                # Fallback: read all manifest text files in deterministic order.
                for _, href in sorted(manifest.items(), key=lambda item: item[1]):
                    inner_path = _resolve_inner_path(opf_path, href)
                    if inner_path not in zf.namelist():
                        continue
                    chapter_text = _read_zip_text(zf, inner_path)
                    plain = strip_html_to_text(chapter_text)
                    if plain:
                        sections.append(plain)
    except ClawLearnError:
        raise
    except Exception as exc:
        raise build_error(
            error_code="INPUT_EPUB_READ_FAILED",
            cause="EPUB parsing failed.",
            detail=f"path={path}, reason={exc}",
            next_steps=["Check whether the EPUB file is valid and not encrypted"],
            exit_code=ExitCode.INPUT_ERROR,
        ) from exc

    merged = "\n\n".join(part for part in sections if part.strip()).strip()
    if not merged:
        raise build_error(
            error_code="INPUT_EPUB_EMPTY",
            cause="EPUB contains no readable text content.",
            detail=f"path={path}",
            next_steps=["Try another EPUB file or convert EPUB to plain text first"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    return EpubReadResult(title=title, text=merged)
