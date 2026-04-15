"""Upload file materialization utilities for the web run service."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any


def safe_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe or "input"


def materialize_uploaded_file(uploaded_file: Any, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(uploaded_file, (str, Path)):
        src = Path(uploaded_file)
        if not src.exists():
            raise ValueError(f"Uploaded file path does not exist: {src}")
        dst = tmp_dir / f"{safe_stem(src.stem)}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst

    if isinstance(uploaded_file, dict):
        src_path = uploaded_file.get("path") or uploaded_file.get("name")
        if src_path:
            src = Path(str(src_path))
            if src.exists():
                dst = tmp_dir / f"{safe_stem(src.stem)}{src.suffix}"
                shutil.copyfile(src, dst)
                return dst

    name = getattr(uploaded_file, "name", "input.txt")
    suffix = Path(str(name)).suffix
    stem = Path(str(name)).stem
    dst = tmp_dir / f"{safe_stem(stem)}{suffix}"
    if not hasattr(uploaded_file, "read"):
        raise ValueError("Unsupported uploaded file payload.")
    data = uploaded_file.read()
    if isinstance(data, str):
        dst.write_text(data, encoding="utf-8")
    else:
        dst.write_bytes(data)
    return dst
