"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_lines(path: Path, lines: Iterable[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

