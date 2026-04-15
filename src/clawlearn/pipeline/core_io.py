"""Core IO helpers shared by pipeline domains."""

from __future__ import annotations

from pathlib import Path


def resolve_input_path(*, workspace_root: Path, input_value: str) -> Path:
    file_path = Path(input_value)
    if file_path.is_absolute():
        return file_path
    return (workspace_root / file_path).resolve()

