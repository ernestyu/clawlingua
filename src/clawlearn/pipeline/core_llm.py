"""Core LLM helpers shared by pipeline domains."""

from __future__ import annotations

from typing import Any


def iter_batches(items: list[Any], size: int) -> list[list[Any]]:
    chunk_size = max(1, int(size or 1))
    if chunk_size <= 1:
        return [[item] for item in items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

