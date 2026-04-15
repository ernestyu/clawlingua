"""Core chunking helpers shared by pipeline domains."""

from __future__ import annotations

from typing import Any

from ..chunking.splitter import split_into_chunks


def chunk_document(
    *,
    run_id: str,
    text: str,
    max_chars: int,
    min_chars: int,
    overlap_sentences: int,
    material_profile: str,
    difficulty: str,
) -> list[Any]:
    return split_into_chunks(
        run_id=run_id,
        text=text,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_sentences=overlap_sentences,
        material_profile=material_profile,
        difficulty=difficulty,
    )

