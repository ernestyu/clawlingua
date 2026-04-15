"""Chunk model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    run_id: str
    chunk_id: str
    chunk_index: int
    source_text: str
    char_count: int
    sentence_count: int
    metadata: dict = Field(default_factory=dict)

