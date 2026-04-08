"""Card model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CardRecord(BaseModel):
    run_id: str
    card_id: str
    chunk_id: str
    source_lang: str
    target_lang: str
    title: str | None = None
    source_url: str | None = None
    text: str
    original: str
    translation: str
    note: str
    audio_file: str | None = None
    audio_field: str | None = None
    target_phrases: list[str] = Field(default_factory=list)

