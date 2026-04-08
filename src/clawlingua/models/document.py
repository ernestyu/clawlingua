"""Document model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    run_id: str
    source_type: Literal["url", "file"]
    source_value: str
    source_lang: str
    target_lang: str
    title: str | None = None
    source_url: str | None = None
    raw_text: str
    cleaned_text: str
    cleaned_markdown: str | None = None
    fetched_at: str | None = None
    metadata: dict = Field(default_factory=dict)

