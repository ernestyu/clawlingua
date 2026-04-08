"""Anki template JSON schema model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..constants import ANKI_FIELDS_ORDER


class CardTemplateSpec(BaseModel):
    name: str
    qfmt: str
    afmt: str


class AnkiTemplateSpec(BaseModel):
    model_name: str
    deck_name: str
    fields: list[str]
    card_templates: list[CardTemplateSpec] = Field(default_factory=list)
    css: str = ""

    def validate_field_order(self) -> None:
        if self.fields != ANKI_FIELDS_ORDER:
            raise ValueError(
                f"fields must equal {ANKI_FIELDS_ORDER}, got {self.fields}"
            )

