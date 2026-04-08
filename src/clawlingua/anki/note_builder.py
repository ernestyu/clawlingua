"""Build ordered Anki note fields."""

from __future__ import annotations

from ..models.card import CardRecord


def build_note_fields(card: CardRecord) -> list[str]:
    return [
        card.text,
        card.original,
        card.translation,
        card.note,
        card.audio_field or "",
    ]

