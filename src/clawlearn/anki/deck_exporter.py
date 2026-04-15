"""Export cards as Anki .apkg."""

from __future__ import annotations

from pathlib import Path

import genanki

from ..errors import build_error
from ..exit_codes import ExitCode
from ..models.card import CardRecord
from ..models.template_schema import AnkiTemplateSpec
from ..utils.hash import stable_int_id
from .note_builder import build_note_fields


def export_apkg(
    *,
    cards: list[CardRecord],
    template: AnkiTemplateSpec,
    output_path: Path,
    media_files: list[Path],
    deck_name_override: str | None = None,
) -> Path:
    try:
        model = genanki.Model(
            stable_int_id(template.model_name),
            template.model_name,
            fields=[{"name": name} for name in template.fields],
            templates=[
                {"name": t.name, "qfmt": t.qfmt, "afmt": t.afmt}
                for t in template.card_templates
            ],
            css=template.css,
            model_type=genanki.Model.CLOZE,
        )
        deck_name = deck_name_override or template.deck_name
        deck = genanki.Deck(stable_int_id(deck_name), deck_name)
        for card in cards:
            note = genanki.Note(model=model, fields=build_note_fields(card))
            deck.add_note(note)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        package = genanki.Package(deck)
        package.media_files = [str(path) for path in media_files]
        package.write_to_file(str(output_path))
        return output_path
    except Exception as exc:
        raise build_error(
            error_code="ANKI_EXPORT_FAILED",
            cause="Anki 导出失败。",
            detail=str(exc),
            next_steps=["检查模板字段顺序与卡片字段是否一致"],
            exit_code=ExitCode.ANKI_EXPORT_ERROR,
        ) from exc

