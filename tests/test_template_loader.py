from pathlib import Path

from clawlingua.anki.template_loader import load_anki_template


def test_load_template_schema() -> None:
    template = load_anki_template(Path("templates/anki_cloze_default.json"))
    assert template.fields == ["Text", "Original", "Translation", "Note", "Audio"]
    assert len(template.card_templates) == 1

