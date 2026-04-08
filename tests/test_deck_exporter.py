from pathlib import Path

from clawlingua.anki.deck_exporter import export_apkg
from clawlingua.models.card import CardRecord
from clawlingua.models.template_schema import AnkiTemplateSpec, CardTemplateSpec


def test_genanki_export_success(tmp_path: Path) -> None:
    template = AnkiTemplateSpec(
        model_name="Test Cloze",
        deck_name="Test Deck",
        fields=["Text", "Original", "Translation", "Note", "Audio"],
        card_templates=[CardTemplateSpec(name="Card 1", qfmt="{{cloze:Text}}", afmt="{{cloze:Text}}")],
        css=".card { font-size: 20px; }",
    )
    media = tmp_path / "audio_000001.mp3"
    media.write_bytes(b"fake")
    card = CardRecord(
        run_id="r1",
        card_id="c1",
        chunk_id="k1",
        source_lang="en",
        target_lang="zh",
        text="I {{c1::like}} this.",
        original="I like this.",
        translation="我喜欢这个。",
        note="note",
        audio_file=media.name,
        audio_field=f"[sound:{media.name}]",
    )
    output = tmp_path / "out.apkg"
    path = export_apkg(
        cards=[card],
        template=template,
        output_path=output,
        media_files=[media],
    )
    assert path.exists()
    assert path.suffix == ".apkg"

