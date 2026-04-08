from clawlingua.anki.note_builder import build_note_fields
from clawlingua.models.card import CardRecord


def test_build_note_fields_order() -> None:
    card = CardRecord(
        run_id="r",
        card_id="c",
        chunk_id="k",
        source_lang="en",
        target_lang="zh",
        text="{{c1::hello}} world",
        original="hello world",
        translation="你好世界",
        note="test note",
        audio_field="[sound:a.mp3]",
    )
    fields = build_note_fields(card)
    assert fields == [
        "{{c1::hello}} world",
        "hello world",
        "你好世界",
        "test note",
        "[sound:a.mp3]",
    ]

