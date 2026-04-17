from __future__ import annotations

from clawlearn.models.card import CardRecord
from clawlearn.pipeline.build_lingua_deck import (
    _build_cloze_text_from_spans,
    _filter_cards_for_export_cloze_format,
    _inject_phrase_hints,
    _normalize_cloze_markup,
)
from clawlearn.pipeline.validators import validate_text_candidate
from clawlearn.utils.text import normalize_for_dedupe


def test_normalize_cloze_markup_is_idempotent_for_double_brace_cloze() -> None:
    text = "X {{c1::<b>abc</b>}}(hint) Y"
    normalized = _normalize_cloze_markup(text)
    assert normalized == text
    assert "{{{c1::" not in normalized


def test_validate_text_candidate_does_not_auto_fix_single_brace_cloze() -> None:
    item = {
        "chunk_id": "chunk_0001",
        "text": "X {c1::<b>abc</b>}(hint) Y",
        "original": "X abc Y",
        "target_phrases": ["abc"],
        "phrase_types": ["strong_collocation"],
        "note_hint": "",
    }
    ok, reason = validate_text_candidate(item, max_sentences=3)
    assert ok is False
    assert reason == "format:missing cloze marker"
    assert item["text"] == "X {c1::<b>abc</b>}(hint) Y"


def test_export_guard_rejects_single_brace_hint_and_triple_brace() -> None:
    valid = CardRecord(
        run_id="run_1",
        card_id="card_valid",
        chunk_id="chunk_0001",
        source_lang="en",
        target_lang="zh",
        text="X {{c1::<b>abc</b>}}(hint) Y",
        original="X abc Y",
        translation="",
        note="",
    )
    invalid_single = CardRecord(
        run_id="run_1",
        card_id="card_single",
        chunk_id="chunk_0002",
        source_lang="en",
        target_lang="zh",
        text="X {abc}(hint) Y",
        original="X abc Y",
        translation="",
        note="",
    )
    invalid_triple = CardRecord(
        run_id="run_1",
        card_id="card_triple",
        chunk_id="chunk_0003",
        source_lang="en",
        target_lang="zh",
        text="X {{{c1::<b>abc</b>}}(hint) Y",
        original="X abc Y",
        translation="",
        note="",
    )
    errors: list[dict] = []
    filtered = _filter_cards_for_export_cloze_format(
        cards=[valid, invalid_single, invalid_triple],
        errors=errors,
    )
    assert [card.card_id for card in filtered] == ["card_valid"]
    assert len(errors) == 2
    assert all(err.get("stage") == "export_cloze_format_guard" for err in errors)
    assert {err.get("card_id") for err in errors} == {"card_single", "card_triple"}


def test_build_cloze_from_spans_and_hint_injection_keeps_canonical_format() -> None:
    sentence = "We made progress today."
    text, phrases = _build_cloze_text_from_spans(
        sentence,
        [{"start": 3, "end": 16, "phrase": "made progress"}],
    )
    assert text == "We {{c1::<b>made progress</b>}}(hint) today."
    assert phrases == ["made progress"]
    hinted = _inject_phrase_hints(
        text=text,
        phrase_to_hint={normalize_for_dedupe("made progress"): "进步表达"},
    )
    assert hinted == "We {{c1::<b>made progress</b>}}(进步表达) today."
    assert "{made progress}(hint)" not in hinted
    assert "{{{c1::" not in hinted
