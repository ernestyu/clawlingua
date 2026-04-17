from __future__ import annotations

from clawlearn.pipeline.validators import validate_text_candidate


def _base_item(*, phrase: str = "made progress", phrase_types: list[str] | None = None) -> dict:
    return {
        "chunk_id": "chunk_0001",
        "text": f"During the weekly review, we {{{{c1::<b>{phrase}</b>}}}}(hint) across three teams today.",
        "original": f"During the weekly review, we {phrase} across three teams today.",
        "target_phrases": [phrase],
        "phrase_types": list(phrase_types or ["strong_collocation"]),
        "note_hint": "",
    }


def test_phrase_types_too_many_labels_is_tagged_as_taxonomy() -> None:
    item = _base_item(phrase_types=["strong_collocation", "phrasal_verb", "stance_positioning"])
    ok, reason = validate_text_candidate(item, max_sentences=3)
    assert ok is False
    assert reason.startswith("taxonomy:too_many_labels:")


def test_phrase_types_invalid_labels_is_tagged_as_taxonomy() -> None:
    item = _base_item(phrase_types=["not_in_taxonomy"])
    ok, reason = validate_text_candidate(item, max_sentences=3)
    assert ok is False
    assert reason.startswith("taxonomy:invalid_labels:")


def test_phrase_types_invalid_combo_is_tagged_as_taxonomy() -> None:
    item = {
        "chunk_id": "chunk_0001",
        "text": "Now {{c1::<b>now</b>}}(hint).",
        "original": "Now now.",
        "target_phrases": ["now"],
        "phrase_types": ["stance_positioning", "concession_contrast"],
        "note_hint": "",
    }
    ok, reason = validate_text_candidate(item, max_sentences=3)
    assert ok is False
    assert reason.startswith("taxonomy:invalid_combo:")


def test_advanced_taxonomy_support_missing_is_tagged_as_taxonomy() -> None:
    item = _base_item(phrase="technology", phrase_types=["phrasal_verb"])
    ok, reason = validate_text_candidate(
        item,
        max_sentences=3,
        difficulty="advanced",
        learning_mode="lingua_expression",
    )
    assert ok is False
    assert reason.startswith("taxonomy:advanced_support_missing:")


def test_non_taxonomy_reason_is_unchanged() -> None:
    item = _base_item()
    item["text"] = "During the weekly review, we made progress across three teams today."
    item["target_phrases"] = []
    ok, reason = validate_text_candidate(item, max_sentences=3)
    assert ok is False
    assert reason == "format:missing cloze marker"
