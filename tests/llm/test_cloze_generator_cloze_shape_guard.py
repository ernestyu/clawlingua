from __future__ import annotations

import pytest

from clawlearn.errors import ClawLearnError
from clawlearn.llm.cloze_generator import _extract_cloze_candidates_from_items


def test_extract_cloze_candidates_raises_on_phrase_shape_mismatch() -> None:
    data = [
        {
            "chunk_id": "chunk_0001",
            "context_sentences": ["We made progress.", "It helped."],
            "phrases": [{"text": "made progress"}],
        }
    ]
    with pytest.raises(ClawLearnError) as exc_info:
        _extract_cloze_candidates_from_items(data=data)
    assert exc_info.value.error_code == "LLM_RESPONSE_SHAPE_INVALID"


def test_extract_cloze_candidates_skips_malformed_items() -> None:
    data = [
        {
            "chunk_id": "chunk_0001",
            "text": "Valid {{c1::<b>phrase</b>}}(hint).",
            "original": "Valid phrase.",
            "target_phrases": ["phrase"],
        },
        {
            "chunk_id": "chunk_0002",
            "text": "Missing original {{c1::<b>x</b>}}(hint).",
            "target_phrases": ["x"],
        },
        {
            "chunk_id": "chunk_0003",
            "text": "Missing phrases {{c1::<b>x</b>}}(hint).",
            "original": "Missing phrases x.",
            "target_phrases": [],
        },
    ]
    candidates = _extract_cloze_candidates_from_items(data=data)
    assert candidates == [
        {
            "chunk_id": "chunk_0001",
            "text": "Valid {{c1::<b>phrase</b>}}(hint).",
            "original": "Valid phrase.",
            "target_phrases": ["phrase"],
            "note_hint": "",
        }
    ]


def test_extract_cloze_candidates_forced_chunk_id_is_applied() -> None:
    data = [
        {
            "chunk_id": "wrong_id",
            "text": "A {{c1::<b>target phrase</b>}}(hint).",
            "original": "A target phrase.",
            "target_phrases": ["target phrase"],
        }
    ]
    candidates = _extract_cloze_candidates_from_items(
        data=data,
        forced_chunk_id="chunk_9999",
    )
    assert candidates[0]["chunk_id"] == "chunk_9999"
