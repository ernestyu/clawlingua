from __future__ import annotations

from clawlearn.llm.cloze_generator import _extract_phrase_candidates_from_items


def test_extract_phrase_candidates_stage1_lite_ignores_reason_and_types() -> None:
    data = [
        {
            "chunk_id": "chunk_0001",
            "context_sentences": ["We made progress today.", "It helped a lot."],
            "phrases": [
                {
                    "text": "made progress",
                    "reason": "good phrase",
                    "phrase_types": ["strong_collocation"],
                }
            ],
        }
    ]
    candidates = _extract_phrase_candidates_from_items(data=data)
    assert candidates == [
        {
            "chunk_id": "chunk_0001",
            "sentence_text": "We made progress today. It helped a lot.",
            "phrase_text": "made progress",
        }
    ]


def test_extract_phrase_candidates_stage1_filters_invalid_shapes_and_rules() -> None:
    data = [
        {
            "chunk_id": "chunk_0002",
            "sentence": "We made progress today and moved forward.",
            "phrases": [
                {"text": "progress"},  # too short
                {"text": "that made progress"},  # invalid lead token
                {"text": "made, progress"},  # forbidden punctuation
                {"text": "moved forward"},  # valid
            ],
        }
    ]
    candidates = _extract_phrase_candidates_from_items(data=data)
    assert candidates == [
        {
            "chunk_id": "chunk_0002",
            "sentence_text": "We made progress today and moved forward.",
            "phrase_text": "moved forward",
        }
    ]


def test_extract_phrase_candidates_stage1_requires_substring_match() -> None:
    data = [
        {
            "chunk_id": "chunk_0003",
            "sentence": "I totally agree with that point.",
            "phrases": [{"text": "agree with"}, {"text": "strong disagreement"}],
        }
    ]
    candidates = _extract_phrase_candidates_from_items(data=data)
    assert candidates == [
        {
            "chunk_id": "chunk_0003",
            "sentence_text": "I totally agree with that point.",
            "phrase_text": "agree with",
        }
    ]


def test_extract_phrase_candidates_stage1_enforces_word_count_upper_bound() -> None:
    data = [
        {
            "chunk_id": "chunk_0004",
            "sentence": "We keep this phrase useful and easy to remember in practice.",
            "phrases": [
                {"text": "phrase useful"},
                {"text": "this phrase useful and easy to remember"},
            ],
        }
    ]
    candidates = _extract_phrase_candidates_from_items(data=data)
    assert candidates == [
        {
            "chunk_id": "chunk_0004",
            "sentence_text": "We keep this phrase useful and easy to remember in practice.",
            "phrase_text": "phrase useful",
        }
    ]
